"""
Portfolio layer — treat the plans as ONE diversified book, not isolated bets.

Any single timing plan lags buy&hold on a trending stock (see WFO). But a book of
many *uncorrelated* risk-managed sleeves is a different animal: the drawdowns
don't line up, so the combined equity curve can have far better risk-adjusted
returns (Sharpe) and much smaller drawdowns than any one plan — the actual case
for running this system.

This builds an equal-risk portfolio over the out-of-sample period:
  * each selected plan is one sleeve (equal weight across sleeves that exist that day),
  * daily portfolio return = mean of the active sleeves' daily strategy returns,
  * compared against an equal-weight buy&hold of the same tickers.

Reports Sharpe / CAGR / max-drawdown / average pairwise correlation, and saves the
equity curves for the dashboard. Runs for the high-confidence and recommended sets.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

import config
import dataio
from strategies.registry import STRATEGIES
from backtest import engine2

ANN = 252


def plan_daily_returns(ticker, strategy_name, params_str, exit_policy,
                       interval="1d", is_fraction=config.IS_FRACTION):
    """Daily strategy returns (0 when flat) over the OOS slice, and the ticker's daily returns."""
    strat = STRATEGIES.get(strategy_name)
    data = dataio.load(ticker, interval)
    if strat is None or data is None:
        return None, None
    df = engine2.clean_ohlcv(data)
    cut = int(len(df) * is_fraction)
    oos = df.iloc[cut:]
    if len(oos) < 30:
        return None, None
    params = engine2.parse_params(params_str)
    entries, exits = engine2.build_signals(strat, oos, params)
    cfg = engine2.EXIT_CONFIGS.get(exit_policy, {})
    pf = engine2._portfolio(oos, entries, exits, cfg, interval)
    try:
        sr = pf.returns()
    except Exception:
        return None, None
    bh = oos["Close"].pct_change().fillna(0.0)
    sr.index = pd.to_datetime(sr.index).tz_localize(None)
    bh.index = pd.to_datetime(bh.index).tz_localize(None)
    return sr, bh


def _metrics(daily: pd.Series) -> dict:
    daily = daily.dropna()
    if len(daily) < 20:
        return {}
    eq = (1 + daily).cumprod()
    cagr = float(eq.iloc[-1] ** (ANN / len(daily)) - 1)
    sharpe = float(daily.mean() / daily.std() * np.sqrt(ANN)) if daily.std() > 0 else np.nan
    dd = float((eq / eq.cummax() - 1).min())
    total = float(eq.iloc[-1] - 1)
    return {"cagr": cagr, "sharpe": sharpe, "max_drawdown": dd, "total_return": total,
            "n_days": int(len(daily))}


def build_portfolio(plans_df: pd.DataFrame, label: str) -> dict:
    strat_cols, bh_cols = {}, {}
    for _, p in plans_df.iterrows():
        sr, bh = plan_daily_returns(p["ticker"], p["strategy"], p["params"], p["exit_policy"])
        if sr is not None and len(sr) > 20:
            strat_cols[p["ticker"]] = sr
            bh_cols[p["ticker"]] = bh
    if not strat_cols:
        return {}
    R = pd.DataFrame(strat_cols).sort_index()
    B = pd.DataFrame(bh_cols).sort_index()

    port = R.mean(axis=1, skipna=True)          # equal weight across existing sleeves
    bench = B.mean(axis=1, skipna=True)          # equal-weight buy&hold of same names

    # average pairwise correlation of the sleeves (lower = better diversification)
    corr = R.corr()
    iu = np.triu_indices_from(corr.values, k=1)
    avg_corr = float(np.nanmean(corr.values[iu])) if len(iu[0]) else np.nan

    port = port.dropna()
    eq = (1 + port).cumprod()
    bench_eq = (1 + bench.reindex(port.index).fillna(0)).cumprod()
    # downsample to ~250 points to keep the embedded chart light
    step = max(1, len(eq) // 250)
    idx = list(range(0, len(eq), step))
    out = {
        "label": label, "n_sleeves": R.shape[1], "avg_correlation": avg_corr,
        "strategy": _metrics(port), "buy_hold": _metrics(bench),
        "equity_dates": [eq.index[i].strftime("%Y-%m-%d") for i in idx],
        "equity_values": [round(float(eq.iloc[i]), 4) for i in idx],
        "bench_values": [round(float(bench_eq.iloc[i]), 4) for i in idx],
    }
    return out


def save(results: list[dict]):
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS portfolio (label TEXT PRIMARY KEY, json TEXT)")
        for r in results:
            if r:
                con.execute("INSERT OR REPLACE INTO portfolio VALUES (?,?)", (r["label"], json.dumps(r)))
        con.commit()
    finally:
        con.close()


def load_plan_sets(interval="1d"):
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        plans = pd.read_sql("SELECT * FROM plans", con)
        try:
            rob = pd.read_sql("SELECT ticker, strategy, exit_policy, psr, mc_p_profit, mc_p_dd_gt_30 FROM robustness", con)
        except Exception:
            rob = pd.DataFrame()
        try:
            liq = pd.read_sql("SELECT ticker, liquidity_tier FROM liquidity", con)
        except Exception:
            liq = pd.DataFrame()
    finally:
        con.close()
    if "interval" in plans.columns:
        plans = plans[plans["interval"] == interval]
    if not rob.empty:
        plans = plans.merge(rob, on=["ticker", "strategy", "exit_policy"], how="left")
    if not liq.empty:
        plans = plans.merge(liq, on="ticker", how="left")
    rec = plans[plans["recommended"] == 1]
    hc = rec
    if {"psr", "mc_p_profit", "mc_p_dd_gt_30", "liquidity_tier"}.issubset(plans.columns):
        hc = rec[(rec["psr"] > 0.90) & (rec["mc_p_profit"] > 0.75)
                 & (rec["mc_p_dd_gt_30"] < 0.25) & rec["liquidity_tier"].isin(["liquid", "ok"])]
    return hc, rec


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    args = ap.parse_args(argv)
    hc, rec = load_plan_sets(args.interval)

    results = []
    for label, dfp in [("high_confidence", hc), ("recommended", rec)]:
        r = build_portfolio(dfp, label)
        if r:
            results.append(r)
    save(results)

    for r in results:
        s, b = r["strategy"], r["buy_hold"]
        print(f"\n=== {r['label'].upper()} portfolio — {r['n_sleeves']} sleeves, "
              f"avg correlation {r['avg_correlation']:.2f} ===")
        print(f"{'':14}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'Total':>10}")
        print(f"{'strategy book':14}{s['cagr']:>8.1%}{s['sharpe']:>9.2f}{s['max_drawdown']:>8.1%}{s['total_return']:>9.1%}")
        print(f"{'equal-wt B&H':14}{b['cagr']:>8.1%}{b['sharpe']:>9.2f}{b['max_drawdown']:>8.1%}{b['total_return']:>9.1%}")
        edge = "BETTER" if s["sharpe"] > b["sharpe"] else "worse"
        print(f"  -> risk-adjusted (Sharpe) is {edge}; drawdown "
              f"{'smaller' if s['max_drawdown'] > b['max_drawdown'] else 'larger'} than buy&hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
