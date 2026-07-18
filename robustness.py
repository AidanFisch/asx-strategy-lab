"""
Tier-1 robustness pass: stress-test every trading plan so we can tell a real
edge from a lucky backtest.

For each plan (chosen on in-sample, evaluated on its out-of-sample holdout) we compute:

  * Monte Carlo bootstrap — resample the plan's trade returns thousands of times
    to get a RANGE of outcomes instead of one number: 5th/50th/95th percentile
    return, the drawdown you'd see in a bad ordering (95th-pct drawdown),
    probability of profit, and probability of a >30% drawdown.
  * Per-year / regime breakdown — the plan's return in each calendar year of the
    holdout, plus worst year and the COVID-crash window. A plan that only works
    in one year is fragile.
  * Probabilistic Sharpe Ratio (PSR) — given the track record length and the
    shape of the return distribution, the probability the TRUE Sharpe is > 0.
    Near 1.0 = the edge is unlikely to be noise; near 0.5 = coin-flip.
  * Multiple-testing context — how many combos were tried for that ticker and
    how many were profitable OOS (if almost all worked, the ticker just trended;
    if few, the plan is distinctive but more prone to overfitting).

Writes a `robustness` table and prints a summary. Fast — reuses the plans.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

import config
import dataio
from strategies.registry import STRATEGIES
from backtest import engine2

N_BOOT = 2000
CRASH_WINDOW = ("2020-02-14", "2020-04-30")


def plan_oos(ticker, strategy_name, params_str, exit_policy, interval="1d",
             is_fraction=config.IS_FRACTION):
    """Rebuild the plan on its OOS slice; return (trade_returns, equity_series)."""
    strat = STRATEGIES.get(strategy_name)
    data = dataio.load(ticker, interval)
    if strat is None or data is None:
        return np.array([]), pd.Series(dtype=float)
    df = engine2.clean_ohlcv(data)
    cut = int(len(df) * is_fraction)
    oos = df.iloc[cut:]
    if len(oos) < 30:
        return np.array([]), pd.Series(dtype=float)
    params = engine2.parse_params(params_str)
    entries, exits = engine2.build_signals(strat, oos, params)
    cfg = engine2.EXIT_CONFIGS.get(exit_policy, {})
    pf = engine2._portfolio(oos, entries, exits, cfg, interval)
    rets = np.asarray(pf.trades.returns.values, dtype=float)
    rets = rets[~np.isnan(rets)]
    try:
        eq = pf.value()
    except Exception:
        eq = pd.Series(dtype=float)
    return rets, eq


def monte_carlo(rets, n=N_BOOT, seed=7) -> dict:
    if len(rets) < 5:
        return {}
    rng = np.random.default_rng(seed)
    finals = np.empty(n)
    dds = np.empty(n)
    m = len(rets)
    for i in range(n):
        samp = rng.choice(rets, size=m, replace=True)
        eq = np.cumprod(1.0 + samp)
        finals[i] = eq[-1] - 1.0
        peak = np.maximum.accumulate(eq)
        dds[i] = (eq / peak - 1.0).min()
    return {
        "mc_ret_p5": float(np.percentile(finals, 5)),
        "mc_ret_p50": float(np.percentile(finals, 50)),
        "mc_ret_p95": float(np.percentile(finals, 95)),
        "mc_dd_median": float(np.percentile(dds, 50)),
        "mc_dd_p95worst": float(np.percentile(dds, 5)),   # 5th pct = a bad ordering
        "mc_p_profit": float((finals > 0).mean()),
        "mc_p_dd_gt_30": float((dds < -0.30).mean()),
    }


def prob_sharpe(rets) -> float:
    """Probability the true (per-trade) Sharpe is > 0, given track length + shape."""
    n = len(rets)
    if n < 5:
        return np.nan
    sd = rets.std(ddof=1)
    if sd == 0:
        return np.nan
    sr = rets.mean() / sd
    sk = float(stats.skew(rets))
    ku = float(stats.kurtosis(rets, fisher=False))  # non-excess
    denom = 1.0 - sk * sr + (ku - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return np.nan
    z = sr * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def yearly_returns(eq: pd.Series) -> dict:
    if eq is None or eq.empty:
        return {}
    out = {}
    for y, g in eq.groupby(eq.index.year):
        if len(g) > 1 and g.iloc[0] != 0:
            out[int(y)] = round(float(g.iloc[-1] / g.iloc[0] - 1.0), 4)
    return out


def window_return(eq: pd.Series, start, end):
    if eq is None or eq.empty:
        return np.nan
    seg = eq.loc[(eq.index >= start) & (eq.index <= end)]
    if len(seg) < 2 or seg.iloc[0] == 0:
        return np.nan
    return float(seg.iloc[-1] / seg.iloc[0] - 1.0)


def trials_context(runs2_ticker: pd.DataFrame) -> dict:
    """Multiple-testing context for a ticker from its full combo set."""
    if runs2_ticker.empty:
        return {"n_trials": 0, "trials_oos_hit_rate": np.nan}
    return {
        "n_trials": int(len(runs2_ticker)),
        "trials_oos_hit_rate": float((runs2_ticker["oos_total_return"] > 0).mean()),
    }


def build(interval="1d") -> pd.DataFrame:
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        plans = pd.read_sql("SELECT * FROM plans", con)
        runs = pd.read_sql("SELECT ticker, oos_total_return FROM runs2", con)
    finally:
        con.close()
    if plans.empty:
        return pd.DataFrame()
    if "interval" in plans.columns:
        plans = plans[plans["interval"] == interval]
    runs_by_tk = {t: g for t, g in runs.groupby("ticker")}

    rows = []
    for _, p in plans.iterrows():
        rets, eq = plan_oos(p["ticker"], p["strategy"], p["params"], p["exit_policy"], interval)
        rec = {
            "ticker": p["ticker"], "strategy": p["strategy"], "exit_policy": p["exit_policy"],
            "recommended": bool(p.get("recommended", False)),
            "oos_n_trades": int(len(rets)),
        }
        rec.update(monte_carlo(rets))
        rec["psr"] = prob_sharpe(rets)
        yr = yearly_returns(eq)
        rec["yearly_json"] = json.dumps(yr)
        rec["n_years"] = len(yr)
        rec["worst_year_ret"] = min(yr.values()) if yr else np.nan
        rec["years_positive"] = float(np.mean([v > 0 for v in yr.values()])) if yr else np.nan
        rec["covid_crash_ret"] = window_return(eq, *CRASH_WINDOW)
        rec.update(trials_context(runs_by_tk.get(p["ticker"], pd.DataFrame(columns=["oos_total_return"]))))
        rows.append(rec)

    return pd.DataFrame(rows)


def save(df: pd.DataFrame):
    if df.empty:
        return
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        con.execute("DROP TABLE IF EXISTS robustness")
        df.to_sql("robustness", con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    args = ap.parse_args(argv)
    df = build(args.interval)
    if df.empty:
        print("No plans to test — run plans.py first.")
        return 1
    save(df)
    rec = df[df["recommended"]]
    print(f"Robustness computed for {len(df)} plans ({len(rec)} recommended).")
    print(f"  median PSR (recommended): {rec['psr'].median():.3f}  "
          f"(prob the edge is real; >0.9 strong)")
    print(f"  recommended with PSR>0.90: {int((rec['psr'] > 0.90).sum())}/{len(rec)}")
    print(f"  recommended with P(profit)>0.75: {int((rec['mc_p_profit'] > 0.75).sum())}/{len(rec)}")
    print(f"  recommended with P(>30% DD)>0.25: {int((rec['mc_p_dd_gt_30'] > 0.25).sum())}/{len(rec)} (fragile)")
    cols = ["ticker", "strategy", "oos_n_trades", "psr", "mc_ret_p5", "mc_ret_p50",
            "mc_ret_p95", "mc_dd_p95worst", "mc_p_profit", "worst_year_ret"]
    show = rec.sort_values("psr", ascending=False)[cols].head(12).copy()
    for c in cols[3:]:
        show[c] = show[c].round(3)
    print("\nMost statistically robust recommended plans (by PSR):")
    print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
