"""
Turn the raw v2 scan (table `runs2`) into a concrete TRADING PLAN per ticker.

For each ticker we:
  1. keep combos with enough trades in BOTH the in-sample and out-of-sample
     periods (min_trades),
  2. CHOOSE the best combo on an IN-SAMPLE metric (default is_sharpe) — never on
     the holdout, so out-of-sample stays an honest test,
  3. re-report its OUT-OF-SAMPLE metrics as the forward estimate,
  4. run a WALK-FORWARD check (several contiguous folds) — a fluke wins one split
     but rarely wins most folds; `wf_consistency` = fraction of folds profitable,
  5. attach the human-readable entry / exit / stop rules for the dashboard and
     the live BUY/SELL alerts.

A plan is flagged `recommended` when it held up out-of-sample AND was consistent
across walk-forward folds.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

import config
import dataio
from strategies.registry import STRATEGIES
from backtest import engine2

EXIT_POLICY_DESC = {
    "signal_only": "Exit only on the strategy's own sell signal (no fixed stop).",
    "sl_10": "10% stop-loss below entry, or the strategy's sell signal.",
    "sl_15": "15% stop-loss below entry, or the strategy's sell signal.",
    "trail_10": "10% trailing stop that ratchets up, or the sell signal.",
    "sl10_tp20": "10% stop-loss / 20% profit target, or the sell signal.",
    "atr_2x": "Stop at 2xATR(14) below entry (volatility stop), or the sell signal.",
}

SELECT_BY = "is_sharpe"
MIN_TRADES = config.MIN_TRADES


def load_runs2(interval="1d", scan="latest") -> pd.DataFrame:
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        df = pd.read_sql("SELECT * FROM runs2", con)
    finally:
        con.close()
    if df.empty:
        return df
    df = df[df["interval"] == interval]
    if scan == "latest" and not df.empty:
        # keep, per ticker, only its most recent scan (lets us add markets
        # incrementally without re-scanning everything into one scan_id)
        latest = df.groupby("ticker")["scan_id"].transform("max")
        df = df[df["scan_id"] == latest]
    return df


def select_best(df: pd.DataFrame, select_by=SELECT_BY, min_trades=MIN_TRADES) -> pd.DataFrame:
    valid = df[(df["is_n_trades"].fillna(0) >= min_trades) &
               (df["oos_n_trades"].fillna(0) >= min_trades) &
               (df[select_by].notna())].copy()
    if valid.empty:
        return valid
    idx = valid.groupby("ticker")[select_by].idxmax()   # choose on in-sample
    best = valid.loc[idx].copy()
    best["beats_bh"] = best["oos_total_return"] > best["oos_buy_hold_return"]
    best["oos_holds"] = (best["oos_total_return"] > 0) & (best["oos_sharpe"] > 0)
    return best


def walk_forward(ticker, strategy_name, params_str, exit_policy, interval="1d", folds=4) -> dict:
    """Backtest the chosen plan on `folds` contiguous slices; report how often it profits."""
    strat = STRATEGIES.get(strategy_name)
    data = dataio.load(ticker, interval)
    if strat is None or data is None or len(data) < folds * 60:
        return {"wf_folds": 0, "wf_positive": 0, "wf_consistency": np.nan, "wf_median_return": np.nan}

    params = engine2.parse_params(params_str)
    cfg = {exit_policy: engine2.EXIT_CONFIGS[exit_policy]}
    n = len(data)
    bounds = np.linspace(0, n, folds + 1).astype(int)
    rets = []
    for i in range(folds):
        seg = data.iloc[bounds[i]:bounds[i + 1]]
        rows = engine2.evaluate(seg, strat, params, interval=interval,
                                exit_configs=cfg, ticker=ticker, min_bars=40)
        r = rows[0].get("total_return", np.nan)
        if r is not None and not (isinstance(r, float) and np.isnan(r)):
            rets.append(float(r))
    if not rets:
        return {"wf_folds": 0, "wf_positive": 0, "wf_consistency": np.nan, "wf_median_return": np.nan}
    pos = sum(1 for r in rets if r > 0)
    return {"wf_folds": len(rets), "wf_positive": pos,
            "wf_consistency": pos / len(rets), "wf_median_return": float(np.median(rets))}


def build_plans(interval="1d", select_by=SELECT_BY, min_trades=MIN_TRADES,
                do_walk_forward=True, wf_min_consistency=0.75) -> pd.DataFrame:
    runs = load_runs2(interval)
    if runs.empty:
        return pd.DataFrame()
    best = select_best(runs, select_by, min_trades)
    if best.empty:
        return best

    records = []
    for _, r in best.iterrows():
        strat = STRATEGIES.get(r["strategy"])
        params = engine2.parse_params(r["params"])
        desc = strat.describe(params) if strat else {"entry": "", "exit": ""}
        rec = {
            "ticker": r["ticker"], "strategy": r["strategy"], "family": r["family"],
            "params": r["params"], "exit_policy": r["exit_policy"],
            "entry_rule": desc["entry"], "exit_rule": desc["exit"],
            "stop_rule": EXIT_POLICY_DESC.get(r["exit_policy"], r["exit_policy"]),
        }
        for k in ["total_return", "cagr", "sharpe", "max_drawdown", "win_rate",
                  "profit_factor", "payoff", "expectancy_R", "n_trades",
                  "avg_ret_pct", "exposure", "buy_hold_return"]:
            rec[f"is_{k}"] = r.get(f"is_{k}")
            rec[f"oos_{k}"] = r.get(f"oos_{k}")
        rec["beats_bh"] = bool(r["beats_bh"])
        rec["oos_holds"] = bool(r["oos_holds"])

        if do_walk_forward:
            rec.update(walk_forward(r["ticker"], r["strategy"], r["params"],
                                    r["exit_policy"], interval))
        records.append(rec)

    plans = pd.DataFrame(records)
    plans["recommended"] = plans["oos_holds"] & plans["beats_bh"] & (
        plans.get("wf_consistency", 0).fillna(0) >= wf_min_consistency)
    # rank recommended first, then by OOS CAGR
    plans = plans.sort_values(["recommended", "oos_cagr"], ascending=[False, False]).reset_index(drop=True)
    return plans


def save_plans(plans: pd.DataFrame, interval="1d"):
    if plans.empty:
        return
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        p = plans.copy()
        p.insert(0, "interval", interval)
        con.execute("DROP TABLE IF EXISTS plans")
        p.to_sql("plans", con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--no-wf", action="store_true", help="skip walk-forward (faster)")
    args = ap.parse_args()
    plans = build_plans(args.interval, do_walk_forward=not args.no_wf)
    if plans.empty:
        print("No plans (scan empty or nothing passed the trade filter).")
    else:
        save_plans(plans, args.interval)
        rec = int(plans["recommended"].sum())
        print(f"Built {len(plans)} plans | recommended: {rec} | "
              f"beat B&H OOS: {int(plans['beats_bh'].sum())} | held OOS: {int(plans['oos_holds'].sum())}")
        cols = ["ticker", "strategy", "exit_policy", "oos_cagr", "oos_total_return",
                "oos_sharpe", "wf_consistency", "recommended"]
        show = plans[cols].head(15).copy()
        for c in ["oos_cagr", "oos_total_return", "oos_sharpe", "wf_consistency"]:
            show[c] = show[c].round(3)
        print(show.to_string(index=False))
