"""
Per-plan trade details: average hold time + the full individual trade log
(entry/exit dates, prices, return, duration) for every plan, on its out-of-sample
holdout period.

Writes:
  * plan_stats  — avg_duration / median_duration / n_oos_trades per plan
  * plan_trades — one row per individual trade (entry_date, exit_date, prices,
                  return, bars held, status)

The dashboard shows avg hold as a column and lists the trades inside a plan's
expanded detail (for the recommended / high-confidence plans).
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

import config
import dataio
from strategies.registry import STRATEGIES
from backtest import engine2


def _col(tr, *names):
    for n in names:
        if n in tr.columns:
            return tr[n]
    return pd.Series([np.nan] * len(tr))


def plan_trade_log(ticker, strategy_name, params_str, exit_policy,
                   interval="1d", is_fraction=config.IS_FRACTION) -> pd.DataFrame:
    strat = STRATEGIES.get(strategy_name)
    data = dataio.load(ticker, interval)
    if strat is None or data is None:
        return pd.DataFrame()
    df = engine2.clean_ohlcv(data)
    cut = int(len(df) * is_fraction)
    oos = df.iloc[cut:]
    if len(oos) < 30:
        return pd.DataFrame()
    params = engine2.parse_params(params_str)
    entries, exits = engine2.build_signals(strat, oos, params)
    cfg = engine2.EXIT_CONFIGS.get(exit_policy, {})
    pf = engine2._portfolio(oos, entries, exits, cfg, interval)

    try:
        tr = pf.trades.records_readable
    except Exception:
        return pd.DataFrame()
    if tr is None or len(tr) == 0:
        return pd.DataFrame()

    dur = np.asarray(pf.trades.duration.values, dtype=float)
    ent = pd.to_datetime(_col(tr, "Entry Timestamp", "Entry Index"), errors="coerce")
    ext = pd.to_datetime(_col(tr, "Exit Timestamp", "Exit Index"), errors="coerce")
    out = pd.DataFrame({
        "entry_date": ent.dt.strftime("%Y-%m-%d"),
        "exit_date": ext.dt.strftime("%Y-%m-%d"),
        "entry_price": pd.to_numeric(_col(tr, "Avg Entry Price"), errors="coerce").round(3),
        "exit_price": pd.to_numeric(_col(tr, "Avg Exit Price"), errors="coerce").round(3),
        "ret": pd.to_numeric(_col(tr, "Return"), errors="coerce").round(4),
        "bars": [int(x) if np.isfinite(x) else None for x in dur] if len(dur) == len(tr) else None,
        "status": _col(tr, "Status").astype(str),
    })
    return out


def build(interval="1d"):
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        plans = pd.read_sql("SELECT ticker, strategy, params, exit_policy, recommended FROM plans", con)
    finally:
        con.close()
    if "interval" in plans.columns:
        plans = plans[plans.get("interval", interval) == interval]

    stats_rows, all_trades = [], []
    for _, p in plans.iterrows():
        tl = plan_trade_log(p["ticker"], p["strategy"], p["params"], p["exit_policy"], interval)
        bars = pd.to_numeric(tl["bars"], errors="coerce").dropna() if not tl.empty else pd.Series(dtype=float)
        stats_rows.append({
            "ticker": p["ticker"], "strategy": p["strategy"], "exit_policy": p["exit_policy"],
            "avg_duration": float(bars.mean()) if len(bars) else np.nan,
            "median_duration": float(bars.median()) if len(bars) else np.nan,
            "n_oos_trades": int(len(tl)),
        })
        if not tl.empty:
            tl = tl.assign(ticker=p["ticker"], strategy=p["strategy"], exit_policy=p["exit_policy"])
            all_trades.append(tl)

    stats = pd.DataFrame(stats_rows)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(
        columns=["ticker", "strategy", "exit_policy", "entry_date", "exit_date",
                 "entry_price", "exit_price", "ret", "bars", "status"])
    return stats, trades


def save(stats, trades):
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        con.execute("DROP TABLE IF EXISTS plan_stats")
        con.execute("DROP TABLE IF EXISTS plan_trades")
        stats.to_sql("plan_stats", con, if_exists="replace", index=False)
        trades.to_sql("plan_trades", con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    args = ap.parse_args(argv)
    stats, trades = build(args.interval)
    if stats.empty:
        print("No plans.")
        return 1
    save(stats, trades)
    print(f"plan_stats: {len(stats)} plans | plan_trades: {len(trades)} individual trades")
    print(f"  median avg-hold across plans: {stats['avg_duration'].median():.0f} trading days")
    ex = stats.dropna(subset=["avg_duration"]).sort_values("n_oos_trades", ascending=False).head(6)
    print(ex[["ticker", "strategy", "exit_policy", "n_oos_trades", "avg_duration", "median_duration"]]
          .round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
