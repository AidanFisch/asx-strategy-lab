"""
Query the leaderboard: rank ticker+strategy+param combos and surface overfitting.

Ranks by in-sample Sharpe (configurable) with a minimum-trades filter, and shows
the out-of-sample metrics alongside so you can see whether a combo held up on data
it never saw. The `oos_holds` flag is a quick robustness heuristic.

Usage
-----
    py -m results.query                       # top 20 by is_sharpe, daily
    py -m results.query --interval 1d --top 30 --min-trades 15
    py -m results.query --sort oos_sharpe     # rank by out-of-sample instead
    py -m results.query --ticker BHP.AX
    py -m results.query --scan latest         # restrict to the most recent scan
"""

from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

import config

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)


def load_runs(interval=None, scan="latest") -> pd.DataFrame:
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        df = pd.read_sql("SELECT * FROM runs", con)
    finally:
        con.close()
    if df.empty:
        return df
    if interval:
        df = df[df["interval"] == interval]
    if scan == "latest" and not df.empty:
        latest = df["scan_id"].max()
        df = df[df["scan_id"] == latest]
    return df


def rank(df: pd.DataFrame, sort="is_sharpe", top=20, min_trades=config.MIN_TRADES,
         ticker=None) -> pd.DataFrame:
    if df.empty:
        return df
    if ticker:
        df = df[df["ticker"] == ticker]

    # require enough trades in BOTH periods to be meaningful
    df = df[(df["is_n_trades"].fillna(0) >= min_trades) &
            (df["oos_n_trades"].fillna(0) >= min_trades)]

    df = df.sort_values(sort, ascending=False).head(top).copy()

    # robustness heuristic: still profitable AND positive Sharpe out-of-sample
    df["oos_holds"] = (df["oos_total_return"] > 0) & (df["oos_sharpe"] > 0)
    return df


DISPLAY = [
    "ticker", "strategy", "params",
    "is_sharpe", "oos_sharpe",
    "is_total_return", "oos_total_return",
    "is_n_trades", "oos_n_trades",
    "oos_max_drawdown", "oos_win_rate", "oos_holds",
]


def main(argv=None):
    p = argparse.ArgumentParser(description="Rank the leaderboard.")
    p.add_argument("--interval", default="1d")
    p.add_argument("--sort", default="is_sharpe",
                   help="column to rank by (e.g. is_sharpe, oos_sharpe, is_total_return)")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--min-trades", type=int, default=config.MIN_TRADES)
    p.add_argument("--ticker", default=None)
    p.add_argument("--scan", default="latest", help="'latest' or 'all'")
    args = p.parse_args(argv)

    df = load_runs(interval=args.interval, scan=args.scan)
    if df.empty:
        print("Leaderboard is empty. Run:  py -m backtest.scanner --interval", args.interval)
        return 1

    ranked = rank(df, sort=args.sort, top=args.top, min_trades=args.min_trades, ticker=args.ticker)
    if ranked.empty:
        print(f"No combos with >= {args.min_trades} trades in both IS and OOS for interval {args.interval}.")
        return 0

    show = ranked[[c for c in DISPLAY if c in ranked.columns]].copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(3)

    held = int(ranked["oos_holds"].sum())
    print(f"\nTop {len(show)} by {args.sort}  (interval={args.interval}, min_trades={args.min_trades})")
    print(f"Out-of-sample robustness: {held}/{len(show)} combos still profitable + positive Sharpe OOS\n")
    print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
