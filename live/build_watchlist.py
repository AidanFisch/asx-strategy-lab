"""
Build live/watchlist.json from the leaderboard: the combos the daily scan will
monitor for fresh signals.

By default it selects combos that HELD UP out-of-sample (oos_holds: still
profitable with positive Sharpe on the holdout) and had enough trades — i.e. the
ones least likely to be overfit. You can widen/narrow with the flags, or hand-edit
watchlist.json afterwards.

Usage
-----
    py -m live.build_watchlist                       # OOS-robust daily combos
    py -m live.build_watchlist --interval 1d --min-oos-sharpe 0.2 --max-per-ticker 2
    py -m live.build_watchlist --all-robust          # ignore per-ticker cap
"""

from __future__ import annotations

import argparse
import json
import sqlite3

import pandas as pd

import config
from backtest import engine


def build(interval="1d", min_trades=config.MIN_TRADES, min_oos_sharpe=0.0,
          max_per_ticker=1, scan="latest") -> list[dict]:
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        df = pd.read_sql("SELECT * FROM runs", con)
    finally:
        con.close()
    if df.empty:
        return []

    df = df[df["interval"] == interval]
    if scan == "latest":
        df = df[df["scan_id"] == df["scan_id"].max()]

    # robust = enough trades both periods, profitable + positive Sharpe OOS
    robust = df[
        (df["is_n_trades"].fillna(0) >= min_trades)
        & (df["oos_n_trades"].fillna(0) >= min_trades)
        & (df["oos_total_return"] > 0)
        & (df["oos_sharpe"] > min_oos_sharpe)
    ].copy()

    robust = robust.sort_values("oos_sharpe", ascending=False)
    if max_per_ticker:
        robust = robust.groupby("ticker", group_keys=False).head(max_per_ticker)

    items = []
    for _, r in robust.iterrows():
        items.append({
            "ticker": r["ticker"],
            "strategy": r["strategy"],
            "params": engine.parse_params(r["params"]),
            "interval": interval,
            "why": f"oos_sharpe={round(float(r['oos_sharpe']),2)}, "
                   f"oos_return={round(float(r['oos_total_return']),2)}",
        })
    return items


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate the live watchlist from the leaderboard.")
    p.add_argument("--interval", default="1d")
    p.add_argument("--min-trades", type=int, default=config.MIN_TRADES)
    p.add_argument("--min-oos-sharpe", type=float, default=0.0)
    p.add_argument("--max-per-ticker", type=int, default=1)
    p.add_argument("--all-robust", action="store_true", help="no per-ticker cap")
    args = p.parse_args(argv)

    items = build(
        interval=args.interval,
        min_trades=args.min_trades,
        min_oos_sharpe=args.min_oos_sharpe,
        max_per_ticker=0 if args.all_robust else args.max_per_ticker,
    )

    config.WATCHLIST_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.WATCHLIST_JSON.write_text(json.dumps(items, indent=2))
    print(f"Wrote {len(items)} watchlist item(s) to {config.WATCHLIST_JSON}")
    for it in items:
        print(f"  {it['ticker']:9} {it['strategy']:18} {it['params']}  ({it['why']})")
    if not items:
        print("No OOS-robust combos found. Loosen --min-oos-sharpe, or the daily scan\n"
              "will fall back to default-param strategies on all cached tickers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
