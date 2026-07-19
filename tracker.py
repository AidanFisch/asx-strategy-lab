"""
Forward paper-trading tracker.

The daily monitor (live/monitor.py) logs every hypothetical BUY into `positions`
and every close into `trades`. This module turns that log into a running
scorecard so you can see whether the strategies actually work GOING FORWARD —
the only test that isn't backward-looking — and whether live results match what
the backtest promised.

It reports:
  * realized closed trades: count, win rate, avg/total return, a compounding
    paper-equity curve, best/worst trades,
  * open positions with unrealised P&L and current stop,
  * expectation check: realized win-rate & avg-return vs each plan's backtested
    out-of-sample figures (are we tracking, or drifting?).

Nothing to show until the monitor has run for a while — that's the point: this
is where forward evidence accumulates. Run the monitor daily (run_scan.cmd / the
GitHub daily workflow) and check back here.
"""

from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd

import config
import dataio


def _read(con, table):
    try:
        return pd.read_sql(f"SELECT * FROM {table}", con)
    except Exception:
        return pd.DataFrame()


def load():
    con = sqlite3.connect(config.SIGNALS_DB)          # live state (small DB)
    try:
        trades = _read(con, "trades")
        positions = _read(con, "positions")
    finally:
        con.close()
    pcon = sqlite3.connect(config.LEADERBOARD_DB)     # research plans
    try:
        plans = _read(pcon, "plans")
    finally:
        pcon.close()
    return trades, positions, plans


def realized_report(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0}
    t = trades.copy()
    t["pnl_pct"] = pd.to_numeric(t["pnl_pct"], errors="coerce")
    t = t.dropna(subset=["pnl_pct"]).sort_values("exit_date")
    if t.empty:
        return {"n": 0}
    eq = (1 + t["pnl_pct"]).cumprod()
    wins = t[t["pnl_pct"] > 0]
    losses = t[t["pnl_pct"] < 0]
    return {
        "n": int(len(t)),
        "win_rate": float((t["pnl_pct"] > 0).mean()),
        "avg_ret": float(t["pnl_pct"].mean()),
        "total_paper_return": float(eq.iloc[-1] - 1),
        "avg_win": float(wins["pnl_pct"].mean()) if len(wins) else np.nan,
        "avg_loss": float(losses["pnl_pct"].mean()) if len(losses) else np.nan,
        "best": float(t["pnl_pct"].max()),
        "worst": float(t["pnl_pct"].min()),
        "first_exit": t["exit_date"].min(),
        "last_exit": t["exit_date"].max(),
        "by_reason": t.groupby("reason")["pnl_pct"].agg(["count", "mean"]).round(3).to_dict("index"),
    }


def open_report(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return positions
    rows = []
    for _, p in positions.iterrows():
        cur = np.nan
        df = dataio.load(p["ticker"], "1d")
        if df is not None and not df.empty:
            cur = float(df["Close"].iloc[-1])
        unreal = (cur / p["entry_price"] - 1) if (cur and p["entry_price"]) else np.nan
        rows.append({"ticker": p["ticker"], "strategy": p["strategy"],
                     "entry_price": p["entry_price"], "current": cur,
                     "unreal_pct": unreal, "stop_level": p["stop_level"]})
    return pd.DataFrame(rows)


def expectation_check(trades, plans) -> pd.DataFrame:
    """Compare realized per-ticker results to the plan's backtested OOS figures."""
    if trades.empty or plans.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["pnl_pct"] = pd.to_numeric(t["pnl_pct"], errors="coerce")
    g = t.groupby("ticker")["pnl_pct"].agg(realized_trades="count",
                                           realized_win_rate=lambda s: (s > 0).mean(),
                                           realized_avg_ret="mean").reset_index()
    cols = [c for c in ["ticker", "oos_win_rate", "oos_avg_ret_pct"] if c in plans.columns]
    return g.merge(plans[cols], on="ticker", how="left")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    args = ap.parse_args(argv)
    trades, positions, plans = load()

    print("=" * 60)
    print("FORWARD PAPER-TRADING SCORECARD")
    print("=" * 60)
    r = realized_report(trades)
    if r["n"] == 0:
        print("\nNo closed paper trades yet.")
        print("Forward tracking accumulates as the daily monitor runs — run")
        print("  py -m live.monitor --interval 1d --refresh   (daily, after close)")
        print("and check back here as trades close.")
    else:
        print(f"\nClosed trades: {r['n']}  ({r['first_exit'][:10]} → {r['last_exit'][:10]})")
        print(f"  win rate: {r['win_rate']:.0%} | avg/trade: {r['avg_ret']:+.2%} | "
              f"paper return: {r['total_paper_return']:+.1%}")
        if "pnl_pct_net" in trades.columns and trades["pnl_pct_net"].notna().any():
            net = pd.to_numeric(trades["pnl_pct_net"], errors="coerce").dropna()
            print(f"  net of CommSec commissions: avg/trade {net.mean():+.2%} | "
                  f"compounded {float((1+net).prod()-1):+.1%}")
        print(f"  avg win: {r['avg_win']:+.2%} | avg loss: {r['avg_loss']:+.2%} | "
              f"best {r['best']:+.1%} | worst {r['worst']:+.1%}")
        print(f"  by exit reason: {r['by_reason']}")
        ec = expectation_check(trades, plans)
        if not ec.empty:
            print("\nRealized vs backtested (per ticker):")
            print(ec.round(3).to_string(index=False))

    op = open_report(positions)
    if op.empty:
        print("\nNo open paper positions.")
    else:
        print(f"\nOpen positions ({len(op)}):")
        show = op.copy()
        for c in ["entry_price", "current", "stop_level"]:
            show[c] = show[c].round(2)
        show["unreal_pct"] = show["unreal_pct"].round(3)
        print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
