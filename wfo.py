"""
Walk-forward optimization (WFO) — the strictest honesty test.

Instead of one in-sample/out-of-sample split, WFO rolls through time: on each
TRAIN window it re-selects the best (strategy, params, exit) across the whole
grid, then trades that pick on the NEXT unseen TEST window, and moves on. The
concatenated test windows are a genuine "how it would have actually traded if I
re-optimized periodically" record — which is exactly how this system runs (the
weekly rescan re-selects plans).

A plan whose ticker survives WFO (test windows mostly positive, beats buy&hold)
is far more trustworthy than one that only looks good on a single split.

Runs on the recommended tickers by default. Writes a `wfo` table.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3

import numpy as np
import pandas as pd

import config
import dataio
from strategies.registry import ALL_STRATEGIES
from backtest import engine2
from backtest.scanner2 import expand_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("wfo")

N_WINDOWS = 5
MIN_TRAIN = 300
MIN_TEST = 40


def select_on_train(train, interval) -> tuple | None:
    """Pick the best (strategy, params, exit_policy) on a train slice by Sharpe."""
    best, best_sr = None, -np.inf
    for strat in ALL_STRATEGIES:
        for params in expand_grid(strat.param_grid):
            try:
                rows = engine2.evaluate(train, strat, params, interval=interval)
            except Exception:
                continue
            for row in rows:
                sr = row.get("sharpe")
                if (row.get("n_trades", 0) >= config.MIN_TRADES and sr is not None
                        and not np.isnan(sr) and sr > best_sr):
                    best_sr, best = sr, (strat, params, row["exit_policy"])
    return best


def wfo_ticker(ticker, interval="1d", n_windows=N_WINDOWS) -> dict | None:
    data = dataio.load(ticker, interval)
    if data is None:
        return None
    df = engine2.clean_ohlcv(data)
    n = len(df)
    if n < MIN_TRAIN + MIN_TEST:
        return None
    bounds = np.linspace(0, n, n_windows + 1).astype(int)

    test_rets, picks, bh = [], [], []
    for i in range(1, n_windows):
        train = df.iloc[: bounds[i]]
        test = df.iloc[bounds[i]: bounds[i + 1]]
        if len(train) < MIN_TRAIN or len(test) < MIN_TEST:
            continue
        pick = select_on_train(train, interval)
        if pick is None:
            continue
        strat, params, exit_policy = pick
        rows = engine2.evaluate(test, strat, params, interval=interval,
                                exit_configs={exit_policy: engine2.EXIT_CONFIGS[exit_policy]},
                                min_bars=MIN_TEST)
        r = rows[0].get("total_return")
        if r is None or (isinstance(r, float) and np.isnan(r)):
            continue
        test_rets.append(float(r))
        picks.append(f"{strat.name}/{exit_policy}")
        c = test["Close"]
        bh.append(float(c.iloc[-1] / c.iloc[0] - 1.0))

    if len(test_rets) < 2:
        return None
    tr = np.array(test_rets)
    compound = float(np.prod(1 + tr) - 1)
    bh_comp = float(np.prod(1 + np.array(bh)) - 1)
    return {
        "ticker": ticker,
        "wfo_windows": len(tr),
        "wfo_total_return": compound,
        "wfo_avg_window_ret": float(tr.mean()),
        "wfo_pct_windows_pos": float((tr > 0).mean()),
        "wfo_worst_window": float(tr.min()),
        "wfo_bh_return": bh_comp,
        "wfo_beats_bh": bool(compound > bh_comp),
        "wfo_n_unique_picks": int(len(set(picks))),
        "wfo_picks": ", ".join(picks),
    }


def build(interval="1d", tickers=None) -> pd.DataFrame:
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        plans = pd.read_sql("SELECT ticker, recommended FROM plans", con)
    except Exception:
        plans = pd.DataFrame()
    con.close()
    if tickers is None:
        tickers = (plans[plans["recommended"] == 1]["ticker"].tolist()
                   if not plans.empty else dataio.available_tickers(interval))

    rows = []
    for i, t in enumerate(tickers, 1):
        r = wfo_ticker(t, interval)
        if r:
            rows.append(r)
        log.info("[%d/%d] %s: %s", i, len(tickers), t,
                 "ok" if r else "skipped (insufficient data)")
    return pd.DataFrame(rows)


def save(df):
    if df.empty:
        return
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        con.execute("DROP TABLE IF EXISTS wfo")
        df.to_sql("wfo", con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--all", action="store_true", help="run on all cached tickers (slow)")
    args = ap.parse_args(argv)
    tickers = dataio.available_tickers(args.interval) if args.all else None
    df = build(args.interval, tickers)
    if df.empty:
        print("No WFO results.")
        return 1
    save(df)
    print(f"\nWFO computed for {len(df)} tickers.")
    print(f"  survived (beat buy&hold walk-forward): {int(df['wfo_beats_bh'].sum())}/{len(df)}")
    print(f"  windows mostly positive (>=75%): {int((df['wfo_pct_windows_pos'] >= 0.75).sum())}/{len(df)}")
    show = df.sort_values("wfo_total_return", ascending=False).head(12)[
        ["ticker", "wfo_windows", "wfo_total_return", "wfo_bh_return", "wfo_pct_windows_pos",
         "wfo_worst_window", "wfo_beats_bh"]].copy()
    for c in ["wfo_total_return", "wfo_bh_return", "wfo_pct_windows_pos", "wfo_worst_window"]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
