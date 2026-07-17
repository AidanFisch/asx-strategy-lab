"""
Phase 3 scanner: backtest every (ticker x strategy x parameter-combo) and store
the results in a SQLite leaderboard.

Overfitting guard
-----------------
Each ticker's history is split by time into an in-sample (IS) portion and an
out-of-sample (OOS) holdout (config.IS_FRACTION). Every combo is evaluated on
BOTH and both metric sets are stored (is_* and oos_* columns). Ranking on IS
while eyeballing OOS makes overfitting visible: a combo with a great is_sharpe
but a poor oos_sharpe is not to be trusted.

Usage
-----
    py -m backtest.scanner                       # all cached daily tickers
    py -m backtest.scanner --interval 1d --limit 5
    py -m backtest.scanner --tickers BHP.AX CBA.AX --interval 1d
    py -m backtest.scanner --interval 15m        # scan cached intraday
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd

import config
import dataio
from strategies import STRATEGY_MODULES, STRATEGIES
from backtest import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("scanner")

# metric keys produced by engine.run_backtest that we split into is_/oos_
_METRIC_KEYS = ["n_bars", "start", "end", "total_return", "cagr", "sharpe",
                "max_drawdown", "n_trades", "win_rate", "profit_factor",
                "avg_pnl", "buy_hold_return"]


def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product of a PARAM_GRID dict -> list of param dicts."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def split_is_oos(data: pd.DataFrame, is_fraction: float):
    """Split by time into (in_sample, out_of_sample) DataFrames."""
    n = len(data)
    cut = int(n * is_fraction)
    return data.iloc[:cut], data.iloc[cut:]


def scan_ticker(ticker: str, interval: str, is_fraction: float) -> list[dict]:
    data = dataio.load(ticker, interval)
    if data is None or data.empty:
        log.warning("%s: no cached data for interval %s; skipping", ticker, interval)
        return []

    is_data, oos_data = split_is_oos(data, is_fraction)
    rows = []
    for strat in STRATEGY_MODULES:
        for params in expand_grid(strat.PARAM_GRID):
            try:
                is_m = engine.run_backtest(is_data, strat, params, ticker=ticker, interval=interval)
                oos_m = engine.run_backtest(oos_data, strat, params, ticker=ticker, interval=interval)
            except Exception as e:
                log.error("%s %s %s: FAILED (%s)", ticker, strat.NAME, params, e)
                continue

            row = {
                "ticker": ticker,
                "strategy": strat.NAME,
                "params": engine._params_str(params),
                "interval": interval,
                "is_fraction": is_fraction,
            }
            for k in _METRIC_KEYS:
                row[f"is_{k}"] = is_m.get(k)
                row[f"oos_{k}"] = oos_m.get(k)
            rows.append(row)
    return rows


def resolve_tickers(args) -> list[str]:
    if args.tickers:
        return [t.strip() for t in args.tickers]
    cached = dataio.available_tickers(args.interval)
    if args.limit:
        cached = cached[: args.limit]
    return cached


def write_results(rows: list[dict], scan_id: str):
    if not rows:
        log.warning("no rows to write")
        return
    df = pd.DataFrame(rows)
    df.insert(0, "scan_id", scan_id)
    df.insert(1, "scan_time", datetime.now(timezone.utc).isoformat())

    config.LEADERBOARD_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        df.to_sql("runs", con, if_exists="append", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_runs_sharpe ON runs(interval, is_sharpe)")
        con.commit()
    finally:
        con.close()
    log.info("wrote %d rows to %s (scan_id=%s)", len(df), config.LEADERBOARD_DB, scan_id)


def run(tickers: list[str], interval: str, is_fraction: float):
    scan_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log.info("scan_id=%s interval=%s tickers=%d is_fraction=%.2f",
             scan_id, interval, len(tickers), is_fraction)
    all_rows = []
    t0 = time.time()
    for i, ticker in enumerate(tickers, 1):
        rows = scan_ticker(ticker, interval, is_fraction)
        all_rows.extend(rows)
        log.info("[%d/%d] %s: %d combos", i, len(tickers), ticker, len(rows))
    write_results(all_rows, scan_id)
    log.info("done: %d rows in %.1fs", len(all_rows), time.time() - t0)
    return scan_id, all_rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Scan tickers x strategies x params into the leaderboard.")
    p.add_argument("--interval", default="1d")
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--is-fraction", type=float, default=config.IS_FRACTION)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tickers = resolve_tickers(args)
    if not tickers:
        log.error("no tickers to scan (nothing cached for interval %s). Run download_data.py first.",
                  args.interval)
        return 1
    run(tickers, args.interval, args.is_fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
