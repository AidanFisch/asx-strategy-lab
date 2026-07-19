"""
Scanner v2: backtest every ticker × strategy × param-combo × exit-policy and
store the FULL result set (not just the best) in SQLite table `runs2`.

As in v1, each ticker's history is split into in-sample (first IS_FRACTION) and
out-of-sample (the tail). Every combination is scored on BOTH; is_* / oos_*
metrics are stored side by side so selection can be done honestly (choose on
in-sample, judge on out-of-sample).

Usage
-----
    py -m backtest.scanner2 --limit 3          # quick timing/smoke test
    py -m backtest.scanner2                     # all cached daily tickers
    py -m backtest.scanner2 --tickers BHP.AX CBA.AX
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
from strategies.registry import ALL_STRATEGIES
from backtest import engine2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("scanner2")

RUNS_TABLE = "runs2"

_METRICS = ["n_bars", "start", "end", "total_return", "cagr", "sharpe", "sortino",
            "max_drawdown", "calmar", "n_trades", "win_rate", "profit_factor",
            "avg_pnl", "avg_ret_pct", "avg_win_pct", "avg_loss_pct", "payoff",
            "exposure", "expectancy_R", "buy_hold_return"]


def expand_grid(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    combos = [dict(zip(keys, c)) for c in itertools.product(*(grid[k] for k in keys))]
    # drop invalid fast>=slow combos where both present
    out = []
    for c in combos:
        if "fast" in c and "slow" in c and c["fast"] >= c["slow"]:
            continue
        out.append(c)
    return out


def split_is_oos(data, frac):
    cut = int(len(data) * frac)
    return data.iloc[:cut], data.iloc[cut:]


def scan_ticker(ticker, interval, frac) -> list[dict]:
    data = dataio.load(ticker, interval)
    if data is None or data.empty:
        return []
    # clean BEFORE splitting so the IS/OOS boundary matches the downstream
    # modules (robustness/portfolio/trade_details), which all clean first
    data = engine2.clean_ohlcv(data)
    is_data, oos_data = split_is_oos(data, frac)
    rows = []
    for strat in ALL_STRATEGIES:
        for params in expand_grid(strat.param_grid):
            try:
                is_rows = engine2.evaluate(is_data, strat, params, interval=interval, ticker=ticker)
                oos_rows = engine2.evaluate(oos_data, strat, params, interval=interval, ticker=ticker)
            except Exception as e:
                log.error("%s %s %s: %s", ticker, strat.name, params, e)
                continue
            oos_by_policy = {r["exit_policy"]: r for r in oos_rows}
            for ir in is_rows:
                orr = oos_by_policy.get(ir["exit_policy"], {})
                row = {
                    "ticker": ticker, "strategy": strat.name, "family": strat.family,
                    "params": engine2._params_str(params), "interval": interval,
                    "exit_policy": ir["exit_policy"], "is_fraction": frac,
                }
                for k in _METRICS:
                    row[f"is_{k}"] = ir.get(k)
                    row[f"oos_{k}"] = orr.get(k)
                rows.append(row)
    return rows


def write_rows(rows, scan_id):
    if not rows:
        log.warning("no rows to write"); return
    df = pd.DataFrame(rows)
    df.insert(0, "scan_id", scan_id)
    df.insert(1, "scan_time", datetime.now(timezone.utc).isoformat())
    config.LEADERBOARD_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        df.to_sql(RUNS_TABLE, con, if_exists="append", index=False)
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_runs2_tk ON {RUNS_TABLE}(interval, ticker)")
        con.commit()
    finally:
        con.close()
    log.info("wrote %d rows to %s (%s)", len(df), RUNS_TABLE, scan_id)


def run(tickers, interval, frac):
    scan_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log.info("scan_id=%s strategies=%d tickers=%d interval=%s",
             scan_id, len(ALL_STRATEGIES), len(tickers), interval)
    all_rows, t0 = [], time.time()
    for i, tk in enumerate(tickers, 1):
        rows = scan_ticker(tk, interval, frac)
        all_rows.extend(rows)
        # flush periodically so a long run isn't lost if interrupted
        if len(all_rows) >= 20000:
            write_rows(all_rows, scan_id); all_rows = []
        log.info("[%d/%d] %s: %d rows (%.1fs elapsed)", i, len(tickers), tk,
                 len(rows), time.time() - t0)
    write_rows(all_rows, scan_id)
    log.info("done in %.1fs", time.time() - t0)
    return scan_id


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--interval", default="1d")
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--universe-file", default=None, help="scan tickers listed in this CSV")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--is-fraction", type=float, default=config.IS_FRACTION)
    args = p.parse_args(argv)

    if args.universe_file:
        tickers = pd.read_csv(args.universe_file)["ticker"].astype(str).str.strip().tolist()
    else:
        tickers = args.tickers or dataio.available_tickers(args.interval)
    if args.limit:
        tickers = tickers[: args.limit]
    if not tickers:
        log.error("no tickers; run download_data.py first"); return 1
    run(tickers, args.interval, args.is_fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
