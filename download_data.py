"""
Phase 1 data pipeline: download daily OR intraday OHLCV from yfinance and cache
it locally as Parquet, with an incremental update mode.

Design notes
------------
* Config-driven: `config.INTERVAL` (or --interval) decides daily vs intraday.
* Incremental cache: each ticker/interval is stored at
      data/raw/<interval>/<TICKER>.parquet
  On re-run we only fetch bars newer than the last cached one and append them.
  This matters most for INTRADAY: Yahoo only serves a short rolling window
  (1m ~7d, 5m/15m/etc ~60d), so the only way to build up real history is to
  fetch regularly and accumulate locally.
* Graceful failures: a missing/delisted/empty ticker is logged and skipped,
  never crashes the whole run.

Usage
-----
    py download_data.py                         # uses config.INTERVAL, full universe
    py download_data.py --interval 1d --tickers BHP.AX CBA.AX
    py download_data.py --interval 5m --limit 5
    py download_data.py --interval 1d --universe   # force full universe.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download")

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------
def load_universe(path=None) -> list[str]:
    """Load tickers from a universe CSV, stripping whitespace and de-duplicating."""
    df = pd.read_csv(path or config.UNIVERSE_CSV)
    tickers = (
        df["ticker"].astype(str).str.strip().replace("", pd.NA).dropna().tolist()
    )
    # preserve order but drop dupes
    seen, out = set(), []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def cache_path(ticker: str, interval: str):
    d = config.raw_dir_for(interval)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{ticker}.parquet"


def read_cache(ticker: str, interval: str) -> pd.DataFrame | None:
    path = cache_path(ticker, interval)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:  # corrupt cache -> treat as no cache
        log.warning("%s: could not read cache (%s); will refetch", ticker, e)
        return None


def earliest_allowed_start(interval: str) -> pd.Timestamp:
    """The earliest datetime Yahoo will serve for this interval, as a UTC timestamp."""
    limit_days = config.lookback_limit_days(interval)
    now = pd.Timestamp.now(tz="UTC")
    if limit_days is None:
        if config.is_intraday(interval):
            # safety fallback; shouldn't happen for known intraday intervals
            return now - pd.Timedelta(days=60)
        start = config.DAILY_START
        return pd.Timestamp(start, tz="UTC")
    # subtract a hair less than the limit so we stay inside Yahoo's window
    return now - pd.Timedelta(days=limit_days - 1)


def compute_fetch_start(ticker: str, interval: str, cached: pd.DataFrame | None,
                        backfill: bool = False):
    """
    Decide the start datetime for this fetch.

    * No cache  -> earliest the interval allows (clamped to Yahoo's window).
    * Have cache -> last cached bar minus an overlap, but never earlier than
                    the interval's allowed window.
    * backfill  -> ignore the cache's start and pull the full history again
                   (merge dedupes), used to extend existing caches backwards.
    """
    allowed = earliest_allowed_start(interval)

    if backfill and not config.is_intraday(interval):
        return None  # period="max"

    if cached is None or cached.empty:
        if not config.is_intraday(interval) and config.DAILY_MAX:
            return None  # signal: use period="max"
        return allowed

    last = cached.index.max()
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    else:
        last = last.tz_convert("UTC")

    start = last - pd.Timedelta(days=config.REFRESH_OVERLAP_DAYS)
    if start < allowed:
        start = allowed
    return start


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_one(ticker: str, interval: str, start) -> pd.DataFrame:
    """
    Pull OHLCV for a single ticker via yfinance. `start` is a Timestamp, or None
    to request period='max' (daily only). Returns a clean OHLCV DataFrame
    (possibly empty).
    """
    tk = yf.Ticker(ticker)
    kwargs = dict(interval=interval, auto_adjust=config.AUTO_ADJUST, actions=False)

    if start is None:
        df = tk.history(period="max", **kwargs)
    else:
        # yfinance 'start' is inclusive; end defaults to now. Pass tz-naive date
        # for daily, full timestamp for intraday.
        if config.is_intraday(interval):
            start_arg = start.tz_convert("UTC")
        else:
            start_arg = start.tz_convert("UTC").date()
        df = tk.history(start=start_arg, **kwargs)

    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLS)

    # keep only OHLCV, in canonical order
    df = df[[c for c in OHLCV_COLS if c in df.columns]].copy()
    df = df.dropna(how="all")
    return df


def merge_and_save(ticker: str, interval: str, cached, fresh) -> tuple[int, int]:
    """Combine cached + fresh bars, de-dupe on the index, persist. Returns (rows_before, rows_after)."""
    if cached is not None and not cached.empty:
        combined = pd.concat([cached, fresh])
    else:
        combined = fresh

    before = 0 if cached is None else len(cached)
    if combined.empty:
        return before, before

    # normalise index -> DatetimeIndex, drop duplicate timestamps (keep latest fetch)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    path = cache_path(ticker, interval)
    combined.to_parquet(path)
    return before, len(combined)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(tickers: list[str], interval: str, backfill: bool = False):
    total = len(tickers)
    ok = skipped = 0
    log.info(
        "Interval=%s  intraday=%s  lookback_limit=%s days  tickers=%d",
        interval,
        config.is_intraday(interval),
        config.lookback_limit_days(interval),
        total,
    )

    for i, ticker in enumerate(tickers, 1):
        try:
            cached = read_cache(ticker, interval)
            start = compute_fetch_start(ticker, interval, cached, backfill=backfill)
            fresh = fetch_one(ticker, interval, start)

            if fresh.empty and (cached is None or cached.empty):
                log.warning("[%d/%d] %s: no data returned; skipping", i, total, ticker)
                skipped += 1
                continue

            before, after = merge_and_save(ticker, interval, cached, fresh)
            added = after - before
            span = ""
            saved = read_cache(ticker, interval)
            if saved is not None and not saved.empty:
                span = f"{saved.index.min().date()} -> {saved.index.max().date()}"
            log.info(
                "[%d/%d] %s: +%d new bars (%d total)  %s",
                i, total, ticker, max(added, 0), after, span,
            )
            ok += 1
        except Exception as e:  # never let one ticker kill the run
            log.error("[%d/%d] %s: FAILED (%s)", i, total, ticker, e)
            skipped += 1
        finally:
            if i < total and config.REQUEST_SLEEP:
                time.sleep(config.REQUEST_SLEEP)

    log.info("Done. interval=%s  ok=%d  skipped=%d  (of %d)", interval, ok, skipped, total)
    return ok, skipped


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Download daily/intraday OHLCV and cache as Parquet.")
    p.add_argument("--interval", default=config.INTERVAL,
                   help=f"yfinance interval (default from config: {config.INTERVAL})")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="explicit ticker list (e.g. BHP.AX CBA.AX); overrides universe")
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N universe tickers (smoke test)")
    p.add_argument("--universe", action="store_true",
                   help="force loading the full universe.csv even if --tickers given")
    p.add_argument("--universe-file", default=None,
                   help="path to an alternative universe CSV (e.g. data/universe_asia.csv)")
    p.add_argument("--backfill", action="store_true",
                   help="re-pull FULL history even for cached tickers (extends caches backwards)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    interval = args.interval

    if interval not in config.INTERVAL_LIMITS:
        log.error("Unknown interval '%s'. Known: %s", interval, ", ".join(config.INTERVAL_LIMITS))
        return 2

    if args.tickers and not args.universe:
        tickers = [t.strip() for t in args.tickers]
    else:
        tickers = load_universe(args.universe_file)
        if args.limit:
            tickers = tickers[: args.limit]

    run(tickers, interval, backfill=args.backfill)
    return 0


if __name__ == "__main__":
    sys.exit(main())
