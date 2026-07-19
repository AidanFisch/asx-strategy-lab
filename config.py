"""
Central configuration for the ASX strategy lab.

The data pipeline reads INTERVAL from here (or from a --interval CLI override)
and decides whether to pull daily or intraday OHLCV accordingly.

yfinance free-data limits are baked into INTERVAL_LIMITS below. They are the
reason the pipeline caches *incrementally*: for intraday timeframes Yahoo only
serves a short rolling window, so the only way to accumulate real history is to
fetch regularly and append to a local Parquet cache over time.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make console output UTF-8 safe (Windows terminals default to cp1252, which
# crashes on emoji in signal alerts). Reconfigure once; harmless elsewhere.
# ---------------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass  # non-reconfigurable stream (e.g. redirected/captured); ignore

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # cached OHLCV lives in data/raw/<interval>/<TICKER>.parquet
UNIVERSE_CSV = DATA_DIR / "universe.csv"

# ---------------------------------------------------------------------------
# What to fetch
# ---------------------------------------------------------------------------
# INTERVAL selects daily vs intraday. Valid yfinance intervals:
#   Daily/positional : 1d, 5d, 1wk, 1mo, 3mo
#   Intraday         : 1m, 2m, 5m, 15m, 30m, 60m (== 1h), 90m
INTERVAL = "1d"

# auto_adjust=True adjusts OHLC for splits & dividends -> better for daily backtests.
AUTO_ADJUST = True

# How much history to try to pull on the FIRST fetch of a ticker/interval.
# For intraday this is automatically clamped to Yahoo's rolling limit below.
# For daily, "max" pulls the full available history.
DAILY_START = "2005-01-01"   # earliest daily bar to request on first fetch
DAILY_MAX = False            # if True, ignore DAILY_START and request period="max"

# When updating an existing cache, re-fetch this many extra days of overlap
# before the last cached bar (guards against partial/last-bar revisions).
REFRESH_OVERLAP_DAYS = 5

# Polite pause between tickers (seconds) to avoid hammering Yahoo.
REQUEST_SLEEP = 0.5

# ---------------------------------------------------------------------------
# Backtest frictions & sizing (Phase 2+)
# ---------------------------------------------------------------------------
# Realistic frictions matter: ignoring them makes strategies look better than
# they'd trade live. Applied on every fill by the backtest engine.
INIT_CASH = 10_000.0     # starting capital per single-ticker backtest
FEES = 0.001             # brokerage as a fraction of trade value (0.1% per side)
SLIPPAGE = 0.0005        # price slippage per fill (5 bps) — proxy for ASX bid/ask spread

# Map interval -> pandas frequency string, used by vectorbt to annualise
# (Sharpe, CAGR). NOTE: intraday annualisation assumes continuous bars and will
# be distorted for exchange-hours-only data — treat intraday risk metrics as
# comparative within the same interval, not absolute.
FREQ_MAP = {
    "1m": "1min", "2m": "2min", "5m": "5min", "15m": "15min",
    "30m": "30min", "60m": "1h", "1h": "1h", "90m": "90min",
    "1d": "1D", "5d": "5D", "1wk": "1W", "1mo": "1MS", "3mo": "3MS",
}


def freq_for(interval: str) -> str:
    return FREQ_MAP.get(interval, "1D")


# ---------------------------------------------------------------------------
# Scanner / leaderboard (Phase 3)
# ---------------------------------------------------------------------------
LEADERBOARD_DB = PROJECT_ROOT / "results" / "leaderboard.db"

# In-sample fraction: the first IS_FRACTION of each ticker's history is used to
# find good params; the remaining tail is the out-of-sample holdout used to
# check the strategy still works on data it never saw. A combo that looks great
# in-sample but falls apart out-of-sample is the classic overfitting red flag.
IS_FRACTION = 0.70

# Default ranking filter: ignore combos with too few trades to be meaningful.
MIN_TRADES = 10

# ---------------------------------------------------------------------------
# Live scan / notifications (Phase 4)
# ---------------------------------------------------------------------------
import os

# Telegram creds come from environment variables so no secrets live in the repo.
# If either is unset, notify/telegram_bot.py runs in dry-run mode (prints only).
#   setx TELEGRAM_BOT_TOKEN "123456:ABC..."   (Windows, new shell after)
#   setx TELEGRAM_CHAT_ID   "123456789"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Position sizing for BUY suggestions (paper): risk the same fraction of capital
# on every trade, sized off the stop distance. Purely advisory in alerts.
CAPITAL = 20_000.0        # notional trading capital
RISK_PER_TRADE = 0.01     # risk 1% of capital per trade (entry -> stop distance)
FALLBACK_POSITION_FRAC = 0.10   # no-stop plans: suggest a flat 10% slice instead

# Brokerage (see brokerage.py — CommSec tiers). MAX_FEE_DRAG_RT caps round-trip
# commission as a fraction of position value; sizes below it get bumped up to
# the cheapest fee-efficient size (with the true risk % reported).
MAX_FEE_DRAG_RT = 0.01    # allow at most 1% of position value lost to commissions

# Live paper-trading state (positions/trades/signals) lives in its OWN small DB,
# separate from the 30MB+ research leaderboard, so the daily cloud run can commit
# it without bloating the repo.
SIGNALS_DB = PROJECT_ROOT / "results" / "live.db"
SIGNALS_TABLE = "signals"

# Watchlist of combos the daily scan monitors (generated from the leaderboard).
WATCHLIST_JSON = PROJECT_ROOT / "live" / "watchlist.json"

# ---------------------------------------------------------------------------
# yfinance free-tier lookback limits, in calendar days, per interval.
# None == effectively unlimited (daily and coarser).
# These match Yahoo's documented behaviour:
#   1m  -> ~7 days retrievable per request (~30d availability, 7d/request)
#   2m..30m, 90m -> ~60 days
#   60m/1h -> ~730 days
#   1d and coarser -> full history
# ---------------------------------------------------------------------------
INTERVAL_LIMITS = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "90m": 60,
    "60m": 730,
    "1h": 730,
    "1d": None,
    "5d": None,
    "1wk": None,
    "1mo": None,
    "3mo": None,
}

INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "1h", "90m"}


def is_intraday(interval: str) -> bool:
    return interval in INTRADAY_INTERVALS


def lookback_limit_days(interval: str):
    """Max calendar days of history Yahoo will serve for this interval (None = unlimited)."""
    return INTERVAL_LIMITS.get(interval, None)


def raw_dir_for(interval: str) -> Path:
    """Cache directory for a given interval, e.g. data/raw/1d or data/raw/5m."""
    return RAW_DIR / interval
