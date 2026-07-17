"""
Phase 4 daily scan (scheduled entrypoint).

For each combo on the watchlist it:
  1. (optionally) refreshes the latest OHLCV into the Parquet cache,
  2. regenerates the strategy's signals,
  3. checks whether the MOST RECENT bar fired a fresh BUY or SELL,
  4. logs any new signal to the `signals` table (deduped), and
  5. sends a Telegram alert.

This is paper-trading: it records hypothetical signals so you can track how the
strategies would have done before risking real money. It never places orders.

Usage
-----
    py -m live.daily_scan                     # scan watchlist, no data refresh
    py -m live.daily_scan --refresh           # update data first, then scan
    py -m live.daily_scan --interval 1d --refresh
    py -m live.daily_scan --dry-run           # never send Telegram (just log)

If watchlist.json is missing/empty, falls back to every cached ticker with each
strategy's DEFAULT_PARAMS so the scan still does something useful.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timezone

import config
import dataio
import download_data
from strategies import STRATEGIES, STRATEGY_MODULES
from backtest import engine
from notify import telegram_bot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("daily_scan")


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------
def load_watchlist(interval: str) -> list[dict]:
    if config.WATCHLIST_JSON.exists():
        try:
            items = json.loads(config.WATCHLIST_JSON.read_text())
            items = [it for it in items if it.get("interval", "1d") == interval]
            if items:
                return items
        except Exception as e:
            log.warning("could not read watchlist (%s); using fallback", e)

    # fallback: default-param strategies on every cached ticker
    log.info("watchlist empty for %s -> fallback to DEFAULT_PARAMS on cached tickers", interval)
    items = []
    for ticker in dataio.available_tickers(interval):
        for strat in STRATEGY_MODULES:
            items.append({
                "ticker": ticker,
                "strategy": strat.NAME,
                "params": dict(strat.DEFAULT_PARAMS),
                "interval": interval,
            })
    return items


# ---------------------------------------------------------------------------
# Signals storage
# ---------------------------------------------------------------------------
def _ensure_table(con):
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {config.SIGNALS_TABLE} (
            detected_at TEXT,
            ticker      TEXT,
            strategy    TEXT,
            params      TEXT,
            interval    TEXT,
            signal      TEXT,
            bar_time    TEXT,
            price       REAL,
            UNIQUE(ticker, strategy, params, interval, bar_time, signal)
        )
    """)


def _already_logged(con, row) -> bool:
    cur = con.execute(
        f"""SELECT 1 FROM {config.SIGNALS_TABLE}
            WHERE ticker=? AND strategy=? AND params=? AND interval=? AND bar_time=? AND signal=?""",
        (row["ticker"], row["strategy"], row["params"], row["interval"],
         row["bar_time"], row["signal"]),
    )
    return cur.fetchone() is not None


def _insert(con, row):
    con.execute(
        f"""INSERT OR IGNORE INTO {config.SIGNALS_TABLE}
            (detected_at, ticker, strategy, params, interval, signal, bar_time, price)
            VALUES (:detected_at,:ticker,:strategy,:params,:interval,:signal,:bar_time,:price)""",
        row,
    )


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------
def detect_signal(item: dict, interval: str):
    """Return a signal dict if the latest bar fired BUY/SELL, else None."""
    ticker = item["ticker"]
    strat = STRATEGIES.get(item["strategy"])
    if strat is None:
        log.warning("%s: unknown strategy %s; skipping", ticker, item["strategy"])
        return None
    params = item.get("params", dict(strat.DEFAULT_PARAMS))

    data = dataio.load(ticker, interval)
    if data is None or data.empty:
        return None
    df = engine._clean_ohlcv(data)
    if df.shape[0] < 30:
        return None

    entries, exits = strat.generate_signals(df, **params)
    entries = entries.reindex(df.index).fillna(False).astype(bool)
    exits = exits.reindex(df.index).fillna(False).astype(bool)

    last_ts = df.index[-1]
    last_price = float(df["Close"].iloc[-1])

    signal = None
    if bool(entries.iloc[-1]):
        signal = "BUY"
    elif bool(exits.iloc[-1]):
        signal = "SELL"
    if signal is None:
        return None

    return {
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "strategy": strat.NAME,
        "params": engine._params_str(params),
        "interval": interval,
        "signal": signal,
        "bar_time": last_ts.isoformat(),
        "price": last_price,
    }


def format_alert(new_signals: list[dict]) -> str:
    lines = ["<b>ASX Strategy Lab — new signals</b>"]
    for s in new_signals:
        emoji = "🟢" if s["signal"] == "BUY" else "🔴"
        lines.append(
            f"{emoji} <b>{s['signal']}</b> {s['ticker']} — {s['strategy']} ({s['params']})\n"
            f"    price {s['price']:.3f} @ {s['bar_time']}"
        )
    lines.append("\n<i>Paper-trading signal, not advice. Past results don't guarantee future returns.</i>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(interval="1d", refresh=False, dry_run=False):
    items = load_watchlist(interval)
    log.info("scanning %d watchlist combos (interval=%s, refresh=%s)", len(items), interval, refresh)

    if refresh and items:
        tickers = sorted({it["ticker"] for it in items})
        log.info("refreshing latest data for %d tickers...", len(tickers))
        download_data.run(tickers, interval)

    con = sqlite3.connect(config.SIGNALS_DB)
    _ensure_table(con)

    fired, new_signals = 0, []
    for it in items:
        try:
            sig = detect_signal(it, interval)
        except Exception as e:
            log.error("%s/%s: detect failed (%s)", it.get("ticker"), it.get("strategy"), e)
            continue
        if sig is None:
            continue
        fired += 1
        if _already_logged(con, sig):
            log.info("  (already logged) %s %s %s @ %s",
                     sig["signal"], sig["ticker"], sig["strategy"], sig["bar_time"])
            continue
        _insert(con, sig)
        new_signals.append(sig)
        log.info("  NEW %s %s %s (%s) price=%.3f @ %s",
                 sig["signal"], sig["ticker"], sig["strategy"], sig["params"],
                 sig["price"], sig["bar_time"])
    con.commit()
    con.close()

    log.info("latest-bar signals fired: %d | new (not previously logged): %d", fired, len(new_signals))

    if new_signals and not dry_run:
        sent = telegram_bot.send_message(format_alert(new_signals))
        log.info("telegram: %s", "sent" if sent else "dry-run/not configured")
    elif new_signals and dry_run:
        log.info("dry-run: %d new signals not sent", len(new_signals))
    else:
        log.info("no new signals to notify")

    return new_signals


def main(argv=None):
    p = argparse.ArgumentParser(description="Daily signal scan + Telegram alerts (paper trading).")
    p.add_argument("--interval", default="1d")
    p.add_argument("--refresh", action="store_true", help="update OHLCV cache before scanning")
    p.add_argument("--dry-run", action="store_true", help="detect + log but never send Telegram")
    args = p.parse_args(argv)
    run(interval=args.interval, refresh=args.refresh, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
