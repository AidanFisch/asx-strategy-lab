"""
Phase 4 validation: prove signal detection, DB logging, dedup, and the Telegram
(dry-run) alert path work — by truncating BHP data to a bar we KNOW fired a
signal, then feeding it through the real daily_scan machinery.
"""

import sqlite3

import dataio
import config
from strategies import rsi_reversion
from backtest import engine
from live import daily_scan
from notify import telegram_bot

TICKER, INTERVAL = "BHP.AX", "1d"
PARAMS = dict(rsi_reversion.DEFAULT_PARAMS)

full = dataio.load(TICKER, INTERVAL)
df = engine._clean_ohlcv(full)
entries, exits = rsi_reversion.generate_signals(df, **PARAMS)

# find the most recent historical bar that fired a BUY entry
buy_bars = entries[entries].index
assert len(buy_bars) > 0, "no BUY signals in history?!"
signal_bar = buy_bars[-1]
truncated = full.loc[:signal_bar]   # make that bar the 'latest' bar
print(f"Chosen BUY bar: {signal_bar.date()}  (truncating history to end there, "
      f"{len(truncated)} bars)")

# monkeypatch dataio.load so detect_signal sees the truncated series as 'latest'
_orig = dataio.load
daily_scan.dataio.load = lambda t, i: truncated if t == TICKER else _orig(t, i)

item = {"ticker": TICKER, "strategy": "rsi_reversion", "params": PARAMS, "interval": INTERVAL}
sig = daily_scan.detect_signal(item, INTERVAL)
print("\ndetect_signal ->", sig)
assert sig and sig["signal"] == "BUY", "expected a BUY on the truncated last bar"

# exercise DB insert + dedup on a temp table row
con = sqlite3.connect(config.SIGNALS_DB)
daily_scan._ensure_table(con)
# start clean in case a prior run left this row behind
con.execute(f"DELETE FROM {config.SIGNALS_TABLE} WHERE bar_time=? AND ticker=?",
            (sig["bar_time"], sig["ticker"])); con.commit()
first = not daily_scan._already_logged(con, sig)
daily_scan._insert(con, sig); con.commit()
second = daily_scan._already_logged(con, sig)   # should now be True (deduped)
con.close()
print(f"\nDB logging: inserted_new={first}  dedup_on_reinsert={second}")
assert first and second, "dedup logic broken"

# render the alert (dry-run: prints, doesn't send unless creds are set)
print("\n--- Telegram alert preview (dry-run) ---")
print(daily_scan.format_alert([sig]))
print("\ntelegram configured:", telegram_bot.is_configured())

# clean up the test row so it doesn't pollute the real signals table
con = sqlite3.connect(config.SIGNALS_DB)
con.execute(f"DELETE FROM {config.SIGNALS_TABLE} WHERE bar_time=? AND ticker=?",
            (sig["bar_time"], sig["ticker"]))
con.commit(); con.close()
print("\n(cleaned up test signal row)")
print("PHASE 4 VALIDATION: PASSED")
