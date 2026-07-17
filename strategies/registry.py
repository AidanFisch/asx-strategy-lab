"""
Strategy registry (v2): a broad, curated roster across five families —
momentum, mean-reversion, breakout (with confirmation), price-action, and
composite/confluence strategies that stack several conditions (e.g. RSI +
Bollinger + a trend price-check) for consistency.

Each Strategy is a spec:
    entry(data, **params) -> bool Series   (when to go long)
    exit (data, **params) -> bool Series   (signal-based exit; stops/targets are
                                            layered on top by the backtest engine)
    param_grid                              (searched by the scanner)
    entry_desc / exit_desc                  (human-readable rule, {param} templated,
                                            shown in the dashboard and BUY/SELL alerts)

The engine adds stop-loss / take-profit / trailing / time exits on top of every
strategy, so "exit" here is only the strategy's own logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from . import primitives as P


@dataclass
class Strategy:
    name: str
    family: str
    entry: Callable
    exit: Callable
    param_grid: dict
    entry_desc: str = ""
    exit_desc: str = ""

    def describe(self, params: dict) -> dict:
        def fmt(s):
            try:
                return s.format(**params)
            except Exception:
                return s
        return {"entry": fmt(self.entry_desc), "exit": fmt(self.exit_desc)}


def _false_like(data):
    return pd.Series(False, index=data.index)


# ===========================================================================
# MOMENTUM / TREND
# ===========================================================================
def _ma_entry(d, fast, slow): return P.cross_up(P.sma(d["Close"], fast), P.sma(d["Close"], slow))
def _ma_exit(d, fast, slow):  return P.cross_down(P.sma(d["Close"], fast), P.sma(d["Close"], slow))

def _macd(d, fast, slow, signal):
    import vectorbt as vbt
    m = vbt.MACD.run(d["Close"], fast_window=fast, slow_window=slow, signal_window=signal)
    return m.macd, m.signal

def _macd_entry(d, fast, slow, signal):
    macd, sig = _macd(d, fast, slow, signal); return P.cross_up(macd, sig)
def _macd_exit(d, fast, slow, signal):
    macd, sig = _macd(d, fast, slow, signal); return P.cross_down(macd, sig)

def _adx_entry(d, adx_th):
    return P.all_true(P.cross_up(P.sma(d["Close"], 20), P.sma(d["Close"], 50)), P.adx(d) > adx_th)
def _adx_exit(d, adx_th):
    return P.cross_down(P.sma(d["Close"], 20), P.sma(d["Close"], 50))

def _roc_entry(d, n, th):
    r = P.roc(d["Close"], n); return P.all_true(r > th, P.rising(r))
def _roc_exit(d, n, th):
    return P.below(P.roc(d["Close"], n), 0)


# ===========================================================================
# MEAN REVERSION
# ===========================================================================
def _rsi_entry(d, period, oversold, overbought): return P.cross_down(P.rsi(d["Close"], period), oversold)
def _rsi_exit(d, period, oversold, overbought):  return P.cross_up(P.rsi(d["Close"], period), overbought)

def _rsi2_entry(d, period, th):
    return P.all_true(P.rsi(d["Close"], period) < th, P.above_sma(d["Close"], 200))
def _rsi2_exit(d, period, th):
    return P.above(d["Close"], P.sma(d["Close"], 5))

def _bbrev_entry(d, n, mult):
    lo, mid, up = P.bbands(d["Close"], n, mult); return P.below(d["Close"], lo)
def _bbrev_exit(d, n, mult):
    lo, mid, up = P.bbands(d["Close"], n, mult); return P.above(d["Close"], mid)

def _z_entry(d, n, zin): return P.below(P.zscore(d["Close"], n), -zin)
def _z_exit(d, n, zin):  return P.above(P.zscore(d["Close"], n), 0)

def _pctma_entry(d, n, x): return P.below(d["Close"], P.sma(d["Close"], n) * (1 - x))
def _pctma_exit(d, n, x):  return P.above(d["Close"], P.sma(d["Close"], n))


# ===========================================================================
# BREAKOUT (with confirmation)
# ===========================================================================
def _donvol_entry(d, n, m): return P.all_true(P.breakout_high(d, n), P.volume_surge(d))
def _donvol_exit(d, n, m):  return P.breakdown_low(d, m)

def _h52_entry(d): return P.all_true(P.new_high(d, 252), P.uptrend(d["Close"]))
def _h52_exit(d):  return P.breakdown_low(d, 50)

def _bbbrk_entry(d, n, mult):
    lo, mid, up = P.bbands(d["Close"], n, mult)
    return P.all_true(P.cross_up(d["Close"], up), P.volume_surge(d))
def _bbbrk_exit(d, n, mult):
    lo, mid, up = P.bbands(d["Close"], n, mult); return P.below(d["Close"], mid)

def _atrch_entry(d, n, k): return P.above(d["Close"], P.sma(d["Close"], n) + k * P.atr(d))
def _atrch_exit(d, n, k):  return P.below(d["Close"], P.sma(d["Close"], n))

def _nbar_entry(d, n, m):
    return P.all_true(P.breakout_high(d, n), P.rising(d["Close"], 1), P.rising(d["Close"].shift(1), 1))
def _nbar_exit(d, n, m): return P.breakdown_low(d, m)


# ===========================================================================
# PRICE ACTION
# ===========================================================================
def _turtle_entry(d, n): return P.breakout_high(d, n)
def _turtle_exit(d, n):  return P.breakdown_low(d, max(int(n / 2), 5))

def _pullback_entry(d, rsi_th):
    return P.all_true(P.uptrend(d["Close"]), P.rsi(d["Close"], 14) < rsi_th, P.rising(d["Close"], 1))
def _pullback_exit(d, rsi_th): return P.above(P.rsi(d["Close"], 14), 65)

def _gap_entry(d, x): return P.all_true(P.gap_up(d, x), P.volume_surge(d))
def _gap_exit(d, x):  return P.below(d["Close"], P.sma(d["Close"], 10))

def _srb_entry(d, n):
    near_support = d["Close"] <= P.rolling_low(d, n) * 1.03
    return P.all_true(near_support, P.rising(d["Close"], 1))
def _srb_exit(d, n): return P.above(d["Close"], P.sma(d["Close"], n))


# ===========================================================================
# COMPOSITE / CONFLUENCE (stacked conditions — the "consistency" plays)
# ===========================================================================
def _rsibb_entry(d, oversold, overbought, n, mult):
    lo, mid, up = P.bbands(d["Close"], n, mult)
    return P.all_true(P.rsi(d["Close"], 14) < oversold,
                      P.below(d["Close"], lo),
                      P.above_sma(d["Close"], 200))
def _rsibb_exit(d, oversold, overbought, n, mult):
    lo, mid, up = P.bbands(d["Close"], n, mult)
    return P.any_true(P.rsi(d["Close"], 14) > overbought, P.above(d["Close"], mid))

def _triple_entry(d, n, m):
    return P.all_true(P.breakout_high(d, n), P.volume_surge(d), P.above_sma(d["Close"], 200))
def _triple_exit(d, n, m): return P.breakdown_low(d, m)

def _rsimacd_entry(d, rsi_th):
    import vectorbt as vbt
    mac = vbt.MACD.run(d["Close"], fast_window=12, slow_window=26, signal_window=9)
    return P.all_true(P.rsi(d["Close"], 14) < rsi_th,
                      P.cross_up(mac.macd, mac.signal),
                      P.above_sma(d["Close"], 200))
def _rsimacd_exit(d, rsi_th):
    import vectorbt as vbt
    mac = vbt.MACD.run(d["Close"], fast_window=12, slow_window=26, signal_window=9)
    return P.any_true(P.rsi(d["Close"], 14) > 65, P.cross_down(mac.macd, mac.signal))

def _squeeze_entry(d, n, mult):
    lo, mid, up = P.bbands(d["Close"], n, mult)
    width = (up - lo) / mid
    squeezed = width < width.rolling(120).quantile(0.25)
    return P.all_true(squeezed.shift(1), P.cross_up(d["Close"], up), P.volume_surge(d))
def _squeeze_exit(d, n, mult):
    lo, mid, up = P.bbands(d["Close"], n, mult); return P.below(d["Close"], mid)


# ===========================================================================
# ROSTER
# ===========================================================================
ALL_STRATEGIES = [
    # --- momentum / trend ---
    Strategy("ma_crossover", "momentum", _ma_entry, _ma_exit,
             {"fast": [10, 20, 50], "slow": [50, 100, 200]},
             "SMA{fast} crosses above SMA{slow}", "SMA{fast} crosses below SMA{slow}"),
    Strategy("macd_momentum", "momentum", _macd_entry, _macd_exit,
             {"fast": [12], "slow": [26], "signal": [9]},
             "MACD({fast},{slow}) crosses above signal({signal})", "MACD crosses below signal"),
    Strategy("adx_trend", "momentum", _adx_entry, _adx_exit,
             {"adx_th": [20, 25]},
             "SMA20>SMA50 while ADX>{adx_th} (strong trend)", "SMA20 crosses below SMA50"),
    Strategy("roc_momentum", "momentum", _roc_entry, _roc_exit,
             {"n": [20, 60], "th": [5]},
             "{n}-day ROC>{th}% and rising", "{n}-day ROC turns negative"),

    # --- mean reversion ---
    Strategy("rsi_reversion", "mean_reversion", _rsi_entry, _rsi_exit,
             {"period": [7, 14], "oversold": [30], "overbought": [70, 80]},
             "RSI{period} drops below {oversold} (oversold)", "RSI{period} rises above {overbought}"),
    Strategy("rsi2_pullback", "mean_reversion", _rsi2_entry, _rsi2_exit,
             {"period": [2, 3], "th": [10, 15]},
             "RSI{period}<{th} while above SMA200 (Connors pullback)", "Close rises back above SMA5"),
    Strategy("bb_reversion", "mean_reversion", _bbrev_entry, _bbrev_exit,
             {"n": [20], "mult": [2.0, 2.5]},
             "Close below lower Bollinger({n},{mult})", "Close back above the middle band"),
    Strategy("zscore_reversion", "mean_reversion", _z_entry, _z_exit,
             {"n": [20], "zin": [2.0, 2.5]},
             "{n}-day z-score below -{zin}", "z-score back above 0"),
    Strategy("pct_below_ma", "mean_reversion", _pctma_entry, _pctma_exit,
             {"n": [50, 100], "x": [0.12]},
             "Close {x:.0%} below SMA{n}", "Close back above SMA{n}"),

    # --- breakout (confirmed) ---
    Strategy("donchian_breakout_vol", "breakout", _donvol_entry, _donvol_exit,
             {"n": [20, 55], "m": [20]},
             "Break above {n}-day high on a volume surge", "Break below {m}-day low"),
    Strategy("high_52w_breakout", "breakout", _h52_entry, _h52_exit,
             {},
             "New 52-week high while in an uptrend", "Break below 50-day low"),
    Strategy("bb_breakout_vol", "breakout", _bbbrk_entry, _bbbrk_exit,
             {"n": [20], "mult": [2.0]},
             "Close breaks above upper Bollinger on volume", "Close back below the middle band"),
    Strategy("atr_channel_breakout", "breakout", _atrch_entry, _atrch_exit,
             {"n": [20], "k": [1.5, 2.5]},
             "Close above SMA{n}+{k}xATR", "Close back below SMA{n}"),
    Strategy("nbar_high_confirmed", "breakout", _nbar_entry, _nbar_exit,
             {"n": [20], "m": [10]},
             "Break {n}-day high, confirmed by 2 rising closes", "Break below {m}-day low"),

    # --- price action ---
    Strategy("turtle_breakout", "price_action", _turtle_entry, _turtle_exit,
             {"n": [20, 55]},
             "Break above {n}-day high (turtle)", "Break below {n}/2-day low"),
    Strategy("pullback_uptrend", "price_action", _pullback_entry, _pullback_exit,
             {"rsi_th": [35, 40]},
             "Uptrend + RSI dips below {rsi_th} + a rising close", "RSI rises above 65"),
    Strategy("gap_up_go", "price_action", _gap_entry, _gap_exit,
             {"x": [0.02, 0.03]},
             "Gap up >{x:.0%} on a volume surge", "Close below SMA10"),
    Strategy("support_bounce", "price_action", _srb_entry, _srb_exit,
             {"n": [20, 50]},
             "Close within 3% of the {n}-day low and turning up", "Close back above SMA{n}"),

    # --- composite / confluence ---
    Strategy("rsi_bb_confluence", "composite", _rsibb_entry, _rsibb_exit,
             {"oversold": [30, 35], "overbought": [70], "n": [20], "mult": [2.0]},
             "RSI<{oversold} AND below lower BB AND above SMA200", "RSI>{overbought} OR back above BB middle"),
    Strategy("triple_confluence_breakout", "composite", _triple_entry, _triple_exit,
             {"n": [20, 55], "m": [20]},
             "{n}-day breakout AND volume surge AND above SMA200", "Break below {m}-day low"),
    Strategy("rsi_macd_confluence", "composite", _rsimacd_entry, _rsimacd_exit,
             {"rsi_th": [40, 45]},
             "RSI<{rsi_th} AND MACD cross-up AND above SMA200", "RSI>65 OR MACD cross-down"),
    Strategy("bb_squeeze_breakout", "composite", _squeeze_entry, _squeeze_exit,
             {"n": [20], "mult": [2.0]},
             "Bollinger squeeze then upper-band break on volume", "Close back below the middle band"),
]

STRATEGIES = {s.name: s for s in ALL_STRATEGIES}


def get(name) -> Strategy:
    return STRATEGIES[name]
