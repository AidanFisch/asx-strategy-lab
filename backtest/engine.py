"""
Backtest engine: run one strategy (with one parameter set) over one ticker's
OHLCV and return a dict of performance metrics.

Execution model & assumptions
-----------------------------
* Long-only, all-in/all-out on a single ticker (init cash = config.INIT_CASH).
* A signal marks a bar; the fill happens at THAT bar's close. This is the
  standard vectorbt convention and carries a mild same-bar lookahead — signals
  are computed from the close we also trade at. It's a known simplification;
  Phase 3+ can switch to next-bar-open fills for extra conservatism.
* Frictions are applied on every fill: `fees` (brokerage) and `slippage`
  (proxy for the ASX bid/ask spread). Both come from config. Ignoring these
  flatters results, so they're on by default.
* Annualised metrics (Sharpe, CAGR) use the interval's pandas freq. Intraday
  annualisation is distorted by exchange-hours-only bars — compare intraday
  metrics within the same interval, not against daily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

import config


def _clean_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Drop all-NaN rows and zero/na-volume bars (opening-auction / illiquid artefacts)."""
    df = data.copy()
    df = df.dropna(subset=["Close"])
    if "Volume" in df.columns:
        df = df[df["Volume"].fillna(0) > 0]
    return df


def build_portfolio(data, strategy, params, interval="1d"):
    """Return (portfolio, entries, exits, close) for a strategy/params on one ticker."""
    df = _clean_ohlcv(data)
    close = df["Close"]
    entries, exits = strategy.generate_signals(df, **params)

    # align + coerce to clean boolean Series on the traded index
    entries = entries.reindex(close.index).fillna(False).astype(bool)
    exits = exits.reindex(close.index).fillna(False).astype(bool)

    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=config.INIT_CASH,
        fees=config.FEES,
        slippage=config.SLIPPAGE,
        freq=config.freq_for(interval),
    )
    return pf, entries, exits, close


def _safe(fn, default=np.nan):
    """Call a metric fn that may blow up / divide-by-zero when there are no trades."""
    try:
        v = fn()
        if v is None:
            return default
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return default
        return v
    except Exception:
        return default


def run_backtest(data, strategy, params, ticker=None, interval="1d") -> dict:
    """
    Backtest one strategy+params on one ticker. Returns a flat dict of metrics,
    safe to write to a leaderboard row. Never raises on empty/degenerate input.
    """
    result = {
        "ticker": ticker,
        "strategy": strategy.NAME,
        "params": _params_str(params),
        "interval": interval,
        **{f"p_{k}": v for k, v in params.items()},
    }

    df = _clean_ohlcv(data)
    if df.shape[0] < 30:  # not enough bars to say anything
        result.update(_empty_metrics(n_bars=df.shape[0]))
        return result

    pf, entries, exits, close = build_portfolio(data, strategy, params, interval)
    trades = pf.trades

    n_trades = int(_safe(lambda: trades.count(), 0) or 0)

    result.update({
        "n_bars": int(df.shape[0]),
        "start": df.index.min().isoformat(),
        "end": df.index.max().isoformat(),
        "total_return": float(_safe(pf.total_return)),
        "cagr": float(_safe(pf.annualized_return)),
        "sharpe": float(_safe(pf.sharpe_ratio)),
        "max_drawdown": float(_safe(pf.max_drawdown)),
        "n_trades": n_trades,
        "win_rate": float(_safe(lambda: trades.win_rate())) if n_trades else np.nan,
        "profit_factor": float(_safe(lambda: trades.profit_factor())) if n_trades else np.nan,
        "avg_pnl": float(_safe(lambda: trades.pnl.mean())) if n_trades else np.nan,
        "buy_hold_return": float(_safe(lambda: close.iloc[-1] / close.iloc[0] - 1.0)),
    })
    return result


def trade_log(data, strategy, params, interval="1d") -> pd.DataFrame:
    """Human-readable trade log for manual validation."""
    pf, *_ = build_portfolio(data, strategy, params, interval)
    tl = pf.trades.records_readable
    return tl


def _params_str(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in params.items())


def parse_params(s: str) -> dict:
    """Inverse of _params_str: 'fast=20,slow=50' -> {'fast':20,'slow':50} with typed values."""
    out = {}
    if not s:
        return out
    for part in s.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def _empty_metrics(n_bars=0) -> dict:
    return {
        "n_bars": int(n_bars), "start": None, "end": None,
        "total_return": np.nan, "cagr": np.nan, "sharpe": np.nan,
        "max_drawdown": np.nan, "n_trades": 0, "win_rate": np.nan,
        "profit_factor": np.nan, "avg_pnl": np.nan, "buy_hold_return": np.nan,
    }
