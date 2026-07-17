"""
Risk-aware backtest engine (v2).

Every strategy from strategies/registry.py is tested WITH an exit policy layered
on top of its own signal exit:
  * stop-loss  (fixed % or ATR-multiple)
  * take-profit (fixed %)
  * trailing stop (%)
The position closes at whichever fires first — signal exit, stop, target, or
trail — which is exactly "sell when the signal says so OR risk says so".

EOD model: signals fire on a daily bar and fill at that bar's close (you place
the order into the close / next session). Frictions: config.FEES + config.SLIPPAGE
on every fill.

`evaluate()` returns one metrics dict per exit config, so the scanner can store
the full grid (strategy x params x exit policy) for every ticker.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import vectorbt as vbt

import config
from strategies import primitives as P

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("future.no_silent_downcasting", True)


# ---------------------------------------------------------------------------
# Exit policies (the risk grid). sl/tp/trail are fractions; atr_mult -> ATR stop.
# ---------------------------------------------------------------------------
EXIT_CONFIGS = {
    "signal_only": {},                              # rely on the strategy's own exit
    "sl_10":       {"sl": 0.10},
    "sl_15":       {"sl": 0.15},
    "trail_10":    {"sl": 0.10, "trail": True},
    "sl10_tp20":   {"sl": 0.10, "tp": 0.20},
    "atr_2x":      {"atr_mult": 2.0},               # stop = 2xATR below entry
}


def clean_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy().dropna(subset=["Close"])
    if "Volume" in df.columns:
        df = df[df["Volume"].fillna(0) > 0]
    return df


def build_signals(strategy, data, params):
    e = strategy.entry(data, **params).reindex(data.index).fillna(False).astype(bool)
    x = strategy.exit(data, **params).reindex(data.index).fillna(False).astype(bool)
    return e, x


def _stop_kwargs(cfg, data):
    """Translate an exit-config dict into vectorbt from_signals stop kwargs."""
    kw = {}
    if "atr_mult" in cfg:
        atr = P.atr(data, 14)
        kw["sl_stop"] = (atr * cfg["atr_mult"] / data["Close"]).clip(lower=1e-4)
    elif "sl" in cfg:
        kw["sl_stop"] = cfg["sl"]
        if cfg.get("trail"):
            kw["sl_trail"] = True
    if "tp" in cfg:
        kw["tp_stop"] = cfg["tp"]
    return kw


def _portfolio(data, entries, exits, cfg, interval):
    close = data["Close"]
    return vbt.Portfolio.from_signals(
        close, entries, exits,
        init_cash=config.INIT_CASH, fees=config.FEES, slippage=config.SLIPPAGE,
        freq=config.freq_for(interval), **_stop_kwargs(cfg, data),
    )


def _safe(fn, default=np.nan):
    try:
        v = fn()
        if v is None:
            return default
        if isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
            return default
        return float(v)
    except Exception:
        return default


def _trade_stats(pf, n_bars):
    """Per-trade risk stats computed from the trade returns array."""
    out = dict(avg_ret=np.nan, avg_win=np.nan, avg_loss=np.nan, payoff=np.nan, exposure=np.nan)
    try:
        rets = np.asarray(pf.trades.returns.values, dtype=float)
        rets = rets[~np.isnan(rets)]
        if rets.size:
            out["avg_ret"] = float(rets.mean())
            wins, losses = rets[rets > 0], rets[rets < 0]
            if wins.size:
                out["avg_win"] = float(wins.mean())
            if losses.size:
                out["avg_loss"] = float(losses.mean())
            if wins.size and losses.size and losses.mean() != 0:
                out["payoff"] = float(wins.mean() / abs(losses.mean()))
    except Exception:
        pass
    try:
        dur = np.asarray(pf.trades.duration.values, dtype=float)
        if n_bars:
            out["exposure"] = float(np.nansum(dur) / n_bars)
    except Exception:
        pass
    return out


def evaluate(data, strategy, params, interval="1d",
             exit_configs=None, ticker=None, min_bars=60) -> list[dict]:
    """Backtest one strategy+params across all exit configs. Returns list of metric dicts."""
    exit_configs = exit_configs or EXIT_CONFIGS
    df = clean_ohlcv(data)
    base = {
        "ticker": ticker, "strategy": strategy.name, "family": strategy.family,
        "params": _params_str(params), "interval": interval,
    }
    if df.shape[0] < min_bars:
        return [{**base, "exit_policy": name, **_empty()} for name in exit_configs]

    entries, exits = build_signals(strategy, df, params)
    close = df["Close"]
    bh = _safe(lambda: close.iloc[-1] / close.iloc[0] - 1.0)
    n_bars = df.shape[0]

    rows = []
    for name, cfg in exit_configs.items():
        row = {**base, "exit_policy": name, "n_bars": int(n_bars),
               "start": df.index.min().isoformat(), "end": df.index.max().isoformat(),
               "buy_hold_return": bh}
        try:
            pf = _portfolio(df, entries, exits, cfg, interval)
            n_trades = int(_safe(lambda: pf.trades.count(), 0) or 0)
            ts = _trade_stats(pf, n_bars)
            sl = cfg.get("sl") or cfg.get("atr_mult")
            row.update({
                "total_return": _safe(pf.total_return),
                "cagr": _safe(pf.annualized_return),
                "sharpe": _safe(pf.sharpe_ratio),
                "sortino": _safe(pf.sortino_ratio),
                "max_drawdown": _safe(pf.max_drawdown),
                "calmar": _safe(pf.calmar_ratio),
                "n_trades": n_trades,
                "win_rate": _safe(lambda: pf.trades.win_rate()) if n_trades else np.nan,
                "profit_factor": _safe(lambda: pf.trades.profit_factor()) if n_trades else np.nan,
                "avg_pnl": _safe(lambda: pf.trades.pnl.mean()) if n_trades else np.nan,
                "avg_ret_pct": ts["avg_ret"], "avg_win_pct": ts["avg_win"],
                "avg_loss_pct": ts["avg_loss"], "payoff": ts["payoff"],
                "exposure": ts["exposure"],
                # expectancy in R (avg return per trade / risk), only meaningful with a stop
                "expectancy_R": (ts["avg_ret"] / sl) if (sl and not np.isnan(ts["avg_ret"])) else np.nan,
            })
        except Exception:
            row.update(_empty())
        rows.append(row)
    return rows


def _params_str(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in params.items())


def parse_params(s: str) -> dict:
    out = {}
    for part in (s or "").split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1); k, v = k.strip(), v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def _empty() -> dict:
    keys = ["n_bars", "total_return", "cagr", "sharpe", "sortino", "max_drawdown",
            "calmar", "n_trades", "win_rate", "profit_factor", "avg_pnl",
            "avg_ret_pct", "avg_win_pct", "avg_loss_pct", "payoff", "exposure",
            "expectancy_R", "buy_hold_return"]
    d = {k: (0 if k == "n_trades" else np.nan) for k in keys}
    d["start"] = d["end"] = None
    return d
