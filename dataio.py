"""Read cached OHLCV Parquet written by the Phase 1 pipeline."""

from __future__ import annotations

import pandas as pd

import config


def load(ticker: str, interval: str = "1d") -> pd.DataFrame | None:
    """Load one ticker's cached OHLCV, or None if not cached."""
    path = config.raw_dir_for(interval) / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df.sort_index()


def available_tickers(interval: str = "1d") -> list[str]:
    """Tickers that have a cached Parquet for this interval."""
    d = config.raw_dir_for(interval)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.parquet"))
