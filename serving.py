"""
Export the small "serving" tables from the big research leaderboard into a tiny
serving.db (a few hundred KB) that gets committed to git.

The daily cloud monitor only needs plans + the validation/liquidity summaries to
rate signals — not the 100k-row raw backtest grid. Keeping those in a small DB
means the daily run commits ~0.5MB instead of a 60MB+ blob every day/week.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

import config

SERVING_TABLES = ["plans", "robustness", "liquidity", "wfo", "plan_stats"]


def export():
    src = sqlite3.connect(config.LEADERBOARD_DB)
    dst = sqlite3.connect(config.SERVING_DB)
    try:
        n = {}
        for t in SERVING_TABLES:
            try:
                df = pd.read_sql(f"SELECT * FROM {t}", src)
                df.to_sql(t, dst, if_exists="replace", index=False)
                n[t] = len(df)
            except Exception as e:
                n[t] = f"skip ({e})"
        dst.commit()
    finally:
        src.close()
        dst.close()
    return n


if __name__ == "__main__":
    n = export()
    print(f"serving.db written: {config.SERVING_DB}")
    for t, c in n.items():
        print(f"  {t}: {c}")
