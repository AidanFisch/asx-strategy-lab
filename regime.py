"""
Market-regime overlay (the one AI/quant addition worth doing).

Idea: don't fight the tide. For each market we classify a daily regime from its
index vs its 200-day moving average (bull = index above its 200d MA). We then
test whether running each portfolio sleeve ONLY when its market is in a bull
regime (otherwise sit in cash) improves the diversified book — lower drawdown,
better Sharpe.

This is a transparent rule, not a black box, and it's cheap. We DELIBERATELY do
not add an ML meta-filter or Qlib: with the book already at Sharpe ~1.6 on a thin
underlying edge, those mostly manufacture overfit confidence.

Compares regime-filtered vs unfiltered books and prints the verdict. Saves a
regime-filtered portfolio ("*_regime") so the dashboard can show it if it helps.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import yfinance as yf

import config
import portfolio as pf_mod

MARKET_INDEX = {
    "Australia": "^AXJO", "Hong Kong": "^HSI", "Japan": "^N225", "Korea": "^KS11",
    "Taiwan": "^TWII", "Singapore": "^STI", "India": "^NSEI", "Other": "^GSPC",
}
SUFFIX_MARKET = {".AX": "Australia", ".HK": "Hong Kong", ".T": "Japan", ".KS": "Korea",
                 ".KQ": "Korea", ".TW": "Taiwan", ".SI": "Singapore", ".NS": "India", ".BO": "India"}


def market_of(t):
    for suf, m in SUFFIX_MARKET.items():
        if str(t).endswith(suf):
            return m
    return "Other"


def market_regimes() -> dict:
    """Per-market daily boolean: True = index above its 200-day MA (bull)."""
    reg = {}
    for market, sym in MARKET_INDEX.items():
        try:
            h = yf.Ticker(sym).history(period="max")["Close"].dropna()
            if len(h) < 220:
                continue
            ok = h > h.rolling(200).mean()
            ok.index = pd.to_datetime(ok.index).tz_localize(None)
            reg[market] = ok
        except Exception:
            continue
    return reg


def _metrics(daily):
    return pf_mod._metrics(daily)


def build(interval="1d"):
    hc, rec = pf_mod.load_plan_sets(interval)
    reg = market_regimes()
    results = []
    for label, dfp in [("recommended", rec), ("high_confidence", hc)]:
        cols, markets = {}, {}
        for _, p in dfp.iterrows():
            sr, _ = pf_mod.plan_daily_returns(p["ticker"], p["strategy"], p["params"], p["exit_policy"])
            if sr is not None and len(sr) > 20:
                cols[p["ticker"]] = sr
                markets[p["ticker"]] = market_of(p["ticker"])
        if not cols:
            continue
        R = pd.DataFrame(cols).sort_index()
        base = R.mean(axis=1, skipna=True)

        F = R.copy()
        for t in R.columns:
            ok = reg.get(markets[t])
            if ok is not None:
                okr = ok.reindex(R.index, method="ffill").fillna(True)
                F[t] = R[t].where(okr, 0.0)   # in cash when regime is bearish
        filt = F.mean(axis=1, skipna=True)

        mb, mf = _metrics(base), _metrics(filt)
        eqf = (1 + filt.dropna()).cumprod()
        step = max(1, len(eqf) // 250)
        idx = list(range(0, len(eqf), step))
        results.append({
            "label": label, "base": mb, "regime": mf,
            "equity_dates": [eqf.index[i].strftime("%Y-%m-%d") for i in idx],
            "equity_values": [round(float(eqf.iloc[i]), 4) for i in idx],
        })
    return results


def save(results):
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS regime (label TEXT PRIMARY KEY, json TEXT)")
        for r in results:
            con.execute("INSERT OR REPLACE INTO regime VALUES (?,?)", (r["label"], json.dumps(r)))
        con.commit()
    finally:
        con.close()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    args = ap.parse_args(argv)
    results = build(args.interval)
    if not results:
        print("No data.")
        return 1
    save(results)
    for r in results:
        b, f = r["base"], r["regime"]
        print(f"\n=== {r['label'].upper()} book: regime overlay vs plain ===")
        print(f"{'':16}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}")
        print(f"{'plain book':16}{b['cagr']:>8.1%}{b['sharpe']:>9.2f}{b['max_drawdown']:>8.1%}")
        print(f"{'regime-filtered':16}{f['cagr']:>8.1%}{f['sharpe']:>9.2f}{f['max_drawdown']:>8.1%}")
        better = f["sharpe"] > b["sharpe"]
        dd_better = f["max_drawdown"] > b["max_drawdown"]
        print(f"  -> regime overlay {'IMPROVES' if better else 'does NOT improve'} Sharpe; "
              f"drawdown {'smaller' if dd_better else 'not smaller'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
