"""
Liquidity layer: is a signal actually tradeable in your size?

A backtest can 'buy' any amount at the close; reality can't. For each ticker we
compute Average Daily dollar Volume (ADV) — median of Close x Volume over the
last ~60 days — and convert it to AUD so names across markets are comparable.
Then we tier it and estimate the biggest position you could take without being
more than ~2% of a day's volume (a rough fill-without-moving-price guide).

Thin small-caps (several of the 'high confidence' gold miners) get flagged so you
don't act on a signal you can't realistically fill.

Writes a `liquidity` table.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import yfinance as yf

import config
import dataio

ADV_WINDOW = 60
MARKET_CCY = {"Australia": "AUD", "Hong Kong": "HKD", "Japan": "JPY", "Korea": "KRW",
              "Taiwan": "TWD", "Singapore": "SGD", "India": "INR", "Other": "USD"}
SUFFIX_MARKET = {".AX": "Australia", ".HK": "Hong Kong", ".T": "Japan", ".KS": "Korea",
                 ".KQ": "Korea", ".TW": "Taiwan", ".SI": "Singapore", ".NS": "India", ".BO": "India"}

# AUD/day thresholds
TIER_LIQUID = 5_000_000
TIER_OK = 1_000_000
POS_FRAC = 0.02   # a position up to 2% of ADV is a reasonable fill assumption


def market_of(t):
    for suf, m in SUFFIX_MARKET.items():
        if str(t).endswith(suf):
            return m
    return "Other"


def fx_to_aud(currencies) -> dict:
    """Latest FX rate: 1 unit of local currency -> AUD, per currency code."""
    rates = {"AUD": 1.0}
    for ccy in sorted(set(currencies) - {"AUD"}):
        rate = np.nan
        for sym in (f"{ccy}AUD=X", f"AUD{ccy}=X"):
            try:
                h = yf.Ticker(sym).history(period="5d")["Close"].dropna()
                if len(h):
                    v = float(h.iloc[-1])
                    rate = v if sym.startswith(ccy) else 1.0 / v
                    break
            except Exception:
                continue
        rates[ccy] = rate
    return rates


def adv_local(ticker, interval="1d", window=ADV_WINDOW):
    df = dataio.load(ticker, interval)
    if df is None or df.empty:
        return np.nan
    seg = df.tail(window)
    dv = (seg["Close"] * seg["Volume"]).replace(0, np.nan).dropna()
    return float(dv.median()) if len(dv) else np.nan


def tier(adv_aud):
    if not np.isfinite(adv_aud):
        return "unknown"
    if adv_aud >= TIER_LIQUID:
        return "liquid"
    if adv_aud >= TIER_OK:
        return "ok"
    return "thin"


def build(interval="1d", tickers=None) -> pd.DataFrame:
    tickers = tickers or dataio.available_tickers(interval)
    markets = {t: market_of(t) for t in tickers}
    ccys = {MARKET_CCY.get(m, "USD") for m in markets.values()}
    fx = fx_to_aud(ccys)

    rows = []
    for t in tickers:
        m = markets[t]
        ccy = MARKET_CCY.get(m, "USD")
        advl = adv_local(t, interval)
        rate = fx.get(ccy, np.nan)
        adv_aud = advl * rate if (np.isfinite(advl) and np.isfinite(rate)) else np.nan
        rows.append({
            "ticker": t, "market": m, "currency": ccy,
            "adv_local": advl, "fx_to_aud": rate, "adv_aud": adv_aud,
            "liquidity_tier": tier(adv_aud),
            "max_pos_aud": adv_aud * POS_FRAC if np.isfinite(adv_aud) else np.nan,
        })
    return pd.DataFrame(rows)


def save(df):
    if df.empty:
        return
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        con.execute("DROP TABLE IF EXISTS liquidity")
        df.to_sql("liquidity", con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    args = ap.parse_args(argv)
    df = build(args.interval)
    if df.empty:
        print("No tickers cached.")
        return 1
    save(df)
    counts = df["liquidity_tier"].value_counts().to_dict()
    print(f"Liquidity computed for {len(df)} tickers: {counts}")

    # flag thin names among the recommended plans
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        plans = pd.read_sql("SELECT ticker, strategy, recommended FROM plans", con)
    except Exception:
        plans = pd.DataFrame()
    con.close()
    if not plans.empty:
        rec = plans[plans["recommended"] == 1].merge(df, on="ticker", how="left")
        thin = rec[rec["liquidity_tier"].isin(["thin", "unknown"])]
        print(f"\nRecommended plans that are THIN (hard to fill): {len(thin)}/{len(rec)}")
        if len(thin):
            s = thin[["ticker", "market", "adv_aud", "max_pos_aud"]].copy()
            s["adv_aud"] = (s["adv_aud"] / 1e6).round(2)
            s["max_pos_aud"] = (s["max_pos_aud"] / 1e3).round(0)
            s.columns = ["ticker", "market", "ADV_$m", "max_pos_$k"]
            print(s.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
