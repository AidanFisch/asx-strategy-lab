"""
Realistic take-home model — what actually reaches your bank after real fills and
the ATO, not what the backtest prints.

Three haircuts the raw backtest understates:
  1. SLIPPAGE — EOD market-on-close fills slip more than the 5bps baked into the
     backtest. We reconstruct a true gross return (adding back the backtest's own
     0.1%/side fee + 5bps slippage) and re-apply a realistic slippage per side.
  2. BROKERAGE — real CommSec commission (fixed-dollar tiers for ASX, %+min for
     International), which is brutal on small parcels — modelled per actual trade
     as the parcel compounds.
  3. TAX — Australian CGT. Trades held < 12 months get NO 50% discount and are
     taxed at your marginal rate. High-turnover trading is almost all short-term,
     so it's taxed at full marginal rate. Losses in a financial year offset gains
     (and net losses carry forward). This is the haircut people forget.

Produces a waterfall — gross -> after slippage -> after brokerage -> after tax —
for the high-confidence book at several parcel sizes, plus the effective annual
take-home. Writes aftertax.json + an `aftertax` row in serving.db for the
dashboard. NOT tax advice; a modelling aid — set MARGINAL_TAX_RATE to your rate.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

import brokerage
import config
from live import monitor

# The backtest's own frictions already inside plan_trades.ret (config.FEES per
# side + config.SLIPPAGE per side, entry and exit). Add back to recover gross.
_BAKED_ROUNDTRIP = 2 * config.FEES + 2 * config.SLIPPAGE


def _market(t: str) -> str:
    return "Australia" if str(t).endswith(".AX") else "International"


def _fy(dt: pd.Timestamp) -> str:
    """Australian financial year label for a date (Jul 1 – Jun 30)."""
    y = dt.year
    return f"FY{y+1}" if dt.month >= 7 else f"FY{y}"


def load_trades(tier="high_conf") -> pd.DataFrame:
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        plans = monitor.enrich_plans(pd.read_sql("SELECT * FROM plans WHERE recommended=1", con))
        pt = pd.read_sql("SELECT * FROM plan_trades WHERE status='Closed'", con)
    finally:
        con.close()
    sel = plans[plans["high_conf"] == True] if tier == "high_conf" else plans  # noqa: E712
    tickers = set(sel["ticker"])
    pt = pt[pt["ticker"].isin(tickers)].copy()
    pt["entry_date"] = pd.to_datetime(pt["entry_date"], errors="coerce")
    pt["exit_date"] = pd.to_datetime(pt["exit_date"], errors="coerce")
    pt = pt.dropna(subset=["entry_date", "exit_date"]).sort_values("entry_date")
    # true gross per-trade return = backtest return + baked-in frictions added back
    pt["gross_ret"] = pt["ret"] + _BAKED_ROUNDTRIP
    pt["hold_days"] = (pt["exit_date"] - pt["entry_date"]).dt.days
    pt["fy"] = pt["exit_date"].map(_fy)
    pt["market"] = pt["ticker"].map(_market)
    return pt


def simulate(pt: pd.DataFrame, stake: float):
    """Compound `stake` per ticker through its trades at four transparency levels.
    Returns dict of end-values + a per-trade ledger (for the tax pass)."""
    slip = config.REAL_SLIPPAGE_PER_SIDE
    ledger = []           # realised $ P&L per trade, at the 'after brokerage' level
    ends = {"gross": 0.0, "slip": 0.0, "broker": 0.0}
    for tk, g in pt.groupby("ticker"):
        mk = _market(tk)
        vg = vs = vb = stake
        for _, tr in g.iterrows():
            r = float(tr["gross_ret"])
            vg *= (1 + r)                                   # pure price edge
            # realistic slippage: lose `slip` on entry and on exit
            r_slip = (1 + r) * (1 - slip) * (1 - slip) - 1
            vs *= (1 + r_slip)
            # + real CommSec brokerage on the compounding parcel
            buyc = brokerage.commission(vb, mk)
            exitv = (vb - buyc) * (1 + r_slip)
            sellc = brokerage.commission(exitv, mk)
            pnl = exitv - sellc - vb
            ledger.append({"fy": tr["fy"], "hold_days": int(tr["hold_days"]), "pnl": pnl})
            vb = vb + pnl
        ends["gross"] += vg
        ends["slip"] += vs
        ends["broker"] += vb
    return ends, pd.DataFrame(ledger)


def apply_tax(ledger: pd.DataFrame) -> tuple[float, list]:
    """Australian CGT across financial years. Losses offset gains within a FY;
    net FY losses carry forward. <12mo gains taxed at full marginal rate; >=12mo
    gains get the 50% discount. Returns (total_tax, per-FY breakdown)."""
    rate = config.MARGINAL_TAX_RATE
    carry = 0.0
    total_tax = 0.0
    rows = []
    for fy in sorted(ledger["fy"].unique()):
        f = ledger[ledger["fy"] == fy]
        short = f[f["hold_days"] < config.CGT_DISCOUNT_DAYS]["pnl"].sum()
        long = f[f["hold_days"] >= config.CGT_DISCOUNT_DAYS]["pnl"].sum()
        net = short + long - carry            # apply carried-forward losses
        if net <= 0:
            carry = -net                      # more losses to carry
            rows.append({"fy": fy, "net_gain": round(short + long), "taxable": 0, "tax": 0})
            continue
        carry = 0.0
        # discount only the long-term portion, and only what survives loss-netting
        # (short-term gains use their discount-free rate first).
        if short >= net:                      # net gain fully explained by short-term
            taxable = net
        else:
            long_kept = net - max(short, 0)
            taxable = max(short, 0) + long_kept * (1 - config.CGT_DISCOUNT)
        tax = taxable * rate
        total_tax += tax
        rows.append({"fy": fy, "net_gain": round(short + long), "taxable": round(taxable), "tax": round(tax)})
    return total_tax, rows


def run(tier="high_conf") -> dict:
    pt = load_trades(tier)
    if pt.empty:
        return {"error": "no trades"}
    n_tickers = pt["ticker"].nunique()
    short_frac = float((pt["hold_days"] < config.CGT_DISCOUNT_DAYS).mean())
    sweep = []
    for stake in (1000, 2500, 5000, 10000, 25000):
        ends, ledger = simulate(pt, float(stake))
        tax, fy_rows = apply_tax(ledger)
        invested = stake * n_tickers
        after_tax = ends["broker"] - tax
        yrs = (pt["exit_date"].max() - pt["entry_date"].min()).days / 365.25
        cagr = (after_tax / invested) ** (1 / yrs) - 1 if after_tax > 0 and yrs > 0 else None
        sweep.append({
            "stake": stake, "invested": round(invested),
            "gross": round(ends["gross"]), "after_slip": round(ends["slip"]),
            "after_broker": round(ends["broker"]), "tax": round(tax),
            "after_tax": round(after_tax),
            "gross_pct": round(ends["gross"] / invested - 1, 4),
            "net_pct": round(after_tax / invested - 1, 4),
            "cagr": round(cagr, 4) if cagr is not None else None,
            "fy": fy_rows,
        })
    return {
        "tier": tier, "n_tickers": int(n_tickers), "n_trades": int(len(pt)),
        "span_start": str(pt["entry_date"].min())[:10], "span_end": str(pt["exit_date"].max())[:10],
        "years": round((pt["exit_date"].max() - pt["entry_date"].min()).days / 365.25, 1),
        "short_term_frac": round(short_frac, 3),
        "assumptions": {"slippage_per_side": config.REAL_SLIPPAGE_PER_SIDE,
                        "marginal_rate": config.MARGINAL_TAX_RATE,
                        "baked_roundtrip": round(_BAKED_ROUNDTRIP, 4)},
        "sweep": sweep,
    }


def save(out: dict):
    (config.PROJECT_ROOT / "aftertax.json").write_text(json.dumps(out), encoding="utf-8")
    con = sqlite3.connect(config.SERVING_DB)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS aftertax (k TEXT PRIMARY KEY, json TEXT)")
        con.execute("INSERT OR REPLACE INTO aftertax VALUES ('at', ?)", (json.dumps(out),))
        con.commit()
    finally:
        con.close()


def main(argv=None):
    out = run("high_conf")
    save(out)
    a = out["assumptions"]
    print("=== Realistic take-home (high-confidence book) ===")
    print(f"{out['n_tickers']} tickers, {out['n_trades']} trades, {out['span_start']}->{out['span_end']} "
          f"(~{out['years']}y).  {out['short_term_frac']*100:.0f}% of trades held <12mo (full marginal-rate tax).")
    print(f"Assumptions: slippage {a['slippage_per_side']*100:.2f}%/side, "
          f"marginal rate {a['marginal_rate']*100:.1f}%, backtest already had "
          f"{a['baked_roundtrip']*100:.2f}% round-trip friction (added back to gross).")
    print()
    print(f"{'parcel':>8} {'invested':>9} {'gross':>10} {'+slip':>10} {'+broker':>10} {'tax':>8} {'AFTER-TAX':>11} {'net%':>7} {'CAGR':>7}")
    for s in out["sweep"]:
        cg = f"{s['cagr']*100:.0f}%" if s["cagr"] is not None else "—"
        print(f"${s['stake']:>7,} ${s['invested']:>8,} ${s['gross']:>9,} ${s['after_slip']:>9,} "
              f"${s['after_broker']:>9,} ${s['tax']:>7,} ${s['after_tax']:>10,} "
              f"{s['net_pct']*100:>6.0f}% {cg:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
