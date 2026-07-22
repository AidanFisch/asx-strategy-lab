"""
Out-of-universe (cross-ticker) generalization test — the single most decisive
check on whether this system has a REAL edge or is just curve-fitting noise.

The overfitting worry: we scan 25 strategies x 49 param sets x hundreds of tickers
and pick the best per ticker. With that many trials, some combos look brilliant by
pure luck. If the "edge" is luck, it will NOT transfer to tickers we didn't select
on. If it's real, it will.

The experiment (done entirely from stored OOS metrics in runs2 — no re-backtesting):

  1. Randomly split the ticker universe in half, stratified by market:
        A = "discovery"  (used to CHOOSE parameters)
        B = "held-out"   (used only to JUDGE them — zero influence on the choice)
  2. For each (strategy, exit_policy), pick the single param config with the best
     median OOS avg-return-per-trade across the DISCOVERY tickers (mimicking exactly
     what the real pipeline does when it selects a plan).
  3. Take that winning config and read its median OOS performance on the HELD-OUT
     tickers. Those tickers had no say in the selection.
  4. Repeat over many random splits so the verdict doesn't depend on one partition.

Read-out:
  * discovery vs held-out median edge (the drop = selection inflation / overfit tax)
  * retention = held-out / discovery  (1.0 = perfect transfer, 0 = pure noise)
  * % of strategies still profitable out-of-universe
  * Spearman rank correlation of per-strategy edge A vs B (are the good strategies
    consistently good regardless of which tickers you look at?)

Writes crossuniverse.json (repo root) + a `crossuniverse` row in serving.db for the
dashboard's "Is the edge real?" panel.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

import config

MIN_TRADES = 5        # ignore OOS records with too few trades to mean anything
N_SPLITS = 200        # random discovery/held-out partitions to average over
METRIC = "oos_avg_ret_pct"   # per-trade edge; comparable across tickers
SEED = 7


def _market(t: str) -> str:
    return "AU" if str(t).endswith(".AX") else "INTL"


def load_runs() -> pd.DataFrame:
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        df = pd.read_sql(
            "SELECT ticker, strategy, exit_policy, params, "
            "oos_avg_ret_pct, oos_sharpe, oos_n_trades, oos_win_rate "
            "FROM runs2", con)
    finally:
        con.close()
    df = df[df["oos_n_trades"] >= MIN_TRADES].copy()
    df["market"] = df["ticker"].map(_market)
    df = df[np.isfinite(df[METRIC])]
    return df


def build_matrix(df: pd.DataFrame):
    """Pivot to a (config x ticker) value matrix so each split is pure numpy.

    Each config = (strategy, exit_policy, params) has exactly one OOS value per
    ticker, so median-over-a-ticker-subset is just a masked column median."""
    df = df.copy()
    df["cfg"] = df["strategy"] + "\x1f" + df["exit_policy"] + "\x1f" + df["params"]
    piv = df.pivot_table(index="cfg", columns="ticker", values=METRIC, aggfunc="mean")
    tickers = list(piv.columns)
    mkt = np.array([_market(t) for t in tickers])
    cfgs = list(piv.index)
    strat = np.array([c.split("\x1f")[0] for c in cfgs])
    ep = np.array([c.split("\x1f")[1] for c in cfgs])
    grp = np.array([f"{s}|{e}" for s, e in zip(strat, ep)])   # strategy+exit group
    return piv.values.astype(float), np.array(tickers), mkt, strat, grp


def one_split(mat, mkt, strat, grp, rng):
    """One random discovery/held-out split -> list of (group, strategy, a, b)."""
    n = mat.shape[1]
    disc = np.zeros(n, dtype=bool)
    for m in np.unique(mkt):                     # stratify by market
        idx = np.where(mkt == m)[0].copy()
        rng.shuffle(idx)
        disc[idx[: len(idx) // 2]] = True
    held = ~disc
    with np.errstate(all="ignore"):
        medA = np.nanmedian(np.where(disc, mat, np.nan), axis=1)
        medB = np.nanmedian(np.where(held, mat, np.nan), axis=1)
    rows = []
    for g in np.unique(grp):
        gi = np.where((grp == g) & np.isfinite(medA))[0]
        if gi.size == 0:
            continue
        best = gi[np.argmax(medA[gi])]           # best params on discovery
        if not np.isfinite(medB[best]):
            continue
        rows.append((g, strat[best], float(medA[best]), float(medB[best])))
    return rows


def run() -> dict:
    df = load_runs()
    mat, tickers, mkt, strat, grp = build_matrix(df)
    rng = np.random.default_rng(SEED)
    # accumulate per (strategy|exit) across splits
    acc: dict[str, dict] = {}
    split_a, split_b, split_rho = [], [], []
    for _ in range(N_SPLITS):
        rows = one_split(mat, mkt, strat, grp, rng)
        if not rows:
            continue
        a_all = [r[2] for r in rows]
        b_all = [r[3] for r in rows]
        split_a.append(np.median(a_all))
        split_b.append(np.median(b_all))
        if len(a_all) >= 4:
            rho = stats.spearmanr(a_all, b_all).correlation
            if np.isfinite(rho):
                split_rho.append(rho)
        for key, sname, a, b in rows:
            d = acc.setdefault(key, {"strat": sname, "a": [], "b": []})
            d["a"].append(a)
            d["b"].append(b)

    per = []
    for key, d in acc.items():
        a = float(np.mean(d["a"]))
        b = float(np.mean(d["b"]))
        per.append({"key": key, "strategy": d["strat"],
                    "discovery": round(a, 4), "heldout": round(b, 4),
                    "retention": round(b / a, 3) if a > 1e-9 else None,
                    "heldout_pos": bool(b > 0), "n": len(d["a"])})
    per.sort(key=lambda r: r["heldout"], reverse=True)

    disc_med = float(np.median(split_a)) if split_a else None
    held_med = float(np.median(split_b)) if split_b else None
    n_strat = len(per)
    n_pos = sum(1 for p in per if p["heldout_pos"])
    out = {
        "metric": METRIC, "min_trades": MIN_TRADES, "n_splits": N_SPLITS,
        "n_tickers": int(df["ticker"].nunique()),
        "n_strategy_configs": n_strat,
        "discovery_edge": round(disc_med, 4) if disc_med is not None else None,
        "heldout_edge": round(held_med, 4) if held_med is not None else None,
        "retention": round(held_med / disc_med, 3) if (disc_med and disc_med > 1e-9) else None,
        "pct_heldout_positive": round(n_pos / n_strat, 3) if n_strat else None,
        "rank_corr": round(float(np.mean(split_rho)), 3) if split_rho else None,
        "per": per,
    }
    return out


def save(out: dict):
    (config.PROJECT_ROOT / "crossuniverse.json").write_text(json.dumps(out), encoding="utf-8")
    con = sqlite3.connect(config.SERVING_DB)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS crossuniverse (k TEXT PRIMARY KEY, json TEXT)")
        con.execute("INSERT OR REPLACE INTO crossuniverse VALUES ('cv', ?)", (json.dumps(out),))
        con.commit()
    finally:
        con.close()


def main(argv=None):
    out = run()
    save(out)
    print("=== Out-of-universe generalization test ===")
    print(f"metric={out['metric']}  tickers={out['n_tickers']}  "
          f"strategy-configs={out['n_strategy_configs']}  splits={out['n_splits']}")
    print(f"Discovery edge (selected on):   {out['discovery_edge']*100:+.2f}% / trade")
    print(f"Held-out edge (never selected): {out['heldout_edge']*100:+.2f}% / trade")
    print(f"Retention (held/disc):          {out['retention']}  "
          f"(1.0 = perfect transfer, ~0 = noise)")
    print(f"Strategies still profitable OOU: {out['pct_heldout_positive']*100:.0f}%")
    print(f"Rank correlation A vs B:         {out['rank_corr']}  "
          f"(are the good strategies consistently good?)")
    print()
    print("Top strategies by held-out edge:")
    for p in out["per"][:8]:
        print(f"  {p['strategy']:26} disc {p['discovery']*100:+5.2f}%  "
              f"held-out {p['heldout']*100:+5.2f}%  retention {p['retention']}")
    print("Worst (overfit / non-transferring):")
    for p in out["per"][-4:]:
        print(f"  {p['strategy']:26} disc {p['discovery']*100:+5.2f}%  "
              f"held-out {p['heldout']*100:+5.2f}%  retention {p['retention']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
