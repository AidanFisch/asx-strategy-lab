"""
Dashboard generator: turn the leaderboard into a "best strategy per ticker" view.

For every ticker it picks the best OUT-OF-SAMPLE-validated combo (ranked by
out-of-sample Sharpe, requiring enough trades in both periods), then renders a
self-contained, theme-aware HTML dashboard you can open in any browser — no
server, no internet. GitHub Actions can regenerate/publish it on a schedule.

Usage
-----
    py -m results.dashboard                       # -> results/dashboard.html
    py -m results.dashboard --rank oos_cagr --min-trades 15
    py -m results.dashboard --out somewhere.html
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone

import pandas as pd

import config

# Metrics you can SELECT each ticker's best combo by. Default is an IN-SAMPLE
# metric on purpose: choosing on out-of-sample would fit to the holdout and make
# the reported OOS returns optimistic (selection bias). Choose on IS, report OOS.
SELECTABLE = ["is_sharpe", "is_total_return", "is_cagr"]
SORTABLE = ["oos_cagr", "oos_sharpe", "oos_total_return", "is_sharpe"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _universe_meta() -> dict:
    """ticker -> {name, sector} from universe.csv (best-effort)."""
    try:
        u = pd.read_csv(config.UNIVERSE_CSV)
        u["ticker"] = u["ticker"].astype(str).str.strip()
        return {r.ticker: {"name": r.get("name", ""), "sector": r.get("sector", "")}
                for r in u.itertuples(index=False)}
    except Exception:
        return {}


def load_latest_runs(interval: str) -> pd.DataFrame:
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        df = pd.read_sql("SELECT * FROM runs", con)
    finally:
        con.close()
    if df.empty:
        return df
    df = df[df["interval"] == interval]
    if df.empty:
        return df
    return df[df["scan_id"] == df["scan_id"].max()].copy()


def best_per_ticker(df: pd.DataFrame, select_by: str, sort_by: str, min_trades: int) -> pd.DataFrame:
    """
    One row per ticker: the combo chosen by `select_by` (an IN-SAMPLE metric) among
    those with enough trades in both periods. Reported returns are OUT-OF-SAMPLE, so
    the selection never sees the holdout it's judged on. Table ordered by `sort_by`.
    """
    valid = df[(df["is_n_trades"].fillna(0) >= min_trades) &
               (df["oos_n_trades"].fillna(0) >= min_trades)].copy()
    valid = valid[valid[select_by].notna()]
    if valid.empty:
        return valid
    idx = valid.groupby("ticker")[select_by].idxmax()   # pick on in-sample
    best = valid.loc[idx].copy()
    best["beats_bh"] = best["oos_total_return"] > best["oos_buy_hold_return"]
    best["oos_holds"] = (best["oos_total_return"] > 0) & (best["oos_sharpe"] > 0)
    return best.sort_values(sort_by, ascending=False)


# ---------------------------------------------------------------------------
# Records / summary for the page
# ---------------------------------------------------------------------------
def _num(x):
    """JSON-safe float (None for NaN/inf)."""
    try:
        f = float(x)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def build_payload(interval: str, select_by: str, sort_by: str, min_trades: int) -> dict:
    df = load_latest_runs(interval)
    meta = _universe_meta()

    if df.empty:
        return {"records": [], "summary": {"error": "no scan data"}, "meta": {}}

    best = best_per_ticker(df, select_by, sort_by, min_trades)

    records = []
    for r in best.itertuples(index=False):
        m = meta.get(r.ticker, {})
        records.append({
            "ticker": r.ticker,
            "name": m.get("name", ""),
            "sector": m.get("sector", ""),
            "strategy": r.strategy,
            "params": r.params,
            "oos_cagr": _num(r.oos_cagr),
            "oos_total_return": _num(r.oos_total_return),
            "oos_sharpe": _num(r.oos_sharpe),
            "oos_max_drawdown": _num(r.oos_max_drawdown),
            "oos_win_rate": _num(r.oos_win_rate),
            "oos_n_trades": int(r.oos_n_trades) if _num(r.oos_n_trades) is not None else 0,
            "oos_buy_hold": _num(r.oos_buy_hold_return),
            "is_total_return": _num(r.is_total_return),
            "beats_bh": bool(r.beats_bh),
            "oos_holds": bool(r.oos_holds),
        })

    n = len(records)
    holds = [x for x in records if x["oos_holds"]]
    beats = [x for x in records if x["beats_bh"]]
    cagrs = sorted(x["oos_cagr"] for x in records if x["oos_cagr"] is not None)
    median_cagr = cagrs[len(cagrs) // 2] if cagrs else None

    # strategy win distribution
    dist = {}
    for x in records:
        dist[x["strategy"]] = dist.get(x["strategy"], 0) + 1
    dist = dict(sorted(dist.items(), key=lambda kv: -kv[1]))

    scan_id = df["scan_id"].iloc[0]
    summary = {
        "interval": interval,
        "select_by": select_by,
        "sort_by": sort_by,
        "min_trades": min_trades,
        "scan_id": scan_id,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_tickers": n,
        "n_holds": len(holds),
        "n_beats_bh": len(beats),
        "median_oos_cagr": median_cagr,
        "strategy_dist": dist,
        "top_strategy": next(iter(dist), None),
    }
    return {"records": records, "summary": summary}


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render_html(payload: dict, standalone: bool = True) -> str:
    data_json = json.dumps(payload)
    body = _TEMPLATE.replace("/*__DATA__*/null", data_json)
    if standalone:
        return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<title>ASX Strategy Lab — Best strategy per ticker</title></head>"
                f"<body>{body}</body></html>")
    return body


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate the best-strategy-per-ticker dashboard.")
    p.add_argument("--interval", default="1d")
    p.add_argument("--select-by", default="is_sharpe", choices=SELECTABLE,
                   help="in-sample metric used to CHOOSE each ticker's best combo (avoids holdout bias)")
    p.add_argument("--sort-by", default="oos_cagr", choices=SORTABLE,
                   help="how the displayed table is ordered")
    p.add_argument("--min-trades", type=int, default=config.MIN_TRADES)
    p.add_argument("--out", default=str(config.PROJECT_ROOT / "results" / "dashboard.html"))
    args = p.parse_args(argv)

    payload = build_payload(args.interval, args.select_by, args.sort_by, args.min_trades)
    html = render_html(payload, standalone=True)
    from pathlib import Path
    Path(args.out).write_text(html, encoding="utf-8")

    s = payload["summary"]
    if s.get("error"):
        print("No data:", s["error"], "- run the scanner first.")
        return 1
    print(f"Dashboard written to {args.out}")
    print(f"  selection metric (in-sample): {s['select_by']}  |  returns shown: out-of-sample")
    print(f"  tickers with a chosen strategy: {s['n_tickers']}")
    print(f"  OOS-robust (held up): {s['n_holds']}  |  beat buy&hold OOS: {s['n_beats_bh']}")
    print(f"  median OOS CAGR: {s['median_oos_cagr']}")
    print(f"  most common winning strategy: {s['top_strategy']}")
    return 0


# The page body: inline CSS + content shell + JS that renders from embedded JSON.
_TEMPLATE = r"""
<style>
  :root{
    --bg:#f7f8fa; --panel:#ffffff; --ink:#1a1d21; --muted:#6b7280; --line:#e5e7eb;
    --accent:#2563eb; --pos:#127a4b; --neg:#c0392b; --chip:#eef2ff; --chipink:#3730a3;
    --shadow:0 1px 2px rgba(0,0,0,.05),0 1px 3px rgba(0,0,0,.06);
  }
  @media (prefers-color-scheme:dark){
    :root{--bg:#0e1116;--panel:#161b22;--ink:#e6edf3;--muted:#8b949e;--line:#2a313c;
      --accent:#4d8bf0;--pos:#3fb950;--neg:#f85149;--chip:#1f2937;--chipink:#a5b4fc;
      --shadow:0 1px 2px rgba(0,0,0,.4);}
  }
  :root[data-theme=dark]{--bg:#0e1116;--panel:#161b22;--ink:#e6edf3;--muted:#8b949e;--line:#2a313c;
    --accent:#4d8bf0;--pos:#3fb950;--neg:#f85149;--chip:#1f2937;--chipink:#a5b4fc;--shadow:0 1px 2px rgba(0,0,0,.4);}
  :root[data-theme=light]{--bg:#f7f8fa;--panel:#fff;--ink:#1a1d21;--muted:#6b7280;--line:#e5e7eb;
    --accent:#2563eb;--pos:#127a4b;--neg:#c0392b;--chip:#eef2ff;--chipink:#3730a3;--shadow:0 1px 2px rgba(0,0,0,.05),0 1px 3px rgba(0,0,0,.06);}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
  .wrap{max-width:1180px;margin:0 auto;padding:28px 20px 60px;}
  h1{font-size:22px;margin:0 0 2px;letter-spacing:-.01em;}
  .sub{color:var(--muted);font-size:13px;margin-bottom:22px;}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:22px;}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;box-shadow:var(--shadow);}
  .card .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em;}
  .card .v{font-size:24px;font-weight:650;margin-top:4px;font-variant-numeric:tabular-nums;}
  .card .v small{font-size:13px;color:var(--muted);font-weight:500;}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);overflow:hidden;}
  .toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:12px 14px;border-bottom:1px solid var(--line);}
  .toolbar input,.toolbar select{background:var(--bg);color:var(--ink);border:1px solid var(--line);
    border-radius:8px;padding:7px 10px;font-size:13px;}
  .toolbar input{flex:1;min-width:160px;}
  .dist{padding:14px;border-bottom:1px solid var(--line);}
  .dist h3{margin:0 0 10px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);}
  .bar{display:flex;align-items:center;gap:10px;margin:5px 0;font-size:13px;}
  .bar .lbl{width:150px;color:var(--ink);}
  .bar .track{flex:1;background:var(--bg);border-radius:6px;overflow:hidden;height:16px;border:1px solid var(--line);}
  .bar .fill{height:100%;background:var(--accent);}
  .bar .cnt{width:34px;text-align:right;color:var(--muted);font-variant-numeric:tabular-nums;}
  .tblwrap{overflow-x:auto;}
  table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;}
  th,td{padding:9px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line);}
  th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left;}
  thead th{position:sticky;top:0;background:var(--panel);cursor:pointer;user-select:none;
    font-size:11px;text-transform:uppercase;letter-spacing:.03em;color:var(--muted);}
  thead th:hover{color:var(--ink);}
  tbody tr:hover{background:var(--chip);}
  .tk{font-weight:650;}
  .nm{color:var(--muted);font-size:12px;}
  .pos{color:var(--pos);} .neg{color:var(--neg);}
  .chip{display:inline-block;background:var(--chip);color:var(--chipink);border-radius:20px;
    padding:2px 9px;font-size:11px;font-weight:600;}
  .flag{font-size:11px;padding:2px 7px;border-radius:6px;border:1px solid var(--line);color:var(--muted);}
  .flag.on{color:var(--pos);border-color:var(--pos);}
  .foot{color:var(--muted);font-size:12px;margin-top:16px;line-height:1.6;}
  .themebtn{margin-left:auto;background:var(--bg);border:1px solid var(--line);color:var(--ink);
    border-radius:8px;padding:7px 10px;cursor:pointer;font-size:13px;}
</style>

<div class="wrap">
  <h1>ASX Strategy Lab — best strategy per ticker</h1>
  <div class="sub" id="subline"></div>

  <div class="cards" id="cards"></div>

  <div class="panel">
    <div class="toolbar">
      <input id="search" placeholder="Filter ticker / name / sector…">
      <select id="stratFilter"><option value="">All strategies</option></select>
      <select id="holdFilter">
        <option value="">All</option>
        <option value="hold">OOS-robust only</option>
        <option value="beat">Beat buy &amp; hold (OOS)</option>
      </select>
      <button class="themebtn" id="themebtn">◐ Theme</button>
    </div>
    <div class="dist" id="dist"></div>
    <div class="tblwrap"><table id="tbl"><thead></thead><tbody></tbody></table></div>
  </div>

  <div class="foot" id="foot"></div>
</div>

<script>
const PAYLOAD = /*__DATA__*/null;

const pct = v => v==null ? "—" : (v*100).toFixed(1)+"%";
const num = v => v==null ? "—" : v.toFixed(2);
const cls = v => v==null ? "" : (v>=0 ? "pos":"neg");

const COLS = [
  {k:"ticker", t:"Ticker", fmt:(v,r)=>`<span class="tk">${v}</span><div class="nm">${r.name||""}</div>`},
  {k:"sector", t:"Sector", fmt:v=>`<span class="nm">${v||""}</span>`},
  {k:"strategy", t:"Best strategy", fmt:(v,r)=>`<span class="chip">${v}</span><div class="nm">${r.params}</div>`},
  {k:"oos_cagr", t:"OOS CAGR", fmt:v=>`<span class="${cls(v)}">${pct(v)}</span>`},
  {k:"oos_total_return", t:"OOS Total", fmt:v=>`<span class="${cls(v)}">${pct(v)}</span>`},
  {k:"oos_buy_hold", t:"Buy&Hold", fmt:v=>`<span class="nm">${pct(v)}</span>`},
  {k:"oos_sharpe", t:"OOS Sharpe", fmt:v=>`<span class="${cls(v)}">${num(v)}</span>`},
  {k:"oos_max_drawdown", t:"OOS MaxDD", fmt:v=>`<span class="neg">${pct(v)}</span>`},
  {k:"oos_win_rate", t:"Win%", fmt:v=>pct(v)},
  {k:"oos_n_trades", t:"Trades", fmt:v=>v},
  {k:"oos_holds", t:"Robust", fmt:v=>`<span class="flag ${v?"on":""}">${v?"✓ holds":"—"}</span>`},
];

let rows = PAYLOAD.records.slice();
let sortK = PAYLOAD.summary.sort_by || "oos_cagr", sortDir = -1;

function renderCards(){
  const s = PAYLOAD.summary;
  const cards = [
    ["Tickers analysed", s.n_tickers],
    ["OOS-robust", `${s.n_holds} <small>/ ${s.n_tickers}</small>`],
    ["Beat buy &amp; hold", `${s.n_beats_bh} <small>/ ${s.n_tickers}</small>`],
    ["Median OOS CAGR", s.median_oos_cagr==null?"—":(s.median_oos_cagr*100).toFixed(1)+"%"],
    ["Top strategy", `<span style="font-size:16px">${s.top_strategy||"—"}</span>`],
  ];
  document.getElementById("cards").innerHTML = cards.map(
    ([k,v])=>`<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  document.getElementById("subline").innerHTML =
    `Best combo per ticker chosen on <b>in-sample ${s.select_by.replace("is_","")}</b>; `
    + `returns shown are <b>out-of-sample</b> (the honest holdout estimate) · `
    + `min ${s.min_trades} trades each period · scan ${s.scan_id} · generated ${s.generated}`;
}

function renderDist(){
  const d = PAYLOAD.summary.strategy_dist||{};
  const max = Math.max(1,...Object.values(d));
  document.getElementById("dist").innerHTML =
    `<h3>Winning strategy distribution</h3>` +
    Object.entries(d).map(([k,v])=>
      `<div class="bar"><div class="lbl">${k}</div>
       <div class="track"><div class="fill" style="width:${100*v/max}%"></div></div>
       <div class="cnt">${v}</div></div>`).join("");
}

function renderHead(){
  document.querySelector("#tbl thead").innerHTML =
    "<tr>"+COLS.map(c=>`<th data-k="${c.k}">${c.t}${sortK===c.k?(sortDir<0?" ▼":" ▲"):""}</th>`).join("")+"</tr>";
  document.querySelectorAll("#tbl thead th").forEach(th=>th.onclick=()=>{
    const k=th.dataset.k; if(sortK===k) sortDir*=-1; else {sortK=k; sortDir=-1;}
    draw();
  });
}

function draw(){
  const q=(document.getElementById("search").value||"").toLowerCase();
  const sf=document.getElementById("stratFilter").value;
  const hf=document.getElementById("holdFilter").value;
  let r = rows.filter(x=>{
    if(sf && x.strategy!==sf) return false;
    if(hf==="hold" && !x.oos_holds) return false;
    if(hf==="beat" && !x.beats_bh) return false;
    if(q){ const hay=(x.ticker+" "+x.name+" "+x.sector).toLowerCase(); if(!hay.includes(q)) return false; }
    return true;
  });
  r.sort((a,b)=>{
    let av=a[sortK], bv=b[sortK];
    if(typeof av==="string"){return sortDir*av.localeCompare(bv);}
    if(av==null)av=-Infinity; if(bv==null)bv=-Infinity;
    return sortDir*(av-bv);
  });
  document.querySelector("#tbl tbody").innerHTML = r.map(x=>
    "<tr>"+COLS.map(c=>`<td>${c.fmt(x[c.k],x)}</td>`).join("")+"</tr>").join("");
  document.getElementById("foot").innerHTML =
    `Showing ${r.length} of ${rows.length} tickers. `+
    `<b>Best strategy</b> = the combo with the highest <b>in-sample</b> ${PAYLOAD.summary.select_by.replace("is_","")} for that ticker; `+
    `every number in the table is its <b>out-of-sample</b> result on the held-out tail it was never chosen on. `+
    `Returns are net of 0.1% brokerage + 5bps slippage. `+
    `<i>Research/education only — not financial advice. Past backtests can mislead; even an out-of-sample win is one historical path, not a guarantee.</i>`;
  renderHead();
}

function initFilters(){
  const strats=[...new Set(rows.map(x=>x.strategy))].sort();
  document.getElementById("stratFilter").innerHTML =
    `<option value="">All strategies</option>`+strats.map(s=>`<option>${s}</option>`).join("");
  ["search","stratFilter","holdFilter"].forEach(id=>{
    document.getElementById(id).addEventListener("input",draw);
  });
  document.getElementById("themebtn").onclick=()=>{
    const cur=document.documentElement.getAttribute("data-theme");
    const next = cur==="dark"?"light":(cur==="light"?"dark":
      (matchMedia("(prefers-color-scheme:dark)").matches?"light":"dark"));
    document.documentElement.setAttribute("data-theme",next);
  };
}

renderCards(); renderDist(); initFilters(); draw();
</script>
"""


if __name__ == "__main__":
    raise SystemExit(main())
