#!/usr/bin/env python
"""Build a simple single-file dashboard from the CSVs in results/.

    python make_dashboard.py            # writes results/dashboard.html and opens it
    python make_dashboard.py --out results --no-open
"""
import argparse
import json
import os
import webbrowser

import pandas as pd

LABELS = {"none": "No mitigation", "rem": "Readout (REM)", "zne": "ZNE", "rem+zne": "REM + ZNE"}

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Error Mitigation Dashboard</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--text:#1c2430;--muted:#6b7686;--line:#e3e6eb;
--c0:#9aa3af;--c1:#2a78d6;--c2:#e8833a;--c3:#1f9d73}
@media(prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1d2128;--text:#e6e9ee;--muted:#8b95a5;--line:#2d333c;
--c0:#6b7686;--c1:#5aa0f0;--c2:#f0a066;--c3:#4cc79d}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:1000px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:22px;margin:0}.sub{color:var(--muted);margin:2px 0 22px}
h2{font-size:15px;margin:0 0 12px}
.grid{display:grid;gap:14px}.kpis{grid-template-columns:repeat(auto-fit,minmax(200px,1fr));margin-bottom:14px}
.two{grid-template-columns:repeat(auto-fit,minmax(380px,1fr));margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.kpi .v{font-size:28px;font-weight:600}.kpi .l{color:var(--muted);font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:0 0 14px;font-size:13px;color:var(--muted)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
select{background:var(--card);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:3px 8px;float:right}
svg{width:100%;height:auto;display:block}svg text{fill:var(--muted);font-size:11px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500}
</style></head><body><main>
<h1>Error Mitigation Dashboard</h1>
<p class="sub">Lower error is better. Mean absolute error of the estimated expectation value vs. the ideal.</p>
<div class="grid kpis" id="kpis"></div>
<div class="legend" id="legend"></div>
<div class="grid two">
  <div class="card"><h2>Overall error</h2><div id="overall"></div></div>
  <div class="card"><h2>Error by circuit</h2><div id="bench"></div></div>
</div>
<div class="card" style="margin-bottom:14px"><h2>Noise sweep
  <select id="sweep"><option value="sweep_gate">Gate noise &times;</option><option value="sweep_readout">Readout noise &times;</option></select></h2>
  <div id="sweepchart"></div></div>
<div class="card"><h2>Summary</h2><div id="table"></div></div>
</main>
<script>
const D=__DATA__;
const S=D.order,C={none:'var(--c0)',rem:'var(--c1)',zne:'var(--c2)','rem+zne':'var(--c3)'};
const f=(x,n=3)=>Number(x).toFixed(n);
const best=D.overall.reduce((a,b)=>a.mean_abs_error<b.mean_abs_error?a:b);
document.getElementById('kpis').innerHTML=[
 [D.labels[best.strategy],'Best strategy'],
 [f(best.error_reduction_x,1)+'×','Error reduction vs. none'],
 [f(best.mean_abs_error),'Best mean abs. error'],
 [f(best.cost_vs_none_x,0)+'×','Circuit cost of best']
].map(([v,l])=>`<div class="card kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');
document.getElementById('legend').innerHTML=S.map(s=>`<span><i style="background:${C[s]}"></i>${D.labels[s]}</span>`).join('');

function bars(el,groups,key){ // groups: [{name, vals:{strategy:value}}]
 const W=460,H=240,L=36,B=26,T=8,max=Math.max(...groups.flatMap(g=>S.map(s=>g.vals[s]||0)))*1.1||1;
 const gw=(W-L)/groups.length,bw=Math.min(24,(gw-12)/S.length);let o='';
 for(let i=0;i<=4;i++){const y=T+(H-T-B)*(1-i/4);o+=`<line x1="${L}" x2="${W}" y1="${y}" y2="${y}" stroke="var(--line)"/><text x="${L-6}" y="${y+4}" text-anchor="end">${f(max*i/4,2)}</text>`}
 groups.forEach((g,gi)=>{const x0=L+gi*gw+(gw-bw*S.length)/2;
  S.forEach((s,si)=>{const v=g.vals[s]||0,h=(H-T-B)*v/max;
   o+=`<rect x="${x0+si*bw}" y="${H-B-h}" width="${bw-2}" height="${h}" rx="2" fill="${C[s]}"><title>${g.name} – ${D.labels[s]}: ${f(v)}</title></rect>`});
  o+=`<text x="${L+gi*gw+gw/2}" y="${H-8}" text-anchor="middle">${g.name}</text>`});
 document.getElementById(el).innerHTML=`<svg viewBox="0 0 ${W} ${H}">${o}</svg>`}

bars('overall',[{name:'All circuits',vals:Object.fromEntries(D.overall.map(r=>[r.strategy,r.mean_abs_error]))}]);
const bn=[...new Set(D.summary.map(r=>r.benchmark))];
bars('bench',bn.map(b=>({name:b,vals:Object.fromEntries(D.summary.filter(r=>r.benchmark===b).map(r=>[r.strategy,r.mean_abs_error]))})));

function sweep(){
 const ex=document.getElementById('sweep').value,rows=D.sweep.filter(r=>r.experiment===ex);
 const xs=[...new Set(rows.map(r=>r.factor))].sort((a,b)=>a-b);
 const W=900,H=260,L=40,B=28,T=8,R=12;
 const val=(s,x)=>{const v=rows.filter(r=>r.strategy===s&&r.factor===x).map(r=>r.abs_error);return v.reduce((a,b)=>a+b,0)/v.length};
 const max=Math.max(...S.flatMap(s=>xs.map(x=>val(s,x))))*1.1||1;
 const px=x=>L+(W-L-R)*(xs.length>1?xs.indexOf(x)/(xs.length-1):.5),py=v=>T+(H-T-B)*(1-v/max);
 let o='';for(let i=0;i<=4;i++){const y=py(max*i/4);o+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="var(--line)"/><text x="${L-6}" y="${y+4}" text-anchor="end">${f(max*i/4,2)}</text>`}
 xs.forEach(x=>o+=`<text x="${px(x)}" y="${H-8}" text-anchor="middle">${x}×</text>`);
 S.forEach(s=>{o+=`<polyline fill="none" stroke="${C[s]}" stroke-width="2.5" points="${xs.map(x=>px(x)+','+py(val(s,x))).join(' ')}"/>`;
  xs.forEach(x=>o+=`<circle cx="${px(x)}" cy="${py(val(s,x))}" r="3.5" fill="${C[s]}"><title>${D.labels[s]} @ ${x}×: ${f(val(s,x))}</title></circle>`)});
 document.getElementById('sweepchart').innerHTML=`<svg viewBox="0 0 ${W} ${H}">${o}</svg>`}
document.getElementById('sweep').onchange=sweep;
if(D.sweep.length)sweep();else document.getElementById('sweepchart').textContent='No sweep data (run without --skip-sweeps).';

document.getElementById('table').innerHTML='<table><tr><th>Strategy</th><th>Mean abs. error</th><th>Error reduction</th><th>Circuit executions</th><th>Total shots</th></tr>'+
 D.overall.map(r=>`<tr><td>${D.labels[r.strategy]}</td><td>${f(r.mean_abs_error)}</td><td>${f(r.error_reduction_x,1)}×</td><td>${r.circuit_executions}</td><td>${r.total_shots}</td></tr>`).join('')+'</table>';
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results", help="folder holding the CSVs (and the dashboard)")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    rd = lambda n: pd.read_csv(os.path.join(a.out, n))
    overall = rd("main_overall.csv")
    data = {
        "labels": LABELS,
        "order": [s for s in LABELS if s in set(overall["strategy"])],
        "overall": overall.to_dict("records"),
        "summary": rd("main_summary.csv").to_dict("records"),
        "sweep": rd("sweep_raw.csv")[["experiment", "factor", "strategy", "abs_error"]].to_dict("records")
        if os.path.exists(os.path.join(a.out, "sweep_raw.csv")) else [],
    }
    path = os.path.abspath(os.path.join(a.out, "dashboard.html"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(HTML.replace("__DATA__", json.dumps(data)))
    print("Wrote", path)
    if not a.no_open:
        webbrowser.open("file:///" + path.replace("\\", "/"))


if __name__ == "__main__":
    main()
