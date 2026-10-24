#!/usr/bin/env python3
"""Food Truck Wars live dashboard. Read-only: never sends orders.

    export NIGHTSHIFT_KEY=fk_...      (never commit the key)
    python3 dashboard.py
    open http://localhost:8000

One background connection long-polls the game API; the browser page polls this local
server. Day-by-day history is saved to dashboard_history.json (gitignored) so the
cash chart survives restarts.
"""
import json, os, sys, threading, time, urllib.request, urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

URL = os.environ.get("NIGHTSHIFT_URL", "https://nightshift.joinbuilders.org").rstrip("/")
KEY = os.environ.get("NIGHTSHIFT_KEY", "")
PORT = int(os.environ.get("DASH_PORT", "8000"))
HIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_history.json")
DEFAULT_COSTS = {"cheap": 2.5, "standard": 3.75, "premium": 5.5}

lock = threading.Lock()
latest = {}
history = {"days": {}}
status = {"error": None}

if os.path.exists(HIST_FILE):
    try:
        history = json.load(open(HIST_FILE))
    except Exception:
        pass


def record(st):
    ld = st.get("last_day") or {}
    d = ld.get("day")
    if not d:
        return
    key = "%s|%s|%s" % (st.get("mode"), st.get("market"), d)
    if key in history["days"]:
        return
    costs = (st.get("rules") or {}).get("costs") or DEFAULT_COSTS
    spots = {}
    for s, sp in (ld.get("spots") or {}).items():
        trucks = []
        for t in sp.get("trucks") or []:
            c = costs.get(t.get("quality"), 3.75)
            trucks.append({"id": t.get("id"), "name": t.get("truck_name"), "price": t.get("price"),
                           "quality": t.get("quality"), "meals": t.get("meals"), "sold": t.get("sold"),
                           "profit": round((t.get("price") or 0) * (t.get("sold") or 0) - (t.get("meals") or 0) * c, 2)})
        spots[s] = {"customers": sp.get("customers"), "hungry": sp.get("hungry"), "trucks": trucks}
    history["days"][key] = {
        "mode": st.get("mode"), "market": st.get("market"), "day": d,
        "weather": ld.get("weather"), "events": [e.get("name") for e in ld.get("events") or []],
        "cash": {t.get("id"): t.get("cash") for t in st.get("trucks") or []},
        "names": {t.get("id"): t.get("truck_name") for t in st.get("trucks") or []},
        "colors": {t.get("id"): t.get("color") for t in st.get("trucks") or []},
        "me": (st.get("you") or {}).get("id"), "you": ld.get("you"), "spots": spots,
    }
    try:
        tmp = HIST_FILE + ".tmp"
        json.dump(history, open(tmp, "w"))
        os.replace(tmp, HIST_FILE)
    except Exception as e:
        print("could not save history:", e)


def poller():
    version, backoff = 0, 1
    while True:
        try:
            req = urllib.request.Request(URL + "/api/v1/state?since=%d" % version,
                                         headers={"Authorization": "Bearer " + KEY})
            with urllib.request.urlopen(req, timeout=35) as r:
                st = json.load(r)
            st["_fetched_ms"] = time.time() * 1000
            version = st.get("version", version)
            with lock:
                latest.clear(); latest.update(st)
                record(st)
            status["error"] = None
            backoff = 1
        except urllib.error.HTTPError as e:
            status["error"] = "API error %d%s" % (e.code, " (wrong key?)" if e.code == 401 else "")
            print(status["error"]); time.sleep(5 if e.code in (401, 429) else 3)
        except Exception as e:
            status["error"] = "network: %s" % e
            print(status["error"]); time.sleep(backoff); backoff = min(backoff * 2, 30)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/state":
            with lock:
                body = dict(latest)
            body["_dash_error"] = status["error"]
            self._send(200, json.dumps(body), "application/json")
        elif path == "/api/history":
            with lock:
                days = sorted(history["days"].values(),
                              key=lambda d: (str(d.get("mode")), d.get("market") or 0, d.get("day") or 0))
            self._send(200, json.dumps(days), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def log_message(self, *a):
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Truck Command</title>
<style>
:root{--bg:#0d0f12;--panel:#15181d;--line:#262b33;--ink:#e8e6e1;--mute:#8a8f98;--gold:#c9a96e;--red:#d9675b;--green:#6fb38a;--blue:#6c9bd2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
header{display:flex;flex-wrap:wrap;gap:20px;align-items:baseline;padding:14px 20px;border-bottom:1px solid var(--line)}
h1{font-size:16px;letter-spacing:.14em;text-transform:uppercase;margin:0;color:var(--gold);font-weight:600}
.kv{color:var(--mute)}.kv b{color:var(--ink);font-weight:600}
#clock{font-variant-numeric:tabular-nums;font-size:22px;font-weight:600}
#err{color:var(--red)}
main{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(320px,1fr);gap:14px;padding:14px 20px}
@media(max-width:1000px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.panel h2{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--mute);margin:0 0 10px;font-weight:600}
svg{width:100%;height:auto;display:block}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{padding:5px 6px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mute);font-weight:500;font-size:12px}
td.r,th.r{text-align:right}
tr.me td{color:var(--gold);font-weight:600}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}
.pos{color:var(--green)}.neg{color:var(--red)}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--blue)}
.ins{margin:0;padding-left:18px}.ins li{margin:4px 0}
.stack{display:grid;gap:14px;align-content:start;min-width:0}.panel{min-width:0}
canvas{width:100%;height:240px;display:block}
.small{font-size:12px;color:var(--mute)}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:9px;margin:1px 3px 1px 0;border:1px solid var(--line)}
.t-under{color:#d9675b;border-color:#5a2c27}.t-camp{color:#c9a96e;border-color:#5a4a2c}.t-chase{color:#e08ad6;border-color:#55304f}
.t-roam{color:#6c9bd2;border-color:#2b3f57}.t-waste{color:#8a8f98}.t-here{color:#0d0f12;background:#d9675b;border-color:#d9675b}
.scroll{overflow-x:auto}
</style></head><body>
<header>
  <h1>Truck Command</h1>
  <span class="kv">mode <b id="mode">-</b></span>
  <span class="kv">day <b id="day">-</b></span>
  <span class="kv">phase <b id="phase">-</b></span>
  <span id="clock">--:--</span>
  <span class="kv">our cash <b id="cash">-</b></span>
  <span class="kv">rank <b id="rank">-</b></span>
  <span id="err"></span>
</header>
<main>
  <div class="stack">
    <div class="panel"><h2>Campus: last day results <span class="small">· number/solid = customers · red arc = share left hungry · dashed = tomorrow's forecast</span></h2><div id="map"></div></div>
    <div class="panel"><h2>Rival intel <span class="small">· labels update from all days since the dashboard started</span></h2><div id="intel"></div></div>
    <div class="panel"><h2>Cash over time</h2><canvas id="chart" width="1000" height="240"></canvas></div>
    <div class="panel"><h2>Our day log</h2><div id="log"></div></div>
  </div>
  <div class="stack">
    <div class="panel"><h2>Standings</h2><div id="stand"></div></div>
    <div class="panel"><h2>Opportunities</h2><ul class="ins" id="ins"></ul></div>
    <div class="panel"><h2>Spot economics <span class="small">· averages per day</span></h2><div id="econ"></div></div>
    <div class="panel"><h2>Tomorrow</h2><div id="fc"></div></div>
    <div class="panel"><h2>News</h2><ul class="ins" id="news"></ul></div>
  </div>
</main>
<script>
const SPOTS={stadium:{x:150,y:390,n:"Ohio Stadium"},thompson:{x:320,y:250,n:"Thompson Library"},oval:{x:470,y:290,n:"The Oval"},
 union:{x:640,y:360,n:"Ohio Union"},highst:{x:770,y:205,n:"High Street"},northres:{x:590,y:120,n:"North Res"}};
const ORDER=["highst","oval","union","northres","thompson","stadium"];
const COST={cheap:2.5,standard:3.75,premium:5.5};
let S=null,H=[];
const esc=s=>String(s==null?"":s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const money=v=>(v<0?"-$":"$")+Math.abs(v||0).toLocaleString(undefined,{maximumFractionDigits:0});
const safeColor=c=>/^#[0-9a-fA-F]{3,8}$/.test(c||"")?c:"#888";

async function tick(){try{const r=await fetch("/api/state");S=await r.json();render();if(H.length){intel();econ();log();chart();}}catch(e){document.getElementById("err").textContent="dashboard server not reachable";}}
async function tickH(){try{const r=await fetch("/api/history");H=await r.json();chart();log();intel();econ();}catch(e){console.error(e)}}
function clock(){
  if(!S||!S.phase_ends_at){return}
  const off=S.server_time&&S._fetched_ms?Date.parse(S.server_time)-S._fetched_ms:0;
  const left=Math.max(0,(Date.parse(S.phase_ends_at)-(Date.now()+off))/1000);
  document.getElementById("clock").textContent=String(Math.floor(left/60)).padStart(2,"0")+":"+String(Math.floor(left%60)).padStart(2,"0");
}
function truckMap(){const m={};(S.trucks||[]).forEach(t=>m[t.id]=t);return m}
function dayProfits(){const p={};const sp=(S.last_day||{}).spots||{};
  for(const s in sp)(sp[s].trucks||[]).forEach(t=>{p[t.id]=(t.price||0)*(t.sold||0)-(t.meals||0)*(COST[t.quality]||3.75)});return p}

function render(){
  const me=(S.you||{}).id, TM=truckMap(), P=dayProfits();
  document.getElementById("err").textContent=S._dash_error||"";
  document.getElementById("mode").textContent=(S.mode||"-")+(S.mode==="final"?" m"+S.market+"/"+S.markets:"");
  document.getElementById("day").textContent=(S.day||0)+" / "+(S.days||"-");
  document.getElementById("phase").textContent=S.phase||"-";
  document.getElementById("cash").textContent=money((S.you||{}).cash);
  const ranked=(S.trucks||[]).slice().sort((a,b)=>(b.cash||0)-(a.cash||0));
  document.getElementById("rank").textContent=(ranked.findIndex(t=>t.id===me)+1||"-")+" of "+ranked.length;
  map(me,TM);
  // standings
  let h='<table><tr><th>#</th><th>Truck</th><th class="r">Cash</th><th class="r">Last day</th><th class="r">Rep</th></tr>';
  ranked.forEach((t,i)=>{const p=P[t.id];h+=`<tr class="${t.id===me?"me":""}"><td>${i+1}</td><td><span class="dot" style="background:${safeColor(t.color)}"></span>${esc(t.truck_name)}<div class="small">${esc(t.company_name)}</div></td><td class="r">${money(t.cash)}</td><td class="r ${p==null?"":p>=0?"pos":"neg"}">${p==null?"-":money(p)}</td><td class="r">${t.reputation_avg==null?"-":(+t.reputation_avg).toFixed(1)}</td></tr>`});
  document.getElementById("stand").innerHTML=h+"</table>";
  // forecast
  const fc=S.forecast||{};const ex=fc.expected_customers||{};const mx=Math.max(1,...Object.values(ex));
  const sp=(S.last_day||{}).spots||{};
  let f=`<div class="small">Weather: ${Object.entries(fc.weather||{}).map(([k,v])=>esc(k)+" "+Math.round(v*100)+"%").join(" · ")||"-"}</div>`;
  f+=`<div class="small" style="margin:4px 0 8px">Events: ${(fc.events||[]).map(e=>"<b>"+esc(e.name)+"</b> @ "+esc(e.spot||"")).join(", ")||"none"}</div>`;
  f+='<table><tr><th>Spot</th><th class="r">Expected</th><th></th><th class="r">Trucks there today</th></tr>';
  ORDER.forEach(s=>{const n=ex[s]||0;const k=((sp[s]||{}).trucks||[]).length;
    f+=`<tr><td>${SPOTS[s].n}</td><td class="r">${n}</td><td style="width:35%"><div class="bar"><i style="width:${100*n/mx}%"></i></div></td><td class="r">${k}</td></tr>`});
  document.getElementById("fc").innerHTML=f+"</table>";
  // opportunities: expected customers per competing truck + yesterday's hungry
  const opp=ORDER.map(s=>{const k=((sp[s]||{}).trucks||[]).filter(t=>t.id!==me).length;const hu=(sp[s]||{}).hungry||0;
    return {s,k,hu,score:(ex[s]||0)/(k+1)}}).sort((a,b)=>b.score-a.score);
  document.getElementById("ins").innerHTML=opp.map(o=>`<li><b>${SPOTS[o.s].n}</b>: ~${Math.round(o.score)} customers per truck tomorrow if rivals stay (${o.k} rival${o.k===1?"":"s"}, ${o.hu} left hungry today)</li>`).join("");
  document.getElementById("news").innerHTML=(S.announcements||[]).slice(-6).reverse().map(a=>`<li>${esc(typeof a==="string"?a:(a.text||a.message||JSON.stringify(a)))}</li>`).join("")||'<li class="small">none</li>';
}

function map(me,TM){
  const sp=(S.last_day||{}).spots||{};const ex=(S.forecast||{}).expected_customers||{};
  let g=`<svg viewBox="0 0 960 520" role="img" aria-label="Campus map">
  <rect width="960" height="520" rx="8" fill="#101317"/>
  <path d="M60 0 C 90 140, 40 300, 80 520" stroke="#1d3347" stroke-width="22" fill="none"/>
  <text x="96" y="30" fill="#4b6176" font-size="12">Olentangy River</text>
  <rect x="890" y="0" width="16" height="520" fill="#22262d"/><text x="884" y="510" fill="#5c626b" font-size="12" text-anchor="end">N High St</text>
  <ellipse cx="470" cy="290" rx="120" ry="70" fill="#14201a" stroke="#1e3327"/>`;
  ORDER.forEach(s=>{const P=SPOTS[s],d=sp[s]||{},c=d.customers||0,hu=d.hungry||0;
    const r=16+Math.sqrt(c)*2.4,rf=16+Math.sqrt(ex[s]||0)*2.4;
    g+=`<circle cx="${P.x}" cy="${P.y}" r="${rf}" fill="none" stroke="#3b4048" stroke-dasharray="4 5"/>`;
    g+=`<circle cx="${P.x}" cy="${P.y}" r="${r}" fill="#6c9bd2" fill-opacity=".16" stroke="#6c9bd2" stroke-opacity=".5"/>`;
    if(c>0&&hu>=c){g+=`<circle cx="${P.x}" cy="${P.y}" r="${r}" fill="none" stroke="#d9675b" stroke-width="4"/>`}
    else if(c>0&&hu>0){const a=2*Math.PI*hu/c,x2=P.x+r*Math.sin(a),y2=P.y-r*Math.cos(a);
      g+=`<path d="M${P.x} ${P.y-r} A${r} ${r} 0 ${a>Math.PI?1:0} 1 ${x2.toFixed(1)} ${y2.toFixed(1)}" stroke="#d9675b" stroke-width="4" fill="none"/>`}
    g+=`<text x="${P.x}" y="${P.y+5}" text-anchor="middle" fill="#e8e6e1" font-size="15" font-weight="700">${c}</text>`;
    const top=P.y-Math.max(r,rf)-8;
    g+=`<text x="${P.x}" y="${top-14}" text-anchor="middle" fill="#e8e6e1" font-size="13" font-weight="600">${P.n}</text>`;
    g+=`<text x="${P.x}" y="${top}" text-anchor="middle" fill="#8a8f98" font-size="11">${hu} hungry · ${ex[s]||0} expected next</text>`;
    (d.trucks||[]).forEach((t,i)=>{const col=safeColor((TM[t.id]||{}).color),y=P.y+r+16+i*22,mine=t.id===me;
      const lbl=`${(t.truck_name||"?").slice(0,16)} · ${(t.quality||"?")[0].toUpperCase()} $${(+t.price).toFixed(2)} · ${t.sold}/${t.meals}`;
      const w=lbl.length*6.6+22;
      g+=`<rect x="${P.x-w/2}" y="${y-12}" width="${w}" height="18" rx="9" fill="#1b1f25" stroke="${mine?"#c9a96e":col}" stroke-width="${mine?2:1}"/>`;
      g+=`<circle cx="${P.x-w/2+10}" cy="${y-3}" r="4" fill="${col}"/>`;
      g+=`<text x="${P.x-w/2+18}" y="${y+1}" fill="${mine?"#c9a96e":"#d6d3cc"}" font-size="11">${esc(lbl)}</text>`;});
  });
  document.getElementById("map").innerHTML=g+"</svg>";
}

function chart(){
  const cv=document.getElementById("chart"),ctx=cv.getContext("2d");const W=cv.width,Hh=cv.height;ctx.clearRect(0,0,W,Hh);
  if(!S||!H.length)return;
  const days=H.filter(d=>d.mode===S.mode&&d.market===S.market);if(!days.length)return;
  const ids=new Set();days.forEach(d=>Object.keys(d.cash||{}).forEach(i=>ids.add(i)));
  let lo=0,hi=1;days.forEach(d=>Object.values(d.cash||{}).forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v)}));
  const L=60,R=20,T=12,B=26,xs=i=>L+(W-L-R)*(days.length<2?0.5:i/(days.length-1)),ys=v=>T+(Hh-T-B)*(1-(v-lo)/(hi-lo||1));
  ctx.strokeStyle="#262b33";ctx.fillStyle="#8a8f98";ctx.font="12px sans-serif";
  for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4,y=ys(v);ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(W-R,y);ctx.stroke();ctx.fillText(money(v),4,y+4)}
  ctx.fillText("day "+days[0].day,L,Hh-6);ctx.textAlign="right";ctx.fillText("day "+days[days.length-1].day,W-R,Hh-6);ctx.textAlign="left";
  const me=(S.you||{}).id,TM=truckMap();
  [...ids].sort(a=>a===me?1:-1).forEach(id=>{ctx.beginPath();ctx.lineWidth=id===me?3:1.5;
    ctx.strokeStyle=id===me?"#c9a96e":safeColor((TM[id]||{}).color);let started=false;
    days.forEach((d,i)=>{const v=(d.cash||{})[id];if(v==null)return;const x=xs(i),y=ys(v);started?ctx.lineTo(x,y):ctx.moveTo(x,y);started=true});ctx.stroke();
    ctx.fillStyle=ctx.strokeStyle;days.forEach((d,i)=>{const v=(d.cash||{})[id];if(v==null)return;ctx.beginPath();ctx.arc(xs(i),ys(v),id===me?4:2.5,0,7);ctx.fill()});
    const last=days[days.length-1].cash||{};if(last[id]!=null){ctx.font="11px sans-serif";ctx.fillText(((TM[id]||{}).truck_name||"").slice(0,14),Math.min(xs(days.length-1)+6,W-90),ys(last[id])-6)}});
}

function log(){
  if(!S)return;const rows=H.filter(d=>d.mode===S.mode&&d.market===S.market&&d.you&&d.you.spot).slice(-12).reverse();
  if(!rows.length){document.getElementById("log").innerHTML='<div class="small">No days recorded yet. History starts when the dashboard starts.</div>';return}
  let h='<table><tr><th>Day</th><th>Weather</th><th>Spot</th><th>Quality</th><th class="r">Price</th><th class="r">Sold/Cooked</th><th class="r">Profit</th></tr>';
  rows.forEach(d=>{const y=d.you;h+=`<tr><td>${d.day}</td><td>${esc(d.weather)}</td><td>${esc((SPOTS[y.spot]||{}).n||y.spot)}</td><td>${esc(y.quality)}</td><td class="r">$${(+y.price).toFixed(2)}</td><td class="r">${y.sold}/${y.meals}</td><td class="r ${y.profit>=0?"pos":"neg"}">${money(y.profit)}</td></tr>`});
  document.getElementById("log").innerHTML=h+"</table>";
}

function curDays(){return S?H.filter(d=>d.mode===S.mode&&d.market===S.market):[]}
function intel(){
  const el=document.getElementById("intel");if(!S)return;const days=curDays();const me=(S.you||{}).id;
  if(!days.length){el.innerHTML='<div class="small">Collecting data. Labels appear after the first recorded day.</div>';return}
  const TM=truckMap(),R={};let mySpotPrev=null;const tomorrowRivals={};
  days.forEach((d,di)=>{
    let mySpot=null;const all=[];
    for(const s in d.spots){(d.spots[s].trucks||[]).forEach(t=>{all.push(t.price);if(t.id===me)mySpot=s})}
    const sorted=all.slice().sort((a,b)=>a-b),med=sorted.length?sorted[Math.floor(sorted.length/2)]:0;
    for(const s in d.spots){const ts=d.spots[s].trucks||[];
      ts.forEach(t=>{if(t.id===me)return;const r=R[t.id]||(R[t.id]={id:t.id,name:t.name,days:0,spots:{},seq:[],prices:[],quals:{},cheapest:0,shared:0,sold:0,meals:0,profit:0,chase:0,belowMed:0});
        r.days++;r.spots[s]=(r.spots[s]||0)+1;r.seq.push(s);r.prices.push(t.price);r.quals[t.quality]=(r.quals[t.quality]||0)+1;
        r.sold+=t.sold||0;r.meals+=t.meals||0;r.profit+=t.profit||0;if(t.price<=med-0.75)r.belowMed++;
        if(ts.length>1){r.shared++;if(ts.every(o=>o.id===t.id||t.price<o.price))r.cheapest++}
        const prevSeq=r.seq.length>1?r.seq[r.seq.length-2]:null;
        if(mySpotPrev&&s===mySpotPrev&&prevSeq!==s)r.chase++;
        if(di===days.length-1)tomorrowRivals[t.id]=s;});}
    mySpotPrev=mySpot||mySpotPrev;});
  const mySpotNow=((S.last_day||{}).you||{}).spot;
  const rows=Object.values(R).sort((a,b)=>b.profit-a.profit);
  let h='<div class="scroll"><table><tr><th>Truck</th><th>Labels</th><th>Home spot</th><th class="r">Avg $</th><th>Quality</th><th class="r">Sell-thru</th><th class="r">Profit/day</th><th class="r">Est. home rep</th><th>Last 5 spots</th></tr>';
  rows.forEach(r=>{
    const top=Object.entries(r.spots).sort((a,b)=>b[1]-a[1])[0],stay=top[1]/r.days,avgP=r.prices.reduce((a,b)=>a+b,0)/r.prices.length;
    const q=Object.entries(r.quals).sort((a,b)=>b[1]-a[1])[0][0],st=r.meals?r.sold/r.meals:0;
    const ra=(TM[r.id]||{}).reputation_avg;const tags=[];
    const under=(r.shared>=1&&r.cheapest/r.shared>=0.6)||(r.days>=2&&r.belowMed/r.days>=0.6);
    if(under)tags.push('<span class="tag t-under">undercutter</span>');
    if(r.days>=3&&stay>=0.7)tags.push('<span class="tag t-camp">camper</span>');
    else if(r.days>=3&&Object.keys(r.spots).length>=3)tags.push('<span class="tag t-roam">roamer</span>');
    if(r.chase>=2)tags.push('<span class="tag t-chase">follows us</span>');
    if(r.days>=2&&st<0.6)tags.push('<span class="tag t-waste">overcooks</span>');
    if(mySpotNow&&tomorrowRivals[r.id]===mySpotNow)tags.push('<span class="tag t-here">at our spot</span>');
    let hr="-";if(ra!=null&&r.days>=3&&stay>=0.7){hr=Math.max(0,Math.min(100,6*ra-250)).toFixed(0)}
    const col=safeColor((TM[r.id]||{}).color);
    h+=`<tr><td><span class="dot" style="background:${col}"></span>${esc(r.name)}</td><td>${tags.join("")||'<span class="small">-</span>'}</td><td>${esc((SPOTS[top[0]]||{}).n||top[0])} <span class="small">${Math.round(stay*100)}%</span></td><td class="r">$${avgP.toFixed(2)}</td><td>${esc(q)}</td><td class="r">${Math.round(st*100)}%</td><td class="r ${r.profit>=0?"pos":"neg"}">${money(r.profit/r.days)}</td><td class="r">${hr}</td><td class="small">${r.seq.slice(-5).map(s=>esc(s)).join(" → ")}</td></tr>`});
  el.innerHTML=h+'</table></div><div class="small" style="margin-top:8px">undercutter = cheapest where it shares a spot, or $0.75+ under the field median · camper = 70%+ of days at one spot · follows us = moved onto our previous spot 2+ times · est. home rep = 6 × avg rep − 250 (campers only)</div>';
}
function econ(){
  const el=document.getElementById("econ");const days=curDays();if(!days.length){el.innerHTML='<div class="small">No data yet.</div>';return}
  let h='<table><tr><th>Spot</th><th class="r">Crowd</th><th class="r">Hungry</th><th class="r">Trucks</th><th class="r">Profit/truck</th></tr>';
  ORDER.forEach(s=>{let c=0,hu=0,tk=0,pf=0,pn=0,n=0;days.forEach(d=>{const x=d.spots[s];if(!x)return;n++;c+=x.customers||0;hu+=x.hungry||0;tk+=(x.trucks||[]).length;(x.trucks||[]).forEach(t=>{pf+=t.profit||0;pn++})});
    if(!n){h+=`<tr><td>${SPOTS[s].n}</td><td class="r" colspan="4">-</td></tr>`;return}
    h+=`<tr><td>${SPOTS[s].n}</td><td class="r">${Math.round(c/n)}</td><td class="r">${Math.round(hu/n)}</td><td class="r">${(tk/n).toFixed(1)}</td><td class="r ${pn&&pf>=0?"pos":"neg"}">${pn?money(pf/pn):"-"}</td></tr>`});
  el.innerHTML=h+'</table><div class="small" style="margin-top:6px">High hungry + low trucks = open territory. Negative profit/truck = a price war.</div>';
}

tick();tickH();setInterval(tick,3000);setInterval(tickH,10000);setInterval(clock,1000);
</script></body></html>"""


def main():
    if not KEY:
        sys.exit("Set NIGHTSHIFT_KEY first (export NIGHTSHIFT_KEY=fk_...). Never commit it.")
    threading.Thread(target=poller, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("dashboard running: open http://localhost:%d  (Ctrl+C to stop)" % PORT)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
