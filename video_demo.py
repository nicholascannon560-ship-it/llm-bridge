"""Free public demo of Video Studio: one short, cheap video per visitor per day.

Embedded by iframe on the North River Labs site. No account and no cookies (they'd be
third-party inside the iframe): the browser keeps the job id it gets back and polls it.

Guards, because a public video generator costs real money and carries our name:
  1. one video per visitor per UTC day (IP kept only as a salted hash), DEMO_VIDEO_PER_IP
  2. a hard spend cap over the last 24h across all demo videos, DEMO_VIDEO_DAILY_USD
  3. every prompt is screened by a small model first; anything unclear is refused
  4. fixed cheap settings (model, 480p, 5s, no audio) so a visitor can't pick an expensive one
Demo jobs are owned by user "demo", which _start_job lets through without a membership.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import time

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

import video_app as va

demo_router = APIRouter()

DEMO_UID = "demo"
MODEL = os.getenv("DEMO_VIDEO_MODEL") or "bytedance/seedance-1-5-pro"
RES = os.getenv("DEMO_VIDEO_RES") or "480p"
SECONDS = int(os.getenv("DEMO_VIDEO_SECONDS") or "5")
ASPECT = os.getenv("DEMO_VIDEO_ASPECT") or "16:9"
DAILY_USD = float(os.getenv("DEMO_VIDEO_DAILY_USD") or "1.5")
PER_IP = int(os.getenv("DEMO_VIDEO_PER_IP") or "1")
STATE_KEY = f"{va.PREFIX}demo.json"
_state_lock = asyncio.Lock()

MOD_PROMPT = """You screen prompts for a free public AI video demo on a small business website.
Reply with exactly one word: ALLOW or BLOCK.
BLOCK: anything sexual or suggestive, nudity, violence or gore, weapons aimed at people, hate or
harassment, drugs, self-harm, real or famous people by name, politicians, copyrighted characters
or brands, children in anything other than an ordinary wholesome scene, or anything meant to deceive.
ALLOW: ordinary scenes such as nature, animals, places, weather, food, objects, abstract visuals,
and everyday people doing everyday things.
Prompt: {p}"""


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _iph(ip: str) -> str:
    return hashlib.sha256(f"video-demo|{ip}".encode()).hexdigest()[:32]


async def _state() -> dict:
    raw = await va._get(STATE_KEY)
    try:
        s = json.loads(raw) if raw else {}
    except Exception:
        s = {}
    if s.get("day") != _today():
        s = {"day": _today(), "ips": {}}
    return s


async def _spent() -> float:
    cutoff = time.time() - 86400
    return round(sum(va._job_cost(j) for j in await va._all_jobs()
                     if j.get("user") == DEMO_UID and j.get("created", 0) >= cutoff), 4)


async def _est() -> float | None:
    m = await va._model(MODEL)
    if not m:
        return None
    est, _ = va.estimate_cost(m, SECONDS, RES, False, "text", 0)
    return va._bill(DEMO_UID, est) if est is not None else None


async def _allowed(prompt: str) -> bool:
    try:
        out = await va._or_chat(MOD_PROMPT.format(p=prompt))
    except Exception:
        return False  # fail closed
    words = out.strip().upper().replace(".", " ").split()
    return bool(words) and words[0] == "ALLOW"


def _err(code: str, status: int) -> JSONResponse:
    return JSONResponse({"error": code}, status_code=status)


@demo_router.get("/video/demo/status")
async def demo_status():
    try:
        est = await _est()
    except Exception:
        est = None
    open_ = est is not None and (await _spent()) + est <= DAILY_USD
    return {"open": open_, "seconds": SECONDS, "resolution": RES}


@demo_router.post("/video/demo/start")
async def demo_start(request: Request):
    b = await va._body(request)
    prompt = " ".join(str(b.get("prompt") or "").split())[:300]
    if len(prompt) < 4:
        return _err("demo-short", 400)
    h = _iph(va._client_ip(request))
    async with _state_lock:
        st = await _state()
        if st["ips"].get(h, 0) >= PER_IP:
            return _err("demo-used", 429)
        try:
            est = await _est()
        except Exception:
            est = None
        if est is None or (await _spent()) + est > DAILY_USD:
            return _err("demo-paused", 503)
        if not await _allowed(prompt):
            return _err("demo-blocked", 400)
        res = await va._start_job(request, {
            "prompt": prompt, "model": MODEL, "duration": SECONDS, "resolution": RES,
            "aspect_ratio": ASPECT, "audio": False, "budget": 0.5,
        }, {"user": DEMO_UID, "base": va._public_base(request)})
        if isinstance(res, JSONResponse):  # refused by OpenRouter or a cap inside _start_job
            print(f"[video-demo] start refused: {res.body[:200]!r}", flush=True)
            return _err("demo-failed", 502)
        st["ips"][h] = st["ips"].get(h, 0) + 1
        await va._put(STATE_KEY, json.dumps(st).encode(), "application/json")
    job = va._find(await va._all_jobs(), res["id"])
    if job is not None and not job.get("src_token"):
        job["src_token"] = secrets.token_urlsafe(24)
        await va._save_jobs()
    return {"id": res["id"], "status": res.get("status")}


@demo_router.get("/video/demo/job/{jid}")
async def demo_job(jid: str, request: Request):
    j = va._find(await va._all_jobs(), jid)
    if not j or j.get("user") != DEMO_UID:
        return _err("not-found", 404)
    va._ensure_poller(j)  # resumes polling after a restart
    out = {"status": j.get("status"), "error": j.get("error")}
    if j.get("clip_key") and j.get("src_token"):
        out["video_url"] = f"{va._public_base(request)}/video/src/{j['src_token']}"
    return out


@demo_router.get("/video/demo", response_class=HTMLResponse)
async def demo_page():
    return HTMLResponse(PAGE)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video Studio demo</title>
<style>
:root{--bg:#0d0f14;--panel:#161a22;--line:#262c38;--text:#eef1f6;--dim:#9aa3b2;--accent:#7c9cff}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--text);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;display:flex;flex-direction:column}
header{display:flex;justify-content:space-between;align-items:baseline;padding:12px 16px;border-bottom:1px solid var(--line)}
header b{font-size:19px}header span{color:var(--dim);font-size:14px}
main{flex:1;overflow:auto;padding:16px;display:grid;gap:12px;align-content:start}
textarea{width:100%;min-height:96px;resize:vertical;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:12px;padding:12px;font:inherit;font-size:16px}
.chips{display:flex;flex-wrap:wrap;gap:8px}.chips button{background:transparent;color:var(--text);border:1px solid var(--line);border-radius:999px;padding:6px 12px;font-size:14px;cursor:pointer}
.go{background:var(--accent);color:#0b1020;border:0;border-radius:999px;padding:12px 18px;font-weight:700;font-size:16px;cursor:pointer}
.go:disabled{opacity:.5}
.note{color:var(--dim);font-size:14px}
.msg{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px}
video{width:100%;border-radius:12px;background:#000;display:block}
a.cta{display:inline-block;background:var(--accent);color:#0b1020;border-radius:999px;padding:10px 16px;font-weight:700;text-decoration:none}
.bar{height:6px;background:var(--line);border-radius:6px;overflow:hidden}.bar i{display:block;height:100%;width:0;background:var(--accent);transition:width 1s linear}
</style></head><body>
<header><b>Video Studio</b><span>Free demo &middot; one short clip</span></header>
<main id="m">
<div class="note">Describe a short scene and we'll generate a 5-second clip with an AI video model. One free clip per visitor per day.</div>
<textarea id="p" maxlength="300" placeholder="e.g. Fog rolling down a river valley at dawn, slow drone shot"></textarea>
<div class="chips" id="chips"></div>
<button class="go" id="go">Make my clip</button>
<div id="out"></div>
</main>
<script>
const ideas=["A red fox trotting through fresh snow at sunrise","Waves crashing on dark rocks under a stormy sky","A cozy cabin window at night with snow falling outside"];
const $=id=>document.getElementById(id), out=$("out"), go=$("go"), p=$("p");
ideas.forEach(t=>{const b=document.createElement("button");b.textContent=t;b.onclick=()=>{p.value=t};$("chips").appendChild(b)});
const say=(h)=>{out.innerHTML=h};
const MSG={"demo-used":"You've already made today's free clip. Come back tomorrow, or try the full Video Studio.",
"demo-paused":"The free demo is resting for today. Come back tomorrow, or try the full Video Studio.",
"demo-blocked":"That idea can't run in the public demo. Try an everyday scene, like nature, places, animals or objects.",
"demo-short":"Write a few words about the scene first.",
"demo-failed":"The video model didn't accept that one. Try a different idea."};
const cta='<p><a class="cta" href="/video" target="_blank" rel="noopener">Open the full Video Studio</a></p>';
fetch("/video/demo/status").then(r=>r.json()).then(d=>{if(!d.open){go.disabled=true;say('<div class="msg">'+MSG["demo-paused"]+'</div>'+cta)}}).catch(()=>{});
go.onclick=async()=>{
  const prompt=p.value.trim(); if(!prompt){say('<div class="msg">'+MSG["demo-short"]+'</div>');return}
  go.disabled=true; say('<div class="msg">Checking your idea…</div>');
  let r,d; try{r=await fetch("/video/demo/start",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({prompt})});d=await r.json()}catch(e){d={error:"demo-failed"}}
  if(!r||!r.ok||!d.id){say('<div class="msg">'+(MSG[d.error]||"Something went wrong. Try again.")+'</div>'+(d.error==="demo-used"||d.error==="demo-paused"?cta:""));go.disabled=(d.error==="demo-used"||d.error==="demo-paused");return}
  const t0=Date.now();
  say('<div class="msg">Generating your clip. This usually takes one to three minutes.<div class="bar" style="margin-top:10px"><i id="b"></i></div><div class="note" id="el" style="margin-top:6px">0s</div></div>');
  const tick=setInterval(()=>{const s=Math.round((Date.now()-t0)/1000);const e=$("el");if(e)e.textContent=s+"s";const b=$("b");if(b)b.style.width=Math.min(95,s/120*100)+"%"},1000);
  const poll=async()=>{
    let j; try{j=await (await fetch("/video/demo/job/"+d.id)).json()}catch(e){return setTimeout(poll,6000)}
    if(j.video_url){clearInterval(tick);say('<video src="'+j.video_url+'" autoplay loop muted playsinline controls></video><div class="note">Made with an AI video model in about '+Math.round((Date.now()-t0)/1000)+' seconds.</div>'+cta);return}
    if(["failed","cancelled","expired","refused"].includes(j.status)||j.error){clearInterval(tick);say('<div class="msg">The video model couldn\\'t finish that one. '+cta+'</div>');return}
    setTimeout(poll,5000)};
  setTimeout(poll,5000);
};
</script></body></html>"""
