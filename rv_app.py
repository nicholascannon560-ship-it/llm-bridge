"""RV Lab: blind remote-viewing trainer + AI model tester, served by the bridge.

GET  /rv                 phone app (HTML shell, holds no data)
GET  /rv/session         {"authed", "configured"}
POST /rv/login           {"pin"} -> signed HttpOnly cookie (path /rv)
POST /rv/new             {"viewer"} -> {"session_id","coordinate","commitment"}
POST /rv/submit          {"session_id","impressions","sketch"?} -> 4 candidate photos A-D
POST /rv/judge           {"session_id","mode":"self"|"ai","pick"?,"judge_model"?} -> reveal
POST /rv/ai_run          {"viewer_model","judge_model","trials"} -> {"run_id"}
GET  /rv/ai_run/{id}     progress + per-trial results of an AI run
GET  /rv/stats           hit rate per viewer vs the 25% chance line
GET  /rv/history         recent finished trials
GET  /rv/models          OpenRouter model list (id, name, vision)

Protocol: the server picks a target photo and 3 decoys before the viewer sees
anything, and publishes sha256("target_id|nonce") as a commitment. The viewer
only ever gets a random coordinate. A judge (the person, or a vision model that
is never told which photo is real) ranks the 4 photos against the session.
Rank 1 on the real target is a hit; chance is 1 in 4.

Auth: RV_PIN, falling back to CART_PIN. Never the bridge key.
Results persist to S3 (AWS_S3_BUCKET, key rv/results.json) with a /tmp fallback.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import math
import os
import random
import re
import secrets
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

rv_router = APIRouter()

RV_APP_VERSION = "1.0.0"  # bump on HTML-only changes so the *.py watch pattern deploys
COOKIE = "rv_session"
SESSION_DAYS = 60
DEFAULT_JUDGE = "z-ai/glm-5.3-flash"
DEFAULT_VIEWER = "z-ai/glm-5.3-flash"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
S3_KEY = "rv/results.json"
LOCAL_STORE = Path(os.getenv("RV_DATA_DIR") or "/tmp") / "rv_results.json"
MAX_TRIALS = 20
PENDING_TTL = 6 * 3600
_HTML_PATH = Path(__file__).with_name("rv_app.html")

_fails: dict[str, list[float]] = {}
_pending: dict[str, dict] = {}
_runs: dict[str, dict] = {}
_tasks: set = set()
_results: list[dict] | None = None
_results_lock = asyncio.Lock()
_pool: list[int] = []
_models_cache: dict = {"t": 0, "data": []}

FALLBACK_IDS = [i for i in range(0, 1085) if i not in {
    86, 97, 105, 138, 148, 150, 205, 207, 224, 226, 245, 246, 262, 285, 286, 298, 303,
    332, 333, 346, 359, 394, 414, 422, 438, 462, 463, 470, 489, 540, 561, 578, 587, 589,
    592, 595, 597, 601, 624, 632, 636, 644, 647, 673, 697, 706, 707, 708, 709, 710, 711,
    712, 713, 714, 720, 725, 734, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 759,
    761, 762, 763, 771, 792, 801, 812, 843, 850, 854, 895, 897, 899, 917, 920, 934, 956,
    963, 968, 1007, 1017, 1030, 1034, 1046}]

VIEWER_PROMPT = """You are the viewer in a controlled remote-viewing experiment. Researchers are measuring whether descriptions produced under blind conditions match a hidden target better than chance, so honest, specific output is exactly what is needed. Nobody expects certainty.

A photograph has already been selected and sealed. You will not be shown it and nothing in this message describes it. Your only reference is the target coordinate: {coord}

Follow the standard protocol. Do not try to name or guess the target as a whole; report raw perceptions.
1. First impressions: 3-6 quick words or fragments.
2. Sensory data: colors, textures, temperature, sounds, smells, light.
3. Dimensional data: shapes, sizes, verticals vs horizontals, open vs enclosed, natural vs man-made, water or land.
4. Sketch: describe in a few sentences what you would draw, where things sit in the frame.
5. Summary: 2-3 sentences pulling the strongest impressions together.

Write it as plain text with those five headings. Be concrete."""

JUDGE_PROMPT = """You are judging a remote-viewing trial. Below is a viewer's session transcript{sketch_note}. After it come four candidate photographs labeled {labels}. Exactly one was the hidden target, but you are not told which, and you must not try to guess from anything except how well each photo matches the session.

Rank all four photos from best match to worst match. Weigh concrete correspondences (dominant colors, shapes, setting, natural vs man-made, water, structures, light, textures) over vague words that would fit anything. Do not favor a position.

SESSION TRANSCRIPT:
---
{transcript}
---

Reply with only JSON, no other text: {{"ranking": ["X","X","X","X"], "reason": "one or two sentences on why your top pick matched best"}}"""


# --------------------------------------------------------------------------- auth
def _pin() -> str:
    return (os.getenv("RV_PIN") or os.getenv("CART_PIN") or "").strip()


def _secret() -> bytes:
    base = os.getenv("UI_SESSION_SECRET") or "rv"
    return hashlib.sha256(f"rv-session|{base}|{_pin()}".encode()).digest()


def _sign(exp: int) -> str:
    return f"{exp}.{hmac.new(_secret(), str(exp).encode(), hashlib.sha256).hexdigest()}"


def _authed(request: Request) -> bool:
    if not _pin():
        return False
    tok = request.cookies.get(COOKIE) or ""
    exp_s, _, mac = tok.partition(".")
    if not exp_s.isdigit() or not mac or int(exp_s) < time.time():
        return False
    return hmac.compare_digest(_sign(int(exp_s)), tok)


def _deny(request: Request):
    if not _pin():
        return JSONResponse({"detail": "RV_PIN (or CART_PIN) is not set on the server"}, status_code=503)
    if not _authed(request):
        return JSONResponse({"detail": "Sign in again"}, status_code=401)
    return None


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "?")


async def _body(request: Request) -> dict:
    try:
        b = await request.json()
        return b if isinstance(b, dict) else {}
    except Exception:
        return {}


# --------------------------------------------------------------------------- storage
def _s3():
    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        return None, None
    try:
        import boto3  # noqa: WPS433
        return bucket, boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION") or "us-east-1")
    except Exception as e:
        print(f"[rv] s3 unavailable: {e!r}", flush=True)
        return None, None


def _load_sync() -> list[dict]:
    bucket, s3 = _s3()
    if s3:
        try:
            obj = s3.get_object(Bucket=bucket, Key=S3_KEY)
            return json.loads(obj["Body"].read())
        except Exception as e:
            if "NoSuchKey" not in repr(e):
                print(f"[rv] s3 load failed: {e!r}", flush=True)
    try:
        return json.loads(LOCAL_STORE.read_text())
    except Exception:
        return []


def _save_sync(rows: list[dict]) -> str:
    data = json.dumps(rows, separators=(",", ":"))
    try:
        LOCAL_STORE.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_STORE.write_text(data)
    except Exception as e:
        print(f"[rv] local save failed: {e!r}", flush=True)
    bucket, s3 = _s3()
    if s3:
        try:
            s3.put_object(Bucket=bucket, Key=S3_KEY, Body=data.encode(), ContentType="application/json")
            return "s3"
        except Exception as e:
            print(f"[rv] s3 save failed: {e!r}", flush=True)
    return "local"


async def _all_results() -> list[dict]:
    global _results
    if _results is None:
        async with _results_lock:
            if _results is None:
                _results = await asyncio.to_thread(_load_sync)
    return _results


async def _record(row: dict) -> None:
    rows = await _all_results()
    async with _results_lock:
        rows.append(row)
        snapshot = list(rows)
    await asyncio.to_thread(_save_sync, snapshot)


# --------------------------------------------------------------------------- targets
def _img_url(pid: int, w: int = 640, h: int = 480) -> str:
    return f"https://picsum.photos/id/{pid}/{w}/{h}"


async def _ensure_pool() -> list[int]:
    global _pool
    if _pool:
        return _pool
    ids: list[int] = []
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            for page in range(1, 12):
                r = await c.get("https://picsum.photos/v2/list", params={"page": page, "limit": 100})
                if r.status_code != 200:
                    break
                batch = r.json()
                if not batch:
                    break
                ids.extend(int(x["id"]) for x in batch)
    except Exception as e:
        print(f"[rv] picsum list failed, using fallback ids: {e!r}", flush=True)
    _pool = ids if len(ids) >= 50 else FALLBACK_IDS
    return _pool


def _coordinate() -> str:
    return f"{random.SystemRandom().randint(1000, 9999)}-{random.SystemRandom().randint(1000, 9999)}"


async def _new_trial(viewer: str, kind: str) -> dict:
    pool = await _ensure_pool()
    rng = random.SystemRandom()
    four = rng.sample(pool, 4)
    target = four[0]
    rng.shuffle(four)
    labels = ["A", "B", "C", "D"]
    nonce = secrets.token_hex(8)
    sid = secrets.token_urlsafe(10)
    trial = {
        "session_id": sid,
        "viewer": viewer,
        "kind": kind,
        "coordinate": _coordinate(),
        "target_id": target,
        "nonce": nonce,
        "commitment": hashlib.sha256(f"{target}|{nonce}".encode()).hexdigest(),
        "candidates": dict(zip(labels, four)),
        "created": time.time(),
    }
    now = time.time()
    for k in [k for k, v in _pending.items() if now - v["created"] > PENDING_TTL]:
        _pending.pop(k, None)
    _pending[sid] = trial
    return trial


def _target_label(trial: dict) -> str:
    return next(k for k, v in trial["candidates"].items() if v == trial["target_id"])


# --------------------------------------------------------------------------- openrouter
async def _or_chat(model: str, messages: list, max_tokens: int, temperature: float, title: str) -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    payload = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "messages": messages}
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(OR_URL, json=payload, headers={
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://kalshiml-production-b2e9.up.railway.app",
            "X-Title": title})
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"OpenRouter {r.status_code}: non-JSON reply")
    if r.status_code >= 400 or data.get("error"):
        err = data.get("error") or {}
        raise RuntimeError(f"{model}: {err.get('message') or r.status_code}")
    msg = ((data.get("choices") or [{}])[0].get("message") or {})
    text = msg.get("content") or ""
    if not text.strip():
        text = msg.get("reasoning") or ""
    if not text.strip():
        raise RuntimeError(f"{model} returned an empty reply")
    return text


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        a, b = text.rfind("{\""), text.rfind("}")
        if a == -1:
            a = text.find("{")
        if a != -1 and b > a:
            return json.loads(text[a:b + 1])
        raise


async def _fetch_b64(url: str) -> str:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
        r = await c.get(url)
        r.raise_for_status()
    return "data:image/jpeg;base64," + base64.b64encode(r.content).decode()


async def _ai_judge(trial: dict, transcript: str, sketch: str | None, model: str) -> dict:
    labels = list(trial["candidates"].keys())
    imgs = await asyncio.gather(*[_fetch_b64(_img_url(trial["candidates"][l], 512, 384)) for l in labels])
    content: list = [{"type": "text", "text": JUDGE_PROMPT.format(
        sketch_note=" and their sketch (the first image)" if sketch else "",
        labels=", ".join(labels), transcript=transcript[:6000])}]
    if sketch:
        content.append({"type": "text", "text": "VIEWER SKETCH:"})
        content.append({"type": "image_url", "image_url": {"url": sketch}})
    for l, img in zip(labels, imgs):
        content.append({"type": "text", "text": f"PHOTO {l}:"})
        content.append({"type": "image_url", "image_url": {"url": img}})
    text = await _or_chat(model, [{"role": "user", "content": content}], 2000, 0, "rv-lab-judge")
    out = _parse_json(text)
    ranking = [str(x).strip().upper()[:1] for x in (out.get("ranking") or [])]
    seen: list[str] = []
    for x in ranking:
        if x in labels and x not in seen:
            seen.append(x)
    if not seen:
        raise RuntimeError(f"judge gave no usable ranking: {text[:200]!r}")
    seen += [l for l in labels if l not in seen]
    return {"ranking": seen, "reason": str(out.get("reason") or "")[:400]}


def _finish(trial: dict, ranking: list[str], judge: str, reason: str, transcript: str) -> dict:
    tl = _target_label(trial)
    rank = ranking.index(tl) + 1
    return {
        "id": trial["session_id"],
        "t": int(time.time()),
        "viewer": trial["viewer"],
        "kind": trial["kind"],
        "judge": judge,
        "coordinate": trial["coordinate"],
        "target_id": trial["target_id"],
        "target_label": tl,
        "nonce": trial["nonce"],
        "commitment": trial["commitment"],
        "candidates": trial["candidates"],
        "ranking": ranking,
        "rank": rank,
        "hit": rank == 1,
        "reason": reason,
        "transcript": transcript[:4000],
    }


def _public_reveal(row: dict) -> dict:
    out = {k: v for k, v in row.items() if k != "transcript"}
    out["images"] = {k: _img_url(v) for k, v in row["candidates"].items()}
    out["verify"] = f"sha256('{row['target_id']}|{row['nonce']}') should equal the commitment"
    return out


# --------------------------------------------------------------------------- stats
def _binom_tail(k: int, n: int, p: float = 0.25) -> float:
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def _summ(rows: list[dict]) -> dict:
    n = len(rows)
    hits = sum(1 for r in rows if r.get("hit"))
    return {
        "n": n, "hits": hits,
        "hit_rate": round(hits / n, 3) if n else None,
        "mean_rank": round(sum(r["rank"] for r in rows) / n, 2) if n else None,
        "p_value": round(_binom_tail(hits, n), 4),
    }


# --------------------------------------------------------------------------- routes
@rv_router.get("/rv", response_class=HTMLResponse)
async def rv_page():
    try:
        html = _HTML_PATH.read_text(encoding="utf-8")
    except Exception:
        return HTMLResponse("rv_app.html missing", status_code=500)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@rv_router.get("/rv/session")
async def rv_session(request: Request):
    return {"authed": _authed(request), "configured": bool(_pin()), "version": RV_APP_VERSION}


@rv_router.post("/rv/login")
async def rv_login(request: Request):
    if not _pin():
        return JSONResponse({"detail": "RV_PIN (or CART_PIN) is not set on the server"}, status_code=503)
    ip, now = _client_ip(request), time.time()
    recent = [t for t in _fails.get(ip, []) if now - t < 900]
    _fails[ip] = recent
    if len(recent) >= 8:
        return JSONResponse({"detail": "Too many tries. Wait 15 minutes."}, status_code=429)
    supplied = str((await _body(request)).get("pin", "")).strip()
    if not supplied or not hmac.compare_digest(supplied, _pin()):
        recent.append(now)
        return JSONResponse({"detail": "Wrong PIN"}, status_code=401)
    _fails.pop(ip, None)
    exp = int(now + SESSION_DAYS * 86400)
    resp = JSONResponse({"ok": True})
    resp.set_cookie(COOKIE, _sign(exp), max_age=SESSION_DAYS * 86400,
                    httponly=True, secure=True, samesite="strict", path="/rv")
    return resp


@rv_router.post("/rv/new")
async def rv_new(request: Request):
    if (d := _deny(request)):
        return d
    viewer = str((await _body(request)).get("viewer") or "me").strip()[:40] or "me"
    t = await _new_trial(viewer, "human")
    return {"session_id": t["session_id"], "coordinate": t["coordinate"], "commitment": t["commitment"]}


@rv_router.post("/rv/submit")
async def rv_submit(request: Request):
    if (d := _deny(request)):
        return d
    b = await _body(request)
    t = _pending.get(str(b.get("session_id", "")))
    if not t:
        return JSONResponse({"detail": "Session expired or unknown. Start a new one."}, status_code=404)
    imp = str(b.get("impressions") or "").strip()
    sketch = str(b.get("sketch") or "")
    if not imp and not sketch:
        return JSONResponse({"detail": "Write or sketch something first"}, status_code=400)
    if t.get("submitted"):
        return JSONResponse({"detail": "Already submitted"}, status_code=409)
    t["submitted"] = time.time()
    t["impressions"] = imp[:6000]
    t["sketch"] = sketch if sketch.startswith("data:image/") and len(sketch) < 3_000_000 else None
    return {"candidates": {k: _img_url(v) for k, v in t["candidates"].items()}}


@rv_router.post("/rv/judge")
async def rv_judge(request: Request):
    if (d := _deny(request)):
        return d
    b = await _body(request)
    t = _pending.get(str(b.get("session_id", "")))
    if not t or not t.get("submitted"):
        return JSONResponse({"detail": "Submit the session first"}, status_code=404)
    labels = list(t["candidates"].keys())
    if b.get("mode") == "ai":
        model = str(b.get("judge_model") or DEFAULT_JUDGE)
        try:
            j = await _ai_judge(t, t["impressions"] or "(sketch only)", t.get("sketch"), model)
        except Exception as e:
            print(f"[rv] judge failed: {e!r}", flush=True)
            return JSONResponse({"detail": f"AI judge failed: {e}"}, status_code=502)
        row = _finish(t, j["ranking"], model, j["reason"], t["impressions"])
    else:
        ranking = [str(x).upper()[:1] for x in (b.get("ranking") or [])]
        pick = str(b.get("pick") or "").upper()[:1]
        if not ranking and pick in labels:
            ranking = [pick]
        ranking = [x for i, x in enumerate(ranking) if x in labels and x not in ranking[:i]]
        if not ranking:
            return JSONResponse({"detail": "Pick the photo that matches best"}, status_code=400)
        partial = len(ranking) < 4
        ranking += [l for l in labels if l not in ranking]
        row = _finish(t, ranking, "self", "", t["impressions"])
        if partial and not row["hit"]:
            row["rank"] = 2.5  # only a top pick was given; a miss averages ranks 2-4
    _pending.pop(t["session_id"], None)
    await _record(row)
    return _public_reveal(row)


async def _run_worker(run: dict) -> None:
    for i in range(run["trials"]):
        if run.get("cancel"):
            break
        step = {"n": i + 1, "status": "viewing"}
        run["log"].append(step)
        try:
            t = await _new_trial(run["viewer_model"], "ai")
            step["coordinate"] = t["coordinate"]
            step["commitment"] = t["commitment"]
            transcript = await _or_chat(run["viewer_model"], [
                {"role": "user", "content": VIEWER_PROMPT.format(coord=t["coordinate"])}],
                1500, 1.0, "rv-lab-viewer")
            t["submitted"] = time.time()
            step["status"] = "judging"
            step["transcript"] = transcript[:4000]
            j = await _ai_judge(t, transcript, None, run["judge_model"])
            row = _finish(t, j["ranking"], run["judge_model"], j["reason"], transcript)
            row["run_id"] = run["id"]
            _pending.pop(t["session_id"], None)
            await _record(row)
            step.update({"status": "done", "result": _public_reveal(row)})
            step["result"]["transcript"] = transcript[:4000]
        except Exception as e:
            print(f"[rv] run {run['id']} trial {i + 1} failed: {e!r}", flush=True)
            step.update({"status": "error", "error": str(e)[:300]})
    done = [s["result"] for s in run["log"] if s.get("status") == "done"]
    run["summary"] = _summ(done)
    run["status"] = "cancelled" if run.get("cancel") else "finished"
    run["finished"] = time.time()


@rv_router.post("/rv/ai_run")
async def rv_ai_run(request: Request):
    if (d := _deny(request)):
        return d
    if any(r["status"] == "running" for r in _runs.values()):
        return JSONResponse({"detail": "A run is already going. Wait for it or cancel it."}, status_code=409)
    b = await _body(request)
    try:
        trials = max(1, min(MAX_TRIALS, int(b.get("trials") or 5)))
    except (TypeError, ValueError):
        trials = 5
    run = {
        "id": secrets.token_urlsafe(8),
        "viewer_model": str(b.get("viewer_model") or DEFAULT_VIEWER).strip()[:120],
        "judge_model": str(b.get("judge_model") or DEFAULT_JUDGE).strip()[:120],
        "trials": trials, "status": "running", "started": time.time(), "log": [],
    }
    _runs[run["id"]] = run
    task = asyncio.create_task(_run_worker(run))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"run_id": run["id"], "trials": trials}


@rv_router.get("/rv/ai_run/{run_id}")
async def rv_ai_run_status(run_id: str, request: Request):
    if (d := _deny(request)):
        return d
    run = _runs.get(run_id)
    if not run:
        return JSONResponse({"detail": "Unknown run (the server may have restarted)"}, status_code=404)
    return run


@rv_router.post("/rv/ai_run/{run_id}/cancel")
async def rv_ai_run_cancel(run_id: str, request: Request):
    if (d := _deny(request)):
        return d
    run = _runs.get(run_id)
    if not run:
        return JSONResponse({"detail": "Unknown run"}, status_code=404)
    run["cancel"] = True
    return {"ok": True}


@rv_router.get("/rv/stats")
async def rv_stats(request: Request):
    if (d := _deny(request)):
        return d
    rows = await _all_results()
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(f"{r['viewer']}|{r['kind']}", []).append(r)
    viewers = []
    for key, rs in by.items():
        v, kind = key.split("|", 1)
        s = _summ(rs)
        s.update({"viewer": v, "kind": kind,
                  "ai_judged": _summ([r for r in rs if r["judge"] != "self"]),
                  "self_judged": _summ([r for r in rs if r["judge"] == "self"])})
        viewers.append(s)
    viewers.sort(key=lambda s: (-(s["n"]), s["viewer"]))
    return {"chance": 0.25, "overall": _summ(rows), "viewers": viewers,
            "storage": "s3" if os.getenv("AWS_S3_BUCKET") else "local (lost on redeploy)"}


@rv_router.get("/rv/history")
async def rv_history(request: Request, limit: int = 30, viewer: str | None = None):
    if (d := _deny(request)):
        return d
    rows = await _all_results()
    if viewer:
        rows = [r for r in rows if r["viewer"] == viewer]
    limit = max(1, min(200, limit))
    out = []
    for r in reversed(rows[-limit:]):
        p = _public_reveal(r)
        p["transcript"] = r.get("transcript", "")
        out.append(p)
    return {"items": out}


@rv_router.get("/rv/models")
async def rv_models(request: Request):
    if (d := _deny(request)):
        return d
    if time.time() - _models_cache["t"] > 3600 or not _models_cache["data"]:
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.get("https://openrouter.ai/api/v1/models")
            data = r.json().get("data") or []
            _models_cache["data"] = sorted(
                [{"id": m["id"], "name": m.get("name") or m["id"],
                  "vision": "image" in ((m.get("architecture") or {}).get("input_modalities") or [])}
                 for m in data if m.get("id")], key=lambda m: m["id"])
            _models_cache["t"] = time.time()
        except Exception as e:
            print(f"[rv] model list failed: {e!r}", flush=True)
    return {"models": _models_cache["data"], "default_judge": DEFAULT_JUDGE, "default_viewer": DEFAULT_VIEWER}
