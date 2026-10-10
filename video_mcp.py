"""Video Studio as an MCP connector, so Claude can make pictures and videos on request.

Mounted on the video-studio service at POST /mcp/{VIDEO_MCP_SECRET} (Streamable HTTP,
JSON responses). It runs as the owner account: jobs land in the same history as the app,
use the same storage, model list and price checks, and pay plain OpenRouter cost.

Spending guards, on top of the app's own budget checks:
  VIDEO_MCP_MAX_JOB_USD  most any one paid call may cost (default 5)
  VIDEO_MCP_DAILY_USD    most all connector calls may spend in 24 hours (default 25)
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
import secrets
import time

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

import video_app as va

mcp_router = APIRouter()
PROTOCOL = "2025-06-18"
_spend: list[tuple[float, float]] = []  # (time, usd) for connector calls, last 24h


def _max_job() -> float:
    try:
        return max(0.05, float(os.getenv("VIDEO_MCP_MAX_JOB_USD") or 5))
    except ValueError:
        return 5.0


def _daily() -> float:
    try:
        return max(0.05, float(os.getenv("VIDEO_MCP_DAILY_USD") or 25))
    except ValueError:
        return 25.0


def _spent() -> float:
    cut = time.time() - 86400
    _spend[:] = [s for s in _spend if s[0] > cut]
    return round(sum(s[1] for s in _spend), 4)


def _guard(cost: float) -> str | None:
    if cost > _max_job() + 1e-9:
        return f"This would cost about ${cost:.2f}, over the ${_max_job():.2f} per-job limit for the connector."
    if _spent() + cost > _daily() + 1e-9:
        return (f"Daily connector limit: ${_spent():.2f} of ${_daily():.2f} used in the last 24 hours, "
                f"and this one is about ${cost:.2f}.")
    return None


def _tok(n: int = 24) -> str:
    """URL-safe id that never ends in - or _ (chat apps trim those off tapped links)."""
    while True:
        t = secrets.token_urlsafe(n)
        if t[-1].isalnum():
            return t


def _secret_ok(s: str) -> bool:
    want = (os.getenv("VIDEO_MCP_SECRET") or "").strip()
    return len(want) >= 24 and hmac.compare_digest(s.encode(), want.encode())


def _as_owner(request: Request) -> Request:
    request.state.user = va.OWNER
    return request


def _detail(out) -> str:
    try:
        return json.loads(out.body).get("detail") or "refused"
    except Exception:
        return "refused"


async def _view_url(request: Request, j: dict) -> str | None:
    """A public, unguessable link to a finished clip (the same kind OpenRouter uses for edits)."""
    if not j.get("clip_key"):
        return None
    if not j.get("src_token"):
        j["src_token"] = _tok(24)
        await va._save_jobs()
    return f"{va._public_base(request)}/video/src/{j['src_token']}"


async def _job_view(request: Request, j: dict) -> dict:
    out = va._public(j)
    out["video_url"] = await _view_url(request, j)
    for k in ("created", "finished"):
        if out.get(k):
            out[k] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(out[k]))
    return {k: v for k, v in out.items() if v not in (None, "", [], False)}


# --------------------------------------------------------------------------- tools
TOOLS = [
    {"name": "list_models",
     "description": "List the video models available, with supported durations, resolutions, aspect ratios and prices. "
                    "Use before make_video to pick a model and valid settings. Pass search to filter by name.",
     "inputSchema": {"type": "object", "properties": {"search": {"type": "string"}}}},
    {"name": "estimate",
     "description": "Price a video before making it. mode is text (prompt only), image (start/end frames) or ref (reference pictures).",
     "inputSchema": {"type": "object", "required": ["model", "duration"], "properties": {
         "model": {"type": "string"}, "duration": {"type": "number"}, "resolution": {"type": "string"},
         "audio": {"type": "boolean"}, "mode": {"type": "string", "enum": ["text", "image", "ref"]},
         "n_images": {"type": "integer"}}}},
    {"name": "upload_image",
     "description": "Upload a picture (base64 PNG/JPEG/WebP, or an https URL) to use as a start frame, end frame or reference. "
                    "Returns an image id and a public URL.",
     "inputSchema": {"type": "object", "properties": {"data_base64": {"type": "string"}, "url": {"type": "string"}}}},
    {"name": "make_image",
     "description": "Make a picture with the image model, or change an existing one by passing from_image (an image id). "
                    "Good for photoreal end frames: pass the sketch as from_image and ask for the same scene made real. "
                    "Costs a few cents. Returns an image id and URL.",
     "inputSchema": {"type": "object", "required": ["prompt"], "properties": {
         "prompt": {"type": "string"}, "from_image": {"type": "string"},
         "aspect": {"type": "string", "enum": ["16:9", "9:16", "1:1", "4:5", "3:4", "4:3"]}}}},
    {"name": "make_video",
     "description": "Start a video. Optional start_image / end_image (image ids) make the model move from one picture to the "
                    "other; ref_images guide style instead. Checks price against budget and the connector limits before running. "
                    "Returns a job; poll job_status until status is completed.",
     "inputSchema": {"type": "object", "required": ["model", "prompt", "duration", "budget"], "properties": {
         "model": {"type": "string"}, "prompt": {"type": "string"}, "duration": {"type": "integer"},
         "resolution": {"type": "string"}, "aspect_ratio": {"type": "string"}, "audio": {"type": "boolean"},
         "start_image": {"type": "string"}, "end_image": {"type": "string"},
         "ref_images": {"type": "array", "items": {"type": "string"}},
         "budget": {"type": "number", "description": "Most this video may cost, in USD"}}}},
    {"name": "job_status",
     "description": "Status of a video job. When it's finished, video_url is a link to watch or download it.",
     "inputSchema": {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}},
    {"name": "list_jobs",
     "description": "Recent videos, newest first.",
     "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
    {"name": "extend",
     "description": "Continue a finished video from its last frame. change optionally describes what happens next.",
     "inputSchema": {"type": "object", "required": ["id", "budget"], "properties": {
         "id": {"type": "string"}, "seconds": {"type": "integer"}, "change": {"type": "string"},
         "budget": {"type": "number"}}}},
    {"name": "upscale",
     "description": "Sharpen a finished clip (up to 20 s). mode precise keeps it faithful; creative adds detail. "
                    "Call with quote_only true first to see the price.",
     "inputSchema": {"type": "object", "required": ["id"], "properties": {
         "id": {"type": "string"}, "factor": {"type": "number"}, "mode": {"type": "string", "enum": ["precise", "creative"]},
         "budget": {"type": "number"}, "quote_only": {"type": "boolean"}}}},
    {"name": "balance",
     "description": "OpenRouter credit left, plus what the connector has spent in the last 24 hours and its limits.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "stitch",
     "description": "Join finished videos, in the order given, into one clip on the server (free, no AI cost). "
                    "crossfade blends each join over that many seconds (0 = hard cut, ~0.15 hides small jumps). "
                    "mask_image (an image id: white = keep, black = drop) makes everything outside the shape pure black, "
                    "e.g. for projection mapping. Video only, no sound. Returns a job; poll job_status for video_url.",
     "inputSchema": {"type": "object", "required": ["jobs"], "properties": {
         "jobs": {"type": "array", "items": {"type": "string"}, "description": "Job ids in play order"},
         "shape": {"type": "string", "enum": ["3:4", "9:16", "16:9", "1:1", "4:5"]},
         "crossfade": {"type": "number"}, "mask_image": {"type": "string"},
         "fit": {"type": "string", "enum": ["fit", "fill"]}, "title": {"type": "string"}}}},
    {"name": "last_frame",
     "description": "Save a finished video's final frame as an image id (free). Use it as the next video's start_image "
                    "so chained clips join exactly where the last one really ended.",
     "inputSchema": {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}},
]


async def _call(name: str, a: dict, request: Request) -> dict:
    req = _as_owner(request)
    base = va._public_base(req)

    if name == "list_models":
        q = str(a.get("search") or "").lower()
        out = []
        for m in await va._models():
            if q and q not in (m["id"] + " " + m["name"]).lower():
                continue
            out.append({"id": m["id"], "name": m["name"], "durations": m["durations"], "resolutions": m["resolutions"],
                        "aspect_ratios": m["aspect_ratios"], "prices": m.get("prices"),
                        "options": m["passthrough"][:12]})
        return {"models": out, "count": len(out)}

    if name == "estimate":
        m = await va._model(str(a.get("model") or ""))
        if not m:
            raise ValueError("Unknown model. Use list_models.")
        cost, basis = va.estimate_cost(m, float(a.get("duration") or 0), str(a.get("resolution") or ""),
                                       bool(a.get("audio", True)), str(a.get("mode") or "text"), int(a.get("n_images") or 0))
        return {"cost_usd": cost, "basis": basis, "per_job_limit": _max_job(), "spent_24h": _spent(), "daily_limit": _daily()}

    if name == "upload_image":
        raw = None
        if a.get("data_base64"):
            s = str(a["data_base64"])
            if "," in s[:100]:
                s = s.split(",", 1)[1]
            raw = base64.b64decode(s)
        elif str(a.get("url") or "").startswith("https://"):
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
                r = await c.get(a["url"])
            r.raise_for_status()
            raw = r.content
        if not raw:
            raise ValueError("Send data_base64 or an https url")
        if len(raw) > va.MAX_IMAGE_BYTES:
            raise ValueError("Pictures must be under 8 MB")
        kind = va._sniff(raw)
        if not kind:
            raise ValueError("Use a PNG, JPEG or WebP picture")
        iid = _tok(24)
        await va._put(f"{va.PREFIX}images/{iid}", raw, kind[1])
        return {"image_id": iid, "url": f"{base}/video/img/{iid}"}

    if name == "make_image":
        if (g := _guard(va.IMAGE_HOLD)):
            raise ValueError(g)
        prompt = str(a.get("prompt") or "").strip()[:1500]
        if not prompt:
            raise ValueError("Describe the picture")
        content = [{"type": "text", "text": prompt}]
        src = str(a.get("from_image") or "")
        if src:
            if not va.IMG_ID.match(src):
                raise ValueError("from_image must be an image id")
            data = await va._get(f"{va.PREFIX}images/{src}")
            if not data:
                raise ValueError("That image id wasn't found")
            kind = va._sniff(data) or ("jpg", "image/jpeg")
            content.append({"type": "image_url", "image_url": {"url": f"data:{kind[1]};base64,{base64.b64encode(data).decode()}"}})
        payload = {"model": va.IMAGE_MODEL, "modalities": ["image", "text"], "messages": [{"role": "user", "content": content}]}
        if a.get("aspect") in ("9:16", "16:9", "1:1", "4:5", "3:4", "4:3"):
            payload["image_config"] = {"aspect_ratio": a["aspect"]}
        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.post(f"{va.OR_BASE}/chat/completions", json=payload, headers=va._or_headers())
        data = r.json()
        if r.status_code >= 400 or data.get("error"):
            raise ValueError(f"Image model refused: {(data.get('error') or {}).get('message') or r.status_code}")
        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        url = (((msg.get("images") or [{}])[0].get("image_url") or {}).get("url")) or ""
        if not url.startswith("data:"):
            raise ValueError("The model didn't return a picture; try rewording")
        img = base64.b64decode(url.split(",", 1)[1])
        cost = float((data.get("usage") or {}).get("cost") or va.IMAGE_HOLD)
        _spend.append((time.time(), cost))
        kind = va._sniff(img) or ("png", "image/png")
        iid = _tok(24)
        await va._put(f"{va.PREFIX}images/{iid}", img, kind[1])
        return {"image_id": iid, "url": f"{base}/video/img/{iid}", "cost_usd": round(cost, 4)}

    if name == "make_video":
        budget = round(float(a.get("budget") or 0), 2)
        if budget <= 0:
            raise ValueError("Set a budget (USD)")
        m = await va._model(str(a.get("model") or ""))
        if not m:
            raise ValueError("Unknown model. Use list_models.")
        images = []
        for key, role in (("start_image", "first"), ("end_image", "last")):
            if a.get(key):
                images.append({"id": str(a[key]), "role": role})
        for rid in (a.get("ref_images") or [])[:va.MAX_REFS]:
            images.append({"id": str(rid), "role": "ref"})
        mode = "image" if any(i["role"] != "ref" for i in images) else "ref" if images else "text"
        est, basis = va.estimate_cost(m, float(a.get("duration") or 0), str(a.get("resolution") or ""),
                                      bool(a.get("audio", False)), mode, len(images))
        if est is None:
            raise ValueError(f"Can't price {m['name']} ({basis}); pick another model.")
        if (g := _guard(est)):
            raise ValueError(g)
        spec = {"model": m["id"], "prompt": str(a.get("prompt") or ""), "duration": a.get("duration"),
                "resolution": a.get("resolution") or "", "aspect_ratio": a.get("aspect_ratio") or "",
                "audio": bool(a.get("audio", False)), "budget": min(budget, _max_job()), "images": images}
        out = await va._start_job(req, spec, {"user": "owner", "base": base})
        if isinstance(out, Response):
            raise ValueError(_detail(out))
        _spend.append((time.time(), float(out.get("estimate") or est)))
        return {"job": out, "next": "Poll job_status with this id; videos usually take 1-5 minutes."}

    if name == "job_status":
        j = va._find(await va._all_jobs(), str(a.get("id") or ""))
        if not j or (j.get("user") or "owner") != "owner":
            raise ValueError("No such video")
        va._ensure_poller(j)
        return await _job_view(req, j)

    if name == "list_jobs":
        n = max(1, min(40, int(a.get("limit") or 10)))
        jobs = [j for j in await va._all_jobs() if (j.get("user") or "owner") == "owner" and not j.get("superseded")]
        return {"jobs": [await _job_view(req, j) for j in reversed(jobs[-n:])]}

    if name == "extend":
        src = va._find(await va._all_jobs(), str(a.get("id") or ""))
        if not src or (src.get("user") or "owner") != "owner" or not src.get("clip_key"):
            raise ValueError("That video isn't ready yet")
        budget = min(round(float(a.get("budget") or 0), 2), _max_job())
        if (g := _guard(budget)):
            raise ValueError(g)
        src["base"] = src.get("base") or base
        out = await va._extend(src, auto=False, seconds=int(a["seconds"]) if a.get("seconds") else None, budget=budget,
                               change=str(a.get("change") or "").strip()[:1500] or None)
        _spend.append((time.time(), float(out.get("estimate") or budget)))
        return {"job": out}

    if name == "upscale":
        mode = "precise" if a.get("mode") != "creative" else "creative"
        j, plan, err = await va._upscale_quote(req, str(a.get("id") or ""), a.get("factor"), mode)
        if err:
            raise ValueError(_detail(err))
        if a.get("quote_only"):
            return {"quote": plan}
        if (g := _guard(plan["cost"])):
            raise ValueError(g)
        budget = round(float(a.get("budget") or plan["cost"] + 0.05), 2)
        url = await _view_url(req, j)
        extra = {"user": "owner", "base": base, "video_ref_url": url + ("?song=1" if j.get("scored_key") else ""),
                 "no_duration": True, "est_override": plan["cost"], "est_basis": plan["basis"], "parent_id": j["id"],
                 "edit_mode": "upscale", "change": f"Upscaled {plan['factor']:g}x ({mode}) to {plan['width']}x{plan['height']}"}
        if plan["can_tune"]:
            extra["provider_params"] = {"upscale_factor": plan["factor"], "creativity": 0 if mode == "precise" else 1}
            extra["provider_slugs"] = ["black-forest-labs", "bfl"]
        spec = {"model": va.UPSCALE_MODEL, "prompt": (j.get("base_prompt") or j.get("prompt") or "Upscale this video.")[:1000],
                "duration": max(1, int(plan["seconds"] + 0.999)), "resolution": "", "aspect_ratio": "",
                "audio": True, "budget": budget, "images": []}
        out = await va._start_job(req, spec, extra)
        if isinstance(out, Response):
            raise ValueError(_detail(out))
        _spend.append((time.time(), plan["cost"]))
        return {"job": out}

    if name == "balance":
        info = {"connector_spent_24h": _spent(), "connector_daily_limit": _daily(), "per_job_limit": _max_job(),
                "app_spent_24h": await va._spent_24h()}
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.get(f"{va.OR_BASE}/credits", headers=va._or_headers())
            d = (r.json() or {}).get("data") or {}
            if "total_credits" in d:
                info["openrouter_credit_left"] = round(float(d["total_credits"]) - float(d.get("total_usage") or 0), 2)
        except Exception as e:
            info["openrouter_credit_error"] = str(e)[:120]
        return info

    if name == "stitch":
        ids = [str(x) for x in (a.get("jobs") or []) if x]
        shape = a.get("shape") if a.get("shape") in va.SHAPES else "3:4"
        try:
            job = await va.start_stitch(ids, shape=shape, crossfade=float(a.get("crossfade") or 0),
                                        mask_image=str(a.get("mask_image") or "") or None,
                                        fit="fill" if a.get("fit") == "fill" else "fit",
                                        title=str(a.get("title") or "Stitch")[:80])
        except RuntimeError as e:
            raise ValueError(str(e))
        return {"job": await _job_view(req, job), "next": "Poll job_status with this id; stitching takes about 1-4 minutes."}

    if name == "last_frame":
        try:
            iid = await va.save_last_frame(str(a.get("id") or ""))
        except RuntimeError as e:
            raise ValueError(str(e))
        return {"image_id": iid, "url": f"{base}/video/img/{iid}"}

    raise ValueError(f"Unknown tool {name}")


# --------------------------------------------------------------------------- transport
def _rpc(id_, result=None, error=None) -> JSONResponse:
    body = {"jsonrpc": "2.0", "id": id_}
    body.update({"error": error} if error else {"result": result})
    return JSONResponse(body)


@mcp_router.get("/mcp/{secret}")
async def mcp_get(secret: str):
    return Response(status_code=405 if _secret_ok(secret) else 404, headers={"Allow": "POST"})


@mcp_router.post("/mcp/{secret}")
async def mcp_post(secret: str, request: Request):
    if not _secret_ok(secret):
        return Response("Not found", status_code=404)
    try:
        msg = await request.json()
    except Exception:
        return _rpc(None, error={"code": -32700, "message": "parse error"})
    if isinstance(msg, list):  # batches aren't used by Claude; answer the first request
        msg = next((m for m in msg if isinstance(m, dict) and "id" in m), msg[0] if msg else {})
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if mid is None:  # notification (e.g. notifications/initialized)
        return Response(status_code=202)
    if method == "initialize":
        return _rpc(mid, {"protocolVersion": params.get("protocolVersion") or PROTOCOL,
                          "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": {"name": "video-studio", "version": va.VIDEO_APP_VERSION},
                          "instructions": "Make pictures and videos with Video Studio. Price things with estimate before "
                                          "make_video, poll job_status for results, and stay within the budget the user gives."})
    if method == "ping":
        return _rpc(mid, {})
    if method == "tools/list":
        return _rpc(mid, {"tools": TOOLS})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        try:
            out = await _call(str(name), args if isinstance(args, dict) else {}, request)
            print(f"VIDEO_MCP call {name} ok", flush=True)
            return _rpc(mid, {"content": [{"type": "text", "text": json.dumps(out, indent=1, default=str)}]})
        except Exception as e:
            print(f"VIDEO_MCP call {name} failed: {str(e)[:200]}", flush=True)
            return _rpc(mid, {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True})
    return _rpc(mid, error={"code": -32601, "message": f"unknown method {method}"})
