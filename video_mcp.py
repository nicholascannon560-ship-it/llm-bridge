"""Video Studio as an MCP connector, so Claude can make pictures and videos on request.

Two doors, same tools (Streamable HTTP, JSON responses):
  POST /mcp                    customers: OAuth sign-in (video_oauth.py); every call runs as the
                               signed-in account, sees only its own videos and pays from its credit.
  POST /mcp/{VIDEO_MCP_SECRET} the owner's private link: runs as the owner account.

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
from video_oauth import challenge, user_for_token
import video_projects as vp
import video_routing as vr
import video_pictures as vpic
import video_shapes as vsh

mcp_router = APIRouter()
PROTOCOL = "2025-06-18"
_spend: dict[str, list[tuple[float, float]]] = {}  # uid -> (time, usd) for connector calls, last 24h


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


def _spent(uid: str) -> float:
    cut = time.time() - 86400
    rows = _spend[uid] = [s for s in _spend.get(uid, []) if s[0] > cut]
    return round(sum(s[1] for s in rows), 4)


def _note(uid: str, cost: float) -> None:
    _spend.setdefault(uid, []).append((time.time(), float(cost)))


def _guard(cost: float, uid: str) -> str | None:
    """Per-account connector limits. Customers are also bounded by their prepaid credit."""
    if cost > _max_job() + 1e-9:
        return f"This would cost about ${cost:.2f}, over the ${_max_job():.2f} per-job limit for the connector."
    if _spent(uid) + cost > _daily() + 1e-9:
        return (f"Daily connector limit: ${_spent(uid):.2f} of ${_daily():.2f} used in the last 24 hours, "
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




def _detail(out) -> str:
    try:
        return json.loads(out.body).get("detail") or "refused"
    except Exception:
        return "refused"


MAKERS = {"upload_image", "make_image", "make_shape", "fill_shape", "make_video", "extend", "upscale", "stitch", "loop", "bundle", "last_frame"}


async def _no_project(uid: str, ref) -> str:
    names = [p["name"] for p in await vp.mine(uid)]
    return (f"No project called {ref!r}. Your projects: {', '.join(names) if names else 'none yet'}. "
            f"Use create_project to make it.")


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
    {"name": "list_projects",
     "description": "List this account's projects (folders of videos, pictures and edits) with counts, and which one "
                    "is active. New work goes to the active project unless a tool is given a project.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "create_project",
     "description": "Make a new project and make it the active one.",
     "inputSchema": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}},
    {"name": "use_project",
     "description": "Switch the active project (shared with the Video Studio app). Pass \"unsorted\" for no project.",
     "inputSchema": {"type": "object", "required": ["project"], "properties": {"project": {"type": "string"}}}},
    {"name": "project_contents",
     "description": "What's in a project: its videos, edited videos (stitches, loops, zips, editor renders) and pictures. "
                    "Defaults to the active project.",
     "inputSchema": {"type": "object", "properties": {"project": {"type": "string"}, "limit": {"type": "integer"}}}},
    {"name": "move_to_project",
     "description": "Move videos or pictures into a project. ids are job ids (videos) or image ids (pictures).",
     "inputSchema": {"type": "object", "required": ["ids", "project"], "properties": {
         "ids": {"type": "array", "items": {"type": "string"}},
         "kind": {"type": "string", "enum": ["video", "picture"]}, "project": {"type": "string"}}}},
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
     "inputSchema": {"type": "object", "properties": {"project": {"type": "string", "description": "Project name or id to save into (default: the active project)"}, "data_base64": {"type": "string"}, "url": {"type": "string"}}}},
    {"name": "make_image",
     "description": "Make a picture. Leave model out (auto) and the router picks the best picture model for the job: "
                    "readable text/design, photoreal, art, editing (from_image), combining pictures (ref_images), "
                    "transparent cutouts, SVG vectors, style matching, 4K, or a cheap draft. Pass from_image (an image "
                    "id) to change an existing picture. Returns an image id, URL, cost and which model made it.",
     "inputSchema": {"type": "object", "required": ["prompt"], "properties": {"project": {"type": "string", "description": "Project name or id to save into (default: the active project)"}, 
         "prompt": {"type": "string"}, "from_image": {"type": "string", "description": "Image id of a picture to change"},
         "ref_images": {"type": "array", "items": {"type": "string"}, "description": "Image ids to combine or use as references"},
         "aspect": {"type": "string", "description": "Shape, e.g. 16:9, 9:16, 1:1, 4:5, 3:4, 4:3, 21:9"},
         "transparent": {"type": "boolean", "description": "Transparent background (cutouts, stickers)"},
         "vector": {"type": "boolean", "description": "Return an SVG vector (logos, icons)"},
         "resolution": {"type": "string", "description": "Size tier such as 1K, 2K or 4K (default: best the model offers below 4K)"},
         "model": {"type": "string", "description": "Picture model id, or omit / \"auto\" to route"},
         "tier": {"type": "string", "enum": ["best", "budget"], "description": "best quality (default) or cheapest good option"},
         "budget": {"type": "number", "description": "Most this picture may cost, in USD (optional)"}}}},
    {"name": "make_shape",
     "description": "Projection mapping: turn a photo of a physical cutout (or a ready white-on-black mask) into a "
                    "clean mask picture: white = the cutout, black = everything else. Upload the photo first "
                    "(upload_image). A two-tone picture is just cleaned up for free; a real photo is traced by a "
                    "picture model (a few cents). Returns mask_id, usable as mask_image in loop and stitch.",
     "inputSchema": {"type": "object", "required": ["image"], "properties": {
         "image": {"type": "string", "description": "Image id of the cutout photo or mask"},
         "name": {"type": "string", "description": "Name for the shape, e.g. Castle"},
         "project": {"type": "string", "description": "Project name or id to save into (default: the active project)"}}}},
    {"name": "fill_shape",
     "description": "Projection mapping: make a picture that fills a cutout's shape (mask_id from make_shape), "
                    "composed to fit the outline, with everything outside the shape forced to pure black. Uses the "
                    "models best at mask-guided editing. For a looping clip, then make_video with start_image and "
                    "end_image both = the returned image_id, and loop it (pingpong) with mask_image = mask_id.",
     "inputSchema": {"type": "object", "required": ["mask_id", "prompt"], "properties": {
         "mask_id": {"type": "string"}, "prompt": {"type": "string", "description": "What should fill the shape"},
         "model": {"type": "string", "description": "Picture model id, or omit for the best shape-filling model"},
         "tier": {"type": "string", "enum": ["best", "budget"]}, "budget": {"type": "number"},
         "project": {"type": "string", "description": "Project name or id to save into (default: the active project)"}}}},
    {"name": "list_shapes",
     "description": "Cutout shapes (masks) saved on this account, newest first.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "list_image_models",
     "description": "Every picture model available, with what each can do (editing, how many pictures it takes, "
                    "transparent background, SVG, sizes, quality levels, shapes) and its price per picture.",
     "inputSchema": {"type": "object", "properties": {"q": {"type": "string", "description": "Optional text filter"}}}},
    {"name": "pick_image_model",
     "description": "Preview which picture model auto-routing would use, and why, without making anything. Same inputs as make_image.",
     "inputSchema": {"type": "object", "required": ["prompt"], "properties": {
         "prompt": {"type": "string"}, "from_image": {"type": "string"}, "ref_images": {"type": "array", "items": {"type": "string"}},
         "aspect": {"type": "string"}, "transparent": {"type": "boolean"}, "vector": {"type": "boolean"},
         "resolution": {"type": "string"}, "tier": {"type": "string", "enum": ["best", "budget"]}, "budget": {"type": "number"}}}},
    {"name": "make_video",
     "description": "Start a video. Optional start_image / end_image (image ids) make the model move from one picture to the "
                    "other; ref_images guide style instead. Checks price against budget and the connector limits before running. "
                    "Returns a job; poll job_status until status is completed.",
     "inputSchema": {"type": "object", "required": ["prompt", "duration", "budget"], "properties": {"tier": {"type": "string", "enum": ["best", "budget"], "description": "Only when model is auto/omitted: best quality (default) or cheapest good option"}, "project": {"type": "string", "description": "Project name or id to save into (default: the active project)"}, 
         "model": {"type": "string", "description": "Model id, or omit / \"auto\" to let the router pick the best model for the job"}, "prompt": {"type": "string"}, "duration": {"type": "integer"},
         "resolution": {"type": "string"}, "aspect_ratio": {"type": "string"}, "audio": {"type": "boolean"},
         "start_image": {"type": "string"}, "end_image": {"type": "string"},
         "ref_images": {"type": "array", "items": {"type": "string"}},
         "budget": {"type": "number", "description": "Most this video may cost, in USD"}}}},
    {"name": "pick_model",
     "description": "Preview which model auto-routing would use for a video, and why, without making it. Same inputs as make_video.",
     "inputSchema": {"type": "object", "required": ["prompt", "duration"], "properties": {
         "prompt": {"type": "string"}, "duration": {"type": "integer"}, "budget": {"type": "number"},
         "resolution": {"type": "string"}, "aspect_ratio": {"type": "string"}, "audio": {"type": "boolean"},
         "start_image": {"type": "string"}, "ref_images": {"type": "array", "items": {"type": "string"}},
         "tier": {"type": "string", "enum": ["best", "budget"]}}}},
    {"name": "routing",
     "description": "Show the auto-routing table (which models are tried, in order, for each job type). kind is video "
                    "(default) or picture. The owner can change one job's list by passing job and models (model ids, "
                    "best first); an empty list restores the default.",
     "inputSchema": {"type": "object", "properties": {"kind": {"type": "string", "enum": ["video", "picture"]}, "job": {"type": "string"},
         "models": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "job_status",
     "description": "Status of a video job. When it's finished, video_url is a link to watch or download it.",
     "inputSchema": {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}},
    {"name": "list_jobs",
     "description": "Recent videos, newest first.",
     "inputSchema": {"type": "object", "properties": {"project": {"type": "string", "description": "Only this project (name or id); \"unsorted\" for none"}, "limit": {"type": "integer"}}}},
    {"name": "extend",
     "description": "Continue a finished video from its last frame. change optionally describes what happens next.",
     "inputSchema": {"type": "object", "required": ["id", "budget"], "properties": {"project": {"type": "string", "description": "Project name or id to save into (default: the active project)"}, 
         "id": {"type": "string"}, "seconds": {"type": "integer"}, "change": {"type": "string"},
         "budget": {"type": "number"}}}},
    {"name": "upscale",
     "description": "Sharpen a finished clip (up to 20 s). mode precise keeps it faithful; creative adds detail. "
                    "Call with quote_only true first to see the price.",
     "inputSchema": {"type": "object", "required": ["id"], "properties": {"project": {"type": "string", "description": "Project name or id to save into (default: the active project)"}, 
         "id": {"type": "string"}, "factor": {"type": "number"}, "mode": {"type": "string", "enum": ["precise", "creative"]},
         "budget": {"type": "number"}, "quote_only": {"type": "boolean"}}}},
    {"name": "balance",
     "description": "Credit left on this Video Studio account, plus what the connector has spent in the last 24 hours and its limits.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "stitch",
     "description": "Join finished videos, in the order given, into one clip on the server (free, no AI cost). "
                    "crossfade blends each join over that many seconds (0 = hard cut, ~0.15 hides small jumps). "
                    "mask_image (an image id: white = keep, black = drop) makes everything outside the shape pure black, "
                    "e.g. for projection mapping. Video only, no sound. Returns a job; poll job_status for video_url.",
     "inputSchema": {"type": "object", "required": ["jobs"], "properties": {"project": {"type": "string", "description": "Project name or id to save into (default: the active project)"}, 
         "jobs": {"type": "array", "items": {"type": "string"}, "description": "Job ids in play order"},
         "shape": {"type": "string", "enum": ["3:4", "9:16", "16:9", "1:1", "4:5"]},
         "crossfade": {"type": "number"}, "mask_image": {"type": "string"},
         "fit": {"type": "string", "enum": ["fit", "fill"]}, "title": {"type": "string"}}}},
    {"name": "loop",
     "description": "Cut one finished video for a VJ deck (free, no AI cost). mode pingpong plays the cut forward then "
                    "backward, so it loops seamlessly; mode trim just cuts it. start = seconds in, length = seconds "
                    "(0 = to the end). mask_image burns in a shape (white = keep) like stitch. Returns a job; poll job_status.",
     "inputSchema": {"type": "object", "required": ["id"], "properties": {"project": {"type": "string", "description": "Project name or id to save into (default: the active project)"}, 
         "id": {"type": "string"}, "mode": {"type": "string", "enum": ["pingpong", "trim"]},
         "start": {"type": "number"}, "length": {"type": "number"},
         "shape": {"type": "string", "enum": ["3:4", "9:16", "16:9", "1:1", "4:5"]},
         "mask_image": {"type": "string"}, "fit": {"type": "string", "enum": ["fit", "fill"]},
         "title": {"type": "string"}}}},
    {"name": "bundle",
     "description": "Pack finished videos into one zip file you can download with a single link (free). names = file names "
                    "inside the zip, in the same order as jobs (optional). Keep each zip to about 20 clips. Returns a job; "
                    "poll job_status, then give the person the video_url: it downloads as a .zip.",
     "inputSchema": {"type": "object", "required": ["jobs"], "properties": {"project": {"type": "string", "description": "Project name or id to save into (default: the active project)"}, 
         "jobs": {"type": "array", "items": {"type": "string"}}, "names": {"type": "array", "items": {"type": "string"}},
         "title": {"type": "string"}}}},
    {"name": "last_frame",
     "description": "Save a finished video's final frame as an image id (free). Use it as the next video's start_image "
                    "so chained clips join exactly where the last one really ended.",
     "inputSchema": {"type": "object", "required": ["id"], "properties": {"project": {"type": "string", "description": "Project name or id to save into (default: the active project)"}, "id": {"type": "string"}}}},
]


async def _call(name: str, a: dict, request: Request, user: dict) -> dict:
    request.state.user = user
    req, uid = request, user["id"]
    base = va._public_base(req)

    if name == "list_projects":
        return await vp.overview(uid)

    if name == "create_project":
        p = await vp.create(uid, a.get("name"))
        return {"created": p["name"], "id": p["id"], "active": p["name"]}

    if name == "use_project":
        ok, p = await vp.resolve(uid, a.get("project"))
        if not ok:
            raise ValueError(await _no_project(uid, a.get("project")))
        await vp.set_active(uid, p["id"] if p else None)
        return {"active": p["name"] if p else "Unsorted"}

    if name == "project_contents":
        ref = a.get("project")
        ok, p = await vp.resolve(uid, ref) if ref is not None else (True, {"id": await vp.active(uid)})
        if not ok:
            raise ValueError(await _no_project(uid, ref))
        pid = (p or {}).get("id")
        n = max(1, min(100, int(a.get("limit") or 30)))
        rows = [j for j in await va._all_jobs() if va._owns(uid, j) and j.get("project") == pid and not j.get("superseded")]
        vids = [await _job_view(req, j) for j in reversed(rows) if not vp.is_edited(j)][:n]
        edited = [await _job_view(req, j) for j in reversed(rows) if vp.is_edited(j)][:n]
        pics = [{**x, "url": base + x["url"]} for x in await vp.pictures(uid, pid, n)]
        return {"project": await vp.name_of(uid, pid), "videos": vids, "edited": edited, "pictures": pics}

    if name == "move_to_project":
        ok, p = await vp.resolve(uid, a.get("project"))
        if not ok:
            raise ValueError(await _no_project(uid, a.get("project")))
        ids = [str(x) for x in (a.get("ids") or [])]
        kinds = [a["kind"]] if a.get("kind") in ("video", "picture") else ["video", "picture"]
        moved = 0
        for k in kinds:
            moved += await vp.move(uid, k, ids, p["id"] if p else None)
        return {"moved": moved, "to": p["name"] if p else "Unsorted"}

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
        if cost is not None:
            cost = va._bill(uid, cost)
        return {"cost_usd": cost, "basis": basis, "per_job_limit": _max_job(), "spent_24h": _spent(uid), "daily_limit": _daily()}

    if name == "upload_image":
        raw = None
        if a.get("data_base64"):
            s = str(a["data_base64"])
            if "," in s[:100]:
                s = s.split(",", 1)[1]
            raw = base64.b64decode(s)
        elif str(a.get("url") or "").startswith("https://") and user.get("owner"):
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
                r = await c.get(a["url"])
            r.raise_for_status()
            raw = r.content
        if not raw:
            raise ValueError("Send the picture as data_base64" + ("" if not user.get("owner") else " or an https url"))
        if len(raw) > va.MAX_IMAGE_BYTES:
            raise ValueError("Pictures must be under 8 MB")
        kind = va._sniff(raw)
        if not kind:
            raise ValueError("Use a PNG, JPEG or WebP picture")
        iid = _tok(24)
        await va._put(f"{va.PREFIX}images/{iid}", raw, kind[1])
        await va._index_image(iid, uid, "upload")
        return {"image_id": iid, "url": f"{base}/video/img/{iid}"}

    if name in ("pick_model", "make_video") and str(a.get("model") or "auto").lower() == "auto":
        budget = round(float(a.get("budget") or 0), 2)
        r = await vr.route(uid, str(a.get("prompt") or ""), duration=float(a.get("duration") or 0),
                           aspect=str(a.get("aspect_ratio") or ""), resolution=str(a.get("resolution") or ""),
                           audio=a.get("audio"), start=bool(a.get("start_image") or a.get("end_image")),
                           refs=len(a.get("ref_images") or []), budget=budget, tier=str(a.get("tier") or "best"))
        if name == "pick_model":
            return r
        a = {**a, "model": r["model"], "resolution": r["resolution"], "audio": r["audio"]}
        routed = r
    else:
        routed = None
        if name == "pick_model":
            return {"model": a.get("model"), "note": "A model was named, so nothing was routed."}

    if name == "routing" and a.get("kind") == "picture":
        if a.get("job"):
            if not user.get("owner"):
                raise ValueError("Only the owner can change routing")
            await vpic.set_proute(str(a["job"]), [str(x) for x in (a.get("models") or [])])
        names = {m["id"]: m["name"] for m in await vpic.catalog()}
        t = await vpic.ptable()
        return {k: {"label": vpic.PJOBS[k], "models": [names.get(x, x + " (not available)") for x in t[k]], "ids": t[k],
                    "is_default": t[k] == vpic.PDEFAULTS[k]} for k in vpic.PJOBS}

    if name == "make_shape":
        out = await vsh.make_shape(uid, str(a.get("image") or ""), str(a.get("name") or ""))
        _note(uid, out["cost"])
        return {**out, "url": f"{base}{out['url']}"}

    if name == "fill_shape":
        if (g := _guard(va.IMAGE_HOLD, uid)):
            raise ValueError(g)
        out = await vsh.fill_shape(uid, str(a.get("mask_id") or ""), str(a.get("prompt") or ""),
                                   tier=str(a.get("tier") or "best"), budget=float(a.get("budget") or 0),
                                   model=str(a.get("model") or ""))
        _note(uid, out["cost"])
        return {**out, "url": f"{base}{out['url']}"}

    if name == "list_shapes":
        return {"shapes": [{**s, "url": f"{base}{s['url']}"} for s in await vsh.list_shapes(uid)]}

    if name == "list_image_models":
        q = str(a.get("q") or "").lower()
        rows = [m for m in await vpic.describe_models()
                if not q or q in (m["id"] + m["name"] + m["about"] + " ".join(m["can"])).lower()]
        return {"models": rows, "count": len(rows),
                "note": "Token-priced models show an estimate; it self-corrects from what pictures actually cost."}

    if name in ("pick_image_model", "make_image"):
        src = str(a.get("from_image") or "")
        refs = [str(x) for x in (a.get("ref_images") or [])][:16]
        kw = dict(aspect=str(a.get("aspect") or ""), clear=bool(a.get("transparent")),
                  vector=True if a.get("vector") else None, resolution=str(a.get("resolution") or ""),
                  tier=str(a.get("tier") or "best"), budget=float(a.get("budget") or 0), model=str(a.get("model") or ""))
        if name == "pick_image_model":
            return await vpic.route_picture(uid, str(a.get("prompt") or ""), has_src=bool(src), refs=len(refs), **kw)
        if (g := _guard(va.IMAGE_HOLD, uid)):
            raise ValueError(g)
        out = await vpic.make_picture(uid, str(a.get("prompt") or ""), src_ids=[src] if src else [], ref_ids=refs, **kw)
        _note(uid, out["cost"])
        res = {"image_id": None if out["vector"] else out["id"], "url": f"{base}{out['url']}", "cost_usd": out["cost"],
               "made_with": out["name"], "routed": {k: out[k] for k in ("job_label", "tier", "reason", "resolution", "quality")}}
        if out["vector"]:
            res["note"] = "SVG vector: open the url to view or download. It can't be used as a video frame."
        return res

    if name == "routing":
        if a.get("job"):
            if not user.get("owner"):
                raise ValueError("Only the owner can change routing")
            await vr.set_route(str(a["job"]), [str(x) for x in (a.get("models") or [])])
        names = {m["id"]: m["name"] for m in await va._models()}
        t = await vr.table()
        return {k: {"label": vr.JOBS[k], "models": [names.get(x, x) for x in t[k]], "ids": t[k],
                    "is_default": t[k] == vr.DEFAULTS[k]} for k in vr.JOBS}

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
        if (g := _guard(va._bill(uid, est), uid)):
            raise ValueError(g)
        spec = {"model": m["id"], "prompt": str(a.get("prompt") or ""), "duration": a.get("duration"),
                "resolution": a.get("resolution") or "", "aspect_ratio": a.get("aspect_ratio") or "",
                "audio": bool(a.get("audio", False)), "budget": min(budget, _max_job()), "images": images}
        out = await va._start_job(req, spec, {"user": uid, "base": base})
        if isinstance(out, Response):
            raise ValueError(_detail(out))
        _note(uid, float(out.get("estimate") or est))
        res = {"job": out, "next": "Poll job_status with this id; videos usually take 1-5 minutes."}
        if routed:
            res["routed"] = {k: routed[k] for k in ("name", "resolution", "audio", "job_label", "tier", "reason")}
        return res

    if name == "job_status":
        j = va._find(await va._all_jobs(), str(a.get("id") or ""))
        if not va._owns(uid, j):
            raise ValueError("No such video")
        va._ensure_poller(j)
        return await _job_view(req, j)

    if name == "list_jobs":
        n = max(1, min(40, int(a.get("limit") or 10)))
        jobs = [j for j in await va._all_jobs() if va._owns(uid, j) and not j.get("superseded")]
        if a.get("project") is not None:
            ok, p = await vp.resolve(uid, a["project"])
            if not ok:
                raise ValueError(await _no_project(uid, a["project"]))
            jobs = [j for j in jobs if j.get("project") == (p["id"] if p else None)]
        return {"jobs": [await _job_view(req, j) for j in reversed(jobs[-n:])]}

    if name == "extend":
        src = va._find(await va._all_jobs(), str(a.get("id") or ""))
        if not va._owns(uid, src) or not src.get("clip_key"):
            raise ValueError("That video isn't ready yet")
        budget = min(round(float(a.get("budget") or 0), 2), _max_job())
        if (g := _guard(budget, uid)):
            raise ValueError(g)
        src["base"] = src.get("base") or base
        out = await va._extend(src, auto=False, seconds=int(a["seconds"]) if a.get("seconds") else None, budget=budget,
                               change=str(a.get("change") or "").strip()[:1500] or None)
        _note(uid, float(out.get("estimate") or budget))
        return {"job": out}

    if name == "upscale":
        mode = "precise" if a.get("mode") != "creative" else "creative"
        j, plan, err = await va._upscale_quote(req, str(a.get("id") or ""), a.get("factor"), mode)
        if err:
            raise ValueError(_detail(err))
        plan = {**plan, "cost_billed": va._bill(uid, plan["cost"])}
        if a.get("quote_only"):
            return {"quote": plan}
        if (g := _guard(plan["cost_billed"], uid)):
            raise ValueError(g)
        budget = round(float(a.get("budget") or plan["cost_billed"] + 0.05), 2)
        url = await _view_url(req, j)
        extra = {"user": uid, "base": base, "video_ref_url": url + ("?song=1" if j.get("scored_key") else ""),
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
        _note(uid, plan["cost_billed"])
        return {"job": out}

    if name == "balance":
        info = {"connector_spent_24h": _spent(uid), "connector_daily_limit": _daily(), "per_job_limit": _max_job()}
        if not user.get("owner"):  # customers see their own credit, never the business's accounts
            fresh = await va._user_by_id(uid)
            info.update(credit=va._credit(fresh), markup=va.USAGE_MARKUP,
                        note="Add credit in Video Studio > Account.")
            return info
        info["app_spent_24h"] = await va._spent_24h()
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.get(f"{va.OR_BASE}/credits", headers=va._or_headers())
                d = (r.json() or {}).get("data") or {}
                if "total_credits" in d:
                    info["openrouter_credit_left"] = round(float(d["total_credits"]) - float(d.get("total_usage") or 0), 2)
                if va._owner_wallet_on():
                    info["owner_credit"] = va._credit(await va._owner_rec())
                    k = await c.get(f"{va.OR_BASE}/key", headers=va._or_headers("owner"))
                    kd = (k.json() or {}).get("data") or {}
                    info["owner_key_spent_total"] = round(float(kd.get("usage") or 0), 4)
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
                                        title=str(a.get("title") or "Stitch")[:80], user=uid)
        except RuntimeError as e:
            raise ValueError(str(e))
        return {"job": await _job_view(req, job), "next": "Poll job_status with this id; stitching takes about 1-4 minutes."}

    if name == "loop":
        shape = a.get("shape") if a.get("shape") in va.SHAPES else "3:4"
        try:
            job = await va.start_loop(str(a.get("id") or ""), shape=shape, start=float(a.get("start") or 0),
                                      length=float(a.get("length") or 0),
                                      mode="pingpong" if a.get("mode") != "trim" else "trim",
                                      mask_image=str(a.get("mask_image") or "") or None,
                                      fit="fill" if a.get("fit") == "fill" else "fit",
                                      title=str(a.get("title") or "Loop")[:80], user=uid)
        except RuntimeError as e:
            raise ValueError(str(e))
        return {"job": await _job_view(req, job), "next": "Poll job_status with this id."}

    if name == "bundle":
        try:
            job = await va.start_bundle([str(x) for x in (a.get("jobs") or []) if x],
                                        [str(x) for x in (a.get("names") or [])],
                                        title=str(a.get("title") or "clips")[:60], user=uid)
        except RuntimeError as e:
            raise ValueError(str(e))
        return {"job": await _job_view(req, job), "next": "Poll job_status; when ready, video_url downloads the zip."}

    if name == "last_frame":
        try:
            iid = await va.save_last_frame(str(a.get("id") or ""), user=uid)
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


@mcp_router.get("/mcp")
async def mcp_get_oauth(request: Request):
    return Response(status_code=405, headers={"Allow": "POST"})


@mcp_router.post("/mcp")
async def mcp_post_oauth(request: Request):
    """The customer door: needs a signed-in account (OAuth bearer token)."""
    user = await user_for_token(request)
    if not user:
        return challenge(request)
    return await _handle(request, user)


@mcp_router.post("/mcp/{secret}")
async def mcp_post(secret: str, request: Request):
    if not _secret_ok(secret):
        return Response("Not found", status_code=404)
    return await _handle(request, await va._owner_rec())


async def _handle(request: Request, user: dict):
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
                                          "make_video, poll job_status for results, and stay within the budget the user gives. "
                                          "Work is filed in projects: when the user names a project, pass it as project on "
                                          "each tool (or call use_project); tell the user which project things were saved to. "
                                          "Leave make_video's and make_image's model out (auto) unless the user names one: the router picks the "
                                          "best model for the job; pass tier budget only if they ask for cheap. Tell the user "
                                          "which model was used and why."})
    if method == "ping":
        return _rpc(mid, {})
    if method == "tools/list":
        return _rpc(mid, {"tools": TOOLS})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        args = args if isinstance(args, dict) else {}
        token = None
        try:
            if name in MAKERS and args.get("project") is not None:
                ok, p = await vp.resolve(user["id"], args["project"])
                if not ok:
                    raise ValueError(await _no_project(user["id"], args["project"]))
                token = va._PROJECT.set(p["id"] if p else None)
            out = await _call(str(name), args, request, user)
            if name in MAKERS and isinstance(out, dict):
                job = out.get("job") if isinstance(out.get("job"), dict) else None
                pid = job.get("project") if job and "project" in job else (
                    va._PROJECT.get() if token is not None else await vp.active(user["id"]))
                out["saved_to_project"] = await vp.name_of(user["id"], pid)
            print(f"VIDEO_MCP call {name} ok ({'owner' if user.get('owner') else 'customer'})", flush=True)
            if token is not None:
                va._PROJECT.reset(token)
            return _rpc(mid, {"content": [{"type": "text", "text": json.dumps(out, indent=1, default=str)}]})
        except Exception as e:
            print(f"VIDEO_MCP call {name} failed: {str(e)[:200]}", flush=True)
            if token is not None:
                va._PROJECT.reset(token)
            return _rpc(mid, {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True})
    return _rpc(mid, error={"code": -32601, "message": f"unknown method {method}"})
