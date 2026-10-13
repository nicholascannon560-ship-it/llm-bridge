"""Auto model routing for Video Studio.

route() reads a request (prompt + what's attached + length/shape/sound/budget), labels the job, then walks
that job's ranked model list and returns the first model that can actually do it: it must support the
length and shape, be priceable for the input mode, and fit the budget. Default is best quality: each model
is tried at its highest resolution (below 4K unless 4K is asked for) that fits, before moving down the list.
tier="budget" (or a request that asks for a cheap draft) puts the cheap draft models first.

The ranked lists below are defaults from public blind-vote arenas (Arena and Artificial Analysis,
Aug-Sep 2026), limited to models available through OpenRouter. The owner can override any list; overrides
are stored in {PREFIX}routing.json and survive deploys.

Routes (signed in): POST /video/route  -> {model, resolution, audio, estimate, job, reason, tried}
                    GET  /video/routing -> current table; POST /video/routing (owner) {job, models: [...]}
"""
from __future__ import annotations

import json
import re

from fastapi import APIRouter, Request

import video_app as va

routing_router = APIRouter()
ROUTING_KEY = f"{va.PREFIX}routing.json"

JOBS = {
    "animate_photo": "Bring a photo or start frame to life",
    "scene": "A shot made from a text description (realistic or cinematic)",
    "dialogue": "People talking, singing or with synced sound",
    "characters": "Keep the same character or product across shots, using reference pictures",
    "long_shot": "One continuous shot longer than 15 seconds",
    "draft": "Cheap, fast test of an idea",
    "uhd": "4K output",
}
DEFAULTS = {
    "animate_photo": ["minimax/hailuo-3-max", "alibaba/wan-3.0", "bytedance/seedance-2.5", "bytedance/seedance-2.0",
                      "x-ai/grok-imagine-video-1.5", "black-forest-labs/flux-3-video"],
    "scene": ["black-forest-labs/flux-3-video", "alibaba/wan-3.0", "bytedance/seedance-2.5", "minimax/hailuo-3",
              "bytedance/seedance-2.0", "minimax/hailuo-3-max", "alibaba/happyhorse-1.1"],
    "dialogue": ["minimax/hailuo-3", "bytedance/seedance-2.0", "bytedance/seedance-2.5", "alibaba/wan-3.0",
                 "google/veo-3.1"],
    "characters": ["bytedance/seedance-2.5", "bytedance/seedance-2.0", "alibaba/wan-3.0", "minimax/hailuo-3-max"],
    "long_shot": ["alibaba/wan-3.0", "bytedance/seedance-2.5", "black-forest-labs/flux-3-video", "alibaba/wan-3.0-prime"],
    "draft": ["x-ai/grok-imagine-video-1.5-lite", "bytedance/seedance-1-5-pro", "bytedance/seedance-2.0-mini",
              "google/veo-3.1-lite"],
    "uhd": ["google/veo-3.1", "bytedance/seedance-2.0", "google/veo-3.1-fast"],
}
RES_RANK = {"4K": 6, "2K": 5, "1080p": 4, "768p": 3, "720p": 2, "480p": 1, "": 0}
_overrides: dict | None = None

CLASSIFY = """You route requests for an AI video generator. Pick the ONE job type that best fits.
Job types:
{jobs}
Facts about the request: start/end picture attached: {start}; reference pictures attached: {refs}; length: {dur}s;
sound wanted: {audio}; 4K asked for: {uhd}.
Request: {prompt}

Rules: people speaking, singing, lip-sync or dialogue -> dialogue. A start picture to animate -> animate_photo
(unless it is about talking). Reference pictures for a consistent character/product -> characters. Only pick
draft if the request asks for something cheap, quick, rough or a test.
Reply with only JSON: {{"job": "<type>", "cheap": true|false, "why": "<under 12 words>"}}"""


async def table() -> dict:
    global _overrides
    if _overrides is None:
        raw = await va._get(ROUTING_KEY)
        try:
            _overrides = json.loads(raw) if raw else {}
        except Exception:
            _overrides = {}
    return {k: list(_overrides.get(k) or v) for k, v in DEFAULTS.items()}


async def set_route(job: str, models: list[str]) -> dict:
    global _overrides
    if job not in JOBS:
        raise ValueError(f"Unknown job type. Use one of: {', '.join(JOBS)}")
    await table()
    known = {m["id"] for m in await va._models()}
    bad = [m for m in models if m not in known]
    if bad:
        raise ValueError(f"Not available models: {', '.join(bad)}")
    if models:
        _overrides[job] = models
    else:
        _overrides.pop(job, None)  # empty list = back to the default
    await va._put(ROUTING_KEY, json.dumps(_overrides).encode(), "application/json")
    return await table()


def _rules(prompt: str, start: bool, refs: int, duration: float, audio: bool, uhd: bool) -> tuple[str, bool]:
    p = prompt.lower()
    cheap = bool(re.search(r"\b(cheap|cheapest|draft|quick test|rough|low[- ]?cost|budget)\b", p))
    talk = bool(re.search(r"\b(talk|talking|says?|saying|speak|speaking|speech|dialogue|sings?|singing|lip[- ]?sync|"
                          r"interview|narrat\w*|monologue|conversation)\b", p)) or ('"' in prompt and audio)
    if uhd:
        return "uhd", cheap
    if talk:
        return "dialogue", cheap
    if refs:
        return "characters", cheap
    if start:
        return "animate_photo", cheap
    if duration > 15:
        return "long_shot", cheap
    return ("draft" if cheap else "scene"), cheap


async def classify(prompt: str, start: bool, refs: int, duration: float, audio: bool, uhd: bool) -> dict:
    job, cheap = _rules(prompt, start, refs, duration, audio, uhd)
    why = "matched the request's wording and attachments"
    try:
        txt = await va._or_chat(CLASSIFY.format(
            jobs="\n".join(f"- {k}: {v}" for k, v in JOBS.items()), start=start, refs=refs, dur=duration or "?",
            audio=audio, uhd=uhd, prompt=prompt[:1500]))
        m = re.search(r"\{.*\}", txt, re.S)
        d = json.loads(m.group(0)) if m else {}
        if d.get("job") in JOBS:
            job, why = d["job"], str(d.get("why") or why)[:120]
            cheap = cheap or bool(d.get("cheap"))
    except Exception:
        pass  # the rules above are the fallback
    # Hard facts beat the classifier.
    if uhd:
        job = "uhd"
    elif refs and job not in ("dialogue", "characters"):
        job = "characters"
    if duration > 15 and job not in ("long_shot", "characters"):
        job = "long_shot"
    return {"job": job, "cheap": cheap, "why": why}


def _res_options(m: dict, want: str, uhd: bool) -> list[str]:
    res = m.get("resolutions") or [""]
    if want:
        return [want] if want in res else []
    allowed = [r for r in res if uhd or RES_RANK.get(r, 0) < RES_RANK["4K"]] or res
    return sorted(allowed, key=lambda r: -RES_RANK.get(r, 0))


async def route(uid: str, prompt: str, *, duration: float, aspect: str = "", resolution: str = "", audio=None,
                start: bool = False, refs: int = 0, budget: float = 0, tier: str = "best") -> dict:
    uhd = resolution.upper() == "4K" or bool(re.search(r"\b4k\b", prompt.lower()))
    c = await classify(prompt, start, refs, duration, bool(audio), uhd)
    job = c["job"]
    sound = bool(audio) if audio is not None else job == "dialogue"
    cheap = tier == "budget" or c["cheap"]
    t = await table()
    order = (t["draft"] + t[job]) if cheap and job != "draft" else t[job]
    if job != "scene":
        order += [x for x in t["scene"] if x not in order]  # anything sensible beats nothing
    models = {m["id"]: m for m in await va._models()}
    mode = "image" if start else "ref" if refs else "text"
    n_img = (1 if start else 0) + refs
    tried = []
    for mid in dict.fromkeys(order):
        m = models.get(mid)
        if not m:
            continue
        if duration and m.get("durations") and int(duration) not in [int(d) for d in m["durations"]]:
            tried.append(f"{m['name']}: can't do {int(duration)}s"); continue
        if aspect and m.get("aspect_ratios") and aspect not in m["aspect_ratios"]:
            tried.append(f"{m['name']}: no {aspect}"); continue
        opts = _res_options(m, resolution, uhd)
        if not opts:
            tried.append(f"{m['name']}: no {resolution}"); continue
        if cheap:
            opts = list(reversed(opts))
        for res in opts:
            cost, _ = va.estimate_cost(m, float(duration or 0), res, sound, mode, n_img)
            if cost is None:
                continue
            billed = va._bill(uid, cost)
            if budget and billed > budget + 1e-9:
                continue
            reason = (f"{'Cheapest good option' if cheap else 'Best available'} for: {JOBS[job].lower()}"
                      f"{' (top pick was skipped)' if tried else ''}.")
            return {"model": m["id"], "name": m["name"], "resolution": res, "audio": sound, "estimate": billed,
                    "job": job, "job_label": JOBS[job], "tier": "budget" if cheap else "best", "reason": reason,
                    "classifier": c["why"], "tried": tried}
        tried.append(f"{m['name']}: over ${budget:.2f} budget" if budget else f"{m['name']}: can't price this")
    raise ValueError("No model can do this within the settings. " + "; ".join(tried[:6]))


# --------------------------------------------------------------------------- routes
@routing_router.post("/video/route")
async def route_endpoint(request: Request):
    if (d := await va._deny(request)):
        return d
    b = await va._body(request)
    imgs = b.get("images") or []
    try:
        out = await route(va._uid(request), str(b.get("prompt") or ""), duration=float(b.get("duration") or 0),
                          aspect=str(b.get("aspect_ratio") or ""), resolution=str(b.get("resolution") or ""),
                          audio=b.get("audio"), start=any(i.get("role") in ("first", "last") for i in imgs),
                          refs=sum(1 for i in imgs if i.get("role") == "ref"), budget=float(b.get("budget") or 0),
                          tier=str(b.get("tier") or "best"))
    except ValueError as e:
        return va._err(str(e))
    return out


@routing_router.get("/video/routing")
async def routing_get(request: Request):
    if (d := await va._deny(request)):
        return d
    names = {m["id"]: m["name"] for m in await va._models()}
    t = await table()
    return {"jobs": [{"job": k, "label": JOBS[k], "models": [{"id": x, "name": names.get(x, x + " (not available)")}
                                                              for x in t[k]], "default": t[k] == DEFAULTS[k]}
                     for k in JOBS]}


@routing_router.post("/video/routing")
async def routing_set(request: Request):
    if (d := await va._deny(request)):
        return d
    if not request.state.user.get("owner"):
        return va._err("Only the owner can change routing", 403)
    b = await va._body(request)
    try:
        await set_route(str(b.get("job") or ""), [str(x) for x in (b.get("models") or [])])
    except ValueError as e:
        return va._err(str(e))
    return await routing_get(request)
