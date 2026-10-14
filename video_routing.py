"""Auto model routing for Video Studio.

route() reads a request (prompt + what's attached + length/shape/sound/budget), labels the job, then walks
that job's ranked model list and returns the first model that can actually do it: it must support the
length and shape, be priceable for the input mode, and fit the budget. Default is best quality: each model
is tried at its highest resolution (below 4K unless 4K is asked for) that fits, before moving down the list.
tier="budget" (or a request that asks for a cheap draft) puts the cheap draft models first.

The ranked lists below are defaults from public blind-vote arenas (Arena and Artificial Analysis,
Aug-Sep 2026), limited to models available through OpenRouter. The owner can override any list; overrides
are stored in {PREFIX}routing.json and survive deploys.

The app sends chain=true: lengths past a model's longest clip are built from parts (seg x parts), and the
reply lists every model that can do the job within budget ("candidates"), so the page only offers those.
For changing an existing clip it also sends video_in (edit: only models that take a video in), fit_length
(the model's length closest to the clip's) and prefer (try this model first, e.g. the clip's own).

Routes (signed in): POST /video/route  -> {model, resolution, audio, estimate, seg, parts, job, reason, tried,
                                           candidates: [{model, name, resolution, estimate, seg, parts}]}
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
    "restyle": "Change an existing video (the video goes in)",
}
# Picked by the request itself, never by the classifier.
FIXED_JOBS = {"restyle"}
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
    # Empty = every model that can take a video in, video-editing models first (see route()).
    "restyle": [],
}
RESTYLE_HINT = re.compile(r"aleph|edit|lucy|restyle", re.I)
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


_classify_cache: dict = {}


async def classify(prompt: str, start: bool, refs: int, duration: float, audio: bool, uhd: bool,
                   chain: bool = False) -> dict:
    key = (prompt[:1500], start, refs, duration > 15, audio, uhd, chain)
    if key in _classify_cache:
        return dict(_classify_cache[key])
    out = await _classify(prompt, start, refs, duration, audio, uhd, chain)
    if len(_classify_cache) > 300:
        _classify_cache.clear()
    _classify_cache[key] = dict(out)
    return out


async def _classify(prompt: str, start: bool, refs: int, duration: float, audio: bool, uhd: bool,
                    chain: bool) -> dict:
    job, cheap = _rules(prompt, start, refs, 0 if chain else duration, audio, uhd)
    why = "matched the request's wording and attachments"
    try:
        txt = await va._or_chat(CLASSIFY.format(
            jobs="\n".join(f"- {k}: {v}" for k, v in JOBS.items() if k not in FIXED_JOBS), start=start, refs=refs, dur=duration or "?",
            audio=audio, uhd=uhd, prompt=prompt[:1500]))
        m = re.search(r"\{.*\}", txt, re.S)
        d = json.loads(m.group(0)) if m else {}
        if d.get("job") in JOBS and d.get("job") not in FIXED_JOBS:
            job, why = d["job"], str(d.get("why") or why)[:120]
            cheap = cheap or bool(d.get("cheap"))
    except Exception:
        pass  # the rules above are the fallback
    # Hard facts beat the classifier.
    if uhd:
        job = "uhd"
    elif refs and job not in ("dialogue", "characters"):
        job = "characters"
    if duration > 15 and job not in ("long_shot", "characters") and not chain:
        job = "long_shot"
    return {"job": job, "cheap": cheap, "why": why}


def _res_options(m: dict, want: str, uhd: bool) -> list[str]:
    res = m.get("resolutions") or [""]
    if want:
        return [want] if want in res else []
    allowed = [r for r in res if uhd or RES_RANK.get(r, 0) < RES_RANK["4K"]] or res
    return sorted(allowed, key=lambda r: -RES_RANK.get(r, 0))


def _plan(m: dict, duration: float, chain: bool, fit: bool = False) -> tuple[int, int] | None:
    """(seconds per part, parts) for this model, or None if it can't make this length.
    chain=True (the app) builds lengths past the model's longest clip from parts, each starting on the
    last frame of the one before, the same way the app's long videos work."""
    T = int(duration or 0)
    ds = sorted({int(d) for d in (m.get("durations") or [])})
    if not ds or not T or T in ds:
        return T, 1
    if fit:  # changing an existing clip: the model's length closest to it
        return min(ds, key=lambda d: (abs(d - T), -d)), 1
    if chain and T > ds[-1]:
        parts = -(-T // ds[-1])
        if parts <= 12:
            return ds[-1], parts
    return None


def _takes_video(m: dict) -> bool:
    """Whether OpenRouter prices this model for a video going in (i.e. it can edit a clip)."""
    return any(va.estimate_cost(m, 5.0, r, snd, "video", 0)[0] is not None
               for r in (m.get("resolutions") or [""]) for snd in (False, True))


async def route(uid: str, prompt: str, *, duration: float, aspect: str = "", resolution: str = "", audio=None,
                start: bool = False, refs: int = 0, budget: float = 0, tier: str = "best",
                chain: bool = False, video_in: bool = False, fit_length: bool = False, prefer: str = "") -> dict:
    """video_in: the job sends an existing clip to the model (edits), so only models that take a video in.
    fit_length: use each model's length closest to `duration` (for changing a clip of a set length).
    prefer: a model to try first if it can do the job (e.g. keep the clip's model when extending it)."""
    uhd = resolution.upper() == "4K" or bool(re.search(r"\b4k\b", prompt.lower()))
    if video_in:
        c = {"job": "restyle", "cheap": False, "why": "an existing video is being changed"}
    else:
        c = await classify(prompt, start, refs, duration, bool(audio), uhd, chain)
    job = c["job"]
    sound = bool(audio) if audio is not None else job == "dialogue"
    cheap = tier == "budget" or c["cheap"]
    t = await table()
    models = {m["id"]: m for m in await va._models()}
    if video_in:  # every model that can take a video in qualifies (the price check below proves it)
        order = list(t["restyle"]) + sorted(models, key=lambda i: (not RESTYLE_HINT.search(i), i))
    else:
        order = (t["draft"] + t[job]) if cheap and job != "draft" else list(t[job])
        if job != "scene":
            order += [x for x in t["scene"] if x not in order]  # anything sensible beats nothing
    if prefer and prefer in models:
        order = [prefer] + [x for x in order if x != prefer]
    mode = "video" if video_in else "image" if start else "ref" if refs else "text"
    n_img = refs if video_in else (1 if start else 0) + refs
    tried, candidates, best = [], [], None
    for mid in dict.fromkeys(order):
        m = models.get(mid)
        if not m or (video_in and not _takes_video(m)):
            continue
        plan = _plan(m, duration, chain, fit_length)
        if plan is None:
            tried.append(f"{m['name']}: can't do {int(duration)}s"); continue
        seg, parts = plan
        if aspect and m.get("aspect_ratios") and aspect not in m["aspect_ratios"]:
            tried.append(f"{m['name']}: no {aspect}"); continue
        opts = _res_options(m, resolution, uhd)
        if not opts:
            tried.append(f"{m['name']}: no {resolution}"); continue
        if cheap:
            opts = list(reversed(opts))
        priced, pick = False, None
        for res in opts:
            cost, _ = va.estimate_cost(m, float(seg), res, sound, mode, n_img)
            more = 0.0
            if cost is not None and parts > 1:  # later parts start from the previous part's last frame
                more, _ = va.estimate_cost(m, float(seg), res, sound, "image", 1)
            if cost is None or more is None:
                continue
            priced = True
            first = va._bill(uid, cost)
            total = round(first + va._bill(uid, more) * (parts - 1), 4)
            # the server checks first-part price x parts against the budget, so fit both ways
            if budget and max(total, first * parts) > budget + 1e-9:
                continue
            pick = {"model": m["id"], "name": m["name"], "resolution": res, "estimate": total, "seg": seg,
                    "parts": parts}
            break
        if pick:
            candidates.append(pick)
            if best is None:
                best = pick
        else:
            tried.append(f"{m['name']}: over ${budget:.2f} budget" if priced and budget
                         else f"{m['name']}: can't price this")
    if best is None:
        if video_in and not tried:
            raise ValueError("No model can do this within the settings. None of the current models take a video in.")
        raise ValueError("No model can do this within the settings. " + "; ".join(tried[:6]))
    reason = (f"{'Cheapest good option' if cheap else 'Best available'} for: {JOBS[job].lower()}"
              f"{' (top pick was skipped)' if tried and order and best['model'] != order[0] else ''}.")
    return {**best, "audio": sound, "job": job, "job_label": JOBS[job], "tier": "budget" if cheap else "best",
            "reason": reason, "classifier": c["why"], "tried": tried, "candidates": candidates}


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
                          tier=str(b.get("tier") or "best"), chain=bool(b.get("chain")),
                          video_in=bool(b.get("video_in")), fit_length=bool(b.get("fit_length")),
                          prefer=str(b.get("prefer") or ""))
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
