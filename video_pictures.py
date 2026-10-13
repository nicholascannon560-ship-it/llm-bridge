"""Pictures for Video Studio: every OpenRouter picture model, routed to the one that fits the job.

Catalog: GET {OR}/images/models (live, cached 10 min) gives each model's real limits: resolutions, shapes,
how many reference pictures, quality levels, transparent backgrounds, SVG output. Pricing comes from each
model's endpoint record (cached a day): per picture (exact), per megapixel (computed) or per token
(estimated, then corrected by the running average of what that model actually charged).

Routing: the request is labelled (text/design, photo, art, edit, combine, cutout, vector, style match,
4K, draft); that job's ranked list is walked and the first model that can actually do it within budget
wins, at its best quality/resolution (or cheapest, for drafts and tier=budget). Defaults come from the
Arena / Artificial Analysis / BudgetPixel image arenas (Sep-Oct 2026); the owner can override any list.

Generation: POST {OR}/images, one shared call for the app and the Claude connector. SVG results are kept
apart from the picture library and served with a script-blocking CSP.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import secrets
import time

import httpx
from fastapi import APIRouter, Request, Response

import video_app as va

pictures_router = APIRouter()
COSTS_KEY = f"{va.PREFIX}picture_costs.json"
PROUTING_KEY = f"{va.PREFIX}picture_routing.json"
_cat = {"t": 0.0, "data": []}
_prices: dict[str, tuple[float, list]] = {}
_learned: dict | None = None
_overrides: dict | None = None

PJOBS = {
    "text_design": "Readable text or design: posters, signs, logos, labels, menus, infographics",
    "photo": "Photoreal picture",
    "art": "Illustration, painting, anime, 3D or other stylised art",
    "edit": "Change an existing picture",
    "combine": "Combine several pictures, or keep a character/product consistent from references",
    "cutout": "Transparent background: cutouts, stickers, product shots",
    "vector": "Vector SVG: logos and icons that scale",
    "style_match": "New picture in the exact style of reference pictures",
    "uhd": "4K or print size",
    "draft": "Cheap, quick test",
}
SUN, GI2, MAI = "openai/gpt-image-2.5-sunburst", "openai/gpt-image-2", "microsoft/mai-image-2.6"
NB21, NBPRO = "google/gemini-nano-banana-2.1", "google/gemini-3-pro-image"
PDEFAULTS = {
    "text_design": [SUN, MAI, GI2, NBPRO, "qwen/qwen-image-3-pro", "sourceful/riverflow-v2.5-pro",
                    "openai/gpt-image-2.5-flare"],
    "photo": [SUN, MAI, "x-ai/grok-imagine-image-2.0", NB21, "bytedance-seed/seedream-5-0-pro", "meta/muse-image",
              "black-forest-labs/flux-3-image"],
    "art": [SUN, "meta/muse-image", MAI, "recraft/recraft-v4.1-pro", "black-forest-labs/flux-3-image", NB21,
            "bytedance-seed/seedream-5-0-pro"],
    "edit": [SUN, GI2, MAI, "meta/muse-image", NB21, NBPRO, "bytedance-seed/seedream-5-0-pro"],
    "combine": [NB21, SUN, GI2, "bytedance-seed/seedream-5-0-pro", "black-forest-labs/flux-3-image",
                "tencent/hy-image-v3.5-preview", NBPRO],
    "cutout": [SUN, "openai/gpt-image-2.5-flare", "sourceful/riverflow-v2.5-pro", "openai/gpt-image-1",
               "sourceful/riverflow-v2.5-fast"],
    "vector": ["recraft/recraft-v4.1-pro-vector", "recraft/recraft-v4.1-vector", "recraft/recraft-v4-pro-vector",
               "recraft/recraft-v4-vector", "recraft/recraft-v4-styles-pro-vector"],
    "style_match": ["recraft/recraft-v4-styles-pro", "recraft/recraft-v4-styles", SUN, NB21],
    "uhd": [NB21, NBPRO, "bytedance-seed/seedream-4.5", "sourceful/riverflow-v2.5-pro",
            "black-forest-labs/flux-3-image", "tencent/hy-image-v3.5-preview"],
    "draft": ["meta/muse-image", "bytedance-seed/seedream-5-0-flash", "google/gemini-3.1-flash-lite-image",
              "recraft/recraft-v4.1-flash", "microsoft/mai-image-2.6-flash", "openai/gpt-image-2.5-flare"],
}
RES_RANK = {"4K": 7, "2K": 6, "1.5K": 5, "1K": 4, "768": 3, "512": 2, "": 0}
RES_MP = {"512": 0.26, "768": 0.59, "1K": 1.05, "1.5K": 2.36, "2K": 4.19, "4K": 16.8, "": 1.05}
Q_RANK = {"max": 6, "xhigh": 5, "high": 4, "medium": 3, "auto": 2, "low": 1}


# --------------------------------------------------------------------------- catalog
def _caps(m: dict) -> dict:
    sp = m.get("supported_parameters") or {}
    ev = lambda k: [str(v) for v in ((sp.get(k) or {}).get("values") or [])]
    takes_image = "image" in ((m.get("architecture") or {}).get("input_modalities") or [])
    ref = sp.get("input_references") or {}
    fmts = ev("output_format")
    return {
        "id": m["id"], "name": m.get("name") or m["id"], "about": (m.get("description") or "")[:240],
        "resolutions": ev("resolution"), "aspects": ev("aspect_ratio"), "qualities": ev("quality"),
        "backgrounds": ev("background"), "formats": fmts, "vector": fmts == ["svg"],
        "refs_min": int(ref.get("min", 0) or 0) if ref else 0,
        "refs_max": int(ref.get("max", 0) or 0) if ref else (1 if takes_image else 0),
        "seed": "seed" in sp,
    }


async def catalog(force: bool = False) -> list[dict]:
    if not force and _cat["data"] and time.time() - _cat["t"] < 600:
        return _cat["data"]
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{va.OR_BASE}/images/models", headers=va._or_headers())
    r.raise_for_status()
    data = [_caps(m) for m in (r.json().get("data") or []) if m.get("id")]
    _cat.update(t=time.time(), data=data)
    return data


async def pricing(mid: str) -> list:
    hit = _prices.get(mid)
    if hit and time.time() - hit[0] < 86400:
        return hit[1]
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(f"{va.OR_BASE}/images/models/{mid}/endpoints", headers=va._or_headers())
        eps = (r.json().get("endpoints") or []) if r.status_code < 400 else []
        rows = eps[0].get("pricing") or [] if eps else []
    except Exception:
        rows = []
    _prices[mid] = (time.time(), rows)
    return rows


def _out_tokens(mid: str, res: str, quality: str) -> float:
    if mid.startswith("openai/"):
        return {"low": 272, "medium": 1056, "high": 4160, "xhigh": 6200, "max": 8300}.get(quality or "high", 4160)
    if "gemini" in mid:
        return {"512": 750, "1K": 1290, "2K": 1680, "4K": 2520}.get(res or "1K", 1290)
    return 1300 * max(1.0, RES_MP.get(res, 1.05))


async def _learned_costs() -> dict:
    global _learned
    if _learned is None:
        raw = await va._get(COSTS_KEY)
        try:
            _learned = json.loads(raw) if raw else {}
        except Exception:
            _learned = {}
    return _learned


async def estimate(mid: str, res: str, quality: str, n_refs: int) -> tuple[float, str]:
    """Raw (pre-markup) cost of one picture, and how sure we are: exact / computed / estimated / learned."""
    learned = (await _learned_costs()).get(f"{mid}|{res}|{quality}")
    if learned and learned[1] >= 2:
        return round(learned[0], 4), "learned"
    total, sure = 0.0, "exact"
    for p in await pricing(mid):
        unit, what, usd = p.get("unit"), p.get("billable"), float(p.get("cost_usd") or 0)
        if unit == "image":
            total += usd * (n_refs if what == "input_image" else 1 if what == "output_image" else 0)
        elif unit == "megapixel":
            total += usd * (RES_MP.get(res, 1.05) if what == "output_image" else 1.05 * n_refs if what == "input_image" else 0)
            sure = "computed" if sure == "exact" else sure
        elif unit == "token":
            n = (_out_tokens(mid, res, quality) if what == "output_image" else 800 * n_refs if what == "input_image"
                 else 250 if what == "input_text" else 0)
            total += usd * n
            sure = "estimated"
        else:
            sure = "estimated"
    if total <= 0:
        return (round(learned[0], 4), "learned") if learned else (0.10, "estimated")
    return round(total, 4), sure


async def _learn(mid: str, res: str, quality: str, cost: float) -> None:
    d = await _learned_costs()
    k = f"{mid}|{res}|{quality}"
    avg, n = d.get(k, [cost, 0])
    n = min(n + 1, 20)
    d[k] = [round(avg + (cost - avg) / n, 5), n]
    await va._put(COSTS_KEY, json.dumps(d).encode(), "application/json")


# --------------------------------------------------------------------------- routing table
async def ptable() -> dict:
    global _overrides
    if _overrides is None:
        raw = await va._get(PROUTING_KEY)
        try:
            _overrides = json.loads(raw) if raw else {}
        except Exception:
            _overrides = {}
    return {k: list(_overrides.get(k) or v) for k, v in PDEFAULTS.items()}


async def set_proute(job: str, models: list[str]) -> dict:
    global _overrides
    if job not in PJOBS:
        raise ValueError(f"Unknown picture job. Use one of: {', '.join(PJOBS)}")
    await ptable()
    known = {m["id"] for m in await catalog()}
    if (bad := [m for m in models if m not in known]):
        raise ValueError(f"Not available picture models: {', '.join(bad)}")
    if models:
        _overrides[job] = models
    else:
        _overrides.pop(job, None)
    await va._put(PROUTING_KEY, json.dumps(_overrides).encode(), "application/json")
    return await ptable()


# --------------------------------------------------------------------------- classify + route
CLASSIFY = """You route requests for an AI picture generator. Pick the ONE job type that best fits.
Job types:
{jobs}
Facts: picture to change attached: {src}; reference pictures attached: {refs}; transparent background asked: {clear}.
Request: {prompt}

Rules: words that must appear in the picture, or a poster/sign/logo/label/menu/infographic -> text_design.
A logo or icon wanted as SVG/vector -> vector. One attached picture to change -> edit. Several pictures to
merge or a character to keep consistent -> combine. "In the style of" attached references -> style_match.
Only pick draft if the request asks for cheap, quick, rough or a test.
Reply with only JSON: {{"job": "<type>", "cheap": true|false, "why": "<under 12 words>"}}"""

_W = lambda *w: re.compile(r"\b(" + "|".join(w) + r")\b", re.I)
RX = {
    "vector": _W(r"svg", r"vector", r"scalable"),
    "cutout": _W(r"transparent", r"no background", r"cut-?out", r"sticker", r"isolated", r"alpha"),
    "uhd": _W(r"4k", r"print", r"printing", r"high[- ]res(olution)?", r"large format", r"billboard"),
    "style": re.compile(r"\b(in the (same )?style of (these|this|the|my) (reference|picture|image)s?|match(ing)? (the |their |its )?style)\b", re.I),
    "text": _W(r"poster", r"sign", r"signage", r"logo", r"headline", r"caption", r"label", r"menu", r"infographic",
               r"flyer", r"title", r"typography", r"lettering", r"words?", r"text", r"says", r"reads",
               r"comic", r"diagram", r"chart", r"meme", r"business card", r"book cover", r"thumbnail"),
    "art": _W(r"illustration", r"illustrated", r"painting", r"watercolou?r", r"anime", r"manga", r"cartoon", r"drawing",
              r"sketch", r"3d render", r"pixel art", r"oil", r"stylized", r"stylised", r"concept art", r"clip ?art",
              r"children'?s book", r"low poly"),
    "draft": _W(r"cheap", r"cheapest", r"draft", r"quick", r"rough", r"test", r"budget"),
}


def _rules(prompt: str, has_src: bool, refs: int, clear: bool, vector) -> tuple[str, bool]:
    cheap = bool(RX["draft"].search(prompt))
    if vector or (vector is None and RX["vector"].search(prompt)):
        return "vector", cheap
    if clear or RX["cutout"].search(prompt):
        return "cutout", cheap
    if refs and RX["style"].search(prompt):
        return "style_match", cheap
    total = refs + (1 if has_src else 0)
    if total >= 2:
        return "combine", cheap
    if has_src:
        return "edit", cheap
    if RX["uhd"].search(prompt):
        return "uhd", cheap
    if RX["text"].search(prompt) or re.search(r"[\"“][^\"”]{2,}[\"”]", prompt):
        return "text_design", cheap
    if RX["art"].search(prompt):
        return "art", cheap
    return ("draft" if cheap else "photo"), cheap


async def classify(prompt: str, has_src: bool, refs: int, clear: bool, vector=None) -> dict:
    job, cheap = _rules(prompt, has_src, refs, clear, vector)
    why = "matched the request's wording and attachments"
    try:
        txt = await va._or_chat(CLASSIFY.format(jobs="\n".join(f"- {k}: {v}" for k, v in PJOBS.items()),
                                                src=has_src, refs=refs, clear=clear, prompt=prompt[:1500]))
        mm = re.search(r"\{.*\}", txt, re.S)
        d = json.loads(mm.group(0)) if mm else {}
        if d.get("job") in PJOBS:
            job, why = d["job"], str(d.get("why") or why)[:120]
            cheap = cheap or bool(d.get("cheap"))
    except Exception:
        pass
    # Hard facts beat the classifier.
    total = refs + (1 if has_src else 0)
    if vector is True:
        job = "vector"
    elif clear:
        job = "cutout"
    elif job == "style_match" and not total:
        job = "art"
    elif job in ("edit", "combine") and not total:
        job = "photo"
    elif total >= 2 and job in ("photo", "art", "edit"):
        job = "combine"
    elif total == 1 and job in ("photo", "art"):
        job = "edit"
    if vector is False and job == "vector":
        job = "text_design"
    return {"job": job, "cheap": cheap, "why": why}


def _fits(m: dict, job: str, aspect: str, n_refs: int, clear: bool) -> str | None:
    """None if this model can do the job, else the reason it can't."""
    if job == "vector" and not m["vector"]:
        return "no SVG output"
    if job != "vector" and m["vector"]:
        return "SVG only"
    if clear and "transparent" not in m["backgrounds"]:
        return "no transparent background"
    if job == "uhd" and "4K" not in m["resolutions"]:
        return "no 4K"
    if n_refs > m["refs_max"]:
        return f"takes {m['refs_max']} picture{'s' if m['refs_max'] != 1 else ''} at most"
    if n_refs < m["refs_min"]:
        return f"needs {m['refs_min']} reference picture{'s' if m['refs_min'] != 1 else ''}"
    if aspect and aspect != "auto":
        if m["aspects"] and aspect not in m["aspects"]:
            return f"no {aspect}"
        if not m["aspects"] and aspect != "1:1":
            return "can't set the shape"
    return None


def _options(m: dict, job: str, resolution: str, cheap: bool, low_ok: bool) -> list[tuple[str, str]]:
    """(resolution, quality) settings to try, best first. Under best quality a model is only tried at
    its full quality ("high"; "max" for 4K/print) unless low_ok, the last-resort pass."""
    res = m["resolutions"] or [""]
    if resolution:
        res = [resolution] if resolution in res else []
    elif job != "uhd":
        res = [r for r in res if r != "4K"] or res
    res = sorted(res, key=lambda r: -RES_RANK.get(r, 0))
    cap = Q_RANK["max"] if job == "uhd" else Q_RANK["high"]
    qs = sorted([q for q in m["qualities"] if q != "auto" and Q_RANK.get(q, 0) <= cap], key=lambda q: -Q_RANK.get(q, 0))
    if qs and not cheap and not low_ok:
        top = Q_RANK[qs[0]]
        qs = [q for q in qs if Q_RANK[q] >= min(top, Q_RANK["high"])]
    combos = [(r, q) for q in (qs or [""]) for r in res]  # quality first, then size
    return list(reversed(combos)) if cheap else combos


async def route_picture(uid: str, prompt: str, *, aspect: str = "", has_src: bool = False, refs: int = 0,
                        clear: bool = False, vector=None, resolution: str = "", tier: str = "best",
                        budget: float = 0, model: str = "") -> dict:
    n_refs = refs + (1 if has_src else 0)
    cat = {m["id"]: m for m in await catalog()}
    if model and model.lower() != "auto":
        m = cat.get(model)
        if not m:
            raise ValueError("Unknown picture model. Use list_image_models.")
        job = "vector" if m["vector"] else "edit" if has_src else "photo"
        c, order = {"job": job, "cheap": tier == "budget", "why": "model chosen by name"}, [model]
    else:
        c = await classify(prompt, has_src, refs, clear, vector)
        job = c["job"]
        t = await ptable()
        cheap0 = tier == "budget" or c["cheap"]
        order = (t["draft"] + t[job]) if cheap0 and job not in ("draft", "vector", "cutout", "style_match") else t[job]
        if job not in ("vector", "cutout", "style_match", "uhd"):
            order += [x for x in t["photo"] + t["edit"] if x not in order]
    cheap = tier == "budget" or c["cheap"]
    tried = []
    passes = [False] if cheap else [False, True]  # second pass: lower quality settings, as a last resort
    for low_ok in passes:
        for mid in dict.fromkeys(order):
            m = cat.get(mid)
            if not m:
                continue
            if (why := _fits(m, job, aspect, n_refs, clear)):
                if not low_ok:
                    tried.append(f"{m['name']}: {why}")
                continue
            opts = _options(m, job, resolution, cheap, low_ok)
            if not opts:
                if not low_ok:
                    tried.append(f"{m['name']}: no {resolution}")
                continue
            for res, q in opts:
                raw, sure = await estimate(mid, res, q, n_refs)
                billed = va._bill(uid, raw)
                if budget and billed > budget + 1e-9:
                    continue
                return {"model": mid, "name": m["name"], "resolution": res, "quality": q, "estimate": round(billed, 4),
                        "price_basis": sure, "job": job, "job_label": PJOBS[job], "tier": "budget" if cheap else "best",
                        "reason": (f"{'Cheapest good option' if cheap else 'Best available'} for: {PJOBS[job].split(':')[0].lower()}"
                                   f"{' (top pick was skipped)' if tried else ''}."),
                        "classifier": c["why"], "tried": tried, "vector": m["vector"]}
            if not low_ok:
                tried.append(f"{m['name']}: over ${budget:.2f} budget")
    raise ValueError("No picture model can do this with these settings. " + "; ".join(tried[:6]))


# --------------------------------------------------------------------------- generate
async def make_picture(uid: str, prompt: str, *, aspect: str = "", src_ids: list[str] | None = None,
                       ref_ids: list[str] | None = None, clear: bool = False, vector=None, resolution: str = "",
                       tier: str = "best", budget: float = 0, model: str = "") -> dict:
    """Route, hold credit, generate, store, charge actual cost. Raises ValueError with a readable message."""
    prompt = (prompt or "").strip()[:4000]
    if not prompt:
        raise ValueError("Describe the picture")
    srcs, refs = list(src_ids or [])[:1], list(ref_ids or [])
    for i in srcs + refs:
        if not va.IMG_ID.match(str(i)):
            raise ValueError("Pictures must be image ids")
    r = await route_picture(uid, prompt, aspect=aspect, has_src=bool(srcs), refs=len(refs), clear=clear,
                            vector=vector, resolution=resolution, tier=tier, budget=budget, model=model)
    hold_amt = max(0.02, round(r["estimate"] * 1.3, 4))
    held = await va._image_hold(uid, hold_amt)
    if isinstance(held, Response):
        raise ValueError(json.loads(held.body).get("detail") or "Not enough credit")
    try:
        inputs = []
        for iid in srcs + refs:
            data = await va._get(f"{va.PREFIX}images/{iid}")
            if not data:
                raise ValueError("A picture you attached wasn't found")
            kind = va._sniff(data) or ("jpg", "image/jpeg")
            inputs.append({"type": "image_url", "image_url": {
                "url": f"data:{kind[1]};base64,{base64.b64encode(data).decode()}"}})
        payload = {"model": r["model"], "prompt": prompt}
        if aspect and aspect != "auto":
            payload["aspect_ratio"] = aspect
        if r["resolution"]:
            payload["resolution"] = r["resolution"]
        if r["quality"]:
            payload["quality"] = r["quality"]
        if clear:
            payload["background"] = "transparent"
        if r["vector"]:
            payload["output_format"] = "svg"
        if inputs:
            payload["input_references"] = inputs
        async with httpx.AsyncClient(timeout=300) as c:
            resp = await c.post(f"{va.OR_BASE}/images", json=payload, headers=va._or_headers(uid))
        data = resp.json()
        if resp.status_code >= 400 or data.get("error"):
            raise ValueError(f"{r['name']} refused: {(data.get('error') or {}).get('message') or resp.status_code}")
        first = (data.get("data") or [{}])[0]
        if not first.get("b64_json"):
            raise ValueError(f"{r['name']} didn't return a picture; try rewording")
        img = base64.b64decode(first["b64_json"])
        real = (data.get("usage") or {}).get("cost")
    except Exception as e:
        await va._release(uid, held, "picture failed")
        raise ValueError(str(e)) from None
    if real is not None:
        await _learn(r["model"], r["resolution"], r["quality"], float(real))
    billed = va._bill(uid, float(real)) if real is not None else r["estimate"]
    await va._charge_small(uid, held, billed, "picture")
    iid = secrets.token_urlsafe(24)
    if r["vector"] or first.get("media_type") == "image/svg+xml" or img.lstrip()[:5] in (b"<svg ", b"<?xml"):
        await va._put(f"{va.PREFIX}vectors/{iid}", img, "image/svg+xml")
        url = f"/video/svg/{iid}"
    else:
        kind = va._sniff(img) or ("png", "image/png")
        await va._put(f"{va.PREFIX}images/{iid}", img, kind[1])
        await va._index_image(iid, uid, "made", prompt)
        url = f"/video/img/{iid}"
    return {"id": iid, "url": url, "vector": url.startswith("/video/svg/"), "cost": round(billed, 4),
            "model": r["model"], "name": r["name"], "resolution": r["resolution"], "quality": r["quality"],
            "job_label": r["job_label"], "reason": r["reason"], "tier": r["tier"]}


async def describe_models() -> list[dict]:
    cat = await catalog()
    sem = asyncio.Semaphore(8)

    async def one(m):
        async with sem:
            raw, sure = await estimate(m["id"], "1K" if "1K" in m["resolutions"] else (m["resolutions"] or [""])[0],
                                       "high" if "high" in m["qualities"] else "", 0)
        can = []
        if m["vector"]:
            can.append("SVG vector output")
        if m["refs_max"]:
            can.append(f"edits/uses up to {m['refs_max']} picture{'s' if m['refs_max'] > 1 else ''}"
                       + (" (needs at least one style reference)" if m["refs_min"] else ""))
        if "transparent" in m["backgrounds"]:
            can.append("transparent background")
        if m["resolutions"]:
            can.append("sizes " + "/".join(m["resolutions"]))
        if m["qualities"]:
            can.append("quality " + "/".join(q for q in m["qualities"] if q != "auto"))
        if m["aspects"]:
            can.append(f"{len([a for a in m['aspects'] if a != 'auto'])} shapes")
        return {"id": m["id"], "name": m["name"], "can": can,
                "price": f"{'about ' if sure != 'exact' else ''}${raw:.3f} per picture", "about": m["about"]}
    return list(await asyncio.gather(*(one(m) for m in cat)))


# --------------------------------------------------------------------------- routes
@pictures_router.get("/video/pictures/models")
async def picture_models(request: Request):
    if (d := await va._deny(request)):
        return d
    return {"models": [{"id": m["id"], "name": m["name"], "vector": m["vector"]} for m in await catalog()]}


@pictures_router.post("/video/pictures/route")
async def picture_route(request: Request):
    if (d := await va._deny(request)):
        return d
    b = await va._body(request)
    try:
        return await route_picture(va._uid(request), str(b.get("prompt") or ""), aspect=str(b.get("aspect") or ""),
                                   has_src=bool(b.get("from_image")), refs=len(b.get("ref_images") or []),
                                   clear=bool(b.get("transparent")), tier=str(b.get("tier") or "best"),
                                   budget=float(b.get("budget") or 0), model=str(b.get("model") or ""))
    except ValueError as e:
        return va._err(str(e))


@pictures_router.get("/video/svg/{iid}")
async def picture_svg(iid: str):
    if not va.IMG_ID.match(iid):
        return va._err("Not found", 404)
    data = await va._get(f"{va.PREFIX}vectors/{iid}")
    if not data:
        return va._err("Not found", 404)
    return Response(data, media_type="image/svg+xml", headers={
        "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:; sandbox",
        "X-Content-Type-Options": "nosniff", "Cache-Control": "public, max-age=86400", "X-Robots-Tag": "noindex"})
