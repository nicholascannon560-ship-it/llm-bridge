"""Cutout shapes for projection mapping.

make_shape: a photo of the physical cutout (or a ready black/white mask) -> a clean mask picture
            (white = the cutout, black = everything else). A picture that is already two-tone is
            just cleaned up (free); a real photo is traced by a picture model, then thresholded.
fill_shape: mask + description -> a picture composed to fit the shape, routed to the models best
            at mask-guided editing, then the mask is burned in so nothing lands outside the cutout.
Masks are ordinary picture ids, so they plug straight into loop/stitch mask_image for video.
Shapes are listed per account in {PREFIX}shapes.json.
"""
from __future__ import annotations

import asyncio
import io
import json
import secrets
import time

from fastapi import APIRouter, Request

import video_app as va
import video_pictures as vpic

shapes_router = APIRouter()
SHAPES_KEY = f"{va.PREFIX}shapes.json"
_shapes: list | None = None

# Best first at filling a given outline (mask-guided editing), from the image-edit arenas.
FILL_MODELS = ["google/gemini-nano-banana-2.1", "openai/gpt-image-2.5-sunburst", "black-forest-labs/flux-3-image",
               "openai/gpt-image-2", "google/gemini-3-pro-image", "bytedance-seed/seedream-5-0-pro"]
TRACE_MODELS = ["google/gemini-nano-banana-2.1", "openai/gpt-image-2.5-sunburst", "openai/gpt-image-2"]
# Picture, video, loop frame and mask must share one shape for the mask to line up with the video,
# so fills use the shapes every video model and the loop renderer support; the mask is padded to match.
VIDEO_SHAPES = ["16:9", "9:16", "1:1"]
ASPECTS = {"1:1": 1.0, "5:4": 1.25, "4:3": 4 / 3, "3:2": 1.5, "16:9": 16 / 9, "21:9": 21 / 9,
           "4:5": 0.8, "3:4": 0.75, "2:3": 2 / 3, "9:16": 9 / 16}

TRACE_PROMPT = ("This is a photo of a flat cutout shape. Output a clean two-tone mask of it: the cutout's exact "
                "silhouette filled solid pure white (#FFFFFF), everything else solid pure black (#000000). "
                "No shading, texture, outline, text or background detail. Keep the same framing.")
FILL_PROMPT = ("The white area in this picture is the exact outline of a physical cutout that this picture will be "
               "projected onto. Paint the following inside the white area, filling it edge to edge and composed to "
               "fit the shape: {desc}. Everything outside the white area must stay solid pure black. Keep the "
               "outline exactly where it is.")


# --------------------------------------------------------------------------- image work (runs in a thread)
def _open(data: bytes):
    from PIL import Image, ImageOps
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data)))


def _two_tone(data: bytes) -> bool:
    """True if the picture is already a mask: nearly every pixel near black or near white."""
    im = _open(data).convert("L")
    im.thumbnail((256, 256))
    h = im.histogram()
    total = sum(h) or 1
    return (sum(h[:48]) + sum(h[208:])) / total >= 0.96 and sum(h[208:]) / total > 0.02


def _clean_mask(data: bytes) -> tuple[bytes, int, int]:
    """Threshold to pure black/white, smooth jagged edges, white = cutout. Returns PNG bytes, width, height."""
    from PIL import ImageFilter
    im = _open(data).convert("L")
    im = im.point(lambda v: 255 if v >= 128 else 0).filter(ImageFilter.MedianFilter(5))
    w, h = im.size
    border = [im.getpixel((x, 0)) for x in range(0, w, 8)] + [im.getpixel((x, h - 1)) for x in range(0, w, 8)] + \
             [im.getpixel((0, y)) for y in range(0, h, 8)] + [im.getpixel((w - 1, y)) for y in range(0, h, 8)]
    if sum(1 for v in border if v > 127) > len(border) / 2:  # shape drawn black on white: flip it
        im = im.point(lambda v: 255 - v)
    if not im.getbbox():
        raise ValueError("Couldn't find a shape in that picture. Photograph the cutout against a plain, "
                         "contrasting background, or upload a white-on-black mask.")
    out = io.BytesIO()
    im.save(out, "PNG", optimize=True)
    return out.getvalue(), w, h


def _burn_in(picture: bytes, mask: bytes) -> bytes:
    """Force everything outside the mask to pure black (mask scaled to the picture, soft 1px edge)."""
    from PIL import Image, ImageFilter
    pic = _open(picture).convert("RGB")
    m = _open(mask).convert("L").resize(pic.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.8))
    out = io.BytesIO()
    Image.composite(pic, Image.new("RGB", pic.size, (0, 0, 0)), m).save(out, "PNG", optimize=True)
    return out.getvalue()


def _pad_to(mask: bytes, aspect: str) -> bytes | None:
    """Letterbox the mask with black to the given shape so a model that can't do the mask's own shape
    doesn't stretch the outline. None if the mask is already within 3% of that shape."""
    from PIL import Image
    im = _open(mask).convert("L")
    w, h = im.size
    target = ASPECTS[aspect]
    if abs(w / h - target) / target <= 0.03:
        return None
    nw, nh = (round(h * target), h) if w / h < target else (w, round(w / target))
    canvas = Image.new("L", (nw, nh), 0)
    canvas.paste(im, ((nw - w) // 2, (nh - h) // 2))
    out = io.BytesIO()
    canvas.save(out, "PNG", optimize=True)
    return out.getvalue()


def _size(data: bytes) -> tuple[int, int]:
    return _open(data).size


def _closest_aspect(w: int, h: int, allowed: list[str] | None = None) -> str:
    r = w / max(h, 1)
    pool = [a for a in ASPECTS if not allowed or a in allowed] or list(ASPECTS)
    return min(pool, key=lambda a: abs(ASPECTS[a] - r) / ASPECTS[a])


# --------------------------------------------------------------------------- shape records
async def _all() -> list:
    global _shapes
    if _shapes is None:
        raw = await va._get(SHAPES_KEY)
        try:
            _shapes = json.loads(raw) if raw else []
        except Exception:
            _shapes = []
    return _shapes


async def _record(uid: str, mask_id: str, name: str, how: str, w: int, h: int) -> dict:
    rows = await _all()
    rec = {"mask_id": mask_id, "user": uid or "owner", "name": (name or "Cutout")[:80], "how": how,
           "w": w, "h": h, "aspect": _closest_aspect(w, h, VIDEO_SHAPES), "t": time.time(), "project": va._PROJECT.get()}
    rows.append(rec)
    await va._put(SHAPES_KEY, json.dumps(rows[-2000:]).encode(), "application/json")
    return rec


async def list_shapes(uid: str) -> list[dict]:
    return [{k: r[k] for k in ("mask_id", "name", "aspect", "how")} | {"url": f"/video/img/{r['mask_id']}"}
            for r in reversed(await _all()) if r["user"] == (uid or "owner")]


async def _mask_bytes(uid: str, mask_id: str) -> bytes:
    if not va.IMG_ID.match(str(mask_id)):
        raise ValueError("mask_id must be a picture id")
    data = await va._get(f"{va.PREFIX}images/{mask_id}")
    if not data:
        raise ValueError("That mask wasn't found")
    return data


# --------------------------------------------------------------------------- make + fill
async def make_shape(uid: str, image_id: str, name: str = "") -> dict:
    """Photo of the cutout (or a ready mask) -> clean white-on-black mask saved as a picture."""
    if not va.IMG_ID.match(str(image_id)):
        raise ValueError("image must be a picture id (upload the cutout photo first)")
    src = await va._get(f"{va.PREFIX}images/{image_id}")
    if not src:
        raise ValueError("That picture wasn't found")
    cost, how = 0.0, "already a mask: cleaned up"
    if not await asyncio.to_thread(_two_tone, src):
        last = None
        for mid in TRACE_MODELS:
            try:
                out = await vpic.make_picture(uid, TRACE_PROMPT, src_ids=[image_id], model=mid)
                break
            except ValueError as e:
                last = e
                if "credit" in str(e).lower():
                    raise
        else:
            raise ValueError(f"Couldn't trace the cutout: {last}")
        src = await va._get(f"{va.PREFIX}images/{out['id']}")
        cost, how = out["cost"], f"traced from the photo by {out['name']}"
        mask, w, h = await asyncio.to_thread(_clean_mask, src)
        await va._put(f"{va.PREFIX}images/{out['id']}", mask, "image/png")  # replace the raw trace with the clean mask
        mask_id = out["id"]
    else:
        mask, w, h = await asyncio.to_thread(_clean_mask, src)
        mask_id = secrets.token_urlsafe(24)
        await va._put(f"{va.PREFIX}images/{mask_id}", mask, "image/png")
        await va._index_image(mask_id, uid, "mask", name or "Cutout mask")
    rec = await _record(uid, mask_id, name, how, w, h)
    return {"mask_id": mask_id, "url": f"/video/img/{mask_id}", "aspect": rec["aspect"], "how": how,
            "cost": round(cost, 4), "name": rec["name"]}


async def fill_shape(uid: str, mask_id: str, prompt: str, *, tier: str = "best", budget: float = 0,
                     model: str = "") -> dict:
    """Fill the mask's shape with the description; outside the shape is guaranteed pure black."""
    desc = (prompt or "").strip()
    if not desc:
        raise ValueError("Describe what should fill the shape")
    mask = await _mask_bytes(uid, mask_id)
    w, h = await asyncio.to_thread(_size, mask)
    cat = {m["id"]: m for m in await vpic.catalog()}
    order = [model] if model and model.lower() != "auto" else FILL_MODELS
    if tier == "budget" and not model:
        order = ["bytedance-seed/seedream-5-0-flash", "google/gemini-3.1-flash-lite-image"] + order
    full = FILL_PROMPT.format(desc=desc[:3000])
    tried = []
    for mid in order:
        m = cat.get(mid)
        if not m:
            continue
        shapes = [a for a in VIDEO_SHAPES if a in m["aspects"]]
        if not shapes:
            tried.append(f"{m['name']}: can't set a video shape")
            continue
        aspect = _closest_aspect(w, h, shapes)
        try:
            await vpic.route_picture(uid, full, aspect=aspect, has_src=True,
                                     model=mid, tier=tier, budget=budget)
        except ValueError as e:
            tried.append(f"{m['name']}: {str(e)[:60]}")
            continue
        use_id, use_mask = mask_id, mask
        padded = await asyncio.to_thread(_pad_to, mask, aspect)
        if padded:  # keep the outline's true proportions inside the model's shape
            use_id = secrets.token_urlsafe(24)
            await va._put(f"{va.PREFIX}images/{use_id}", padded, "image/png")
            await va._index_image(use_id, uid, "mask", f"Cutout mask padded to {aspect}")
            use_mask = padded
        out = await vpic.make_picture(uid, full, aspect=aspect, src_ids=[use_id],
                                      model=mid, tier=tier, budget=budget)
        pic = await va._get(f"{va.PREFIX}images/{out['id']}")
        await va._put(f"{va.PREFIX}images/{out['id']}", await asyncio.to_thread(_burn_in, pic, use_mask), "image/png")
        return {"image_id": out["id"], "url": out["url"], "cost": out["cost"], "made_with": out["name"],
                "mask_id": use_id, "aspect": aspect, "padded": bool(padded),
                "next": (f"For a looping projection clip: make_video with aspect_ratio {aspect} and start_image and "
                         f"end_image both set to this image_id, then loop that video with shape {aspect}, mode "
                         "pingpong and mask_image set to this mask_id (same shape everywhere keeps the edge exact).")}
    raise ValueError("No picture model could fill this shape with these settings. " + "; ".join(tried[:5]))


# --------------------------------------------------------------------------- app routes
@shapes_router.get("/video/shapes")
async def shapes_get(request: Request):
    if (d := await va._deny(request)):
        return d
    return {"shapes": await list_shapes(va._uid(request))}


@shapes_router.post("/video/shapes/mask")
async def shapes_mask(request: Request):
    if (d := await va._deny(request)):
        return d
    b = await va._body(request)
    try:
        return await make_shape(va._uid(request), str(b.get("image") or ""), str(b.get("name") or ""))
    except ValueError as e:
        return va._err(str(e), 402 if "credit" in str(e).lower() else 400)


@shapes_router.post("/video/shapes/fill")
async def shapes_fill(request: Request):
    """{"prompt", "mask_id"? | "image"? (a cutout photo; a mask is made first), "tier"?, "model"?}"""
    if (d := await va._deny(request)):
        return d
    b = await va._body(request)
    uid = va._uid(request)
    try:
        mask_id, made = str(b.get("mask_id") or ""), None
        if not mask_id:
            made = await make_shape(uid, str(b.get("image") or ""))
            mask_id = made["mask_id"]
        out = await fill_shape(uid, mask_id, str(b.get("prompt") or ""), tier=str(b.get("tier") or "best"),
                               model=str(b.get("model") or ""))
        if made:
            out["cost"] = round(out["cost"] + made["cost"], 4)
            out["mask_how"] = made["how"]
        return out
    except ValueError as e:
        return va._err(str(e), 402 if "credit" in str(e).lower() else 400)
