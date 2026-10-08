"""Video Studio: AI video generation through OpenRouter, served by the bridge.

GET  /video                     phone app (HTML shell, holds no data)
GET  /video/session             {"authed","configured","daily_cap","spent_today"}
POST /video/login               {"pin"} -> signed HttpOnly cookie (path /video)
GET  /video/models              OpenRouter video models with durations, resolutions, prices
POST /video/prompt              {"idea"} -> {"prompt"} (GLM rewrites a rough idea into a shot description)
POST /video/estimate            {"model","duration","resolution","audio"} -> {"cost"}
POST /video/generate            {"model","prompt","duration","resolution","aspect_ratio",
                                 "audio","budget","song_id"?,"song_start"?} -> job
GET  /video/jobs                recent jobs (newest first); resumes polling for unfinished ones
GET  /video/jobs/{id}           one job
POST /video/jobs/{id}/song      {"song_id","song_start"} -> re-score a finished clip, no new generation
GET  /video/clip/{id}           the mp4 (Range supported for iOS); ?song=1 for the scored version, ?dl=1 to download
POST /video/songs               {"name","data"(base64)} upload a song (mp3/m4a/wav, <= 15 MB)
GET  /video/songs               uploaded songs
POST /video/images              {"data"(base64 jpeg/png/webp)} upload a picture prompt -> {"id","url"}
GET  /video/img/{id}            the picture, PUBLIC on purpose: the video provider has to fetch it.
                                The id is 24 random bytes, so it can't be guessed or listed.

POST /video/jobs/{id}/edit       {"mode":"edit"|"remake","change", model/duration/resolution/aspect/budget/audio,
                                 "keep_look"?, "song_id"?} -> a NEW job; the original is never touched
                                 edit   = video-to-video: the finished clip goes in as a video reference
                                          (only models that accept video input; OpenRouter says if not)
                                 remake = GLM folds the change into the old prompt, reuses the old
                                          pictures, and (keep_look) starts on the old clip's first frame
POST /video/jobs/{id}/extend     {"seconds","budget","change"?} -> a NEW job that continues from the clip's
                                 last frame and is joined onto the end (the original stays)
     generate also takes "parts": N > 1 builds a long video automatically, one part
     after another, each starting on the last frame of the one before, joined as it goes.
GET  /video/src/{token}         the original clip, PUBLIC on purpose for video-to-video (random token)

Pictures: generate takes "images": [{"id","role"}], role = first (start frame),
last (end frame) or ref (style/content reference). Frames win over refs when
both are sent, so refs are dropped with a warning in that case.

Money guards, both enforced server-side:
  - per-video budget: every request carries "budget"; the job is refused if the
    estimate from OpenRouter's listed price exceeds it. Unknown price = refused.
  - optional daily cap: only if VIDEO_DAILY_USD is set above 0 (off by default),
    over a rolling 24h, using actual cost once known and the estimate until then.

Auth: VIDEO_PIN, falling back to CART_PIN. Never the bridge key.
Storage: S3 (AWS_S3_BUCKET, prefix video/) with a /tmp fallback.
Songs are muxed with the static ffmpeg from imageio-ffmpeg.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

video_router = APIRouter()

VIDEO_APP_VERSION = "2.10.0"  # bump on HTML-only changes so the *.py watch pattern deploys
COOKIE = "video_session"
SESSION_DAYS = 60
OR_BASE = "https://openrouter.ai/api/v1"
PROMPT_MODEL = os.getenv("VIDEO_PROMPT_MODEL") or "z-ai/glm-5.3-flash"
# Each deployment keeps its own data: the bridge uses video/, the Video Studio service studio/.
PREFIX = (os.getenv("VIDEO_S3_PREFIX") or "video/").strip("/") + "/"
JOBS_KEY = f"{PREFIX}jobs.json"
SONGS_KEY = f"{PREFIX}songs.json"
USERS_KEY = f"{PREFIX}users.json"
LOCAL_DIR = Path(os.getenv("VIDEO_DATA_DIR") or "/tmp") / PREFIX.strip("/")
MAX_SONG_BYTES = 15 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_REFS = 4
IMG_ID = re.compile(r"^[A-Za-z0-9_-]{30,40}$")
IMG_TYPES = {b"\xff\xd8\xff": ("jpg", "image/jpeg"), b"\x89PNG": ("png", "image/png"), b"RIFF": ("webp", "image/webp")}
POLL_EVERY = 15
POLL_GIVE_UP = 30 * 60
_HTML_PATH = Path(__file__).with_name("video_app.html")
RES_TOKENS = ("480p", "720p", "768p", "1080p", "1k", "2k", "4k")
SONG_EXT = {"mp3": "audio/mpeg", "m4a": "audio/mp4", "aac": "audio/aac", "wav": "audio/wav",
            "ogg": "audio/ogg"}

_fails: dict[str, list[float]] = {}
_tasks: dict[str, asyncio.Task] = {}
_jobs: list[dict] | None = None
_songs: list[dict] | None = None
_users: dict | None = None  # {"users": [...], "events": [stripe event ids already handled]}
_codes: dict[str, dict] = {}  # email -> {"hash", "exp", "tries"}
_lock = asyncio.Lock()
_models_cache: dict = {"t": 0.0, "data": []}


def _daily_cap() -> float | None:
    """None = no daily cap (the default). Set VIDEO_DAILY_USD > 0 to turn one on."""
    try:
        v = float(os.getenv("VIDEO_DAILY_USD") or "0")
    except ValueError:
        return None
    return v if v > 0 else None


# --------------------------------------------------------------------------- auth
def _pin() -> str:
    return (os.getenv("VIDEO_PIN") or os.getenv("CART_PIN") or "").strip()


def _secret() -> bytes:
    base = os.getenv("UI_SESSION_SECRET") or "video"
    return hashlib.sha256(f"video-session|{base}|{_pin()}".encode()).digest()


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


USER_COOKIE = "studio_user"
OWNER = {"id": "owner", "owner": True, "email": None}


def _user_sign(uid: str, exp: int) -> str:
    mac = hmac.new(_secret(), f"user|{uid}|{exp}".encode(), hashlib.sha256).hexdigest()
    return f"{uid}.{exp}.{mac}"


async def _current_user(request: Request) -> dict | None:
    if _authed(request):
        return OWNER
    tok = request.cookies.get(USER_COOKIE) or ""
    parts = tok.split(".")
    if len(parts) != 3 or not parts[1].isdigit() or int(parts[1]) < time.time():
        return None
    if not hmac.compare_digest(_user_sign(parts[0], int(parts[1])), tok):
        return None
    u = await _user_by_id(parts[0])
    return u if u and not u.get("blocked") else None


async def _deny(request: Request):
    """Resolve the signed-in person onto request.state.user, or return a 401."""
    u = await _current_user(request)
    if not u:
        return JSONResponse({"detail": "Sign in again"}, status_code=401)
    request.state.user = u
    return None


def _uid(request: Request) -> str:
    return request.state.user["id"]


def _owns(request_or_uid, obj: dict | None) -> bool:
    uid = request_or_uid if isinstance(request_or_uid, str) else _uid(request_or_uid)
    return bool(obj) and (obj.get("user") or "owner") == uid


def _mine(request: Request, rows: list[dict], rid: str) -> dict | None:
    o = _find(rows, rid)
    return o if _owns(request, o) else None


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "?")


async def _body(request: Request) -> dict:
    try:
        b = await request.json()
        return b if isinstance(b, dict) else {}
    except Exception:
        return {}


def _err(msg: str, code: int = 400) -> JSONResponse:
    return JSONResponse({"detail": msg}, status_code=code)


# --------------------------------------------------------------------------- storage
def _s3():
    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        return None, None
    try:
        import boto3  # noqa: WPS433
        return bucket, boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION") or "us-east-1")
    except Exception as e:
        print(f"[video] s3 unavailable: {e!r}", flush=True)
        return None, None


def _local(key: str) -> Path:
    return LOCAL_DIR / key.replace("/", "__")


def _put_sync(key: str, data: bytes, ctype: str) -> None:
    try:
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        _local(key).write_bytes(data)
    except Exception as e:
        print(f"[video] local save failed {key}: {e!r}", flush=True)
    bucket, s3 = _s3()
    if s3:
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=ctype)
        except Exception as e:
            print(f"[video] s3 save failed {key}: {e!r}", flush=True)


def _get_sync(key: str) -> bytes | None:
    p = _local(key)
    if p.exists():
        try:
            return p.read_bytes()
        except Exception:
            pass
    bucket, s3 = _s3()
    if s3:
        try:
            data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            try:
                LOCAL_DIR.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            except Exception:
                pass
            return data
        except Exception as e:
            if "NoSuchKey" not in repr(e):
                print(f"[video] s3 load failed {key}: {e!r}", flush=True)
    return None


def _delete_sync(key: str) -> bool:
    """Remove a stored file everywhere. True if S3 accepted the delete (or there's no S3)."""
    try:
        _local(key).unlink(missing_ok=True)
    except Exception:
        pass
    bucket, s3 = _s3()
    if not s3:
        return True
    try:
        s3.delete_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:
        print(f"[video] s3 delete failed {key}: {e!r}", flush=True)
        return False


async def _delete(*keys: str | None) -> None:
    for k in keys:
        if k:
            await asyncio.to_thread(_delete_sync, k)


async def _put(key: str, data: bytes, ctype: str) -> None:
    await asyncio.to_thread(_put_sync, key, data, ctype)


async def _get(key: str) -> bytes | None:
    return await asyncio.to_thread(_get_sync, key)


async def _load_list(key: str) -> list[dict]:
    raw = await _get(key)
    if not raw:
        return []
    try:
        rows = json.loads(raw)
        return rows if isinstance(rows, list) else []
    except Exception:
        return []


async def _all_jobs() -> list[dict]:
    global _jobs
    if _jobs is None:
        async with _lock:
            if _jobs is None:
                _jobs = await _load_list(JOBS_KEY)
    return _jobs


async def _all_songs() -> list[dict]:
    global _songs
    if _songs is None:
        async with _lock:
            if _songs is None:
                _songs = await _load_list(SONGS_KEY)
    return _songs


async def _save_jobs() -> None:
    rows = await _all_jobs()
    async with _lock:
        snapshot = json.dumps(rows[-300:], separators=(",", ":")).encode()
    await _put(JOBS_KEY, snapshot, "application/json")


async def _save_songs() -> None:
    rows = await _all_songs()
    async with _lock:
        snapshot = json.dumps(rows, separators=(",", ":")).encode()
    await _put(SONGS_KEY, snapshot, "application/json")


def _find(rows: list[dict], rid: str) -> dict | None:
    return next((r for r in rows if r.get("id") == rid), None)


# --------------------------------------------------------------------------- pricing
RES_SIDE = {"480p": 480, "720p": 720, "768p": 768, "1080p": 1080, "1k": 1080, "2k": 1440, "4k": 2160}
MODES = ("text", "image", "ref", "video")


def _sku_tags(key: str) -> dict | None:
    """Read one of OpenRouter's pricing SKU names. Seen in the wild (Oct 2026):
    duration_seconds_768p, reference_duration_seconds_768p, cents_per_second_output,
    cents_per_video_output_second_720p, cents_per_image_input, duration_seconds_with_audio,
    text_to_video_duration_seconds_720p, image_to_video_..., video_tokens_without_audio,
    video_tokens_with_video_input, minimum_cents_per_generation, reference_images."""
    k = key.lower()
    t = {"res": next((r for r in RES_SIDE if re.search(rf"(^|[_-]){r}($|[_-])", k)), None),
         "cents": "cent" in k, "audio": None, "mode": None, "kind": None}
    if "without_audio" in k or "no_audio" in k or "silent" in k:
        t["audio"] = False
    elif "audio" in k:
        t["audio"] = True
    if "reference" in k or "video_input" in k:
        t["mode"] = "refvid"
    elif "image_to_video" in k:
        t["mode"] = "image"
    elif "text_to_video" in k:
        t["mode"] = "text"
    if "continuation" in k or "megapixel" in k or "upscale" in k:
        t["kind"] = "skip"
    elif "minimum" in k:
        t["kind"] = "min"
    elif "reference_images" in k or ("image" in k and "input" in k):
        t["kind"] = "per_image"
    elif "token" in k:
        t["kind"] = "token"
    elif "second" in k:
        t["kind"] = "per_sec"
    elif "video" in k or "generation" in k:
        t["kind"] = "flat"
    else:
        return None
    return t


def _tokens_per_second(res: str) -> float:
    side = RES_SIDE.get((res or "720p").lower(), 720)
    return side * side * 16 / 9 * 24 / 1024  # Seedance-style: w*h*fps/1024, assumes 16:9-sized frames


def price_terms(model: dict, resolution: str, audio: bool, mode: str) -> dict | None:
    """-> {"ps": $/second, "min": $ per video floor, "img": $ per input picture, "basis": str} or None."""
    skus = []
    for k, v in (model.get("pricing_skus") or {}).items():
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        t = _sku_tags(str(k))
        if not t or t["kind"] == "skip":
            continue
        kl = str(k).lower()
        scale = (0.01 if t["cents"] else 1) / (1e6 if "million" in kl else 1e3 if "thousand" in kl else 1)
        t.update(key=str(k), val=val * scale)
        skus.append(t)
    res = (resolution or "").lower()
    mclass = "refvid" if mode in ("ref", "video") else mode

    def respick(lst):
        exact = [x for x in lst if x["res"] == res]
        return exact or [x for x in lst if x["res"] is None]

    def rate(kind):
        comp = [x for x in skus if x["kind"] == kind and x["audio"] in (None, audio)
                and x["mode"] in (None, mclass)]
        base = respick([x for x in comp if x["audio"] is None and x["mode"] != "refvid"])
        if mclass == "refvid":
            rv = respick([x for x in comp if x["mode"] == "refvid"])
            if rv:
                return max(x["val"] for x in rv), rv
        aud = respick([x for x in comp if x["audio"] is not None and x["mode"] != "refvid"])
        used = base + aud
        if not used:
            return None, []
        if aud and not audio:  # an explicit "without audio" price replaces the base one
            return max(x["val"] for x in aud), aud
        return max(x["val"] for x in used), used

    ps, used = rate("per_sec")
    if ps is None:
        tok, used = rate("token")
        if tok is not None:
            ps = tok * _tokens_per_second(res)
    flat, fused = rate("flat")
    if ps is None and flat is None:
        return None
    mins = [x["val"] for x in skus if x["kind"] == "min"]
    imgs = [x["val"] for x in skus if x["kind"] == "per_image"]
    basis = ", ".join(sorted({x["key"] for x in used + fused}))
    return {"ps": ps or 0.0, "flat": flat or 0.0, "min": max(mins) if mins else 0.0,
            "img": max(imgs) if imgs else 0.0, "basis": basis}


def estimate_cost(model: dict, duration: float, resolution: str, audio: bool,
                  mode: str = "text", n_images: int = 0) -> tuple[float | None, str]:
    t = price_terms(model, resolution, audio, mode)
    if t is None:
        return None, f"unrecognised pricing: {', '.join(sorted((model.get('pricing_skus') or {})))}"
    cost = max(t["ps"] * float(duration) + t["flat"], t["min"]) + t["img"] * n_images
    return round(cost, 4), t["basis"]


def price_table(model: dict) -> dict:
    """Every resolution x audio x mode combination, so the page can price options instantly
    with the same rules the server enforces."""
    out = {}
    for res in (model.get("resolutions") or [""]):
        for audio in (True, False):
            for mode in MODES:
                t = price_terms(model, res, audio, mode)
                out[f"{res}|{int(audio)}|{mode}"] = None if t is None else \
                    {k: round(t[k], 6) for k in ("ps", "flat", "min", "img")}
    return out


def _public_base(request: Request) -> str:
    dom = os.getenv("VIDEO_PUBLIC_DOMAIN") or os.getenv("RAILWAY_PUBLIC_DOMAIN")
    if dom:
        return f"https://{dom.strip().rstrip('/')}"
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    return f"https://{host}"


def _sniff(data: bytes) -> tuple[str, str] | None:
    for magic, kind in IMG_TYPES.items():
        if data.startswith(magic):
            if magic == b"RIFF" and data[8:12] != b"WEBP":
                return None
            return kind
    return None


USAGE_MARKUP = max(0.0, float(os.getenv("VIDEO_USAGE_MARKUP") or "0.015"))


def _bill(uid: str, cost: float) -> float:
    """What a customer pays for usage: OpenRouter's cost plus the margin (covers the 1.5%
    instant-payout fee). The owner always pays plain cost."""
    return round(float(cost), 4) if uid == "owner" else round(float(cost) * (1 + USAGE_MARKUP), 4)


def _job_cost(j: dict) -> float:
    if j.get("actual_cost") is not None:
        return float(j["actual_cost"])
    if j.get("status") in ("failed", "cancelled", "expired", "refused"):
        return 0.0
    return float(j.get("estimate") or 0)


async def _spent_24h() -> float:
    cutoff = time.time() - 86400
    return round(sum(_job_cost(j) for j in await _all_jobs() if j.get("created", 0) >= cutoff), 4)


# --------------------------------------------------------------------------- openrouter
def _or_headers() -> dict:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    return {"Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://kalshiml-production-b2e9.up.railway.app",
            "X-Title": "Video Studio"}


async def _models(force: bool = False) -> list[dict]:
    if not force and _models_cache["data"] and time.time() - _models_cache["t"] < 600:
        return _models_cache["data"]
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{OR_BASE}/videos/models", headers=_or_headers())
    r.raise_for_status()
    out = []
    for m in (r.json().get("data") or []):
        if not m.get("id"):
            continue
        out.append({
            "id": m["id"],
            "name": m.get("name") or m["id"],
            "durations": [d for d in (m.get("supported_durations") or []) if isinstance(d, (int, float))],
            "resolutions": list(m.get("supported_resolutions") or []),
            "aspect_ratios": list(m.get("supported_aspect_ratios") or []),
            "pricing_skus": m.get("pricing_skus") or {},
            "passthrough": list(m.get("allowed_passthrough_parameters") or []),
        })
        out[-1]["prices"] = price_table(out[-1])
    _models_cache.update(t=time.time(), data=out)
    return out


async def _model(mid: str) -> dict | None:
    return next((m for m in await _models() if m["id"] == mid), None)


async def _or_chat(prompt: str) -> str:
    payload = {"model": PROMPT_MODEL, "max_tokens": 700, "temperature": 0.7,
               "messages": [{"role": "user", "content": prompt}]}
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{OR_BASE}/chat/completions", json=payload, headers=_or_headers())
    data = r.json()
    if r.status_code >= 400 or data.get("error"):
        raise RuntimeError((data.get("error") or {}).get("message") or f"HTTP {r.status_code}")
    msg = ((data.get("choices") or [{}])[0].get("message") or {})
    text = (msg.get("content") or "").strip() or (msg.get("reasoning") or "").strip()
    if not text:
        raise RuntimeError("prompt model returned nothing")
    return text


PROMPT_REWRITE = """Rewrite this rough idea as a single shot description for an AI video model.
Say what is on screen, the motion, the camera move, the lighting and the mood, in 2-4 plain sentences.
Video models reward concrete visual detail. Do not add on-screen text, logos or real people's names.
Reply with only the description, no preamble and no quotes.

Idea: {idea}"""



REMAKE_PROMPT = """You edit prompts for an AI video model.
Here is the prompt that made the current video, and a change the user wants.
Write the new prompt: keep everything from the old prompt that the change doesn't touch,
apply the change, and stay concrete about subject, motion, camera, light and mood.
Reply with only the new prompt, no preamble and no quotes.

Old prompt: {old}

Change: {change}"""


# --------------------------------------------------------------------------- ffmpeg
def _ffmpeg() -> str:
    import imageio_ffmpeg  # noqa: WPS433
    return imageio_ffmpeg.get_ffmpeg_exe()


def _probe_duration(path: str) -> float | None:
    p = subprocess.run([_ffmpeg(), "-hide_banner", "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", p.stderr)
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def mux_song_sync(video: bytes, song: bytes, song_ext: str, start: float) -> bytes:
    """Lay the song over the clip: starts `start` seconds into the song, stops with
    the video, fades out over the last second. The video stream is copied, not re-encoded."""
    with tempfile.TemporaryDirectory() as d:
        vp, sp, op = f"{d}/v.mp4", f"{d}/s.{song_ext}", f"{d}/o.mp4"
        Path(vp).write_bytes(video)
        Path(sp).write_bytes(song)
        dur = _probe_duration(vp) or 0
        fade = f"afade=t=out:st={max(0.0, dur - 1.0):.2f}:d=1" if dur > 2 else "anull"
        cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
               "-i", vp, "-ss", f"{max(0.0, start):.2f}", "-i", sp,
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
               "-af", fade, "-c:a", "aac", "-b:a", "192k", "-shortest",
               "-movflags", "+faststart", op]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if p.returncode != 0 or not Path(op).exists():
            raise RuntimeError(f"ffmpeg failed: {(p.stderr or '').strip()[-400:]}")
        return Path(op).read_bytes()


def first_frame_sync(video: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        vp, op = f"{d}/v.mp4", f"{d}/f.jpg"
        Path(vp).write_bytes(video)
        p = subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", vp,
                            "-frames:v", "1", "-q:v", "2", op], capture_output=True, text=True, timeout=60)
        if p.returncode != 0 or not Path(op).exists():
            raise RuntimeError(f"couldn't grab the first frame: {(p.stderr or '').strip()[-300:]}")
        return Path(op).read_bytes()


def last_frame_sync(video: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        vp, op = f"{d}/v.mp4", f"{d}/f.jpg"
        Path(vp).write_bytes(video)
        p = subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-sseof", "-0.25", "-i", vp,
                            "-update", "1", "-q:v", "2", op], capture_output=True, text=True, timeout=60)
        if p.returncode != 0 or not Path(op).exists():
            raise RuntimeError(f"couldn't grab the last frame: {(p.stderr or '').strip()[-300:]}")
        return Path(op).read_bytes()


def _probe(path: str) -> tuple[int, int, bool]:
    p = subprocess.run([_ffmpeg(), "-hide_banner", "-i", path], capture_output=True, text=True)
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", p.stderr)
    w, h = (int(m.group(1)), int(m.group(2))) if m else (1280, 720)
    return w - w % 2, h - h % 2, "Audio:" in p.stderr


def concat_sync(first: bytes, second: bytes) -> bytes:
    """Join two clips end to end. Re-encodes so different sizes/frame rates still line up;
    the second is scaled to the first's frame. Keeps sound only if both clips have it."""
    with tempfile.TemporaryDirectory() as d:
        a, b, o = f"{d}/a.mp4", f"{d}/b.mp4", f"{d}/o.mp4"
        Path(a).write_bytes(first)
        Path(b).write_bytes(second)
        w, h, a_snd = _probe(a)
        _, _, b_snd = _probe(b)
        sound = a_snd and b_snd
        vf = (f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v0];"
              f"[1:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v1];")
        if sound:
            graph = vf + "[0:a]aresample=44100[a0];[1:a]aresample=44100[a1];[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
            maps = ["-map", "[v]", "-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
        else:
            graph = vf + "[v0][v1]concat=n=2:v=1:a=0[v]"
            maps = ["-map", "[v]"]
        cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", a, "-i", b,
               "-filter_complex", graph, *maps, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", o]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if p.returncode != 0 or not Path(o).exists():
            raise RuntimeError(f"couldn't join the clips: {(p.stderr or '').strip()[-400:]}")
        return Path(o).read_bytes()


async def _apply_song(job: dict, song_id: str, start: float) -> None:
    song = _find(await _all_songs(), song_id)
    if not song:
        raise RuntimeError("song not found")
    video = await _get(job["clip_key"])
    audio = await _get(song["key"])
    if not video or not audio:
        raise RuntimeError("clip or song file is missing from storage")
    out = await asyncio.to_thread(mux_song_sync, video, audio, song["ext"], start)
    key = f"{PREFIX}clips/{job['id']}-song.mp4"
    await _put(key, out, "video/mp4")
    job.update(scored_key=key, song_id=song_id, song_name=song["name"], song_start=start, song_error=None)


# --------------------------------------------------------------------------- job runner
async def _poll(job: dict) -> None:
    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
            while time.time() - started < POLL_GIVE_UP:
                r = await c.get(job["polling_url"], headers=_or_headers())
                if r.status_code >= 400:
                    raise RuntimeError(f"poll HTTP {r.status_code}: {r.text[:200]}")
                st = r.json()
                status = st.get("status") or "pending"
                cost = (st.get("usage") or {}).get("cost")
                if cost is not None:
                    job["actual_cost"] = float(cost)
                # "completed" is only shown once the file is safely stored, so a
                # restart mid-download leaves the job resumable instead of stuck.
                shown = "in_progress" if status == "completed" else status
                if shown != job.get("status"):
                    job["status"] = shown
                    await _save_jobs()
                if status == "completed":
                    url = (st.get("unsigned_urls") or [f"{OR_BASE}/videos/{job['or_id']}/content?index=0"])[0]
                    v = await c.get(url, headers=_or_headers(), timeout=180)
                    if v.status_code >= 400 or not v.content:
                        raise RuntimeError(f"download HTTP {v.status_code}")
                    key = f"{PREFIX}clips/{job['id']}.mp4"
                    src = _find(await _all_jobs(), job["extend_of"]) if job.get("extend_of") else None
                    if src and src.get("clip_key"):
                        await _put(f"{PREFIX}clips/{job['id']}-part.mp4", v.content, "video/mp4")
                        before = await _get(src["clip_key"])
                        joined = await asyncio.to_thread(concat_sync, before or b"", v.content)
                        await _put(key, joined, "video/mp4")
                        job["total_seconds"] = (src.get("total_seconds") or src.get("duration") or 0) + job["duration"]
                    else:
                        await _put(key, v.content, "video/mp4")
                        job["total_seconds"] = job["duration"]
                    job["clip_key"] = key
                    if job.get("song_id") and not job.get("chain_remaining"):
                        try:
                            await _apply_song(job, job["song_id"], float(job.get("song_start") or 0))
                        except Exception as e:
                            job["song_error"] = str(e)[:300]
                    job.update(status="ready", finished=time.time())
                    await _settle(job, job.get("actual_cost"))
                    await _save_jobs()
                    if (job.get("parts_total") or 1) > 1 and not job.get("chain_remaining"):
                        await _drop_chain_parts(job)
                    if job.get("chain_remaining"):
                        try:
                            await _extend(job, auto=True)
                        except Exception as e:
                            job["chain_error"] = f"Stopped at {job['total_seconds']}s: {e}"[:300]
                            job["chain_remaining"] = 0
                            await _save_jobs()
                    return
                if status in ("failed", "cancelled", "expired"):
                    job["error"] = str(st.get("error") or status)[:300]
                    job["finished"] = time.time()
                    await _settle(job, 0.0)
                    await _unhide_parent(job)
                    await _save_jobs()
                    return
                await asyncio.sleep(POLL_EVERY)
        job.update(status="failed", error="gave up waiting after 30 minutes", finished=time.time())
        await _settle(job, 0.0)
        await _unhide_parent(job)
        await _save_jobs()
    except Exception as e:
        print(f"[video] poll {job.get('id')} error: {e!r}", flush=True)
        job["poll_error"] = str(e)[:300]
        await _save_jobs()


async def _unhide_parent(job: dict) -> None:
    """If an automatic part fails, bring back the video built so far instead of losing it."""
    if job.get("extend_of") and (job.get("part_no") or 1) > 1:
        src = _find(await _all_jobs(), job["extend_of"])
        if src and src.get("superseded"):
            src["superseded"] = False
            src["chain_remaining"] = 0
            src["chain_error"] = f"Stopped at {src.get('total_seconds') or src.get('duration')}s: the next part failed."


def _ensure_poller(job: dict) -> None:
    if job.get("status") not in ("pending", "in_progress", "completed") or not job.get("polling_url"):
        return
    t = _tasks.get(job["id"])
    if t and not t.done():
        return
    _tasks[job["id"]] = asyncio.create_task(_poll(job))


def _public(job: dict) -> dict:
    keep = ("id", "created", "finished", "model", "prompt", "duration", "resolution", "aspect_ratio",
            "audio", "budget", "estimate", "actual_cost", "status", "error", "poll_error",
            "song_id", "song_name", "song_start", "song_error", "images", "image_note",
            "parent_id", "edit_mode", "change", "total_seconds", "parts_total", "part_no",
            "chain_remaining", "chain_error", "chain_estimate", "charged", "options_note", "uploaded")
    out = {k: job.get(k) for k in keep}
    out["has_clip"] = bool(job.get("clip_key"))
    out["has_scored"] = bool(job.get("scored_key"))
    return out


# --------------------------------------------------------------------------- routes
@video_router.get("/video", response_class=HTMLResponse)
async def video_page():
    try:
        html = _HTML_PATH.read_text(encoding="utf-8")
    except Exception:
        return HTMLResponse("video_app.html missing", status_code=500)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@video_router.get("/video/session")
async def video_session(request: Request):
    _ensure_sweeper()
    u = await _current_user(request)
    out = {"authed": bool(u), "configured": True, "version": VIDEO_APP_VERSION, "retain_days": RETAIN_DAYS,
           "markup": 0 if (not u or u.get("owner")) else USAGE_MARKUP,
           "accounts": _accounts_on(), "email_codes": _email_codes_on(), "billing": _billing_ready()}
    if u:
        out.update(daily_cap=_daily_cap(), spent_today=await _spent_24h(), account=_account(u))
    return out


@video_router.post("/video/login")
async def video_login(request: Request):
    if not _pin():
        return _err("VIDEO_PIN (or CART_PIN) is not set on the server", 503)
    ip, now = _client_ip(request), time.time()
    recent = [t for t in _fails.get(ip, []) if now - t < 900]
    _fails[ip] = recent
    if len(recent) >= 8:
        return _err("Too many tries. Wait 15 minutes.", 429)
    supplied = str((await _body(request)).get("pin", "")).strip()
    if not supplied or not hmac.compare_digest(supplied, _pin()):
        recent.append(now)
        return _err("Wrong PIN", 401)
    _fails.pop(ip, None)
    exp = int(now + SESSION_DAYS * 86400)
    resp = JSONResponse({"ok": True})
    resp.set_cookie(COOKIE, _sign(exp), max_age=SESSION_DAYS * 86400,
                    httponly=True, secure=True, samesite="strict", path="/video")
    return resp


@video_router.get("/video/models")
async def video_models(request: Request):
    if (d := await _deny(request)):
        return d
    try:
        models = await _models(force=request.query_params.get("refresh") == "1")
    except Exception as e:
        return _err(f"Could not load models from OpenRouter: {e}", 502)
    return {"models": models}


@video_router.post("/video/prompt")
async def video_prompt(request: Request):
    if (d := await _deny(request)):
        return d
    idea = str((await _body(request)).get("idea", "")).strip()[:1500]
    if not idea:
        return _err("Type an idea first")
    try:
        text = await _or_chat(PROMPT_REWRITE.format(idea=idea))
    except Exception as e:
        return _err(f"Prompt rewrite failed: {e}", 502)
    return {"prompt": text.strip().strip('"')[:2000]}


@video_router.post("/video/estimate")
async def video_estimate(request: Request):
    if (d := await _deny(request)):
        return d
    b = await _body(request)
    m = await _model(str(b.get("model", "")))
    if not m:
        return _err("Unknown model")
    cost, basis = estimate_cost(m, float(b.get("duration") or 0), str(b.get("resolution") or ""),
                                bool(b.get("audio", True)), str(b.get("mode") or "text"), int(b.get("n_images") or 0))
    return {"cost": cost, "basis": basis}



@video_router.post("/video/generate")
async def video_generate(request: Request):
    if (d := await _deny(request)):
        return d
    return await _start_job(request, await _body(request))


async def _start_job(request: Request, b: dict, extra: dict | None = None):
    extra = extra or {}
    prompt = str(b.get("prompt", "")).strip()[:2500]
    if not prompt:
        return _err("Write a prompt first")
    try:
        m = await _model(str(b.get("model", "")))
    except Exception as e:
        return _err(f"Could not load models from OpenRouter: {e}", 502)
    if not m:
        return _err("Unknown model. Refresh the model list.")
    try:
        duration = int(float(b.get("duration") or 0))
        budget = round(float(b.get("budget") or 0), 2)
    except (TypeError, ValueError):
        return _err("Duration and budget must be numbers")
    resolution = str(b.get("resolution") or "")
    aspect = str(b.get("aspect_ratio") or "")
    song_id = str(b.get("song_id") or "") or None
    audio = bool(b.get("audio", True)) and not song_id
    if m["durations"] and duration not in m["durations"] and not extra.get("no_duration"):
        return _err(f"{m['name']} supports {m['durations']} seconds")
    if duration <= 0:
        return _err("Pick a duration")
    if m["resolutions"] and resolution not in m["resolutions"]:
        return _err(f"{m['name']} supports {', '.join(m['resolutions'])}")
    if m["aspect_ratios"] and aspect and aspect not in m["aspect_ratios"]:
        return _err(f"{m['name']} supports {', '.join(m['aspect_ratios'])}")
    if budget <= 0:
        return _err("Set a budget for this video")
    uid = extra.get("user") or (request.state.user["id"] if request is not None else "owner")
    if song_id and not _owns(uid, _find(await _all_songs(), song_id)):
        return _err("That song is gone. Pick another.")

    images, frames, refs, image_note = [], [], [], None
    raw_imgs = b.get("images") or []
    if not isinstance(raw_imgs, list):
        return _err("Pictures came through wrong. Re-add them.")
    base = extra.get("base") or _public_base(request)
    for it in raw_imgs[:8]:
        iid, role = str((it or {}).get("id", "")), str((it or {}).get("role", ""))
        if not IMG_ID.match(iid) or role not in ("first", "last", "ref"):
            return _err("Pictures came through wrong. Re-add them.")
        url = f"{base}/video/img/{iid}"
        if role in ("first", "last"):
            if any(f["frame_type"] == f"{role}_frame" for f in frames):
                return _err(f"Only one {'starting' if role == 'first' else 'ending'} picture is allowed")
            frames.append({"type": "image_url", "image_url": {"url": url}, "frame_type": f"{role}_frame"})
        else:
            refs.append({"type": "image_url", "image_url": {"url": url}})
        images.append({"id": iid, "role": role})
    if len(refs) > MAX_REFS:
        return _err(f"Use at most {MAX_REFS} reference pictures")
    if frames and refs:
        refs = []
        image_note = "Reference pictures were skipped: models use start/end pictures instead when both are given."

    try:
        parts = max(1, min(12, int(b.get("parts") or 1)))
    except (TypeError, ValueError):
        parts = 1
    mode = "video" if extra.get("video_ref_url") else "image" if frames else "ref" if refs else "text"
    n_img = len(frames) + len(refs)
    if extra.get("est_override") is not None:  # priced by the caller (upscaling is per megapixel-second)
        est, basis = round(float(extra["est_override"]), 4), extra.get("est_basis") or "fixed"
    else:
        est, basis = estimate_cost(m, duration, resolution, audio, mode, n_img)
    if est is None:
        return _err(f"Can't price {m['name']} ({basis}), so it won't run. Pick another model.")
    est = _bill(uid, est)  # customers pay usage plus the margin; the owner pays cost
    if est * parts > budget + 1e-9:
        what = f"This {duration * parts}s video ({parts} parts)" if parts > 1 else "This video"
        return _err(f"{what} would cost about ${est * parts:.2f}, over your ${budget:.2f} budget. "
                    f"Shorten it, lower the resolution, or raise the budget.")
    cap, spent = _daily_cap(), await _spent_24h()
    if cap is not None and spent + est * parts > cap:
        return _err(f"Daily cap reached: ${spent:.2f} of ${cap:.2f} used in the last 24 hours, "
                    f"this one is about ${est * parts:.2f}.")

    # Customers pay from prepaid credit: hold this part's estimate now, settle to the real
    # charge when it finishes, refund it if it fails. A long video must be covered in full up front.
    hold = 0.0
    if uid != "owner":
        u = await _user_by_id(uid)
        if not u or u.get("blocked"):
            return _err("Sign in again", 401)
        if not _is_member(u):
            return _err("Start a membership to make videos.", 402)
        need = est * parts if extra.get("part_no") in (None, 1) else est
        if _credit(u) + 1e-9 < need:
            if u.get("auto_reload") and u["id"] not in _reloading:
                _kick_autoreload(u)
                return _err("Topping up your credit from your card. Try again in a few seconds.", 402)
            return _err(f"This needs about ${need:.2f} of credit and you have ${_credit(u):.2f}. Add credit to continue.", 402)
        hold = round(est, 4)
        await _ledger(u, -hold, f"Hold for a {duration}s video")

    payload = {"model": m["id"], "prompt": prompt}
    if not extra.get("no_duration"):  # talking videos run as long as the speech
        payload["duration"] = duration
    if resolution:
        payload["resolution"] = resolution
    if aspect:
        payload["aspect_ratio"] = aspect
    if not audio:
        payload["generate_audio"] = False
    if extra.get("audio_url"):  # talking video: lip-sync the picture to this voice track
        payload["input_references"] = [{"type": "audio_url", "audio_url": {"url": extra["audio_url"]}}]
        if frames:
            payload["frame_images"] = frames
    elif extra.get("video_ref_url"):
        payload["input_references"] = [{"type": "video_url", "video_url": {"url": extra["video_ref_url"]}}] + refs
    elif frames:
        payload["frame_images"] = frames
    elif refs:
        payload["input_references"] = refs
    if extra.get("provider_params"):
        # Provider-specific knobs go under provider.options keyed by provider slug; only the
        # slug that actually serves the request is forwarded, so naming a couple is harmless.
        payload["provider"] = {"options": {slug: {"parameters": extra["provider_params"]}
                                           for slug in extra.get("provider_slugs") or []}}
    if extra.get("previous_job_id"):
        payload["previous_job_id"] = extra["previous_job_id"]

    def _msg(data, r):
        e = data.get("error") if isinstance(data, dict) else None
        return (e.get("message") if isinstance(e, dict) else e) or f"HTTP {r.status_code}"

    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{OR_BASE}/videos", json=payload, headers=_or_headers())
            data = r.json()
            options_dropped = False
            for _ in range(3):
                if r.status_code < 400:
                    break
                low = str(_msg(data, r)).lower()
                if "provider" in payload and r.status_code in (400, 422) and not options_dropped:
                    # Extra provider options are nice-to-haves: if refused, run without them.
                    payload.pop("provider")
                    options_dropped = True
                elif "previous_job_id" in payload and "previous" in low:
                    # A hint some providers don't take: retry without it.
                    payload.pop("previous_job_id")
                elif payload.get("generate_audio") is False and "generate_audio" in low:
                    # Some models (HeyGen) always render sound. A song replaces it anyway,
                    # so let the model make sound, but re-check the price with audio on.
                    est2, basis2 = estimate_cost(m, duration, resolution, True, mode, n_img)
                    est2 = _bill(uid, est2) if est2 is not None else None
                    if est2 is None or est2 * parts > budget + 1e-9:
                        await _release(uid, hold, "Over budget")
                        return _err(f"{m['name']} always makes its own sound, which brings this to about "
                                    f"${(est2 or 0) * parts:.2f}, over your ${budget:.2f} budget.")
                    if uid != "owner" and est2 > est:
                        u2 = await _user_by_id(uid)
                        extra_hold = round(est2 - est, 4)
                        if _credit(u2) + 1e-9 < extra_hold:
                            await _release(uid, hold, "Not enough credit")
                            return _err("Not enough credit for this model's sound. Add credit to continue.", 402)
                        await _ledger(u2, -extra_hold, "Hold for model sound")
                        hold = round(hold + extra_hold, 4)
                    est, basis = est2, basis2
                    payload.pop("generate_audio")
                    audio = True
                else:
                    break
                r = await c.post(f"{OR_BASE}/videos", json=payload, headers=_or_headers())
                data = r.json()
    except Exception as e:
        await _release(uid, hold, "OpenRouter did not answer")
        return _err(f"OpenRouter did not answer: {e}", 502)
    if r.status_code >= 400 or not data.get("id"):
        await _release(uid, hold, "Job refused")
        msg = str(_msg(data, r))
        if extra.get("video_ref_url") and any(w in msg.lower() for w in ("video", "reference", "input")):
            return _err(f"{m['name']} couldn't edit this video ({msg}). Try a model that takes video "
                        f"input, or use Remake with changes, which works on every model.", 502)
        return _err(f"OpenRouter refused the job: {msg}", 502)

    job = {"id": secrets.token_urlsafe(8), "created": time.time(), "model": m["id"], "prompt": prompt,
           "duration": duration, "resolution": resolution, "aspect_ratio": aspect, "audio": audio,
           "budget": budget, "estimate": est, "price_basis": basis, "or_id": data["id"],
           "polling_url": data.get("polling_url") or f"{OR_BASE}/videos/{data['id']}",
           "status": data.get("status") or "pending", "song_id": song_id,
           "song_start": float(b.get("song_start") or 0), "images": images, "image_note": image_note,
           "parent_id": extra.get("parent_id"), "edit_mode": extra.get("edit_mode"),
           "change": extra.get("change"), "base": base, "talk": extra.get("talk"),
           "options_note": "Some extra options weren't accepted, so it ran without them." if options_dropped else None,
           "base_prompt": extra.get("base_prompt") or prompt,
           "extend_of": extra.get("extend_of"),
           "parts_total": extra.get("parts_total") or parts,
           "part_no": extra.get("part_no") or 1,
           "chain_remaining": extra["chain_remaining"] if "chain_remaining" in extra else parts - 1,
           "chain_estimate": extra.get("chain_estimate") or (round(est * parts, 4) if parts > 1 else None),
           # what each later part may spend; the total was already checked against the budget above
           "user": uid, "hold": hold,
           "part_budget": extra.get("part_budget") or (round(max(budget / parts, est) + 0.01, 2) if parts > 1 else None)}
    jobs = await _all_jobs()
    async with _lock:
        jobs.append(job)
    await _save_jobs()
    _ensure_poller(job)
    return _public(job)


@video_router.get("/video/jobs")
async def video_jobs(request: Request):
    if (d := await _deny(request)):
        return d
    jobs = await _all_jobs()
    for j in jobs:
        _ensure_poller(j)
    mine = [j for j in jobs if _owns(request, j)]
    return {"jobs": [_public(j) for j in reversed(mine[-80:]) if not j.get("superseded")][:60],
            "spent_today": await _spent_24h(), "daily_cap": _daily_cap()}


@video_router.get("/video/jobs/{jid}")
async def video_job(jid: str, request: Request):
    if (d := await _deny(request)):
        return d
    j = _mine(request, await _all_jobs(), jid)
    if not j:
        return _err("No such video", 404)
    _ensure_poller(j)
    return _public(j)


@video_router.post("/video/jobs/{jid}/song")
async def video_job_song(jid: str, request: Request):
    if (d := await _deny(request)):
        return d
    j = _mine(request, await _all_jobs(), jid)
    if not j or not j.get("clip_key"):
        return _err("That video isn't ready yet", 404)
    b = await _body(request)
    if not _owns(request, _find(await _all_songs(), str(b.get("song_id", "")))):
        return _err("That song is gone. Pick another.", 404)
    try:
        await _apply_song(j, str(b.get("song_id", "")), float(b.get("song_start") or 0))
    except Exception as e:
        j["song_error"] = str(e)[:300]
        await _save_jobs()
        return _err(f"Couldn't add the song: {e}", 500)
    await _save_jobs()
    return _public(j)


@video_router.get("/video/clip/{jid}")
async def video_clip(jid: str, request: Request):
    if (d := await _deny(request)):
        return d
    j = _mine(request, await _all_jobs(), jid)
    scored = request.query_params.get("song") == "1"
    key = (j or {}).get("scored_key" if scored else "clip_key")
    if not key:
        return _err("No file for that video", 404)
    data = await _get(key)
    if not data:
        return _err("File missing from storage", 404)
    size = len(data)
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"}
    if request.query_params.get("dl") == "1":
        headers["Content-Disposition"] = f'attachment; filename="video-{jid}{"-song" if scored else ""}.mp4"'
    rng = request.headers.get("range", "")
    m = re.match(r"bytes=(\d*)-(\d*)", rng)
    if m and (m.group(1) or m.group(2)):
        if m.group(1):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
        else:
            start, end = max(0, size - int(m.group(2))), size - 1
        end = min(end, size - 1)
        if start > end:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return Response(data[start:end + 1], status_code=206, media_type="video/mp4", headers=headers)
    return Response(data, media_type="video/mp4", headers=headers)


@video_router.get("/video/songs")
async def video_songs(request: Request):
    if (d := await _deny(request)):
        return d
    return {"songs": [{"id": s["id"], "name": s["name"]} for s in reversed(await _all_songs()) if _owns(request, s)]}


@video_router.post("/video/songs")
async def video_song_upload(request: Request):
    if (d := await _deny(request)):
        return d
    b = await _body(request)
    name = str(b.get("name") or "song").strip()[:120]
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in SONG_EXT:
        return _err("Use an mp3, m4a, aac, wav or ogg file")
    raw = str(b.get("data") or "")
    if "," in raw[:100]:
        raw = raw.split(",", 1)[1]
    try:
        audio = base64.b64decode(raw, validate=True)
    except Exception:
        return _err("That file didn't come through. Try again.")
    if not audio:
        return _err("Empty file")
    if len(audio) > MAX_SONG_BYTES:
        return _err("Songs must be under 15 MB")
    sid = secrets.token_urlsafe(6)
    key = f"{PREFIX}songs/{sid}.{ext}"
    await _put(key, audio, SONG_EXT[ext])
    songs = await _all_songs()
    async with _lock:
        songs.append({"id": sid, "name": name, "ext": ext, "key": key, "created": time.time(), "user": _uid(request)})
    await _save_songs()
    return {"id": sid, "name": name}


@video_router.post("/video/images")
async def video_image_upload(request: Request):
    if (d := await _deny(request)):
        return d
    raw = str((await _body(request)).get("data") or "")
    if "," in raw[:100]:
        raw = raw.split(",", 1)[1]
    try:
        img = base64.b64decode(raw, validate=True)
    except Exception:
        return _err("That picture didn't come through. Try again.")
    if not img:
        return _err("Empty picture")
    if len(img) > MAX_IMAGE_BYTES:
        return _err("Pictures must be under 8 MB")
    kind = _sniff(img)
    if not kind:
        return _err("Use a JPEG, PNG or WebP picture")
    iid = secrets.token_urlsafe(24)
    await _put(f"{PREFIX}images/{iid}", img, kind[1])
    return {"id": iid, "url": f"/video/img/{iid}"}


@video_router.get("/video/img/{iid}")
async def video_image(iid: str):
    # Deliberately no PIN check: OpenRouter's provider fetches this URL.
    if not IMG_ID.match(iid):
        return _err("Not found", 404)
    data = await _get(f"{PREFIX}images/{iid}")
    if not data:
        return _err("Not found", 404)
    kind = _sniff(data) or ("jpg", "image/jpeg")
    return Response(data, media_type=kind[1], headers={"Cache-Control": "public, max-age=86400",
                                                       "X-Robots-Tag": "noindex"})


@video_router.post("/video/jobs/{jid}/edit")
async def video_job_edit(jid: str, request: Request):
    if (d := await _deny(request)):
        return d
    orig = _mine(request, await _all_jobs(), jid)
    if not orig or not orig.get("clip_key"):
        return _err("That video isn't ready yet", 404)
    b = await _body(request)
    mode = str(b.get("mode") or "edit")
    change = str(b.get("change") or "").strip()[:1500]
    if mode not in ("edit", "remake"):
        return _err("Unknown edit mode")
    if not change:
        return _err("Say what to change")
    spec = {k: b.get(k) for k in ("model", "duration", "resolution", "aspect_ratio", "budget", "audio",
                                  "song_id", "song_start")}
    for k in ("model", "duration", "resolution", "aspect_ratio"):
        if spec.get(k) in (None, ""):
            spec[k] = orig.get(k)
    extra = {"parent_id": orig["id"], "edit_mode": mode, "change": change}
    user_imgs = [i for i in (b.get("images") or []) if isinstance(i, dict)][:8]

    if mode == "edit":
        if not orig.get("src_token"):
            orig["src_token"] = secrets.token_urlsafe(24)
            await _save_jobs()
        extra["video_ref_url"] = f"{_public_base(request)}/video/src/{orig['src_token']}"
        if spec["model"] == orig.get("model") and orig.get("or_id"):
            extra["previous_job_id"] = orig["or_id"]
        spec["prompt"] = (f"Edit the input video: {change}. "
                          f"Keep everything else about the video the same.")
        spec["images"] = [{"id": i.get("id"), "role": "ref"} for i in user_imgs]
    else:
        try:
            spec["prompt"] = (await _or_chat(REMAKE_PROMPT.format(old=orig.get("prompt", ""), change=change)))\
                .strip().strip('"')[:2500]
        except Exception as e:
            print(f"[video] remake rewrite failed, falling back: {e!r}", flush=True)
            spec["prompt"] = f"{orig.get('prompt', '')} Change: {change}"[:2500]
        imgs = user_imgs if user_imgs else [dict(i) for i in (orig.get("images") or [])]
        if b.get("keep_look", True) and not any(i.get("role") == "first" for i in user_imgs):
            try:
                frame = await asyncio.to_thread(first_frame_sync, await _get(orig["clip_key"]) or b"")
                fid = secrets.token_urlsafe(24)
                await _put(f"{PREFIX}images/{fid}", frame, "image/jpeg")
                imgs = [i for i in imgs if i.get("role") != "first"] + [{"id": fid, "role": "first"}]
            except Exception as e:
                print(f"[video] keep_look frame failed: {e!r}", flush=True)
        spec["images"] = imgs
    return await _start_job(request, spec, extra)


@video_router.get("/video/src/{token}")
async def video_src(token: str, request: Request):
    # Deliberately no PIN check: the provider fetches this for video-to-video edits.
    if not IMG_ID.match(token):
        return _err("Not found", 404)
    j = next((x for x in await _all_jobs() if x.get("src_token") == token), None)
    key = (j or {}).get("scored_key") if request.query_params.get("song") == "1" else None
    key = key or (j or {}).get("clip_key")
    data = await _get(key) if key else None
    if not data:
        return _err("Not found", 404)
    return Response(data, media_type="video/mp4", headers={"Cache-Control": "public, max-age=86400",
                                                           "X-Robots-Tag": "noindex"})


CONTINUE = " Continue the same scene smoothly from the opening frame: same characters, look, lighting and camera style."


async def _extend(src: dict, *, auto: bool, seconds: int | None = None, budget: float | None = None,
                  change: str | None = None):
    """Start a job that picks up from src's last frame. Returns the job dict, raises on refusal."""
    if not src.get("clip_key"):
        raise RuntimeError("that video isn't ready")
    clip = await _get(src["clip_key"])
    if not clip:
        raise RuntimeError("the video file is missing")
    frame = await asyncio.to_thread(last_frame_sync, clip)
    fid = secrets.token_urlsafe(24)
    await _put(f"{PREFIX}images/{fid}", frame, "image/jpeg")
    base_prompt = src.get("base_prompt") or src.get("prompt") or ""
    prompt = (f"{change.strip()}." if change else base_prompt) + CONTINUE
    if auto:
        left = src["chain_remaining"] - 1
        seg_budget = src.get("part_budget") or src.get("budget") or 0
    else:
        left = 0
        seg_budget = budget or 0
    spec = {"model": src["model"], "prompt": prompt[:2500], "duration": seconds or src["duration"],
            "resolution": src.get("resolution"), "aspect_ratio": src.get("aspect_ratio"),
            "audio": src.get("audio", True), "budget": seg_budget,
            "song_id": src.get("song_id"), "song_start": src.get("song_start") or 0,
            "images": [{"id": fid, "role": "first"}]}
    extra = {"user": src.get("user") or "owner", "base": src.get("base"), "extend_of": src["id"], "parent_id": src["id"],
             "edit_mode": "extend", "change": change or (None if auto else "continued"),
             "base_prompt": base_prompt, "chain_remaining": left,
             "parts_total": src.get("parts_total") if auto else 1,
             "part_no": (src.get("part_no") or 1) + 1 if auto else 1,
             "chain_estimate": src.get("chain_estimate") if auto else None,
             "part_budget": src.get("part_budget") if auto else None}
    out = await _start_job(None, spec, extra)
    if isinstance(out, Response):
        try:
            detail = json.loads(out.body).get("detail")
        except Exception:
            detail = "refused"
        raise RuntimeError(detail)
    if auto:
        src["superseded"] = True
        await _save_jobs()
    return out


@video_router.post("/video/jobs/{jid}/extend")
async def video_job_extend(jid: str, request: Request):
    if (d := await _deny(request)):
        return d
    src = _mine(request, await _all_jobs(), jid)
    if not src or not src.get("clip_key"):
        return _err("That video isn't ready yet", 404)
    b = await _body(request)
    try:
        seconds = int(float(b.get("seconds") or src["duration"]))
        budget = round(float(b.get("budget") or 0), 2)
    except (TypeError, ValueError):
        return _err("Seconds and budget must be numbers")
    if not src.get("base"):
        src["base"] = _public_base(request)
    try:
        return await _extend(src, auto=False, seconds=seconds, budget=budget,
                             change=str(b.get("change") or "").strip()[:1500] or None)
    except Exception as e:
        return _err(f"Couldn't extend: {e}", 400)


# =========================================================================== accounts & billing
# Customers sign in with an emailed code, pay a monthly membership through Stripe, and
# spend prepaid credit at cost. The owner (PIN) never pays here.
STRIPE_API = "https://api.stripe.com/v1"
CODE_TTL = 15 * 60
LEDGER_KEEP = 200
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _packs() -> dict[str, str]:
    """VIDEO_CREDIT_PRICES="10:price_a,25:price_b,50:price_c" -> {"10": "price_a", ...}"""
    out = {}
    for part in (os.getenv("VIDEO_CREDIT_PRICES") or "").split(","):
        amt, _, pid = part.strip().partition(":")
        if amt.strip().isdigit() and pid.strip():
            out[amt.strip()] = pid.strip()
    return out


def _accounts_on() -> bool:
    """Customer accounts (passkeys) only where VIDEO_ACCOUNTS=1 (the Video Studio service), never on the bridge."""
    return os.getenv("VIDEO_ACCOUNTS") == "1"


def _email_codes_on() -> bool:
    """The older emailed-code sign-in, kept switched off unless a verified sender exists."""
    return _accounts_on() and os.getenv("VIDEO_EMAIL_CODES") == "1" and bool(os.getenv("RESEND_API_KEY"))


TRIAL_DAYS = int(os.getenv("VIDEO_TRIAL_DAYS") or "30")


def _trial_ok(u: dict) -> bool:
    """One free month per person: never had a membership before."""
    return TRIAL_DAYS > 0 and not u.get("subscription") and not u.get("had_trial")


def _billing_ready() -> bool:
    return bool(os.getenv("STRIPE_SECRET_KEY") and os.getenv("VIDEO_MEMBERSHIP_PRICE"))


async def _users_doc() -> dict:
    global _users
    if _users is None:
        async with _lock:
            if _users is None:
                raw = await _get(USERS_KEY)
                try:
                    doc = json.loads(raw) if raw else {}
                except Exception:
                    doc = {}
                _users = {"users": doc.get("users") or [], "events": doc.get("events") or [],
                          "funding": doc.get("funding") or []}
    return _users


async def _save_users() -> None:
    doc = await _users_doc()
    async with _lock:
        doc["events"] = doc["events"][-500:]
        if doc.get("funding"):
            done = [f for f in doc["funding"] if f["status"] != "waiting"]
            doc["funding"] = [f for f in doc["funding"] if f["status"] == "waiting"] + done[-200:]
        snapshot = json.dumps(doc, separators=(",", ":")).encode()
    await _put(USERS_KEY, snapshot, "application/json")


async def _user_by_id(uid: str) -> dict | None:
    return next((u for u in (await _users_doc())["users"] if u["id"] == uid), None)


async def _user_by_email(email: str) -> dict | None:
    e = email.strip().lower()
    return next((u for u in (await _users_doc())["users"] if u["email"] == e), None)


async def _user_by_customer(cus: str) -> dict | None:
    return next((u for u in (await _users_doc())["users"] if cus and u.get("stripe_customer") == cus), None)


def _is_member(u: dict) -> bool:
    if u.get("comp"):  # free membership granted by the owner
        return True
    if u.get("sub_status") not in ("active", "trialing"):
        return False
    end = u.get("period_end")
    return not end or end + 3 * 86400 > time.time()  # small grace while a renewal settles


def _credit(u: dict | None) -> float:
    return round(float((u or {}).get("credit") or 0), 4)


async def _ledger(u: dict, amount: float, note: str) -> None:
    async with _lock:
        u["credit"] = round(_credit(u) + amount, 4)
        u.setdefault("ledger", []).append({"t": time.time(), "amount": round(amount, 4), "note": note,
                                           "balance": u["credit"]})
        u["ledger"] = u["ledger"][-LEDGER_KEEP:]
    await _save_users()
    if amount < 0:
        _kick_autoreload(u)


async def _release(uid: str, hold: float, why: str) -> None:
    if uid == "owner" or not hold:
        return
    u = await _user_by_id(uid)
    if u:
        await _ledger(u, hold, f"Refund: {why}")


async def _settle(job: dict, actual: float | None) -> None:
    """Swap a job's hold for what it really cost. Runs once per job."""
    if job.get("settled") or (job.get("user") or "owner") == "owner":
        job["settled"] = True
        return
    job["settled"] = True
    u = await _user_by_id(job["user"])
    if not u:
        return
    hold = float(job.get("hold") or 0)
    cost = _bill(job["user"], float(actual)) if actual is not None else hold
    job["charged"] = round(cost, 4)
    diff = round(hold - cost, 4)
    if abs(diff) >= 0.0001:
        await _ledger(u, diff, "Refund: failed video" if cost == 0 else
                      ("Adjust to actual cost" if diff > 0 else "Actual cost was higher than the hold"))


def _account(u: dict) -> dict:
    return {"id": u["id"], "email": u.get("email"), "owner": bool(u.get("owner")),
            "member": True if u.get("owner") else _is_member(u),
            "status": "owner" if u.get("owner") else (u.get("sub_status") or "none"),
            "period_end": u.get("period_end"), "credit": None if u.get("owner") else _credit(u),
            "packs": sorted(int(k) for k in _packs()), "billing": _billing_ready(),
            "has_customer": bool(u.get("stripe_customer")), "passkeys": len(u.get("passkeys") or []),
            "trial_offer": TRIAL_DAYS if (not u.get("owner") and _trial_ok(u)) else 0,
            "card": u.get("card") if isinstance(u.get("card"), dict) else None,
            "auto_reload": u.get("auto_reload"), "reload_error": u.get("reload_error"),
            "trial_end": u.get("trial_end") if u.get("sub_status") == "trialing" else None,
            "ledger": [] if u.get("owner") else list(reversed((u.get("ledger") or [])[-30:])),
            "funding": _funding_summary() if u.get("owner") else None}


def _set_user_cookie(resp: Response, uid: str) -> None:
    exp = int(time.time() + SESSION_DAYS * 86400)
    resp.set_cookie(USER_COOKIE, _user_sign(uid, exp), max_age=SESSION_DAYS * 86400,
                    httponly=True, secure=True, samesite="lax", path="/video")


async def _send_code_email(email: str, code: str) -> None:
    key, sender = os.getenv("RESEND_API_KEY"), os.getenv("VIDEO_EMAIL_FROM") or os.getenv("ALERT_EMAIL_FROM")
    if not key or not sender:
        raise RuntimeError("email isn't set up on the server")
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post("https://api.resend.com/emails", headers={"Authorization": f"Bearer {key}"}, json={
            "from": sender, "to": [email], "subject": f"Your Video Studio code: {code}",
            "text": f"Your sign-in code is {code}\n\nIt works for 15 minutes. If you didn't ask for it, ignore this email."})
    if r.status_code >= 400:
        raise RuntimeError(f"email service said {r.status_code}: {r.text[:200]}")


@video_router.post("/video/auth/start")
async def video_auth_start(request: Request):
    if not _email_codes_on():
        return _err("Sign-up isn't available here.", 404)
    email = str((await _body(request)).get("email") or "").strip().lower()[:200]
    if not EMAIL_RE.match(email):
        return _err("Enter a real email address")
    ip, now = _client_ip(request), time.time()
    recent = [t for t in _fails.get("mail:" + ip, []) if now - t < 3600]
    if len(recent) >= 10:
        return _err("Too many codes asked for. Try again in an hour.", 429)
    _fails["mail:" + ip] = recent + [now]
    code = f"{secrets.randbelow(1_000_000):06d}"
    _codes[email] = {"hash": hashlib.sha256(f"{email}|{code}".encode()).hexdigest(), "exp": now + CODE_TTL, "tries": 0}
    try:
        await _send_code_email(email, code)
    except Exception as e:
        print(f"[video] code email failed: {e!r}", flush=True)
        return _err("Couldn't send the email. Try again in a minute.", 502)
    return {"ok": True}


@video_router.post("/video/auth/verify")
async def video_auth_verify(request: Request):
    if not _email_codes_on():
        return _err("Sign-up isn't available here.", 404)
    b = await _body(request)
    email = str(b.get("email") or "").strip().lower()
    code = re.sub(r"\D", "", str(b.get("code") or ""))
    rec = _codes.get(email)
    if not rec or rec["exp"] < time.time():
        return _err("That code expired. Ask for a new one.", 401)
    rec["tries"] += 1
    if rec["tries"] > 6:
        _codes.pop(email, None)
        return _err("Too many wrong codes. Ask for a new one.", 429)
    if not hmac.compare_digest(rec["hash"], hashlib.sha256(f"{email}|{code}".encode()).hexdigest()):
        return _err("Wrong code", 401)
    _codes.pop(email, None)
    u = await _user_by_email(email)
    if not u:
        u = {"id": secrets.token_urlsafe(9).replace("-", "a").replace("_", "b"), "email": email,
             "created": time.time(), "credit": 0.0, "ledger": []}
        doc = await _users_doc()
        async with _lock:
            doc["users"].append(u)
        await _save_users()
    if u.get("blocked"):
        return _err("This account is closed.", 403)
    resp = JSONResponse({"ok": True})
    _set_user_cookie(resp, u["id"])
    return resp


@video_router.post("/video/logout")
async def video_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(USER_COOKIE, path="/video")
    resp.delete_cookie(COOKIE, path="/video")
    return resp


@video_router.get("/video/account")
async def video_account(request: Request):
    if (d := await _deny(request)):
        return d
    u = request.state.user
    if not u.get("owner") and u.get("stripe_customer") and "card" not in u:
        try:
            await _saved_card(u)
        except Exception as e:
            print(f"[video] card lookup failed: {e!r}", flush=True)
    return _account(u)


# --------------------------------------------------------------------------- stripe
def _form(params: dict, prefix: str = "") -> list[tuple[str, str]]:
    out = []
    for k, v in params.items():
        key = f"{prefix}[{k}]" if prefix else str(k)
        if isinstance(v, dict):
            out += _form(v, key)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                out += _form(item, f"{key}[{i}]") if isinstance(item, dict) else [(f"{key}[{i}]", str(item))]
        elif v is not None:
            out.append((key, "true" if v is True else "false" if v is False else str(v)))
    return out


async def _stripe(method: str, path: str, params: dict | None = None, idem: str | None = None) -> dict:
    key = os.getenv("STRIPE_SECRET_KEY")
    if not key:
        raise RuntimeError("payments aren't set up yet")
    async with httpx.AsyncClient(timeout=30) as c:
        if method == "GET":
            r = await c.get(f"{STRIPE_API}{path}", auth=(key, ""), params=_form(params or {}))
        else:
            r = await c.request(method, f"{STRIPE_API}{path}", auth=(key, ""),
                                content=urlencode(_form(params or {})).encode(),
                                headers={"Content-Type": "application/x-www-form-urlencoded",
                                         **({"Idempotency-Key": idem} if idem else {})})
    data = r.json()
    if r.status_code >= 400:
        raise RuntimeError((data.get("error") or {}).get("message") or f"Stripe {r.status_code}")
    return data


async def _ensure_customer(u: dict) -> str:
    if u.get("stripe_customer"):
        return u["stripe_customer"]
    cus = await _stripe("POST", "/customers", {"email": u["email"], "metadata": {"app": "video_studio", "user_id": u["id"]}})
    u["stripe_customer"] = cus["id"]
    await _save_users()
    return cus["id"]


def _customer_only(request: Request):
    u = request.state.user
    if u.get("owner"):
        return _err("The owner account doesn't pay. Sign in with an email account to test payments.")
    if not _billing_ready():
        return _err("Payments aren't set up yet.", 503)
    return None


@video_router.post("/video/billing/subscribe")
async def video_subscribe(request: Request):
    if (d := await _deny(request)) or (d := _customer_only(request)):
        return d
    u, base = request.state.user, _public_base(request)
    if _is_member(u):
        return _err("Your membership is already active.")
    try:
        cus = await _ensure_customer(u)
        sess = await _stripe("POST", "/checkout/sessions", {
            "mode": "subscription", "customer": cus, "client_reference_id": u["id"],
            "line_items": [{"price": os.getenv("VIDEO_MEMBERSHIP_PRICE"), "quantity": 1}],
            "success_url": f"{base}/video?paid=member", "cancel_url": f"{base}/video",
            "metadata": {"app": "video_studio", "kind": "membership", "user_id": u["id"]},
            "subscription_data": {"metadata": {"app": "video_studio", "user_id": u["id"]},
                                  **({"trial_period_days": TRIAL_DAYS} if _trial_ok(u) else {})}})
    except Exception as e:
        return _err(f"Couldn't start checkout: {e}", 502)
    return {"url": sess["url"]}


@video_router.post("/video/billing/topup")
async def video_topup(request: Request):
    if (d := await _deny(request)) or (d := _customer_only(request)):
        return d
    u, base = request.state.user, _public_base(request)
    pack = str((await _body(request)).get("pack") or "")
    price = _packs().get(pack)
    if not price:
        return _err("Pick a credit amount")
    try:
        cus = await _ensure_customer(u)
        sess = await _stripe("POST", "/checkout/sessions", {
            "mode": "payment", "customer": cus, "client_reference_id": u["id"],
            "line_items": [{"price": price, "quantity": 1}],
            "success_url": f"{base}/video?paid=credit", "cancel_url": f"{base}/video",
            "payment_intent_data": {"setup_future_usage": "off_session",
                                    "metadata": {"app": "video_studio", "kind": "credit_checkout", "user_id": u["id"]}},
            "metadata": {"app": "video_studio", "kind": "credit", "user_id": u["id"], "credit_cents": str(int(pack) * 100)}})
    except Exception as e:
        return _err(f"Couldn't start checkout: {e}", 502)
    return {"url": sess["url"]}


@video_router.post("/video/billing/portal")
async def video_portal(request: Request):
    if (d := await _deny(request)) or (d := _customer_only(request)):
        return d
    u = request.state.user
    if not u.get("stripe_customer"):
        return _err("You don't have a membership to manage yet.")
    try:
        sess = await _stripe("POST", "/billing_portal/sessions",
                             {"customer": u["stripe_customer"], "return_url": f"{_public_base(request)}/video"})
    except Exception as e:
        return _err(f"Couldn't open billing: {e}", 502)
    return {"url": sess["url"]}


def _verify_stripe_sig(payload: bytes, header: str, secret: str, tolerance: int = 300) -> bool:
    parts = dict(p.split("=", 1) for p in header.split(",") if "=" in p)
    sigs = [p.split("=", 1)[1] for p in header.split(",") if p.startswith("v1=")]
    t = parts.get("t", "")
    if not t.isdigit() or abs(time.time() - int(t)) > tolerance or not sigs:
        return False
    expected = hmac.new(secret.encode(), f"{t}.".encode() + payload, hashlib.sha256).hexdigest()
    return any(hmac.compare_digest(expected, s) for s in sigs)


def _period_end(sub: dict) -> float | None:
    end = sub.get("current_period_end")
    if not end:  # newer API versions keep it on the subscription items
        items = ((sub.get("items") or {}).get("data") or [])
        end = max((i.get("current_period_end") or 0 for i in items), default=0) or None
    return float(end) if end else None


@video_router.post("/video/stripe/webhook")
async def video_stripe_webhook(request: Request):
    secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    payload = await request.body()
    if not secret or not _verify_stripe_sig(payload, request.headers.get("stripe-signature", ""), secret):
        return _err("bad signature", 400)
    ev = json.loads(payload)
    doc = await _users_doc()
    if ev.get("id") in doc["events"]:
        return {"ok": True, "duplicate": True}
    obj = (ev.get("data") or {}).get("object") or {}
    meta = obj.get("metadata") or {}
    kind = ev.get("type", "")
    if kind == "checkout.session.completed" and meta.get("app") == "video_studio":
        u = await _user_by_id(meta.get("user_id") or obj.get("client_reference_id") or "")
        if u:
            if obj.get("customer") and not u.get("stripe_customer"):
                u["stripe_customer"] = obj["customer"]
            u.pop("card", None)
            if meta.get("kind") == "credit" and obj.get("payment_status") == "paid":
                cents = int(meta.get("credit_cents") or obj.get("amount_total") or 0)
                await _ledger(u, cents / 100, f"Added ${cents / 100:.2f} credit")
                if _funding_fa():
                    async with _lock:
                        doc.setdefault("funding", []).append({"id": obj.get("id"), "cents": cents,
                                                              "user": u["id"], "t": time.time(), "status": "waiting"})
            elif meta.get("kind") == "membership" and obj.get("subscription"):
                u["subscription"] = obj["subscription"]
                u["sub_status"] = u.get("sub_status") or "active"
    elif kind.startswith("customer.subscription.") and (meta.get("app") == "video_studio"
                                                        or await _user_by_customer(obj.get("customer"))):
        u = await _user_by_id(meta.get("user_id") or "") or await _user_by_customer(obj.get("customer"))
        if u:
            u["subscription"] = obj.get("id")
            u["sub_status"] = "canceled" if kind.endswith(".deleted") else obj.get("status")
            if obj.get("trial_end"):
                u["had_trial"] = True
                u["trial_end"] = obj.get("trial_end")
            if u["sub_status"] in ("active", "trialing"):
                u.pop("files_purged_at", None)
            u["period_end"] = _period_end(obj)
            u["cancel_at_period_end"] = bool(obj.get("cancel_at_period_end"))
    async with _lock:
        doc["events"].append(ev.get("id"))
    await _save_users()
    return {"ok": True}


# --------------------------------------------------------------------------- passkeys (Face ID)
# Sign-in with no email service: the phone keeps a private key behind Face ID/Touch ID and
# proves it holds it. Passkeys sync through iCloud Keychain / Google Password Manager, so a
# new phone on the same account keeps working. Email is only a label (and Stripe receipts).
_pk_pending: dict[str, dict] = {}


def _rp(request: Request) -> tuple[str, str]:
    host = (os.getenv("VIDEO_PUBLIC_DOMAIN") or os.getenv("RAILWAY_PUBLIC_DOMAIN")
            or request.headers.get("x-forwarded-host") or request.headers.get("host") or "").split(":")[0].strip()
    return host, f"https://{host}"


def _pk_stash(data: dict) -> str:
    now = time.time()
    for k in [k for k, v in _pk_pending.items() if v["exp"] < now]:
        _pk_pending.pop(k, None)
    tok = secrets.token_urlsafe(16)
    _pk_pending[tok] = {**data, "exp": now + 300}
    return tok


def _pk_take(tok: str) -> dict | None:
    rec = _pk_pending.pop(str(tok or ""), None)
    return rec if rec and rec["exp"] >= time.time() else None


@video_router.post("/video/passkey/register/start")
async def video_pk_register_start(request: Request):
    if not _accounts_on():
        return _err("Sign-up isn't available here.", 404)
    import webauthn  # noqa: WPS433
    from webauthn.helpers.structs import (AuthenticatorSelectionCriteria, PublicKeyCredentialDescriptor,
                                          ResidentKeyRequirement, UserVerificationRequirement)
    from webauthn.helpers import base64url_to_bytes
    me = await _current_user(request)
    if me and me.get("owner"):
        return _err("The owner signs in with the PIN.")
    if me:  # signed in: add another passkey to this account
        uid, email = me["id"], me["email"]
    else:
        email = str((await _body(request)).get("email") or "").strip().lower()[:200]
        if not EMAIL_RE.match(email):
            return _err("Enter your email so we can label your account and send receipts")
        if await _user_by_email(email):
            return _err("There's already an account for that email. Tap Sign in with Face ID.", 409)
        uid = secrets.token_urlsafe(9).replace("-", "a").replace("_", "b")
    rp_id, _ = _rp(request)
    existing = (me or {}).get("passkeys") or []
    opts = webauthn.generate_registration_options(
        rp_id=rp_id, rp_name="Video Studio", user_id=uid.encode(), user_name=email, user_display_name=email,
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.REQUIRED, user_verification=UserVerificationRequirement.REQUIRED),
        exclude_credentials=[PublicKeyCredentialDescriptor(id=base64url_to_bytes(p["id"])) for p in existing])
    tok = _pk_stash({"kind": "register", "challenge": opts.challenge, "uid": uid, "email": email,
                     "adding": bool(me)})
    return {"token": tok, "options": json.loads(webauthn.options_to_json(opts))}


@video_router.post("/video/passkey/register/finish")
async def video_pk_register_finish(request: Request):
    if not _accounts_on():
        return _err("Sign-up isn't available here.", 404)
    import webauthn  # noqa: WPS433
    from webauthn.helpers import bytes_to_base64url
    b = await _body(request)
    rec = _pk_take(b.get("token"))
    if not rec or rec["kind"] != "register":
        return _err("That took too long. Try again.", 400)
    rp_id, origin = _rp(request)
    try:
        v = webauthn.verify_registration_response(credential=b.get("credential"), expected_challenge=rec["challenge"],
                                                  expected_rp_id=rp_id, expected_origin=origin,
                                                  require_user_verification=True)
    except Exception as e:
        print(f"[video] passkey register failed: {e!r}", flush=True)
        return _err("Face ID setup didn't go through. Try again.", 400)
    key = {"id": bytes_to_base64url(v.credential_id), "pk": bytes_to_base64url(v.credential_public_key),
           "count": v.sign_count, "created": time.time()}
    if rec["adding"]:
        u = await _user_by_id(rec["uid"])
        if not u:
            return _err("Sign in again", 401)
        u.setdefault("passkeys", []).append(key)
        await _save_users()
    else:
        if await _user_by_email(rec["email"]):
            return _err("There's already an account for that email. Tap Sign in with Face ID.", 409)
        u = {"id": rec["uid"], "email": rec["email"], "created": time.time(), "credit": 0.0, "ledger": [],
             "passkeys": [key]}
        doc = await _users_doc()
        async with _lock:
            doc["users"].append(u)
        await _save_users()
    resp = JSONResponse({"ok": True})
    _set_user_cookie(resp, u["id"])
    return resp


@video_router.post("/video/passkey/login/start")
async def video_pk_login_start(request: Request):
    if not _accounts_on():
        return _err("Sign-in isn't available here.", 404)
    import webauthn  # noqa: WPS433
    from webauthn.helpers.structs import UserVerificationRequirement
    rp_id, _ = _rp(request)
    opts = webauthn.generate_authentication_options(rp_id=rp_id,
                                                     user_verification=UserVerificationRequirement.REQUIRED)
    tok = _pk_stash({"kind": "login", "challenge": opts.challenge})
    return {"token": tok, "options": json.loads(webauthn.options_to_json(opts))}


@video_router.post("/video/passkey/login/finish")
async def video_pk_login_finish(request: Request):
    if not _accounts_on():
        return _err("Sign-in isn't available here.", 404)
    import webauthn  # noqa: WPS433
    from webauthn.helpers import base64url_to_bytes
    ip, now = _client_ip(request), time.time()
    recent = [t for t in _fails.get("pk:" + ip, []) if now - t < 900]
    if len(recent) >= 15:
        return _err("Too many tries. Wait 15 minutes.", 429)
    b = await _body(request)
    rec = _pk_take(b.get("token"))
    if not rec or rec["kind"] != "login":
        return _err("That took too long. Try again.", 400)
    cred = b.get("credential") or {}
    cid = str(cred.get("id") or cred.get("rawId") or "")
    u, key = None, None
    for x in (await _users_doc())["users"]:
        key = next((p for p in (x.get("passkeys") or []) if p["id"] == cid), None)
        if key:
            u = x
            break
    if not u:
        _fails["pk:" + ip] = recent + [now]
        return _err("No account uses that passkey. Create an account first.", 404)
    rp_id, origin = _rp(request)
    try:
        v = webauthn.verify_authentication_response(
            credential=cred, expected_challenge=rec["challenge"], expected_rp_id=rp_id, expected_origin=origin,
            credential_public_key=base64url_to_bytes(key["pk"]), credential_current_sign_count=key.get("count") or 0,
            require_user_verification=True)
    except Exception as e:
        _fails["pk:" + ip] = recent + [now]
        print(f"[video] passkey login failed: {e!r}", flush=True)
        return _err("Face ID sign-in didn't match. Try again.", 401)
    if u.get("blocked"):
        return _err("This account is closed.", 403)
    key["count"] = v.new_sign_count
    key["used"] = time.time()
    await _save_users()
    resp = JSONResponse({"ok": True})
    _set_user_cookie(resp, u["id"])
    return resp


# =========================================================================== editor
# A simple timeline: pick finished clips, trim them, add captions, pick an output shape and
# a song, and the server renders one video. Rendering is free (no AI involved), and the
# result lands in the clip list like any other clip, so it can be saved, re-scored or
# edited further.
EDITS_KEY = f"{PREFIX}edits.json"
SHAPES = {"9:16": (720, 1280), "16:9": (1280, 720), "1:1": (720, 720), "4:5": (720, 900)}
_edits: list[dict] | None = None


async def _all_edits() -> list[dict]:
    global _edits
    if _edits is None:
        async with _lock:
            if _edits is None:
                _edits = await _load_list(EDITS_KEY)
    return _edits


async def _save_edits() -> None:
    rows = await _all_edits()
    async with _lock:
        snapshot = json.dumps(rows[-300:], separators=(",", ":")).encode()
    await _put(EDITS_KEY, snapshot, "application/json")


def _caption_png(text: str, w: int, h: int, pos: str) -> bytes:
    """Caption as a transparent PNG the size of the frame: white text, dark outline, wrapped."""
    import io
    from PIL import Image, ImageDraw, ImageFont
    size = max(28, int(w * 0.065))
    font = ImageFont.load_default(size=size)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    words, lines, line = text.split(), [], ""
    for wd in words:
        trial = (line + " " + wd).strip()
        if d.textlength(trial, font=font) > w * 0.86 and line:
            lines.append(line)
            line = wd
        else:
            line = trial
    if line:
        lines.append(line)
    lines = lines[:4]
    lh = int(size * 1.25)
    block = lh * len(lines)
    y = {"top": int(h * 0.08), "middle": (h - block) // 2}.get(pos, h - block - int(h * 0.1))
    for ln in lines:
        tw = d.textlength(ln, font=font)
        d.text(((w - tw) / 2, y), ln, font=font, fill=(255, 255, 255, 255),
               stroke_width=max(2, size // 12), stroke_fill=(0, 0, 0, 230))
        y += lh
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def render_edit_sync(items: list[dict], clips: list[bytes], shape: str, fit: str, keep_sound: bool,
                     song: bytes | None, song_ext: str, song_start: float, song_volume: float,
                     fade: bool) -> bytes:
    """items[i] = {"start","end","caption","caption_pos"}; clips[i] = the source mp4 bytes."""
    w, h = SHAPES.get(shape, SHAPES["9:16"])
    with tempfile.TemporaryDirectory() as d:
        inputs, filt, labels, total = [], [], [], 0.0
        idx = 0
        for i, (it, data) in enumerate(zip(items, clips)):
            vp = f"{d}/c{i}.mp4"
            Path(vp).write_bytes(data)
            dur_src = _probe_duration(vp) or 0
            start = max(0.0, float(it.get("start") or 0))
            end = float(it.get("end") or 0) or dur_src
            end = min(end, dur_src) if dur_src else end
            if end - start < 0.2:
                raise RuntimeError(f"clip {i + 1} is trimmed to nothing")
            seg = end - start
            total += seg
            _, _, has_snd = _probe(vp)
            inputs += ["-i", vp]
            vi = idx
            idx += 1
            if fit == "fill":
                sc = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
            else:
                sc = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
            filt.append(f"[{vi}:v]trim={start:.3f}:{end:.3f},setpts=PTS-STARTPTS,{sc},setsar=1,fps=30,format=yuv420p[v{i}]")
            vlabel = f"[v{i}]"
            cap = str(it.get("caption") or "").strip()
            if cap:
                cp = f"{d}/cap{i}.png"
                Path(cp).write_bytes(_caption_png(cap[:200], w, h, str(it.get("caption_pos") or "bottom")))
                inputs += ["-loop", "1", "-t", f"{seg:.3f}", "-i", cp]
                ci = idx
                idx += 1
                filt.append(f"{vlabel}[{ci}:v]overlay=0:0:shortest=1[vc{i}]")
                vlabel = f"[vc{i}]"
            labels.append(vlabel)
            if keep_sound and has_snd:
                filt.append(f"[{vi}:a]atrim={start:.3f}:{end:.3f},asetpts=PTS-STARTPTS,aresample=44100,"
                            f"aformat=channel_layouts=stereo[a{i}]")
            else:
                filt.append(f"anullsrc=r=44100:cl=stereo,atrim=0:{seg:.3f}[a{i}]")
            labels.append(f"[a{i}]")
        n = len(items)
        filt.append("".join(labels) + f"concat=n={n}:v=1:a=1[vcat][acat]")
        vout = "[vcat]"
        if fade and total > 2:
            filt.append(f"[vcat]fade=t=in:st=0:d=0.5,fade=t=out:st={total - 0.6:.2f}:d=0.6[vf]")
            vout = "[vf]"
        aout = "[acat]"
        if song:
            sp = f"{d}/song.{song_ext}"
            Path(sp).write_bytes(song)
            inputs += ["-ss", f"{max(0.0, song_start):.2f}", "-i", sp]
            si = idx
            idx += 1
            vol = max(0.0, min(1.5, song_volume))
            under = 0.35 if keep_sound else 1.0
            filt.append(f"[{si}:a]atrim=0:{total:.3f},asetpts=PTS-STARTPTS,aresample=44100,"
                        f"aformat=channel_layouts=stereo,volume={vol:.2f},"
                        f"afade=t=out:st={max(0.0, total - 1.5):.2f}:d=1.5[song]")
            filt.append(f"[acat]volume={under}[base];[base][song]amix=inputs=2:duration=first:dropout_transition=0[amix]")
            aout = "[amix]"
        out = f"{d}/out.mp4"
        cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *inputs,
               "-filter_complex", ";".join(filt), "-map", vout, "-map", aout,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{total:.3f}", out]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if p.returncode != 0 or not Path(out).exists():
            raise RuntimeError(f"render failed: {(p.stderr or '').strip()[-500:]}")
        return Path(out).read_bytes()


def _edit_public(e: dict) -> dict:
    return {k: e.get(k) for k in ("id", "title", "items", "shape", "fit", "keep_sound", "song_id",
                                  "song_start", "song_volume", "fade", "status", "error", "job_id",
                                  "created", "updated")}


@video_router.get("/video/edits")
async def video_edits(request: Request):
    if (d := await _deny(request)):
        return d
    rows = [e for e in await _all_edits() if _owns(request, e)]
    return {"edits": [_edit_public(e) for e in reversed(rows[-40:])]}


@video_router.post("/video/edits")
async def video_edit_save(request: Request):
    """Create or update a timeline. Body: id?, title, items[{job,start,end,caption,caption_pos}],
    shape, fit (fit|fill), keep_sound, song_id, song_start, song_volume, fade."""
    if (d := await _deny(request)):
        return d
    b = await _body(request)
    jobs = await _all_jobs()
    items = []
    for it in (b.get("items") or [])[:30]:
        j = _mine(request, jobs, str((it or {}).get("job") or ""))
        if not j or not j.get("clip_key"):
            return _err("One of the clips is gone or not ready. Remove it and try again.")
        try:
            start = max(0.0, float(it.get("start") or 0))
            end = float(it.get("end") or 0)
        except (TypeError, ValueError):
            return _err("Trim times must be numbers")
        items.append({"job": j["id"], "start": start, "end": end,
                      "caption": str(it.get("caption") or "")[:200],
                      "caption_pos": it.get("caption_pos") if it.get("caption_pos") in ("top", "middle", "bottom") else "bottom",
                      "scored": bool(it.get("scored")) and bool(j.get("scored_key"))})
    song_id = str(b.get("song_id") or "") or None
    if song_id and not _owns(request, _find(await _all_songs(), song_id)):
        return _err("That song is gone. Pick another.")
    rows = await _all_edits()
    e = _mine(request, rows, str(b.get("id") or "")) if b.get("id") else None
    if not e:
        e = {"id": secrets.token_urlsafe(8), "user": _uid(request), "created": time.time(), "status": "draft"}
        async with _lock:
            rows.append(e)
    e.update(title=str(b.get("title") or "Untitled edit")[:80], items=items,
             shape=b.get("shape") if b.get("shape") in SHAPES else "9:16",
             fit="fill" if b.get("fit") == "fill" else "fit", keep_sound=bool(b.get("keep_sound", True)),
             song_id=song_id, song_start=float(b.get("song_start") or 0),
             song_volume=float(b.get("song_volume") if b.get("song_volume") is not None else 1.0),
             fade=bool(b.get("fade", True)), updated=time.time())
    if e.get("status") in ("ready", "failed"):
        e["status"] = "draft"
    await _save_edits()
    return _edit_public(e)


async def _render(e: dict) -> None:
    try:
        jobs = await _all_jobs()
        clips = []
        for it in e["items"]:
            j = _find(jobs, it["job"])
            key = (j or {}).get("scored_key") if it.get("scored") else (j or {}).get("clip_key")
            data = await _get(key) if key else None
            if not data:
                raise RuntimeError("a clip's file is missing")
            clips.append(data)
        song, ext = None, "mp3"
        if e.get("song_id"):
            s = _find(await _all_songs(), e["song_id"])
            if s:
                song, ext = await _get(s["key"]), s["ext"]
        out = await asyncio.to_thread(render_edit_sync, e["items"], clips, e["shape"], e["fit"], e["keep_sound"],
                                      song, ext, e.get("song_start") or 0, e.get("song_volume") or 1.0, e.get("fade", True))
        jid = secrets.token_urlsafe(8)
        key = f"{PREFIX}clips/{jid}.mp4"
        await _put(key, out, "video/mp4")
        total = sum((float(i["end"]) or 0) - float(i["start"]) for i in e["items"] if float(i["end"] or 0) > 0)
        job = {"id": jid, "created": time.time(), "finished": time.time(), "model": "editor",
               "prompt": f"Edit: {e['title']}", "duration": round(total, 1) or None, "total_seconds": round(total, 1) or None,
               "resolution": "720p", "aspect_ratio": e["shape"], "audio": True, "budget": 0, "estimate": 0,
               "actual_cost": 0, "status": "ready", "clip_key": key, "user": e.get("user") or "owner",
               "settled": True, "edit_mode": "timeline", "change": e["title"], "parent_id": e["items"][0]["job"]}
        all_jobs = await _all_jobs()
        async with _lock:
            all_jobs.append(job)
        await _save_jobs()
        e.update(status="ready", job_id=jid, error=None)
    except Exception as ex:
        print(f"[video] render {e.get('id')} failed: {ex!r}", flush=True)
        e.update(status="failed", error=str(ex)[:400])
    await _save_edits()


@video_router.post("/video/edits/{eid}/render")
async def video_edit_render(eid: str, request: Request):
    if (d := await _deny(request)):
        return d
    e = _mine(request, await _all_edits(), eid)
    if not e:
        return _err("No such edit", 404)
    if not e.get("items"):
        return _err("Add at least one clip")
    if e.get("status") == "rendering":
        return _edit_public(e)
    e["status"] = "rendering"
    await _save_edits()
    _tasks["edit:" + eid] = asyncio.create_task(_render(e))
    return _edit_public(e)


@video_router.post("/video/edits/{eid}/delete")
async def video_edit_delete(eid: str, request: Request):
    if (d := await _deny(request)):
        return d
    rows = await _all_edits()
    e = _mine(request, rows, eid)
    if e:
        async with _lock:
            rows.remove(e)
        await _save_edits()
    return {"ok": True}


# =========================================================================== characters
# A saved character = a name, a short description and up to 4 pictures. Tapping it in the
# composer adds the pictures as references and the description to the prompt, so the same
# person or product shows up across clips.
CHARS_KEY = f"{PREFIX}characters.json"
_chars: list[dict] | None = None


async def _all_chars() -> list[dict]:
    global _chars
    if _chars is None:
        async with _lock:
            if _chars is None:
                _chars = await _load_list(CHARS_KEY)
    return _chars


async def _save_chars() -> None:
    rows = await _all_chars()
    async with _lock:
        snapshot = json.dumps(rows, separators=(",", ":")).encode()
    await _put(CHARS_KEY, snapshot, "application/json")


@video_router.get("/video/characters")
async def video_chars(request: Request):
    if (d := await _deny(request)):
        return d
    return {"characters": [{k: c.get(k) for k in ("id", "name", "desc", "images")}
                           for c in await _all_chars() if _owns(request, c)]}


@video_router.post("/video/characters")
async def video_char_save(request: Request):
    if (d := await _deny(request)):
        return d
    b = await _body(request)
    name = str(b.get("name") or "").strip()[:60]
    if not name:
        return _err("Give the character a name")
    imgs = [str(i) for i in (b.get("images") or []) if IMG_ID.match(str(i))][:4]
    if not imgs:
        return _err("Add at least one picture of them")
    rows = await _all_chars()
    c = _mine(request, rows, str(b.get("id") or "")) if b.get("id") else None
    if not c:
        c = {"id": secrets.token_urlsafe(6), "user": _uid(request), "created": time.time()}
        async with _lock:
            rows.append(c)
    c.update(name=name, desc=str(b.get("desc") or "").strip()[:400], images=imgs)
    await _save_chars()
    return {k: c.get(k) for k in ("id", "name", "desc", "images")}


@video_router.post("/video/characters/{cid}/delete")
async def video_char_delete(cid: str, request: Request):
    if (d := await _deny(request)):
        return d
    rows = await _all_chars()
    c = _mine(request, rows, cid)
    if c:
        async with _lock:
            rows.remove(c)
        await _save_chars()
    return {"ok": True}


# =========================================================================== picture maker
# Make or change a picture with an image model, to use as a Start picture, a reference, or
# a character photo. Customers pay what OpenRouter charges (held, then settled).
IMAGE_MODEL = os.getenv("VIDEO_IMAGE_MODEL") or "google/gemini-3.1-flash-image-preview"
IMAGE_HOLD = 0.15


async def _charge_small(uid: str, hold: float, actual: float | None, what: str) -> None:
    if uid == "owner":
        return
    u = await _user_by_id(uid)
    if u and actual is not None and abs(hold - actual) >= 0.0001:
        await _ledger(u, round(hold - actual, 4), f"Adjust {what} to actual cost")


@video_router.post("/video/images/generate")
async def video_image_generate(request: Request):
    """{"prompt", "aspect"?, "from_image"?} -> {"id","url"}. from_image edits an existing picture."""
    if (d := await _deny(request)):
        return d
    u = request.state.user
    b = await _body(request)
    prompt = str(b.get("prompt") or "").strip()[:1500]
    if not prompt:
        return _err("Describe the picture")
    if not u.get("owner"):
        if not _is_member(u):
            return _err("Start a membership to make pictures.", 402)
        if _credit(u) < IMAGE_HOLD:
            return _err(f"Making a picture needs about ${IMAGE_HOLD:.2f} of credit. Add credit to continue.", 402)
        await _ledger(u, -IMAGE_HOLD, "Hold for a picture")
    content = [{"type": "text", "text": prompt}]
    src = str(b.get("from_image") or "")
    if IMG_ID.match(src):
        data = await _get(f"{PREFIX}images/{src}")
        if data:
            kind = _sniff(data) or ("jpg", "image/jpeg")
            content.append({"type": "image_url", "image_url": {
                "url": f"data:{kind[1]};base64,{base64.b64encode(data).decode()}"}})
    payload = {"model": IMAGE_MODEL, "modalities": ["image", "text"],
               "messages": [{"role": "user", "content": content}]}
    aspect = str(b.get("aspect") or "")
    if aspect in ("9:16", "16:9", "1:1", "4:5", "3:4", "4:3"):
        payload["image_config"] = {"aspect_ratio": aspect}
    try:
        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.post(f"{OR_BASE}/chat/completions", json=payload, headers=_or_headers())
        data = r.json()
        if r.status_code >= 400 or data.get("error"):
            raise RuntimeError((data.get("error") or {}).get("message") or f"HTTP {r.status_code}")
        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        url = (((msg.get("images") or [{}])[0].get("image_url") or {}).get("url")) or ""
        if not url.startswith("data:"):
            raise RuntimeError("the model didn't return a picture; try rewording")
        img = base64.b64decode(url.split(",", 1)[1])
        cost = (data.get("usage") or {}).get("cost")
    except Exception as e:
        if not u.get("owner"):
            await _ledger(await _user_by_id(u["id"]), IMAGE_HOLD, "Refund: picture failed")
        return _err(f"Couldn't make the picture: {e}", 502)
    kind = _sniff(img) or ("png", "image/png")
    iid = secrets.token_urlsafe(24)
    await _put(f"{PREFIX}images/{iid}", img, kind[1])
    await _charge_small(u["id"], IMAGE_HOLD, _bill(u["id"], float(cost)) if cost is not None else None, "picture")
    return {"id": iid, "url": f"/video/img/{iid}", "cost": cost}


# =========================================================================== talking video
# A photo that talks: lip-synced to a typed script (the model's own voice) or to an uploaded
# voice recording. Runs on HeyGen Avatar IV through the same job, budget and credit path.
TALK_MODEL = os.getenv("VIDEO_TALK_MODEL") or "heygen/avatar-iv"


@video_router.get("/video/aud/{token}")
async def video_audio_public(token: str):
    # Deliberately no PIN check: the talking-video model downloads the voice track from here.
    if not IMG_ID.match(token):
        return _err("Not found", 404)
    s = next((x for x in await _all_songs() if x.get("pub") == token), None)
    data = await _get(s["key"]) if s else None
    if not data:
        return _err("Not found", 404)
    return Response(data, media_type=SONG_EXT.get(s["ext"], "audio/mpeg"),
                    headers={"Cache-Control": "public, max-age=86400", "X-Robots-Tag": "noindex"})


def _audio_seconds_sync(data: bytes, ext: str) -> float | None:
    with tempfile.TemporaryDirectory() as d:
        pth = f"{d}/a.{ext}"
        Path(pth).write_bytes(data)
        return _probe_duration(pth)


@video_router.post("/video/talk")
async def video_talk(request: Request):
    """{"image", "script"? | "voice_id"?, "aspect_ratio", "resolution", "budget"}"""
    if (d := await _deny(request)):
        return d
    b = await _body(request)
    image = str(b.get("image") or "")
    if not IMG_ID.match(image):
        return _err("Add a photo of the face that should talk")
    script = str(b.get("script") or "").strip()[:3000]
    voice_id = str(b.get("voice_id") or "")
    extra = {"no_duration": True}
    if voice_id:
        s = _find(await _all_songs(), voice_id)
        if not _owns(request, s):
            return _err("That recording is gone. Upload it again.")
        audio = await _get(s["key"])
        secs = await asyncio.to_thread(_audio_seconds_sync, audio or b"", s["ext"]) if audio else None
        if not secs:
            return _err("Couldn't read that recording")
        if not s.get("pub"):
            s["pub"] = secrets.token_urlsafe(24)
            await _save_songs()
        extra["audio_url"] = f"{_public_base(request)}/video/aud/{s['pub']}"
        prompt = script or "Speak naturally to the camera with matching expressions."
        duration = int(secs + 0.999)
        extra["talk"] = {"voice": s["name"]}
    elif script:
        words = len(script.split())
        duration = max(3, int(words / 2.4 + 1.5))  # about 2.4 spoken words per second
        prompt = script
        extra["talk"] = {"script": True}
    else:
        return _err("Type what they should say, or pick a voice recording")
    if duration > 120:
        return _err("Keep it under 2 minutes")
    opts = {}
    if b.get("captions"):
        opts["caption"] = True
    if b.get("remove_background"):
        opts["remove_background"] = True
    if b.get("expressiveness") in ("low", "medium", "high"):
        opts["expressiveness"] = b["expressiveness"]
    if str(b.get("motion") or "").strip():
        opts["motion_prompt"] = str(b["motion"]).strip()[:300]
    if opts:
        extra["provider_params"] = opts
        extra["provider_slugs"] = ["heygen"]
    spec = {"model": TALK_MODEL, "prompt": prompt, "duration": duration,
            "resolution": b.get("resolution") or "720p", "aspect_ratio": b.get("aspect_ratio") or "9:16",
            "audio": True, "budget": b.get("budget"), "images": [{"id": image, "role": "first"}]}
    extra["change"] = "Talking video"
    extra["edit_mode"] = "talk"
    return await _start_job(request, spec, extra)


@video_router.get("/video/song/{sid}")
async def video_song_file(sid: str, request: Request):
    """The person's own uploaded song or recording, so the page can play it and read its length."""
    if (d := await _deny(request)):
        return d
    s = _mine(request, await _all_songs(), sid)
    data = await _get(s["key"]) if s else None
    if not data:
        return _err("Not found", 404)
    return Response(data, media_type=SONG_EXT.get(s["ext"], "audio/mpeg"),
                    headers={"Cache-Control": "private, max-age=3600", "Accept-Ranges": "none"})


# =========================================================================== upscale
# FLUX Video Upscale: sharpens a finished clip 1.5-3x (to 1080p/2K/4K). Priced per output
# megapixel-second; precise keeps faces/products faithful, creative invents more detail.
UPSCALE_MODEL = os.getenv("VIDEO_UPSCALE_MODEL") or "black-forest-labs/flux-video-upscale"
UPSCALE_MAX_SECONDS, UPSCALE_MAX_MB, UPSCALE_MAX_SIDE, UPSCALE_CAP_MP = 20, 50, 2560, 14.4


def upscale_plan(w: int, h: int, secs: float, factor: float, mode: str, skus: dict) -> dict:
    """Output size and cost. Output frames are capped near 14.4 MP (4K)."""
    factor = max(1.5, min(3.0, float(factor)))
    ow, oh = w * factor, h * factor
    mp = ow * oh / 1e6
    if mp > UPSCALE_CAP_MP:
        k = (UPSCALE_CAP_MP / mp) ** 0.5
        ow, oh, mp = ow * k, oh * k, UPSCALE_CAP_MP
    key = next((k for k in skus if "megapixel" in k.lower() and mode in k.lower()), None)
    if key is None:
        return {"error": "can't price upscaling right now"}
    rate = float(skus[key]) / (100 if "cent" in key.lower() else 1)
    return {"factor": factor, "width": int(ow), "height": int(oh), "mp": round(mp, 3),
            "cost": round(rate * mp * secs, 4), "rate": rate, "basis": key}


def _video_info_sync(data: bytes) -> tuple[int, int, float]:
    with tempfile.TemporaryDirectory() as d:
        pth = f"{d}/v.mp4"
        Path(pth).write_bytes(data)
        w, h, _ = _probe(pth)
        return w, h, _probe_duration(pth) or 0.0


async def _upscale_quote(request: Request, jid: str, factor, mode: str):
    j = _mine(request, await _all_jobs(), jid)
    if not j or not j.get("clip_key"):
        return None, None, _err("That video isn't ready yet", 404)
    m = await _model(UPSCALE_MODEL)
    if not m:
        return None, None, _err("The upscaler isn't available on OpenRouter right now.", 503)
    data = await _get(j.get("scored_key") or j["clip_key"])
    if not data:
        return None, None, _err("The video file is missing", 404)
    if len(data) > UPSCALE_MAX_MB * 1024 * 1024:
        return None, None, _err(f"Upscaling takes clips up to {UPSCALE_MAX_MB} MB")
    w, h, secs = await asyncio.to_thread(_video_info_sync, data)
    if secs > UPSCALE_MAX_SECONDS + 0.3:
        return None, None, _err(f"Upscaling takes clips up to {UPSCALE_MAX_SECONDS} seconds. "
                                f"Trim it in the editor first (this one is {secs:.0f}s).")
    if max(w, h) > UPSCALE_MAX_SIDE:
        return None, None, _err("This clip is already larger than the upscaler accepts")
    plan = upscale_plan(w, h, secs, factor or 2, "precise" if mode == "precise" else "creative", m["pricing_skus"])
    if plan.get("error"):
        return None, None, _err(plan["error"], 503)
    plan.update(src_width=w, src_height=h, seconds=round(secs, 2),
                can_tune=all(k in m["passthrough"] for k in ("upscale_factor", "creativity")))
    if not plan["can_tune"]:  # the provider will use its defaults: 2x, creative
        plan.update(upscale_plan(w, h, secs, 2, "creative", m["pricing_skus"]), note="Using the default 2x creative mode")
    return j, plan, None


@video_router.post("/video/jobs/{jid}/upscale/quote")
async def video_upscale_quote(jid: str, request: Request):
    if (d := await _deny(request)):
        return d
    b = await _body(request)
    _, plan, err = await _upscale_quote(request, jid, b.get("factor"), str(b.get("mode") or "precise"))
    if not err and not request.state.user.get("owner"):
        plan = {**plan, "cost": _bill(request.state.user["id"], plan["cost"])}
    return err or plan


@video_router.post("/video/jobs/{jid}/upscale")
async def video_upscale(jid: str, request: Request):
    """{"factor": 1.5-3, "mode": "precise"|"creative", "budget"} -> a new job with the sharper clip."""
    if (d := await _deny(request)):
        return d
    b = await _body(request)
    mode = "precise" if b.get("mode") == "precise" else "creative"
    j, plan, err = await _upscale_quote(request, jid, b.get("factor"), mode)
    if err:
        return err
    if not j.get("src_token"):
        j["src_token"] = secrets.token_urlsafe(24)
        await _save_jobs()
    extra = {"video_ref_url": f"{_public_base(request)}/video/src/{j['src_token']}" + ("?song=1" if j.get("scored_key") else ""),
             "no_duration": True, "est_override": plan["cost"], "est_basis": plan["basis"],
             "parent_id": j["id"], "edit_mode": "upscale",
             "change": f"Upscaled {plan['factor']:g}x ({mode}) to {plan['width']}x{plan['height']}"}
    if plan["can_tune"]:
        extra["provider_params"] = {"upscale_factor": plan["factor"], "creativity": 0 if mode == "precise" else 1}
        extra["provider_slugs"] = ["black-forest-labs", "bfl"]
    spec = {"model": UPSCALE_MODEL, "prompt": (j.get("base_prompt") or j.get("prompt") or "Upscale this video.")[:1000],
            "duration": max(1, int(plan["seconds"] + 0.999)), "resolution": "", "aspect_ratio": "",
            "audio": True, "budget": b.get("budget"), "images": []}
    return await _start_job(request, spec, extra)


# =========================================================================== your own videos
# Upload a video from the phone (sent as the raw file, not base64). It's converted to a
# standard mp4 and becomes a clip like any other: restyle it with Edit, add songs or
# captions, sharpen it, or use it in the editor.
MAX_UPLOAD_MB = int(os.getenv("VIDEO_MAX_UPLOAD_MB") or "200")
MAX_UPLOAD_SECONDS = 180


def normalize_upload_sync(data: bytes) -> tuple[bytes, float, int, int]:
    """Any phone video (mov/HEVC/mp4) -> h264/aac mp4, long side at most 1920, 30 fps."""
    with tempfile.TemporaryDirectory() as d:
        src, out = f"{d}/in", f"{d}/out.mp4"
        Path(src).write_bytes(data)
        secs = _probe_duration(src)
        if not secs:
            raise RuntimeError("that file isn't a video this app can read")
        if secs > MAX_UPLOAD_SECONDS + 0.5:
            raise RuntimeError(f"videos can be up to {MAX_UPLOAD_SECONDS // 60} minutes; this one is {secs / 60:.1f}")
        _, _, snd = _probe(src)
        cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", src,
               "-vf", "scale='if(gt(iw,ih),min(1920,iw),-2)':'if(gt(iw,ih),-2,min(1920,ih))',fps=30,format=yuv420p",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-movflags", "+faststart"]
        cmd += ["-c:a", "aac", "-b:a", "160k"] if snd else ["-an"]
        p = subprocess.run(cmd + [out], capture_output=True, text=True, timeout=900)
        if p.returncode != 0 or not Path(out).exists():
            raise RuntimeError(f"couldn't convert the video: {(p.stderr or '').strip()[-300:]}")
        w, h, _ = _probe(out)
        return Path(out).read_bytes(), secs, w, h


@video_router.post("/video/uploads")
async def video_upload(request: Request):
    """Raw video body (Content-Type video/*), ?name=. Returns the new clip."""
    if (d := await _deny(request)):
        return d
    if int(request.headers.get("content-length") or 0) > MAX_UPLOAD_MB * 1024 * 1024:
        return _err(f"Videos must be under {MAX_UPLOAD_MB} MB")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_UPLOAD_MB * 1024 * 1024:
            return _err(f"Videos must be under {MAX_UPLOAD_MB} MB")
        chunks.append(chunk)
    raw = b"".join(chunks)
    if not raw:
        return _err("Empty file")
    try:
        mp4, secs, w, h = await asyncio.to_thread(normalize_upload_sync, raw)
    except Exception as e:
        return _err(str(e)[:300])
    jid = secrets.token_urlsafe(8)
    key = f"{PREFIX}clips/{jid}.mp4"
    await _put(key, mp4, "video/mp4")
    name = str(request.query_params.get("name") or "My video")[:80]
    aspect = "9:16" if h > w * 1.2 else "16:9" if w > h * 1.2 else "1:1"
    job = {"id": jid, "created": time.time(), "finished": time.time(), "model": "upload",
           "prompt": name, "duration": round(secs, 1), "total_seconds": round(secs, 1),
           "resolution": f"{min(w, h)}p", "aspect_ratio": aspect, "audio": True, "budget": 0, "estimate": 0,
           "actual_cost": 0, "status": "ready", "clip_key": key, "user": _uid(request), "settled": True,
           "uploaded": True}
    jobs = await _all_jobs()
    async with _lock:
        jobs.append(job)
    await _save_jobs()
    return _public(job)


# =========================================================================== deleting & cleanup
RETAIN_DAYS = int(os.getenv("VIDEO_RETAIN_DAYS") or "30")
_sweeper: dict = {"task": None, "last": 0.0}


def _job_files(j: dict) -> list[str]:
    return [k for k in (j.get("clip_key"), j.get("scored_key"),
                        f"{PREFIX}clips/{j['id']}-part.mp4" if j.get("extend_of") else None) if k]


async def _forget_jobs(doomed: list[dict]) -> int:
    for j in doomed:
        await _delete(*_job_files(j))
    jobs = await _all_jobs()
    ids = {j["id"] for j in doomed}
    async with _lock:
        jobs[:] = [j for j in jobs if j["id"] not in ids]
    if doomed:
        await _save_jobs()
    return len(doomed)


async def _drop_chain_parts(final: dict) -> None:
    """A long video is joined into its last part, so the earlier, hidden parts are dead weight."""
    jobs, doomed, cur = await _all_jobs(), [], final
    while cur.get("extend_of"):
        prev = _find(jobs, cur["extend_of"])
        if not prev or not prev.get("superseded"):
            break
        doomed.append(prev)
        cur = prev
    await _delete(f"{PREFIX}clips/{final['id']}-part.mp4")
    await _forget_jobs(doomed)


@video_router.post("/video/jobs/{jid}/delete")
async def video_job_delete(jid: str, request: Request):
    if (d := await _deny(request)):
        return d
    j = _mine(request, await _all_jobs(), jid)
    if not j:
        return {"ok": True}
    if j.get("status") in ("pending", "in_progress") or j.get("chain_remaining"):
        return _err("Wait until it finishes, then delete it.")
    await _forget_jobs([j])
    return {"ok": True}


@video_router.post("/video/songs/{sid}/delete")
async def video_song_delete(sid: str, request: Request):
    if (d := await _deny(request)):
        return d
    songs = await _all_songs()
    s_ = _mine(request, songs, sid)
    if s_:
        await _delete(s_["key"])
        async with _lock:
            songs.remove(s_)
        await _save_songs()
    return {"ok": True}


async def _purge_user_files(uid: str) -> int:
    """Delete everything a person made (clips, songs, edits, characters). Their account and
    credit record stay, so they can come back to an empty studio with their balance."""
    n = await _forget_jobs([j for j in await _all_jobs() if j.get("user") == uid])
    songs = await _all_songs()
    mine = [x for x in songs if x.get("user") == uid]
    for x in mine:
        await _delete(x["key"])
    if mine:
        async with _lock:
            songs[:] = [x for x in songs if x.get("user") != uid]
        await _save_songs()
    for rows_fn, save_fn in ((_all_edits, _save_edits), (_all_chars, _save_chars)):
        rows = await rows_fn()
        if any(r.get("user") == uid for r in rows):
            async with _lock:
                rows[:] = [r for r in rows if r.get("user") != uid]
            await save_fn()
    return n + len(mine)


async def sweep_once() -> dict:
    """Customers whose membership ended more than RETAIN_DAYS ago, or who signed up and never
    joined within RETAIN_DAYS, lose their files. Owner files are never touched."""
    now, out = time.time(), {"users": 0, "files": 0}
    for u in list((await _users_doc())["users"]):
        if u.get("comp") or _is_member(u) or u.get("files_purged_at"):
            continue
        ended = u.get("period_end") if u.get("sub_status") else None
        since = ended or u.get("created") or now
        if now - since < RETAIN_DAYS * 86400:
            continue
        n = await _purge_user_files(u["id"])
        u["files_purged_at"] = now
        out["users"] += 1
        out["files"] += n
    if out["users"]:
        await _save_users()
        print(f"[video] cleanup removed {out['files']} items for {out['users']} people", flush=True)
    return out


async def _sweep_loop() -> None:
    while True:
        try:
            await sweep_once()
        except Exception as e:
            print(f"[video] cleanup error: {e!r}", flush=True)
        _sweeper["last"] = time.time()
        await asyncio.sleep(6 * 3600)


def _ensure_sweeper() -> None:
    if _accounts_on() and (_sweeper["task"] is None or _sweeper["task"].done()):
        _sweeper["task"] = asyncio.create_task(_sweep_loop())
    if _accounts_on() and _funding_fa() and (_sweeper.get("fund") is None or _sweeper["fund"].done()):
        _sweeper["fund"] = asyncio.create_task(_funding_loop())


# =========================================================================== card funding
# Credit purchases pay for OpenRouter, which is billed to a Stripe-issued card that spends
# from a Stripe financial account. Each paid credit purchase is queued here and, once Stripe
# makes the money available (card payments settle after a couple of business days), moved
# from the payments balance into that financial account with a payout. Idempotent per purchase.
FUND_EVERY = 5 * 60
_funding_state: dict = {"last": None, "error": None, "instant_off_until": 0.0}


def _funding_fa() -> str | None:
    fa = (os.getenv("VIDEO_FUNDING_FA") or "").strip()
    return fa if fa.startswith("fa_") and os.getenv("STRIPE_SECRET_KEY") else None


def _funding_summary() -> dict:
    q = (_users or {}).get("funding") or []
    return {"enabled": bool(_funding_fa()),
            "waiting_cents": sum(f["cents"] for f in q if f["status"] == "waiting"),
            "moved_cents": sum(f.get("moved_cents") or 0 for f in q if f["status"] == "moved"),
            "last_check": _funding_state["last"], "last_error": _funding_state["error"]}


async def fund_once() -> dict:
    """Move queued credit purchases into the card's financial account as money becomes available."""
    fa = _funding_fa()
    doc = await _users_doc()
    waiting = [f for f in doc.get("funding") or [] if f["status"] == "waiting"]
    out = {"moved": 0, "waiting": len(waiting)}
    if not fa or not waiting:
        return out
    bal = await _stripe("GET", "/balance")
    available = sum(int(x.get("amount") or 0) for x in bal.get("available") or [] if x.get("currency") == "usd")
    # Instant Payouts: card money is usable minutes after the sale instead of ~2 days later,
    # for a 1.5% fee. Stripe only shows instant_available once the account is approved for it.
    instant = sum(int(x.get("amount") or 0) for x in bal.get("instant_available") or [] if x.get("currency") == "usd")
    changed = False
    if instant and time.time() > _funding_state["instant_off_until"]:
        for f in sorted(waiting, key=lambda x: x["t"]):
            amount = min(f["cents"], instant)
            if amount < 50 or amount < f["cents"]:
                break
            try:
                po = await _stripe("POST", "/payouts", {"amount": amount, "currency": "usd", "method": "instant",
                                                        "payout_method": fa,
                                                        "description": "Video Studio credit to OpenRouter card (instant)",
                                                        "metadata": {"app": "video_studio", "checkout": f["id"] or ""}},
                                   idem=f"vs-fund-instant-{f['id']}")
            except Exception as e:
                # Not allowed to this destination or not eligible: use standard moves for a day.
                _funding_state["instant_off_until"] = time.time() + 86400
                print(f"[video] instant card funding unavailable, using standard: {e!r}", flush=True)
                break
            f.update(status="moved", moved_cents=amount, payout=po.get("id"), moved_at=time.time(), instant=True)
            instant -= amount
            out["moved"] += amount
            changed = True
        waiting = [f for f in waiting if f["status"] == "waiting"]
    for f in sorted(waiting, key=lambda x: x["t"]):
        amount = f["cents"]
        if available < amount:
            # After 5 days, send what's there (fees can leave the balance a little short).
            if time.time() - f["t"] > 5 * 86400 and available >= 100:
                amount = available
            else:
                break
        po = await _stripe("POST", "/payouts", {"amount": amount, "currency": "usd", "payout_method": fa,
                                                "description": "Video Studio credit to OpenRouter card",
                                                "metadata": {"app": "video_studio", "checkout": f["id"] or ""}},
                           idem=f"vs-fund-{f['id']}")
        f.update(status="moved", moved_cents=amount, payout=po.get("id"), moved_at=time.time())
        available -= amount
        out["moved"] += amount
        changed = True
    if changed:
        await _save_users()
        print(f"[video] moved ${out['moved'] / 100:.2f} of credit purchases to the card account", flush=True)
    return out


async def _funding_loop() -> None:
    while True:
        try:
            await fund_once()
            _funding_state.update(last=time.time(), error=None)
        except Exception as e:
            _funding_state.update(last=time.time(), error=str(e)[:200])
            print(f"[video] card funding error: {e!r}", flush=True)
        await asyncio.sleep(FUND_EVERY)


# =========================================================================== saved card: one-tap credit & auto-reload
# The card from the membership or a credit checkout stays on the customer in Stripe. Credit
# can then be charged to it from inside the app, either on a tap or automatically when the
# balance runs low. Each charge is credited once (PaymentIntent id recorded as handled).
RELOAD_MIN_GAP = 10 * 60
RELOAD_MAX_PER_DAY = 5
_reloading: set[str] = set()


async def _saved_card(u: dict, refresh: bool = False) -> dict | None:
    if not u.get("stripe_customer") or not _billing_ready():
        return None
    if isinstance(u.get("card"), dict) and not refresh:
        return u["card"]
    pms = await _stripe("GET", f"/customers/{u['stripe_customer']}/payment_methods", {"type": "card", "limit": 1})
    pm = (pms.get("data") or [None])[0]
    card = {"id": pm["id"], "brand": (pm.get("card") or {}).get("brand", "card").title(),
            "last4": (pm.get("card") or {}).get("last4", "")} if pm else None
    u["card"] = card or False
    await _save_users()
    return card


async def _charge_card(u: dict, cents: int, why: str, key: str) -> tuple[bool, str]:
    card = await _saved_card(u)
    if not card:
        return False, "No saved card yet. Use Add credit once with checkout and it will be saved."
    try:
        pi = await _stripe("POST", "/payment_intents", {
            "amount": cents, "currency": "usd", "customer": u["stripe_customer"], "payment_method": card["id"],
            "off_session": True, "confirm": True, "description": f"Video Studio credit ({why})",
            "metadata": {"app": "video_studio", "kind": "credit", "user_id": u["id"], "credit_cents": str(cents)}},
            idem=key)
    except Exception as e:
        msg = str(e)
        if "authentication" in msg.lower():
            return False, "Your bank wants to confirm this one. Use Add credit with checkout this time."
        return False, f"The card was declined: {msg[:150]}"
    if pi.get("status") != "succeeded":
        return False, "Your bank wants to confirm this one. Use Add credit with checkout this time."
    doc = await _users_doc()
    if pi["id"] in doc["events"]:
        return True, "already added"
    async with _lock:
        doc["events"].append(pi["id"])
        if _funding_fa():
            doc.setdefault("funding", []).append({"id": pi["id"], "cents": cents, "user": u["id"],
                                                  "t": time.time(), "status": "waiting"})
    await _ledger(u, cents / 100, f"Added ${cents / 100:.2f} credit ({why}, {card['brand']} {card['last4']})")
    return True, "ok"


@video_router.post("/video/billing/charge")
async def video_charge_saved(request: Request):
    """One tap: {"pack": "10", "nonce"} charges the saved card. The nonce makes double taps safe."""
    if (d := await _deny(request)) or (d := _customer_only(request)):
        return d
    u = request.state.user
    b = await _body(request)
    pack = str(b.get("pack") or "")
    if pack not in _packs():
        return _err("Pick a credit amount")
    nonce = re.sub(r"[^A-Za-z0-9_-]", "", str(b.get("nonce") or ""))[:40] or secrets.token_urlsafe(8)
    ok, msg = await _charge_card(u, int(pack) * 100, "one tap", f"vs-tap-{u['id']}-{nonce}")
    if not ok:
        return _err(msg, 402)
    return _account(u)


@video_router.post("/video/billing/autoreload")
async def video_autoreload(request: Request):
    """{"enabled", "below", "amount"}: refill credit with the saved card when it runs low."""
    if (d := await _deny(request)) or (d := _customer_only(request)):
        return d
    u = request.state.user
    b = await _body(request)
    if not b.get("enabled"):
        u["auto_reload"] = None
    else:
        amount = str(b.get("amount") or "10")
        try:
            below = max(0.5, min(50.0, float(b.get("below") or 2)))
        except (TypeError, ValueError):
            return _err("Pick a number")
        if amount not in _packs():
            return _err("Pick a refill amount")
        if not await _saved_card(u, refresh=True):
            return _err("Add credit once with checkout first so there's a card to refill from.")
        u["auto_reload"] = {"below": below, "amount": int(amount)}
        u.pop("reload_error", None)
    await _save_users()
    return _account(u)


def _kick_autoreload(u: dict) -> None:
    ar = u.get("auto_reload")
    if not ar or u.get("owner") or u["id"] in _reloading or _credit(u) >= ar["below"]:
        return
    now = time.time()
    recent = [t for t in u.get("reload_times") or [] if now - t < 86400]
    if recent and now - recent[-1] < RELOAD_MIN_GAP or len(recent) >= RELOAD_MAX_PER_DAY:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _reloading.add(u["id"])

    async def run():
        try:
            u["reload_times"] = recent + [time.time()]
            ok, msg = await _charge_card(u, int(ar["amount"]) * 100, "auto-reload",
                                         f"vs-auto-{u['id']}-{int(time.time() // RELOAD_MIN_GAP)}")
            if not ok:
                u["auto_reload"] = None  # stop retrying a failing card; the customer turns it back on
                u["reload_error"] = f"Auto-reload is off: {msg}"
                await _save_users()
        except Exception as e:
            print(f"[video] auto-reload error for {u['id']}: {e!r}", flush=True)
        finally:
            _reloading.discard(u["id"])

    loop.create_task(run())
