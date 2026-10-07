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
GET  /video/src/{token}         the original clip, PUBLIC on purpose for video-to-video (random token)

Pictures: generate takes "images": [{"id","role"}], role = first (start frame),
last (end frame) or ref (style/content reference). Frames win over refs when
both are sent, so refs are dropped with a warning in that case.

Money guards, both enforced server-side:
  - per-video budget: every request carries "budget"; the job is refused if the
    estimate from OpenRouter's listed price exceeds it. Unknown price = refused.
  - daily cap: VIDEO_DAILY_USD (default 3) over a rolling 24h, using actual
    cost once OpenRouter reports it and the estimate until then.

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

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

video_router = APIRouter()

VIDEO_APP_VERSION = "1.2.0"  # bump on HTML-only changes so the *.py watch pattern deploys
COOKIE = "video_session"
SESSION_DAYS = 60
OR_BASE = "https://openrouter.ai/api/v1"
PROMPT_MODEL = os.getenv("VIDEO_PROMPT_MODEL") or "z-ai/glm-5.3-flash"
JOBS_KEY = "video/jobs.json"
SONGS_KEY = "video/songs.json"
LOCAL_DIR = Path(os.getenv("VIDEO_DATA_DIR") or "/tmp") / "video"
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
_lock = asyncio.Lock()
_models_cache: dict = {"t": 0.0, "data": []}


def _daily_cap() -> float:
    try:
        return max(0.0, float(os.getenv("VIDEO_DAILY_USD") or "3"))
    except ValueError:
        return 3.0


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


def _deny(request: Request):
    if not _pin():
        return JSONResponse({"detail": "VIDEO_PIN (or CART_PIN) is not set on the server"}, status_code=503)
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
def _res_of(sku: str) -> str | None:
    s = sku.lower()
    return next((t for t in RES_TOKENS if t in s), None)


def _audio_of(sku: str) -> bool | None:
    s = sku.lower()
    if any(x in s for x in ("no-audio", "without-audio", "noaudio", "silent", "video-only")):
        return False
    if "audio" in s:
        return True
    return None


def estimate_cost(model: dict, duration: float, resolution: str, audio: bool) -> tuple[float | None, str]:
    """Conservative cost estimate from OpenRouter's listed pricing SKUs.

    Picks the SKUs that match the chosen resolution (or the unmarked base SKU when
    none name it), drops ones that contradict the audio choice, and takes the MAX
    so a guess errs toward refusing rather than overspending. None = unknown price.
    """
    skus = model.get("pricing_skus") or {}
    nums: dict[str, float] = {}
    for k, v in skus.items():
        try:
            nums[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    if not nums:
        return None, "no price listed"
    res = (resolution or "").lower()

    def pool(keys: list[str]) -> list[str]:
        exact = [k for k in keys if _res_of(k) == res]
        base = exact or [k for k in keys if _res_of(k) is None]
        fit = [k for k in base if _audio_of(k) in (None, audio)]
        return fit or base

    per_sec = pool([k for k in nums if "second" in k.lower()])
    if per_sec:
        rate = max(nums[k] for k in per_sec)
        return round(rate * float(duration), 4), f"${rate:g}/s ({', '.join(sorted(per_sec))})"
    per_vid = pool([k for k in nums if "second" not in k.lower() and "token" not in k.lower()])
    if per_vid:
        flat = max(nums[k] for k in per_vid)
        return round(flat, 4), f"${flat:g} per video ({', '.join(sorted(per_vid))})"
    return None, f"unrecognised pricing: {', '.join(sorted(nums))}"


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
        })
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


async def _apply_song(job: dict, song_id: str, start: float) -> None:
    song = _find(await _all_songs(), song_id)
    if not song:
        raise RuntimeError("song not found")
    video = await _get(job["clip_key"])
    audio = await _get(song["key"])
    if not video or not audio:
        raise RuntimeError("clip or song file is missing from storage")
    out = await asyncio.to_thread(mux_song_sync, video, audio, song["ext"], start)
    key = f"video/clips/{job['id']}-song.mp4"
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
                    key = f"video/clips/{job['id']}.mp4"
                    await _put(key, v.content, "video/mp4")
                    job["clip_key"] = key
                    if job.get("song_id"):
                        try:
                            await _apply_song(job, job["song_id"], float(job.get("song_start") or 0))
                        except Exception as e:
                            job["song_error"] = str(e)[:300]
                    job.update(status="ready", finished=time.time())
                    await _save_jobs()
                    return
                if status in ("failed", "cancelled", "expired"):
                    job["error"] = str(st.get("error") or status)[:300]
                    job["finished"] = time.time()
                    await _save_jobs()
                    return
                await asyncio.sleep(POLL_EVERY)
        job.update(status="failed", error="gave up waiting after 30 minutes", finished=time.time())
        await _save_jobs()
    except Exception as e:
        print(f"[video] poll {job.get('id')} error: {e!r}", flush=True)
        job["poll_error"] = str(e)[:300]
        await _save_jobs()


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
            "parent_id", "edit_mode", "change")
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
    authed = _authed(request)
    out = {"authed": authed, "configured": bool(_pin()), "version": VIDEO_APP_VERSION}
    if authed:
        out.update(daily_cap=_daily_cap(), spent_today=await _spent_24h())
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
    if (d := _deny(request)):
        return d
    try:
        models = await _models(force=request.query_params.get("refresh") == "1")
    except Exception as e:
        return _err(f"Could not load models from OpenRouter: {e}", 502)
    return {"models": models}


@video_router.post("/video/prompt")
async def video_prompt(request: Request):
    if (d := _deny(request)):
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
    if (d := _deny(request)):
        return d
    b = await _body(request)
    m = await _model(str(b.get("model", "")))
    if not m:
        return _err("Unknown model")
    cost, basis = estimate_cost(m, float(b.get("duration") or 0), str(b.get("resolution") or ""),
                                bool(b.get("audio", True)))
    return {"cost": cost, "basis": basis}


@video_router.post("/video/generate")
async def video_generate(request: Request):
    if (d := _deny(request)):
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
    if m["durations"] and duration not in m["durations"]:
        return _err(f"{m['name']} supports {m['durations']} seconds")
    if duration <= 0:
        return _err("Pick a duration")
    if m["resolutions"] and resolution not in m["resolutions"]:
        return _err(f"{m['name']} supports {', '.join(m['resolutions'])}")
    if m["aspect_ratios"] and aspect and aspect not in m["aspect_ratios"]:
        return _err(f"{m['name']} supports {', '.join(m['aspect_ratios'])}")
    if budget <= 0:
        return _err("Set a budget for this video")
    if song_id and not _find(await _all_songs(), song_id):
        return _err("That song is gone. Pick another.")

    images, frames, refs, image_note = [], [], [], None
    raw_imgs = b.get("images") or []
    if not isinstance(raw_imgs, list):
        return _err("Pictures came through wrong. Re-add them.")
    base = _public_base(request)
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

    est, basis = estimate_cost(m, duration, resolution, audio)
    if est is None:
        return _err(f"Can't price {m['name']} ({basis}), so it won't run. Pick another model.")
    if est > budget:
        return _err(f"This video would cost about ${est:.2f}, over your ${budget:.2f} budget. "
                    f"Shorten it, lower the resolution, or raise the budget.")
    cap, spent = _daily_cap(), await _spent_24h()
    if spent + est > cap:
        return _err(f"Daily cap reached: ${spent:.2f} of ${cap:.2f} used in the last 24 hours, "
                    f"this one is about ${est:.2f}.")

    payload = {"model": m["id"], "prompt": prompt, "duration": duration}
    if resolution:
        payload["resolution"] = resolution
    if aspect:
        payload["aspect_ratio"] = aspect
    if not audio:
        payload["generate_audio"] = False
    if extra.get("video_ref_url"):
        payload["input_references"] = [{"type": "video_url", "video_url": {"url": extra["video_ref_url"]}}] + refs
    elif frames:
        payload["frame_images"] = frames
    elif refs:
        payload["input_references"] = refs
    if extra.get("previous_job_id"):
        payload["previous_job_id"] = extra["previous_job_id"]

    def _msg(data, r):
        e = data.get("error") if isinstance(data, dict) else None
        return (e.get("message") if isinstance(e, dict) else e) or f"HTTP {r.status_code}"

    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{OR_BASE}/videos", json=payload, headers=_or_headers())
            data = r.json()
            # previous_job_id is a hint some providers don't take; retry once without it.
            if (r.status_code >= 400 and "previous_job_id" in payload
                    and "previous" in str(_msg(data, r)).lower()):
                payload.pop("previous_job_id")
                r = await c.post(f"{OR_BASE}/videos", json=payload, headers=_or_headers())
                data = r.json()
    except Exception as e:
        return _err(f"OpenRouter did not answer: {e}", 502)
    if r.status_code >= 400 or not data.get("id"):
        msg = str(_msg(data, r))
        if extra.get("video_ref_url"):
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
           "change": extra.get("change")}
    jobs = await _all_jobs()
    async with _lock:
        jobs.append(job)
    await _save_jobs()
    _ensure_poller(job)
    return _public(job)


@video_router.get("/video/jobs")
async def video_jobs(request: Request):
    if (d := _deny(request)):
        return d
    jobs = await _all_jobs()
    for j in jobs:
        _ensure_poller(j)
    return {"jobs": [_public(j) for j in reversed(jobs[-60:])],
            "spent_today": await _spent_24h(), "daily_cap": _daily_cap()}


@video_router.get("/video/jobs/{jid}")
async def video_job(jid: str, request: Request):
    if (d := _deny(request)):
        return d
    j = _find(await _all_jobs(), jid)
    if not j:
        return _err("No such video", 404)
    _ensure_poller(j)
    return _public(j)


@video_router.post("/video/jobs/{jid}/song")
async def video_job_song(jid: str, request: Request):
    if (d := _deny(request)):
        return d
    j = _find(await _all_jobs(), jid)
    if not j or not j.get("clip_key"):
        return _err("That video isn't ready yet", 404)
    b = await _body(request)
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
    if (d := _deny(request)):
        return d
    j = _find(await _all_jobs(), jid)
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
    if (d := _deny(request)):
        return d
    return {"songs": [{"id": s["id"], "name": s["name"]} for s in reversed(await _all_songs())]}


@video_router.post("/video/songs")
async def video_song_upload(request: Request):
    if (d := _deny(request)):
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
    key = f"video/songs/{sid}.{ext}"
    await _put(key, audio, SONG_EXT[ext])
    songs = await _all_songs()
    async with _lock:
        songs.append({"id": sid, "name": name, "ext": ext, "key": key, "created": time.time()})
    await _save_songs()
    return {"id": sid, "name": name}


@video_router.post("/video/images")
async def video_image_upload(request: Request):
    if (d := _deny(request)):
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
    await _put(f"video/images/{iid}", img, kind[1])
    return {"id": iid, "url": f"/video/img/{iid}"}


@video_router.get("/video/img/{iid}")
async def video_image(iid: str):
    # Deliberately no PIN check: OpenRouter's provider fetches this URL.
    if not IMG_ID.match(iid):
        return _err("Not found", 404)
    data = await _get(f"video/images/{iid}")
    if not data:
        return _err("Not found", 404)
    kind = _sniff(data) or ("jpg", "image/jpeg")
    return Response(data, media_type=kind[1], headers={"Cache-Control": "public, max-age=86400",
                                                       "X-Robots-Tag": "noindex"})


@video_router.post("/video/jobs/{jid}/edit")
async def video_job_edit(jid: str, request: Request):
    if (d := _deny(request)):
        return d
    orig = _find(await _all_jobs(), jid)
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

    if mode == "edit":
        if not orig.get("src_token"):
            orig["src_token"] = secrets.token_urlsafe(24)
            await _save_jobs()
        extra["video_ref_url"] = f"{_public_base(request)}/video/src/{orig['src_token']}"
        if spec["model"] == orig.get("model") and orig.get("or_id"):
            extra["previous_job_id"] = orig["or_id"]
        spec["prompt"] = (f"Edit the input video: {change}. "
                          f"Keep everything else about the video the same.")
        spec["images"] = []
    else:
        try:
            spec["prompt"] = (await _or_chat(REMAKE_PROMPT.format(old=orig.get("prompt", ""), change=change)))\
                .strip().strip('"')[:2500]
        except Exception as e:
            print(f"[video] remake rewrite failed, falling back: {e!r}", flush=True)
            spec["prompt"] = f"{orig.get('prompt', '')} Change: {change}"[:2500]
        imgs = [dict(i) for i in (orig.get("images") or [])]
        if b.get("keep_look", True):
            try:
                frame = await asyncio.to_thread(first_frame_sync, await _get(orig["clip_key"]) or b"")
                fid = secrets.token_urlsafe(24)
                await _put(f"video/images/{fid}", frame, "image/jpeg")
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
    data = await _get(j["clip_key"]) if j and j.get("clip_key") else None
    if not data:
        return _err("Not found", 404)
    return Response(data, media_type="video/mp4", headers={"Cache-Control": "public, max-age=86400",
                                                           "X-Robots-Tag": "noindex"})
