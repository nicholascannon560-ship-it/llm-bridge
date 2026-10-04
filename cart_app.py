"""Cart Tally — scan-as-you-shop budget app served by the bridge.

GET  /cart        the phone app (HTML shell, holds no data)
POST /cart/login  {"pin": "..."} -> sets a signed, HttpOnly session cookie
POST /cart/read   {"image": "<base64 jpeg>"} -> {"name","price","unit","taxable","note"}

Auth is deliberately separate from BRIDGE_API_KEY: the browser must never hold
the bridge key (it rotates every 24h and unlocks GitHub + Railway). CART_PIN
only unlocks /cart/read, i.e. the worst a leaked PIN can do is spend vision
calls. Changing CART_PIN invalidates every existing session.

Vision goes straight to OpenRouter (the gateway's ChatMessage is text-only).
Model: CART_VISION_MODEL, default below.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

cart_router = APIRouter()

COOKIE = "cart_session"
SESSION_DAYS = 60
MAX_IMAGE_BYTES = 4 * 1024 * 1024
DEFAULT_MODEL = "z-ai/glm-5.3-flash"
CART_APP_VERSION = "1.0.1"  # bump on HTML-only changes so *.py watch pattern triggers a deploy
_HTML_PATH = Path(__file__).with_name("cart_app.html")

# Login throttle: in-memory, per process. Good enough for a single replica.
_fails: dict[str, list[float]] = {}
_FAIL_WINDOW = 15 * 60
_FAIL_MAX = 8

PROMPT = """This photo was taken in a store, aimed at a shelf price tag or a product. Read the item and its price.
Rules:
- "price" is the price for ONE unit, as a number. If the tag says "2 for $5", price is 2.50. If a sale price and a regular price both show, use the sale price.
- If the price is per pound or per weight (e.g. "$3.99/lb"), set "unit" to "lb" (or "oz", "kg") and price to that rate; otherwise unit is null.
- "name" is a short readable product name (brand + item + size if visible), max 40 characters.
- "taxable": true if this item is normally subject to New York sales tax (candy, soda and sweetened drinks, alcohol, paper goods, cleaning supplies, toiletries, pet supplies, hot prepared food, non-food items). false for ordinary grocery food.
- If you cannot read a price, set price to null and say why in "note".
Reply with only JSON, no other text: {"name": string, "price": number|null, "unit": string|null, "taxable": boolean, "note": string|null}"""


def _pin() -> str:
    return (os.getenv("CART_PIN") or "").strip()


def _secret() -> bytes:
    base = os.getenv("UI_SESSION_SECRET") or "cart"
    return hashlib.sha256(f"cart-session|{base}|{_pin()}".encode()).digest()


def _sign(exp: int) -> str:
    mac = hmac.new(_secret(), str(exp).encode(), hashlib.sha256).hexdigest()
    return f"{exp}.{mac}"


def _authed(request: Request) -> bool:
    if not _pin():
        return False
    tok = request.cookies.get(COOKIE) or ""
    exp_s, _, mac = tok.partition(".")
    if not exp_s.isdigit() or not mac:
        return False
    if int(exp_s) < time.time():
        return False
    return hmac.compare_digest(_sign(int(exp_s)), tok)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "?")


@cart_router.get("/cart", response_class=HTMLResponse)
async def cart_page():
    try:
        html = _HTML_PATH.read_text(encoding="utf-8")
    except Exception:
        return HTMLResponse("cart_app.html missing", status_code=500)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@cart_router.get("/cart/session")
async def cart_session(request: Request):
    return {"authed": _authed(request), "configured": bool(_pin())}


@cart_router.post("/cart/login")
async def cart_login(request: Request):
    if not _pin():
        return JSONResponse({"detail": "CART_PIN is not set on the server"}, status_code=503)
    ip = _client_ip(request)
    now = time.time()
    recent = [t for t in _fails.get(ip, []) if now - t < _FAIL_WINDOW]
    _fails[ip] = recent
    if len(recent) >= _FAIL_MAX:
        return JSONResponse({"detail": "Too many tries. Wait 15 minutes."}, status_code=429)
    try:
        body = await request.json()
    except Exception:
        body = {}
    supplied = str((body or {}).get("pin", "")).strip()
    if not supplied or not hmac.compare_digest(supplied, _pin()):
        recent.append(now)
        return JSONResponse({"detail": "Wrong PIN"}, status_code=401)
    _fails.pop(ip, None)
    exp = int(now + SESSION_DAYS * 86400)
    resp = JSONResponse({"ok": True})
    resp.set_cookie(COOKIE, _sign(exp), max_age=SESSION_DAYS * 86400,
                    httponly=True, secure=True, samesite="strict", path="/cart")
    return resp


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            return json.loads(text[a:b + 1])
        raise


def _clean(d: dict) -> dict:
    price = d.get("price")
    try:
        price = None if price is None else round(float(price), 2)
        if price is not None and (price < 0 or price > 100000):
            price = None
    except (TypeError, ValueError):
        price = None
    unit = d.get("unit")
    unit = str(unit)[:4] if unit else None
    note = d.get("note")
    return {
        "name": str(d.get("name") or "Item")[:60],
        "price": price,
        "unit": unit,
        "taxable": bool(d.get("taxable")),
        "note": str(note)[:200] if note else None,
    }


@cart_router.post("/cart/read")
async def cart_read(request: Request):
    if not _pin():
        return JSONResponse({"detail": "CART_PIN is not set on the server"}, status_code=503)
    if not _authed(request):
        return JSONResponse({"detail": "Sign in again"}, status_code=401)
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return JSONResponse({"detail": "OPENROUTER_API_KEY missing"}, status_code=503)
    try:
        body = await request.json()
        b64 = str(body.get("image", ""))
        if "," in b64[:100]:
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64, validate=False)
    except Exception:
        return JSONResponse({"detail": "Send {image: base64}"}, status_code=400)
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        return JSONResponse({"detail": "Image missing or too large"}, status_code=413)
    mime = "image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    data_url = f"data:{mime};base64," + base64.b64encode(raw).decode()

    model = os.getenv("CART_VISION_MODEL") or DEFAULT_MODEL
    payload = {
        "model": model,
        "max_tokens": 400,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    }
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}",
                         "HTTP-Referer": "https://kalshiml-production-b2e9.up.railway.app",
                         "X-Title": "cart-tally"},
                json=payload,
            )
        data = r.json()
    except Exception as e:
        print(f"[cart] openrouter call failed: {e!r}", flush=True)
        return JSONResponse({"detail": "Vision service unreachable"}, status_code=502)
    if r.status_code >= 400 or data.get("error"):
        err = data.get("error") or {}
        print(f"[cart] openrouter {r.status_code} {model}: {err}", flush=True)
        return JSONResponse({"detail": f"Vision model error: {err.get('message') or r.status_code}"}, status_code=502)
    msg = ((data.get("choices") or [{}])[0].get("message") or {})
    text = msg.get("content") or msg.get("reasoning") or ""
    try:
        out = _clean(_parse_json(text))
    except Exception:
        print(f"[cart] unparseable reply from {model}: {text[:300]!r}", flush=True)
        return JSONResponse({"detail": "Couldn't read that tag"}, status_code=422)
    out["model"] = model
    out["ms"] = int((time.time() - t0) * 1000)
    print(f"[cart] read ok {model} {out['ms']}ms price={out['price']}", flush=True)
    return out
