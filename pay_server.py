"""North River Pay: the one place that holds the Square token for every North River Labs app.

Apps never talk to Square. They ask this service for a checkout link and later ask whether
that order was paid; each app keeps its own credit balances.

Start: uvicorn pay_server:app --host 0.0.0.0 --port $PORT
Env:
  PAY_DATA     JSON state file on a volume (default /data/pay.json). Holds the Square token.
  PAY_APPS     {"video-studio": {"key": "...", "name": "Video Studio", "origin": "https://..."}, ...}
  SETUP_CODE   one-time code for /setup (paste the Square token). Each code works once;
               set a new one to rotate the token or switch sandbox/production.
App API (headers X-App-Id + X-App-Key):
  POST /v1/checkout   {"user_ref", "cents", "label"?, "redirect_url"} -> {"order_id", "url"}
  GET  /v1/orders/{id}  -> {"status": "waiting"|"paid", "cents", "paid_cents", "user_ref"}
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import json
import os
import secrets
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

VERSION = "1.0.0"
SQUARE_VERSION = "2025-01-23"
DATA = Path(os.getenv("PAY_DATA") or "/data/pay.json")
MIN_CENTS, MAX_CENTS = 100, 50000

app = FastAPI(title="North River Pay", docs_url=None, redoc_url=None, openapi_url=None)
_lock = asyncio.Lock()
_state: dict | None = None
_tasks: dict = {}


# --------------------------------------------------------------------------- state
def _load() -> dict:
    global _state
    if _state is None:
        try:
            _state = json.loads(DATA.read_text())
        except Exception:
            _state = {}
        _state.setdefault("square", {})
        _state.setdefault("orders", {})
        _state.setdefault("used_codes", [])
    return _state


def _save() -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA.with_suffix(".tmp")
    tmp.write_text(json.dumps(_load(), separators=(",", ":")))
    os.replace(tmp, DATA)


def _apps() -> dict:
    try:
        return json.loads(os.getenv("PAY_APPS") or "{}")
    except Exception:
        return {}


def _err(msg: str, code: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=code)


def _auth(request: Request) -> tuple[str, dict] | None:
    aid = request.headers.get("x-app-id", "")
    key = request.headers.get("x-app-key", "")
    a = _apps().get(aid)
    if not a or not key or not hmac.compare_digest(str(a.get("key", "")), key):
        return None
    return aid, a


# --------------------------------------------------------------------------- square
def _sq() -> dict:
    return _load()["square"]


def _base(env: str) -> str:
    return "https://connect.squareupsandbox.com" if env == "sandbox" else "https://connect.squareup.com"


async def _square(method: str, path: str, body: dict | None = None, token: str | None = None,
                  env: str | None = None) -> dict:
    tok, env = token or _sq().get("token"), env or _sq().get("env") or "production"
    if not tok:
        raise RuntimeError("payments aren't set up yet")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.request(method, f"{_base(env)}{path}", json=body,
                            headers={"Authorization": f"Bearer {tok}", "Square-Version": SQUARE_VERSION})
    data = r.json() if r.content else {}
    if r.status_code >= 400:
        e = (data.get("errors") or [{}])[0]
        raise RuntimeError(e.get("detail") or e.get("code") or f"Square {r.status_code}")
    return data


async def _check_order(oid: str) -> dict:
    """Ask Square whether a waiting order has been paid; mark it paid once."""
    rec = _load()["orders"].get(oid)
    if not rec or rec["status"] != "waiting":
        return rec or {}
    order = (await _square("GET", f"/v2/orders/{oid}")).get("order") or {}
    paid = 0
    for t in order.get("tenders") or []:
        pid = t.get("payment_id") or t.get("id")
        if pid:
            pay = (await _square("GET", f"/v2/payments/{pid}")).get("payment") or {}
            if pay.get("status") == "COMPLETED":
                paid += int((pay.get("amount_money") or {}).get("amount") or 0)
    if paid + 1 >= rec["cents"]:
        async with _lock:
            if rec["status"] == "waiting":
                rec.update(status="paid", paid_cents=paid, paid_at=time.time())
                _save()
                print(f"[pay] {rec['app']} order {oid} paid {paid}c", flush=True)
    return rec


async def _poll_loop() -> None:
    while True:
        try:
            cut = time.time() - 86400
            for oid, rec in list(_load()["orders"].items()):
                if rec["status"] == "waiting" and rec["t"] > cut and _sq().get("token"):
                    await _check_order(oid)
        except Exception as e:
            print(f"[pay] poll error: {e!r}", flush=True)
        await asyncio.sleep(60)


@app.on_event("startup")
async def _start() -> None:
    _load()
    _tasks["poll"] = asyncio.create_task(_poll_loop())


# --------------------------------------------------------------------------- app API
@app.get("/health")
async def health():
    s = _sq()
    return {"ok": True, "version": VERSION, "configured": bool(s.get("token")),
            "env": s.get("env"), "apps": sorted(_apps())}


@app.post("/v1/checkout")
async def checkout(request: Request):
    who = _auth(request)
    if not who:
        return _err("bad app credentials", 401)
    aid, a = who
    if not _sq().get("token"):
        return _err("Payments aren't set up yet.", 503)
    try:
        b = await request.json()
    except Exception:
        b = {}
    try:
        cents = int(b.get("cents") or 0)
    except (TypeError, ValueError):
        cents = 0
    if not MIN_CENTS <= cents <= MAX_CENTS:
        return _err("amount out of range")
    user_ref = str(b.get("user_ref") or "")[:200]
    redirect = str(b.get("redirect_url") or "")
    origin = str(a.get("origin") or "").rstrip("/")
    if not user_ref:
        return _err("user_ref required")
    if not origin or not (redirect == origin or redirect.startswith(origin + "/")):
        return _err("redirect_url must be on the app's own site")
    name = str(a.get("name") or aid)
    label = str(b.get("label") or f"{name} credit ${cents / 100:.2f}")[:255]
    try:
        data = await _square("POST", "/v2/online-checkout/payment-links", {
            "idempotency_key": secrets.token_hex(16),
            "order": {"location_id": _sq()["location"], "reference_id": aid[:40],
                      "line_items": [{"name": label, "quantity": "1",
                                      "base_price_money": {"amount": cents, "currency": "USD"}}],
                      "metadata": {"app": aid[:60], "user_ref": user_ref[:255]}},
            "checkout_options": {"redirect_url": redirect},
            "payment_note": f"{name} · {user_ref}"[:500]})
    except Exception as e:
        return _err(f"Square: {e}", 502)
    link = data.get("payment_link") or {}
    if not link.get("url") or not link.get("order_id"):
        return _err("Square didn't return a checkout link", 502)
    async with _lock:
        st = _load()
        st["orders"][link["order_id"]] = {"app": aid, "user_ref": user_ref, "cents": cents,
                                          "status": "waiting", "t": time.time()}
        cut = time.time() - 30 * 86400  # keep a month of history
        st["orders"] = {k: v for k, v in st["orders"].items() if v["t"] > cut}
        _save()
    return {"order_id": link["order_id"], "url": link["url"]}


@app.get("/v1/orders/{oid}")
async def order(oid: str, request: Request):
    who = _auth(request)
    if not who:
        return _err("bad app credentials", 401)
    rec = _load()["orders"].get(oid)
    if not rec or rec["app"] != who[0]:
        return _err("unknown order", 404)
    if rec["status"] == "waiting" and rec["t"] > time.time() - 86400:
        try:
            rec = await _check_order(oid)
        except Exception as e:
            print(f"[pay] check {oid} failed: {e!r}", flush=True)
    return {"order_id": oid, "status": rec["status"], "cents": rec["cents"],
            "paid_cents": rec.get("paid_cents", 0), "user_ref": rec["user_ref"]}


# --------------------------------------------------------------------------- one-time setup
def _code_ok(code: str) -> bool:
    want = os.getenv("SETUP_CODE") or ""
    if len(want) < 16 or not code or not hmac.compare_digest(want, code):
        return False
    return hashlib.sha256(code.encode()).hexdigest() not in _load()["used_codes"]


PAGE = """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>North River Pay setup</title><style>
body{{font-family:-apple-system,system-ui,sans-serif;max-width:520px;margin:0 auto;padding:24px 18px;
background:#0f1a14;color:#e8efe9}}h1{{font-size:22px}}label{{display:block;margin:16px 0 6px;font-weight:600}}
textarea,select{{width:100%;box-sizing:border-box;font-size:16px;padding:10px;border-radius:10px;border:1px solid #3a5545;
background:#16251c;color:#e8efe9}}textarea{{height:110px;font-family:ui-monospace,monospace}}
button{{margin-top:18px;width:100%;padding:14px;font-size:17px;border:0;border-radius:12px;background:#3fa36b;color:#fff;font-weight:700}}
.msg{{margin-top:16px;padding:12px;border-radius:10px;background:#1d3326}}.bad{{background:#4a1f1f}}p{{color:#a9bcaf}}</style></head>
<body><h1>North River Pay</h1>{body}</body></html>"""


@app.get("/setup", response_class=HTMLResponse)
async def setup_page(code: str = ""):
    if not _code_ok(code):
        return HTMLResponse(PAGE.format(body='<div class="msg bad">This setup link has expired or was already used.</div>'), 403)
    form = f"""<p>Paste your Square access token once. Every app uses it from here.</p>
<form method="post" action="/setup"><input type="hidden" name="code" value="{html.escape(code)}">
<label>Environment</label><select name="env"><option value="production" selected>Production (real money)</option>
<option value="sandbox">Sandbox (testing)</option></select>
<label>Square access token</label><textarea name="token" autocomplete="off" autocapitalize="off" spellcheck="false" required></textarea>
<button type="submit">Save</button></form>"""
    return PAGE.format(body=form)


@app.post("/setup", response_class=HTMLResponse)
async def setup_save(request: Request):
    from urllib.parse import parse_qs
    f = {k: v[0] for k, v in parse_qs((await request.body()).decode()).items()}
    code, token, env = f.get("code", ""), "".join((f.get("token") or "").split()), f.get("env", "production")
    if not _code_ok(code):
        return HTMLResponse(PAGE.format(body='<div class="msg bad">This setup link has expired or was already used.</div>'), 403)
    env = "sandbox" if env == "sandbox" else "production"
    try:
        locs = (await _square("GET", "/v2/locations", token=token, env=env)).get("locations") or []
        live = [l for l in locs if l.get("status") == "ACTIVE"] or locs
        if not live:
            raise RuntimeError("no Square location on this account")
    except Exception as e:
        msg = html.escape(str(e))
        return HTMLResponse(PAGE.format(body=f'<div class="msg bad">Square didn\'t accept that token ({msg}). '
                                             f'Check you copied the {env} token and try again.</div>'
                                             f'<p><a style="color:#8fd1a8" href="/setup?code={html.escape(code)}">Back</a></p>'), 400)
    loc = live[0]
    async with _lock:
        st = _load()
        st["square"] = {"token": token, "env": env, "location": loc["id"],
                        "business": loc.get("business_name") or loc.get("name"), "set_at": time.time()}
        st["used_codes"].append(hashlib.sha256(code.encode()).hexdigest())
        _save()
    print(f"[pay] square token saved ({env}, location {loc['id']})", flush=True)
    biz = html.escape(str(st["square"]["business"] or "your business"))
    return PAGE.format(body=f'<div class="msg">Saved. Payments now go to <b>{biz}</b> ({env}). '
                            f'This link no longer works. You can close this page.</div>')
