"""Per-person sign-in for the Video Studio Claude connector (OAuth 2.1 with PKCE).

A customer adds https://<host>/mcp as a connector in Claude. Claude registers itself
(dynamic client registration), sends the person to /video/oauth/authorize, where they sign
in with Face ID (or the owner PIN) and tap Allow. Claude then trades the one-time code for
an access token. Every connector call carries that token, and the token names exactly one
account, so a customer only ever sees and spends their own videos and credit.

The authorize page lives under /video so the app's sign-in cookies (path /video) reach it.

Storage: one JSON doc ({PREFIX}oauth.json) holding clients and hashed tokens; codes are
kept in memory for five minutes. Tokens are stored as SHA-256 hashes, never in the clear.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import secrets
import time
from urllib.parse import parse_qs, urlencode, urlparse

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

import video_app as va

oauth_router = APIRouter()
OAUTH_KEY = f"{va.PREFIX}oauth.json"
ACCESS_TTL = 30 * 86400      # access tokens last 30 days
REFRESH_TTL = 180 * 86400    # refresh tokens last 180 days and rotate on use
CODE_TTL = 300
_doc: dict | None = None
_codes: dict[str, dict] = {}
_consents: dict[str, dict] = {}


def _base(request: Request) -> str:
    return va._public_base(request)


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


async def _load() -> dict:
    global _doc
    if _doc is None:
        raw = await va._get(OAUTH_KEY)
        try:
            d = json.loads(raw) if raw else {}
        except Exception:
            d = {}
        _doc = {"clients": d.get("clients") or {}, "tokens": d.get("tokens") or {}}
    return _doc


async def _save() -> None:
    d = await _load()
    now = time.time()
    d["tokens"] = {k: v for k, v in d["tokens"].items() if v["exp"] > now}
    await va._put(OAUTH_KEY, json.dumps(d, separators=(",", ":")).encode(), "application/json")


def _redirect_ok(uri: str) -> bool:
    try:
        p = urlparse(uri)
    except Exception:
        return False
    if p.fragment:
        return False
    return p.scheme == "https" or (p.scheme == "http" and p.hostname in ("localhost", "127.0.0.1"))


# --------------------------------------------------------------------------- discovery
def _as_meta(request: Request) -> dict:
    b = _base(request)
    return {"issuer": b,
            "authorization_endpoint": f"{b}/video/oauth/authorize",
            "token_endpoint": f"{b}/video/oauth/token",
            "registration_endpoint": f"{b}/video/oauth/register",
            "revocation_endpoint": f"{b}/video/oauth/revoke",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
            "scopes_supported": ["studio"]}


@oauth_router.get("/.well-known/oauth-authorization-server")
async def as_metadata(request: Request):
    return _as_meta(request)


@oauth_router.get("/.well-known/oauth-authorization-server/{rest:path}")
async def as_metadata_path(rest: str, request: Request):
    return _as_meta(request)


@oauth_router.get("/.well-known/openid-configuration")
async def oidc_metadata(request: Request):
    return _as_meta(request)


def _pr_meta(request: Request) -> dict:
    b = _base(request)
    return {"resource": f"{b}/mcp", "authorization_servers": [b], "scopes_supported": ["studio"],
            "bearer_methods_supported": ["header"], "resource_name": "Video Studio"}


@oauth_router.get("/.well-known/oauth-protected-resource")
async def pr_metadata(request: Request):
    return _pr_meta(request)


@oauth_router.get("/.well-known/oauth-protected-resource/{rest:path}")
async def pr_metadata_path(rest: str, request: Request):
    return _pr_meta(request)


def challenge(request: Request) -> Response:
    """The 401 an unsigned connector call gets: it tells Claude where to sign in."""
    meta = f"{_base(request)}/.well-known/oauth-protected-resource"
    return JSONResponse({"error": "invalid_token", "error_description": "Sign in to Video Studio"},
                        status_code=401, headers={"WWW-Authenticate": f'Bearer resource_metadata="{meta}"'})


async def user_for_token(request: Request) -> dict | None:
    """The account an Authorization: Bearer token belongs to, or None."""
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        return None
    tok = auth[7:].strip()
    if not tok:
        return None
    rec = (await _load())["tokens"].get(_h(tok))
    if not rec or rec.get("kind") != "access" or rec["exp"] < time.time():
        return None
    if rec["uid"] == "owner":
        return await va._owner_rec()
    u = await va._user_by_id(rec["uid"])
    return u if u and not u.get("blocked") else None


# --------------------------------------------------------------------------- registration
@oauth_router.post("/video/oauth/register")
async def register(request: Request):
    try:
        b = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_client_metadata"}, status_code=400)
    uris = [str(u) for u in (b.get("redirect_uris") or []) if isinstance(u, str)][:10]
    if not uris or not all(_redirect_ok(u) for u in uris):
        return JSONResponse({"error": "invalid_redirect_uri",
                             "error_description": "redirect_uris must be https"}, status_code=400)
    d = await _load()
    if len(d["clients"]) > 5000:  # crude cap against junk registrations
        oldest = sorted(d["clients"].items(), key=lambda kv: kv[1].get("t", 0))[:1000]
        for k, _ in oldest:
            d["clients"].pop(k, None)
    cid = "vs_" + secrets.token_urlsafe(18)
    name = str(b.get("client_name") or "An app")[:80]
    d["clients"][cid] = {"redirect_uris": uris, "name": name, "t": time.time()}
    await _save()
    return JSONResponse({"client_id": cid, "client_name": name, "redirect_uris": uris,
                         "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
                         "token_endpoint_auth_method": "none", "client_id_issued_at": int(time.time())},
                        status_code=201)


# --------------------------------------------------------------------------- authorize
def _page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root{{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--muted:#6b6a66;--line:#e2e0da;--go:#1d1d1b;--goink:#fff;--err:#b3261e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--card:#1f1f1d;--ink:#f1f0ec;--muted:#a3a29c;--line:#34332f;--go:#f1f0ec;--goink:#151514}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 -apple-system,system-ui,sans-serif;
display:flex;min-height:100vh;align-items:center;justify-content:center;padding:20px;box-sizing:border-box}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px;max-width:380px;width:100%}}
h1{{font-size:20px;margin:0 0 8px}} p{{color:var(--muted);margin:0 0 16px}}
button,input{{font:inherit;width:100%;box-sizing:border-box;border-radius:10px;padding:12px;margin-top:8px}}
button{{border:0;background:var(--go);color:var(--goink);font-weight:600}}
button.ghost{{background:transparent;color:var(--ink);border:1px solid var(--line)}}
input{{border:1px solid var(--line);background:transparent;color:var(--ink)}}
.err{{color:var(--err);font-size:14px;min-height:1em;margin-top:8px}}
details{{margin-top:16px;color:var(--muted)}}
</style></head><body><div class="card">{body}</div></body></html>""",
                        headers={"Cache-Control": "no-store", "X-Frame-Options": "DENY"})


_LOGIN_JS = """
const unb=s=>{s=s.replace(/-/g,'+').replace(/_/g,'/');while(s.length%4)s+='=';return Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer};
const b64=b=>btoa(String.fromCharCode(...new Uint8Array(b))).replace(/\\+/g,'-').replace(/\\//g,'_').replace(/=+$/,'');
async function post(p,b){const r=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
 const d=await r.json().catch(()=>({}));if(!r.ok)throw new Error(d.detail||('Error '+r.status));return d}
async function faceId(){const e=document.getElementById('err');e.textContent='';
 try{const st=await post('/video/passkey/login/start');const o=st.options;
  const c=await navigator.credentials.get({publicKey:{...o,challenge:unb(o.challenge),allowCredentials:(o.allowCredentials||[]).map(x=>({...x,id:unb(x.id)}))}});
  const r=c.response;await post('/video/passkey/login/finish',{token:st.token,credential:{id:c.id,rawId:b64(c.rawId),type:c.type,
   clientExtensionResults:c.getClientExtensionResults?c.getClientExtensionResults():{},response:{clientDataJSON:b64(r.clientDataJSON),
   authenticatorData:b64(r.authenticatorData),signature:b64(r.signature),userHandle:r.userHandle?b64(r.userHandle):undefined}}});
  location.reload()}catch(x){e.textContent=x.name==='NotAllowedError'?'Face ID was cancelled.':x.message}}
async function pin(){const e=document.getElementById('err');e.textContent='';
 try{await post('/video/login',{pin:document.getElementById('pin').value});location.reload()}catch(x){e.textContent=x.message}}
"""


@oauth_router.get("/video/oauth/authorize")
async def authorize(request: Request):
    q = request.query_params
    d = await _load()
    client = d["clients"].get(q.get("client_id") or "")
    redirect = q.get("redirect_uri") or ""
    if not client or redirect not in client["redirect_uris"]:
        return _page("Can't connect", "<h1>Can't connect</h1><p>This connection request isn't valid. "
                                      "Remove the connector and add it again.</p>")
    state = q.get("state") or ""

    def back(**kw):
        return RedirectResponse(f"{redirect}{'&' if '?' in redirect else '?'}{urlencode({**kw, 'state': state})}", 302)

    if q.get("response_type") != "code":
        return back(error="unsupported_response_type")
    if q.get("code_challenge_method") != "S256" or len(q.get("code_challenge") or "") < 43:
        return back(error="invalid_request", error_description="PKCE S256 is required")

    u = await va._current_user(request)
    if not u:
        return _page("Sign in", f"""<h1>Sign in to Video Studio</h1>
<p>{html.escape(client['name'])} wants to make pictures and videos with your account.</p>
<button onclick="faceId()">Sign in with Face ID</button>
<div class="err" id="err"></div>
<p style="margin-top:16px">No account yet? <a href="/video" target="_blank" style="color:inherit">Create one in Video Studio</a>,
then come back here.</p>
<details><summary>Owner sign-in</summary><input id="pin" type="password" inputmode="numeric" placeholder="PIN">
<button class="ghost" onclick="pin()">Sign in with PIN</button></details>
<script>{_LOGIN_JS}</script>""")

    who = "your owner account" if u.get("owner") else html.escape(u.get("email") or "your account")
    pays = ("Videos it makes are paid from your owner credit at plain cost." if u.get("owner")
            else "Videos and pictures it makes are paid from your Video Studio credit, the same as in the app.")
    tok = secrets.token_urlsafe(24)
    now = time.time()
    for k in [k for k, v in _consents.items() if v["exp"] < now]:
        _consents.pop(k, None)
    _consents[tok] = {"exp": now + 600, "uid": u["id"], "client_id": q.get("client_id"), "redirect_uri": redirect,
                      "challenge": q.get("code_challenge"), "state": state, "resource": q.get("resource") or ""}
    return _page("Allow access", f"""<h1>Allow {html.escape(client['name'])}?</h1>
<p>It will be able to make, list and edit pictures and videos in {who}. {pays}
You can disconnect it any time from Account in Video Studio.</p>
<form method="post" action="/video/oauth/authorize">
<input type="hidden" name="consent" value="{tok}">
<button name="decision" value="allow">Allow</button>
<button class="ghost" name="decision" value="deny">Cancel</button></form>""")


@oauth_router.post("/video/oauth/authorize")
async def authorize_post(request: Request):
    form = {k: v[0] for k, v in parse_qs((await request.body()).decode()).items()}
    c = _consents.pop(form.get("consent") or "", None)
    if not c or c["exp"] < time.time():
        return _page("Expired", "<h1>That took too long</h1><p>Go back to Claude and connect again.</p>")
    u = await va._current_user(request)
    if not u or u["id"] != c["uid"]:  # the consent is bound to whoever saw it
        return _page("Sign in again", "<h1>Sign in again</h1><p>Go back to Claude and connect again.</p>")
    sep = "&" if "?" in c["redirect_uri"] else "?"
    if form.get("decision") != "allow":
        return RedirectResponse(f"{c['redirect_uri']}{sep}{urlencode({'error': 'access_denied', 'state': c['state']})}", 303)
    code = secrets.token_urlsafe(32)
    _codes[code] = {**c, "exp": time.time() + CODE_TTL}
    return RedirectResponse(f"{c['redirect_uri']}{sep}{urlencode({'code': code, 'state': c['state']})}", 303)


# --------------------------------------------------------------------------- tokens
def _tok_err(code: str, desc: str = "", status: int = 400) -> JSONResponse:
    return JSONResponse({"error": code, **({"error_description": desc} if desc else {})}, status_code=status,
                        headers={"Cache-Control": "no-store"})


async def _issue(uid: str, client_id: str) -> dict:
    d = await _load()
    access, refresh, now = secrets.token_urlsafe(32), secrets.token_urlsafe(32), time.time()
    d["tokens"][_h(access)] = {"kind": "access", "uid": uid, "client_id": client_id, "exp": now + ACCESS_TTL, "t": now}
    d["tokens"][_h(refresh)] = {"kind": "refresh", "uid": uid, "client_id": client_id, "exp": now + REFRESH_TTL, "t": now}
    await _save()
    return {"access_token": access, "token_type": "Bearer", "expires_in": ACCESS_TTL,
            "refresh_token": refresh, "scope": "studio"}


async def _form(request: Request) -> dict:
    raw = (await request.body()).decode()
    if (request.headers.get("content-type") or "").startswith("application/json"):
        try:
            return {k: str(v) for k, v in json.loads(raw).items()}
        except Exception:
            return {}
    return {k: v[0] for k, v in parse_qs(raw).items()}


@oauth_router.post("/video/oauth/token")
async def token(request: Request):
    f = await _form(request)
    d = await _load()
    grant = f.get("grant_type")
    if grant == "authorization_code":
        c = _codes.pop(f.get("code") or "", None)
        if not c or c["exp"] < time.time():
            return _tok_err("invalid_grant", "code expired or already used")
        if f.get("client_id") and f["client_id"] != c["client_id"]:
            return _tok_err("invalid_grant", "client mismatch")
        if f.get("redirect_uri") and f["redirect_uri"] != c["redirect_uri"]:
            return _tok_err("invalid_grant", "redirect_uri mismatch")
        verifier = f.get("code_verifier") or ""
        want = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        if not verifier or not secrets.compare_digest(want, c["challenge"]):
            return _tok_err("invalid_grant", "PKCE check failed")
        return JSONResponse(await _issue(c["uid"], c["client_id"]), headers={"Cache-Control": "no-store"})
    if grant == "refresh_token":
        rec = d["tokens"].pop(_h(f.get("refresh_token") or ""), None)  # one use: rotate
        if not rec or rec.get("kind") != "refresh" or rec["exp"] < time.time():
            await _save()
            return _tok_err("invalid_grant", "refresh token expired")
        if f.get("client_id") and f["client_id"] != rec["client_id"]:
            return _tok_err("invalid_grant", "client mismatch")
        return JSONResponse(await _issue(rec["uid"], rec["client_id"]), headers={"Cache-Control": "no-store"})
    return _tok_err("unsupported_grant_type")


@oauth_router.post("/video/oauth/revoke")
async def revoke(request: Request):
    f = await _form(request)
    d = await _load()
    if d["tokens"].pop(_h(f.get("token") or ""), None):
        await _save()
    return Response(status_code=200)


# --------------------------------------------------------------------------- account: connected apps
@oauth_router.get("/video/oauth/connections")
async def connections(request: Request):
    if (deny := await va._deny(request)):
        return deny
    d, uid, now = await _load(), request.state.user["id"], time.time()
    seen: dict[str, dict] = {}
    for rec in d["tokens"].values():
        if rec["uid"] == uid and rec["exp"] > now:
            c = seen.setdefault(rec["client_id"], {"client_id": rec["client_id"], "since": rec.get("t", now),
                                                    "name": (d["clients"].get(rec["client_id"]) or {}).get("name") or "App"})
            c["since"] = min(c["since"], rec.get("t", now))
    return {"connections": sorted(seen.values(), key=lambda c: c["since"])}


@oauth_router.post("/video/oauth/disconnect")
async def disconnect(request: Request):
    """Sign out every connector this person allowed (or one client_id)."""
    if (deny := await va._deny(request)):
        return deny
    try:
        b = await request.json()
    except Exception:
        b = {}
    d, uid, cid = await _load(), request.state.user["id"], str(b.get("client_id") or "")
    before = len(d["tokens"])
    d["tokens"] = {k: v for k, v in d["tokens"].items()
                   if not (v["uid"] == uid and (not cid or v["client_id"] == cid))}
    await _save()
    return {"removed": before - len(d["tokens"])}
