"""
fetch_routes.py — allowlisted outbound HTTP fetch for the bridge.

WHY THIS IS NARROW
  This service holds a GitHub token with repo/workflow/delete_repo/admin:org
  scope and a Railway API token. An endpoint here that fetches an arbitrary URL
  is an SSRF hole pointed directly at those credentials — cloud metadata
  endpoints, internal Railway addresses, localhost admin surfaces. So this is an
  ALLOWLIST, never a blocklist:

    - only hosts in FETCH_ALLOWED_HOSTS (exact, case-insensitive match)
    - GET or POST only; a POST is never carried across a redirect
    - https only
    - every redirect hop re-checked against the same allowlist
    - DNS resolved up front; private, loopback, link-local, reserved and
      multicast addresses refused even if the host is allowlisted
    - hard timeout, hard response size cap, text out (never raw bytes)

  Widening FETCH_ALLOWED_HOSTS toward the open web is a separate, deliberate
  decision. Any agent that can reach untrusted pages through this endpoint must
  not also hold a commit tool.

RESIDUAL RISK (stated rather than papered over)
  The IP check resolves the hostname and validates the answers, then lets httpx
  connect by hostname. A DNS rebind between those two steps is not defeated by
  this design. For an allowlist of first-party Railway domains that is an
  acceptable gap; it would not be if the allowlist were opened up.

CONFIG
  FETCH_ALLOWED_HOSTS   comma-separated hostnames (default: the KalshiML dashboard)
  FETCH_MAX_BYTES       default 1048576, hard cap 5242880
  FETCH_TIMEOUT_S       default 15, hard cap 60
"""
from __future__ import annotations

import ipaddress
import os
import socket
import time
from typing import Optional
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

fetch_router = APIRouter(tags=["fetch"])

_DEFAULT_HOSTS = "kalshiml-production.up.railway.app"
HARD_MAX_BYTES = 5 * 1024 * 1024
HARD_MAX_TIMEOUT = 60.0
MAX_REDIRECTS = 3


def allowed_hosts() -> set[str]:
    raw = os.getenv("FETCH_ALLOWED_HOSTS", _DEFAULT_HOSTS)
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def _check_url(url: str) -> str:
    """Validate scheme, host allowlist and resolved IPs. Returns the hostname."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise HTTPException(400, f"only https is allowed, got {parts.scheme or 'no scheme'!r}")
    host = (parts.hostname or "").lower()
    if not host:
        raise HTTPException(400, "no host in url")
    allowed = allowed_hosts()
    if host not in allowed:
        raise HTTPException(
            403,
            f"host {host!r} is not in FETCH_ALLOWED_HOSTS. Allowed: {sorted(allowed)}",
        )
    try:
        infos = socket.getaddrinfo(host, parts.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise HTTPException(400, f"dns resolution failed for {host!r}: {e}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            raise HTTPException(403, f"{host!r} resolves to a non-public address ({ip}); refused")
    return host


class FetchRequest(BaseModel):
    url: str = Field(..., description="https URL whose host is in FETCH_ALLOWED_HOSTS")
    method: str = Field("GET", description="GET or POST")
    json_body: Optional[dict] = Field(None, description="JSON body, POST only")
    headers: Optional[dict] = Field(None, description="extra request headers, POST or GET")
    timeout_s: Optional[float] = Field(None, description=f"seconds, cap {HARD_MAX_TIMEOUT}")
    max_bytes: Optional[int] = Field(None, description=f"response cap, hard cap {HARD_MAX_BYTES}")


class FetchResponse(BaseModel):
    url: str
    method: str
    final_url: str
    status: int
    content_type: str
    bytes_read: int
    truncated: bool
    redirects: list[str]
    elapsed_ms: int
    text: str


@fetch_router.get("/fetch/allowed_hosts")
async def get_allowed_hosts():
    """What this bridge will currently fetch. Useful before guessing a URL."""
    return {"allowed_hosts": sorted(allowed_hosts()), "https_only": True,
            "max_redirects": MAX_REDIRECTS, "hard_max_bytes": HARD_MAX_BYTES}


@fetch_router.post("/fetch", response_model=FetchResponse)
async def fetch(req: FetchRequest):
    """Fetch an allowlisted URL and return its body as text."""
    # `or` would swallow an explicit 0 and silently substitute the default, so
    # these are None-checks: 0 is a caller error, not "unset".
    timeout_req = req.timeout_s if req.timeout_s is not None else float(os.getenv("FETCH_TIMEOUT_S", "15"))
    bytes_req = req.max_bytes if req.max_bytes is not None else int(os.getenv("FETCH_MAX_BYTES", str(1024 * 1024)))
    if timeout_req <= 0 or bytes_req <= 0:
        raise HTTPException(400, "timeout_s and max_bytes must be positive")
    timeout = min(float(timeout_req), HARD_MAX_TIMEOUT)
    max_bytes = min(int(bytes_req), HARD_MAX_BYTES)

    method = (req.method or "GET").upper()
    if method not in ("GET", "POST"):
        raise HTTPException(400, f"method must be GET or POST, got {method!r}")
    if req.json_body is not None and method != "POST":
        raise HTTPException(400, "json_body is only valid with method=POST")

    # Caller headers are merged over the default UA. Hop-by-hop and host
    # headers are dropped so a caller cannot retarget the request.
    _BANNED_HEADERS = {"host", "content-length", "connection", "transfer-encoding"}
    headers = {"User-Agent": "llm-bridge/fetch"}
    for k, v in (req.headers or {}).items():
        if k.lower() in _BANNED_HEADERS:
            raise HTTPException(400, f"header {k!r} may not be set")
        headers[str(k)] = str(v)

    started = time.time()
    current = req.url
    redirects: list[str] = []

    # Redirects are followed by hand so every hop passes the same allowlist.
    async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client:
        for _ in range(MAX_REDIRECTS + 1):
            _check_url(current)
            try:
                if method == "POST":
                    resp = await client.post(current, headers=headers, json=req.json_body)
                else:
                    resp = await client.get(current, headers=headers)
            except httpx.HTTPError as e:
                raise HTTPException(502, f"fetch failed: {type(e).__name__}: {e}")

            if resp.status_code in (301, 302, 303, 307, 308):
                if method == "POST":
                    raise HTTPException(
                        502,
                        f"refusing to follow a {resp.status_code} redirect on POST — "
                        "the method or body could change silently; call the final URL directly",
                    )
                location = resp.headers.get("location")
                if not location:
                    raise HTTPException(502, f"{resp.status_code} with no Location header")
                current = str(httpx.URL(current).join(location))
                redirects.append(current)
                continue

            body = resp.content[:max_bytes]
            return FetchResponse(
                url=req.url,
                method=method,
                final_url=current,
                status=resp.status_code,
                content_type=resp.headers.get("content-type", ""),
                bytes_read=len(body),
                truncated=len(resp.content) > max_bytes,
                redirects=redirects,
                elapsed_ms=int((time.time() - started) * 1000),
                text=body.decode("utf-8", errors="replace"),
            )

    raise HTTPException(502, f"too many redirects (max {MAX_REDIRECTS})")
