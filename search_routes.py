"""
search_routes.py — web search for the bridge, via DuckDuckGo's HTML endpoint.

WHY THIS EXISTS SEPARATELY FROM fetch_routes.py
  fetch_routes.py is an allowlisted fetch because this service holds
  admin-scoped GitHub and Railway tokens; an arbitrary-URL fetch would be an
  SSRF hole. Search is different: the ONLY upstream host this module contacts
  is hardcoded below (DuckDuckGo). The caller supplies a query string, never
  a URL, so there is no SSRF surface here at all.

  Search RESULTS contain arbitrary URLs, but they are returned as data. To
  actually read one, a caller must pass it through /fetch, which applies the
  FETCH_ALLOWED_HOSTS allowlist as before. Widening that allowlist remains a
  separate, deliberate decision.

  DuckDuckGo HTML search needs no API key and no JavaScript. It rate-limits
  aggressively under automation; callers should keep query volume low and
  tolerate 202/empty responses as "back off", not retry storms.

CONFIG
  SEARCH_TIMEOUT_S    default 15, hard cap 30
  SEARCH_MAX_RESULTS  default 8, hard cap 20
"""
from __future__ import annotations

import html
import os
import re
import time
from typing import Optional
from urllib.parse import parse_qs, urlsplit

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

search_router = APIRouter(tags=["search"])

# Hardcoded upstream — the caller never controls this URL.
DDG_HTML_URL = "https://html.duckduckgo.com/html/"
HARD_MAX_TIMEOUT = 30.0
HARD_MAX_RESULTS = 20

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# DDG HTML results: <a class="result__a" href="//duckduckgo.com/l/?uddg=<enc>...">
# and a snippet in <a class="result__snippet"> or <td class="result-snippet">.
_RESULT_A_RE = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_SNIPPET_RE = re.compile(
    r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(s: str) -> str:
    return html.unescape(_TAG_RE.sub("", s)).strip()


def _unwrap_ddg_redirect(href: str) -> str:
    """DDG wraps result hrefs in /l/?uddg=<urlencoded target>. Unwrap; if the
    link is already direct, return it unchanged. Never return a non-http(s)
    scheme to the caller."""
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    parts = urlsplit(href)
    if "duckduckgo.com" in (parts.hostname or "") and parts.path.startswith("/l/"):
        uddg = parse_qs(parts.query).get("uddg")
        if uddg:
            href = uddg[0]
            parts = urlsplit(href)
    if parts.scheme not in ("http", "https"):
        return ""
    return href


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=400)
    max_results: Optional[int] = Field(None, description=f"hard cap {HARD_MAX_RESULTS}")
    timeout_s: Optional[float] = Field(None, description=f"hard cap {HARD_MAX_TIMEOUT}")


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    n: int
    elapsed_ms: int
    note: str = ""


@search_router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Search the web via DuckDuckGo and return titles, URLs and snippets."""
    max_results = min(
        req.max_results if req.max_results is not None
        else int(os.getenv("SEARCH_MAX_RESULTS", "8")),
        HARD_MAX_RESULTS,
    )
    timeout = min(
        req.timeout_s if req.timeout_s is not None
        else float(os.getenv("SEARCH_TIMEOUT_S", "15")),
        HARD_MAX_TIMEOUT,
    )
    if max_results <= 0 or timeout <= 0:
        raise HTTPException(400, "max_results and timeout_s must be positive")

    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.post(
                DDG_HTML_URL,
                data={"q": req.query},
                headers={"User-Agent": _UA},
            )
    except httpx.HTTPError as e:
        raise HTTPException(502, f"search failed: {type(e).__name__}: {e}")

    if resp.status_code == 202 or resp.status_code >= 500:
        # DDG's rate-limit / anomaly responses — surface as empty, not error.
        return SearchResponse(
            query=req.query, results=[], n=0,
            elapsed_ms=int((time.time() - started) * 1000),
            note=f"duckduckgo returned {resp.status_code}; back off and retry later",
        )
    if resp.status_code >= 400:
        raise HTTPException(502, f"duckduckgo returned {resp.status_code}")

    body = resp.text
    links = _RESULT_A_RE.findall(body)
    snippets = _SNIPPET_RE.findall(body)

    results: list[SearchResult] = []
    for i, (href, title_html) in enumerate(links):
        url = _unwrap_ddg_redirect(href)
        if not url:
            continue
        results.append(SearchResult(
            title=_strip_tags(title_html),
            url=url,
            snippet=_strip_tags(snippets[i]) if i < len(snippets) else "",
        ))
        if len(results) >= max_results:
            break

    return SearchResponse(
        query=req.query,
        results=results,
        n=len(results),
        elapsed_ms=int((time.time() - started) * 1000),
    )


@search_router.get("/search", response_model=SearchResponse)
async def search_get(q: str, max_results: Optional[int] = None):
    """GET convenience wrapper: /search?q=...&max_results=5"""
    return await search(SearchRequest(query=q, max_results=max_results))
