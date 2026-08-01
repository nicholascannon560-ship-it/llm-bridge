"""Local tests for fetch_routes.py — allowlist, SSRF guards, caps, real fetch."""
import os, sys

os.environ["FETCH_ALLOWED_HOSTS"] = "raw.githubusercontent.com, api.github.com, localhost, EXAMPLE-Uppercase.com"

from fastapi import FastAPI
from fastapi.testclient import TestClient
import fetch_routes

app = FastAPI()
app.include_router(fetch_routes.fetch_router)
c = TestClient(app, raise_server_exceptions=False)

RAW = "https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore"

PASS = FAIL = 0
def check(label, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS  {label}")
    else:
        FAIL += 1; print(f"  FAIL  {label}  {extra}")

def post(url, **kw):
    r = c.post("/fetch", json={"url": url, **kw})
    return r.status_code, r.json()

print("\n== allowed_hosts introspection ==")
r = c.get("/fetch/allowed_hosts"); j = r.json()
check("200", r.status_code == 200, j)
check("hosts normalized to lowercase",
      j["allowed_hosts"] == sorted(["raw.githubusercontent.com", "api.github.com", "localhost", "example-uppercase.com"]), j)
check("https_only advertised", j["https_only"] is True)

print("\n== scheme guard ==")
for u, why in [("http://api.github.com/", "plain http"),
               ("file:///etc/passwd", "file scheme"),
               ("ftp://api.github.com/x", "ftp scheme"),
               ("api.github.com/x", "no scheme")]:
    s, j = post(u)
    check(f"{why} -> 400", s == 400, (s, j))

print("\n== allowlist guard ==")
for u, why in [("https://evil.example.net/x", "off-allowlist host"),
               ("https://169.254.169.254/latest/meta-data/", "cloud metadata IP"),
               ("https://api.github.com.evil.net/x", "suffix-spoofed host"),
               ("https://sub.api.github.com/x", "subdomain of an allowed host")]:
    s, j = post(u)
    check(f"{why} -> 403", s == 403, (s, j))

print("\n== private-address guard (host IS allowlisted) ==")
s, j = post("https://localhost/x")
check("localhost resolves private -> 403", s == 403 and "non-public" in j.get("detail", ""), (s, j))

print("\n== bad params ==")
for kw, why in [({"timeout_s": 0}, "zero timeout"), ({"max_bytes": 0}, "zero max_bytes")]:
    s, j = post(RAW, **kw)
    check(f"{why} -> 400", s == 400, (s, j))

print("\n== real fetch (raw.githubusercontent.com is reachable from this sandbox) ==")
s, j = post(RAW, max_bytes=4096)
check("200", s == 200, (s, j))
if s == 200:
    check("status 200 from origin", j["status"] == 200, j["status"])
    check("content type present", bool(j["content_type"]), j["content_type"])
    check("body present", len(j["text"]) > 50, len(j.get("text", "")))
    check("elapsed recorded", j["elapsed_ms"] >= 0)
    check("no redirects", j["redirects"] == [], j["redirects"])

print("\n== size cap ==")
s, j = post(RAW, max_bytes=32)
check("truncated flagged", s == 200 and j["truncated"] is True and j["bytes_read"] == 32, (s, j))
s, j = post(RAW, max_bytes=99_999_999)
check("max_bytes clamped to hard cap", s == 200 and j["bytes_read"] <= fetch_routes.HARD_MAX_BYTES)

print("\n== 404 passthrough, not an exception ==")
s, j = post("https://raw.githubusercontent.com/github/gitignore/main/NoSuchFile.xyz")
check("upstream 404 surfaces as 200 with status=404", s == 200 and j["status"] == 404, (s, j))

print("\n== default allowlist when env unset ==")
saved = os.environ.pop("FETCH_ALLOWED_HOSTS")
check("defaults to the KalshiML dashboard",
      fetch_routes.allowed_hosts() == {"kalshiml-production.up.railway.app"},
      fetch_routes.allowed_hosts())
os.environ["FETCH_ALLOWED_HOSTS"] = saved

print("\n== method guard ==")
s, j = post(RAW, method="PUT")
check("PUT rejected", s == 400, (s, j))
s, j = post(RAW, method="DELETE")
check("DELETE rejected", s == 400, (s, j))
s, j = post(RAW, json_body={"a": 1})
check("json_body on GET rejected", s == 400, (s, j))

print("\n== header guard ==")
for h, why in [({"Host": "evil.net"}, "Host"), ({"Content-Length": "0"}, "Content-Length"),
               ({"Connection": "close"}, "Connection")]:
    s, j = post(RAW, headers=h)
    check(f"{why} header rejected", s == 400, (s, j))
s, j = post(RAW, headers={"X-Custom": "ok"})
check("ordinary custom header allowed", s == 200, (s, j))

print("\n== POST still allowlist-gated ==")
s, j = post("https://evil.example.net/x", method="POST", json_body={"a": 1})
check("off-allowlist POST -> 403", s == 403, (s, j))
s, j = post("http://api.github.com/", method="POST", json_body={})
check("http POST -> 400", s == 400, (s, j))

print("\n== POST reaches the wire ==")
s, j = post("https://api.github.com/graphql", method="POST", json_body={"query": "{viewer{login}}"})
check("POST executed, upstream auth error surfaced", s == 200 and j["method"] == "POST", (s, j))
if s == 200:
    # 401 unauthenticated, or 403 when this sandbox IP is rate-limited — either
    # proves the POST reached GitHub rather than being swallowed locally.
    check("origin rejected the unauthenticated POST", j["status"] in (401, 403), j["status"])

print(f"\n{'='*46}\n  {PASS} passed, {FAIL} failed\n{'='*46}")
sys.exit(1 if FAIL else 0)
