"""Tests for search_routes.py — parsing, unwrap, caps, guards."""
import os, sys
from fastapi import FastAPI
from fastapi.testclient import TestClient
import search_routes

app = FastAPI()
app.include_router(search_routes.search_router)
c = TestClient(app, raise_server_exceptions=False)

PASS = FAIL = 0
def check(label, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS  {label}")
    else:
        FAIL += 1; print(f"  FAIL  {label}  {extra}")

print("\n== unwrap ==")
u = search_routes._unwrap_ddg_redirect("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc")
check("uddg unwrapped", u == "https://example.com/page", u)
u = search_routes._unwrap_ddg_redirect("https://example.com/direct")
check("direct link kept", u == "https://example.com/direct", u)
u = search_routes._unwrap_ddg_redirect("javascript:alert(1)")
check("non-http scheme dropped", u == "", u)
u = search_routes._unwrap_ddg_redirect("//duckduckgo.com/l/?uddg=ftp%3A%2F%2Fevil.net%2Fx")
check("ftp inside uddg dropped", u == "", u)

print("\n== tag stripping ==")
check("tags stripped", search_routes._strip_tags("<b>Hello</b> &amp; <i>bye</i>") == "Hello & bye")

print("\n== param guards ==")
r = c.post("/search", json={"query": ""})
check("empty query -> 422", r.status_code == 422, r.status_code)
r = c.post("/search", json={"query": "x", "max_results": 0})
check("zero max_results -> 400", r.status_code == 400, (r.status_code, r.json()))
r = c.post("/search", json={"query": "x", "timeout_s": 0})
check("zero timeout -> 400", r.status_code == 400, (r.status_code, r.json()))

print("\n== live search ==")
r = c.post("/search", json={"query": "duckduckgo", "max_results": 5})
j = r.json()
check("200", r.status_code == 200, (r.status_code, j))
if r.status_code == 200:
    if j.get("note"):
        print(f"  note: {j['note']} (rate-limited from sandbox IP is acceptable)")
        check("graceful rate-limit note", j["n"] == 0, j)
    else:
        check("got results", j["n"] > 0, j)
        if j["n"]:
            first = j["results"][0]
            check("title present", bool(first["title"]), first)
            check("url is http(s)", first["url"].startswith("http"), first["url"])
        check("max_results respected", j["n"] <= 5, j["n"])

print("\n== GET wrapper ==")
r = c.get("/search", params={"q": "fastapi", "max_results": 3})
check("GET works", r.status_code == 200, r.status_code)

print(f"\n{'='*46}\n  {PASS} passed, {FAIL} failed\n{'='*46}")
sys.exit(1 if FAIL else 0)
