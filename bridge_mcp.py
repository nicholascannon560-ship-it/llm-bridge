"""MCP connector for the bridge.

Exposes the bridge's GitHub / Railway / LLM / issues endpoints as MCP tools so
Claude (or any MCP client) can use the bridge as a custom connector, with no
bridge key to upload or rotate.

URL:   POST /mcp/<BRIDGE_MCP_SECRET>   (Streamable HTTP, JSON responses)
Auth:  the secret in the path. It is separate from BRIDGE_API_KEY, so key
       rotation never breaks the connector. Unset secret = connector off (404).

Tools don't reimplement anything: each one calls the bridge's own HTTP routes
in-process, tagged with a per-process internal token that BridgeAuthMiddleware
accepts. Anyone holding the connector URL has the same power as the bridge key,
so keep the URL private; change BRIDGE_MCP_SECRET to revoke it.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
from typing import Any

import httpx
from fastapi import APIRouter, Request
from starlette.responses import JSONResponse, Response

# Per-process token for the connector's in-process calls. Never leaves memory.
INTERNAL_TOKEN = secrets.token_hex(32)
INTERNAL_HEADER = "x-bridge-internal"

OWNER = os.getenv("GITHUB_OWNER", "nicholascannon560-ship-it")
MAX_OUT = 60_000
PROTOCOLS = ["2025-06-18", "2025-03-26", "2024-11-05"]

router = APIRouter(tags=["mcp"])


def _secret_ok(token: str) -> bool:
    want = os.getenv("BRIDGE_MCP_SECRET") or ""
    return len(want) >= 24 and hmac.compare_digest(token.encode(), want.encode())


def _s(desc: str, **props: Any) -> dict:
    req = [k for k, v in props.items() if v.pop("required", False)]
    return {"type": "object", "properties": props, "required": req, "description": desc}


S = lambda d, **kw: {"type": "string", "description": d, **kw}  # noqa: E731
I = lambda d, **kw: {"type": "integer", "description": d, **kw}  # noqa: E731

TOOLS: list[dict] = [
    {"name": "bridge_health", "description": "Bridge status: version, key age, command queue counters.",
     "inputSchema": _s("")},
    {"name": "list_repos", "description": f"List GitHub repos the bridge can reach (owner {OWNER}).",
     "inputSchema": _s("")},
    {"name": "read_file",
     "description": "Read a file, or list a directory, in a GitHub repo. Use path '' for the repo root.",
     "inputSchema": _s("", repo=S("Repo name, e.g. llm-bridge", required=True),
                       path=S("File or directory path", required=True),
                       ref=S("Branch or commit (default: repo default branch)"),
                       owner=S(f"Repo owner (default {OWNER})"))},
    {"name": "commit_files",
     "description": "Write one or more files to a repo as ONE commit (creates or overwrites). "
                    "Read the current file first; this replaces whole files. Check which branch deploys "
                    "(llm-bridge deploys from Agent-loop, not main).",
     "inputSchema": _s("", repo=S("Repo name", required=True),
                       message=S("Commit message", required=True),
                       files={"type": "array", "required": True, "description": "Files to write",
                              "items": {"type": "object", "required": ["path", "content"],
                                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}},
                       branch=S("Branch (default: repo default branch)"),
                       owner=S(f"Repo owner (default {OWNER})"))},
    {"name": "list_issues", "description": "List issues in a repo.",
     "inputSchema": _s("", repo=S("Repo name", required=True), state=S("open, closed or all"),
                       owner=S(f"Repo owner (default {OWNER})"))},
    {"name": "create_issue", "description": "Open an issue in a repo.",
     "inputSchema": _s("", repo=S("Repo name", required=True), title=S("Title", required=True),
                       body=S("Body (markdown)"), owner=S(f"Repo owner (default {OWNER})"))},
    {"name": "issue_comments", "description": "Read the comments on an issue.",
     "inputSchema": _s("", repo=S("Repo name", required=True), number=I("Issue number", required=True),
                       owner=S(f"Repo owner (default {OWNER})"))},
    {"name": "add_issue_comment",
     "description": "Comment on an issue (e.g. log to change-log: issue 1 KalshiML, 2 llm-bridge, 3 contractor-payroll, "
                    "4 tali, 5 KimiApiBridge-, 6 resolume-bridge).",
     "inputSchema": _s("", repo=S("Repo name", required=True), number=I("Issue number", required=True),
                       body=S("Comment (markdown)", required=True), owner=S(f"Repo owner (default {OWNER})"))},
    {"name": "map_freshness", "description": "Per-slice MAP.md freshness check for one repo (or all).",
     "inputSchema": _s("", repo=S("Repo name (omit for all)"))},
    {"name": "railway_project_services", "description": "List the services in a Railway project.",
     "inputSchema": _s("", project_id=S("Railway project id", required=True))},
    {"name": "railway_service_status", "description": "Latest deployment status of a Railway service.",
     "inputSchema": _s("", service_id=S("Railway service id", required=True))},
    {"name": "railway_deployments", "description": "Recent deployments of a Railway service, newest first.",
     "inputSchema": _s("", service_id=S("Railway service id", required=True))},
    {"name": "railway_logs", "description": "Logs for one Railway deployment.",
     "inputSchema": _s("", deployment_id=S("Deployment id", required=True))},
    {"name": "railway_redeploy", "description": "Redeploy a Railway service's latest build now.",
     "inputSchema": _s("", service_id=S("Railway service id", required=True))},
    {"name": "railway_graphql",
     "description": "Run any Railway GraphQL query or mutation. Unscoped: it can touch every project. "
                    "Introspect before mutating.",
     "inputSchema": _s("", query=S("GraphQL document", required=True),
                       variables={"type": "object", "description": "GraphQL variables"})},
    {"name": "llm_chat",
     "description": "Ask another model through the bridge's LLM gateway (e.g. provider 'moonshot' for Kimi). "
                    "Returns the reply text and usage.",
     "inputSchema": _s("", provider=S("Provider, e.g. moonshot, anthropic, openrouter", required=True),
                       model=S("Model id", required=True),
                       messages={"type": "array", "required": True,
                                 "description": "Chat messages [{role, content}]",
                                 "items": {"type": "object"}},
                       max_tokens=I("Max output tokens"),
                       reasoning_effort=S("low, medium or high (models that support it)"))},
    {"name": "bridge_request",
     "description": "Call any other bridge endpoint directly (see GET /openapi.json for the full list: aws, agent, "
                    "skills, patch, fetch, ...). Path must start with '/'.",
     "inputSchema": _s("", method=S("GET, POST, PATCH, PUT or DELETE", required=True),
                       path=S("Path including any query string, e.g. /llm/usage?days=7", required=True),
                       body={"type": "object", "description": "JSON body for POST/PATCH/PUT"})},
]


async def _call(app, method: str, path: str, body: Any = None) -> Any:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bridge.internal", timeout=180) as c:
        r = await c.request(method, path, json=body, headers={INTERNAL_HEADER: INTERNAL_TOKEN})
    try:
        data = r.json()
    except Exception:
        data = r.text
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {json.dumps(data)[:1500] if not isinstance(data, str) else data[:1500]}")
    return data


async def _tool(app, name: str, a: dict) -> Any:
    owner = a.get("owner") or OWNER
    q = lambda v: httpx.QueryParams(v)  # noqa: E731
    if name == "bridge_health":
        return await _call(app, "GET", "/health")
    if name == "list_repos":
        return await _call(app, "GET", "/repos")
    if name == "read_file":
        path = str(a.get("path") or "").lstrip("/")
        url = f"/contents/{owner}/{a['repo']}/{path}"
        if a.get("ref"):
            url += f"?{q({'ref': a['ref']})}"
        data = await _call(app, "GET", url)
        if isinstance(data, dict) and "entries" in data:
            return {"path": path, "entries": [{"name": e.get("name"), "type": e.get("type"), "path": e.get("path")}
                                              for e in data["entries"]]}
        if isinstance(data, dict) and "content" in data:
            return {"path": data.get("path", path), "sha": data.get("sha"), "size": data.get("size"),
                    "content": data.get("content") or ""}
        return data
    if name == "commit_files":
        body = {"owner": owner, "repo": a["repo"], "message": a["message"], "files": a["files"]}
        if a.get("branch"):
            body["branch"] = a["branch"]
        return await _call(app, "POST", "/commit_tree", body)
    if name == "list_issues":
        return await _call(app, "GET", f"/issues/{owner}/{a['repo']}?{q({'state': a.get('state') or 'open'})}")
    if name == "create_issue":
        return await _call(app, "POST", f"/issues/{owner}/{a['repo']}",
                           {"title": a["title"], "body": a.get("body") or ""})
    if name == "issue_comments":
        return await _call(app, "GET", f"/issues/{owner}/{a['repo']}/{int(a['number'])}/comments?per_page=100")
    if name == "add_issue_comment":
        return await _call(app, "POST", f"/issues/{owner}/{a['repo']}/{int(a['number'])}/comments",
                           {"body": a["body"]})
    if name == "map_freshness":
        return await _call(app, "GET", "/map_freshness" + (f"?{q({'repo': a['repo']})}" if a.get("repo") else ""))
    if name == "railway_project_services":
        return await _call(app, "GET", f"/railway/project/{a['project_id']}/services")
    if name == "railway_service_status":
        return await _call(app, "GET", f"/railway/service/{a['service_id']}/status")
    if name == "railway_deployments":
        return await _call(app, "GET", f"/railway/service/{a['service_id']}/deployments")
    if name == "railway_logs":
        return await _call(app, "GET", f"/railway/deployment/{a['deployment_id']}/logs")
    if name == "railway_redeploy":
        return await _call(app, "POST", f"/railway/service/{a['service_id']}/redeploy")
    if name == "railway_graphql":
        return await _call(app, "POST", "/railway/gql", {"query": a["query"], "variables": a.get("variables") or {}})
    if name == "llm_chat":
        body = {k: a[k] for k in ("provider", "model", "messages", "max_tokens", "reasoning_effort") if a.get(k)}
        return await _call(app, "POST", "/llm/chat", body)
    if name == "bridge_request":
        method = str(a.get("method") or "GET").upper()
        path = str(a.get("path") or "")
        if method not in ("GET", "POST", "PATCH", "PUT", "DELETE") or not path.startswith("/"):
            raise ValueError("method must be GET/POST/PATCH/PUT/DELETE and path must start with '/'")
        if path.startswith("/mcp") or path.startswith("/rotate_key"):
            raise ValueError("that path isn't available through the connector")
        return await _call(app, method, path, a.get("body") if method != "GET" else None)
    raise ValueError(f"unknown tool {name}")


def _text(out: Any) -> str:
    s = out if isinstance(out, str) else json.dumps(out, indent=1, default=str)
    return s if len(s) <= MAX_OUT else s[:MAX_OUT] + f"\n... [truncated, {len(s)} chars total]"


async def _one(app, msg: dict) -> dict | None:
    if not isinstance(msg, dict) or msg.get("id") is None:
        return None  # notification
    mid, method, params = msg["id"], msg.get("method"), msg.get("params") or {}
    ok = lambda r: {"jsonrpc": "2.0", "id": mid, "result": r}  # noqa: E731
    err = lambda c, m: {"jsonrpc": "2.0", "id": mid, "error": {"code": c, "message": m}}  # noqa: E731
    if method == "initialize":
        want = params.get("protocolVersion")
        return ok({"protocolVersion": want if want in PROTOCOLS else PROTOCOLS[0],
                   "capabilities": {"tools": {"listChanged": False}},
                   "serverInfo": {"name": "llm-bridge", "title": "LLM Bridge", "version": "1.0.0"},
                   "instructions": "Nicholas's bridge: GitHub repos, Railway, issues/change-log and an LLM gateway. "
                                   "Read files before committing; llm-bridge deploys from branch Agent-loop."})
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": TOOLS})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if not any(t["name"] == name for t in TOOLS):
            return err(-32602, f"unknown tool {name}")
        try:
            out = await _tool(app, name, args)
            print(f"[mcp] {name} ok", flush=True)
            return ok({"content": [{"type": "text", "text": _text(out)}], "isError": False})
        except Exception as e:  # tool errors go back to the model, not as protocol errors
            print(f"[mcp] {name} failed: {e!r}"[:300], flush=True)
            return ok({"content": [{"type": "text", "text": str(e)[:4000]}], "isError": True})
    return err(-32601, f"method not found: {method}")


@router.api_route("/mcp/{token}", methods=["GET", "POST", "DELETE"], include_in_schema=False)
async def mcp(token: str, request: Request):
    if not _secret_ok(token):
        return Response("Not found", status_code=404)
    if request.method != "POST":
        return Response(status_code=405, headers={"Allow": "POST"})
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}, 400)
    app = request.app
    if isinstance(body, list):
        out = [r for r in [await _one(app, m) for m in body] if r]
        return JSONResponse(out) if out else Response(status_code=202)
    r = await _one(app, body)
    return JSONResponse(r, headers={"Cache-Control": "no-store"}) if r else Response(status_code=202)
