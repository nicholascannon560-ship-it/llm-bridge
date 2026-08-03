"""Tool registry for the autonomous agent harness.

Each tool has:
  - An OpenAI-compatible function schema (for the LLM)
  - An async execution handler (for the bridge to run)

All handlers assume they are running inside the bridge process where
llm_gateway and railway_extension are importable.
"""

import base64
import json
import os
import traceback
from typing import Any, Dict, List
from urllib.parse import urlparse
from datetime import datetime, timezone

# Import bridge modules when running inside the bridge
try:
    from llm_gateway import ChatRequest, ChatMessage, get_router
    from railway_extension import (
        railway_query,
        set_service_variable,
        list_projects,
        list_services,
        get_service_status,
        get_logs,
        redeploy_service,
        BRIDGE_SERVICE_ID,
    )
    BRIDGE_MODE = True
except ImportError:
    BRIDGE_MODE = False

GITHUB_API = "https://api.github.com"
OWNER = os.getenv("GITHUB_OWNER", "nicholascannon560-ship-it")

# Hard allow-list for http_get. Agents may only reach these hosts.
HTTP_ALLOW_HOSTS = {
    "kalshiml-production.up.railway.app",
}

HTTP_MAX_BYTES = int(os.getenv("AGENT_HTTP_MAX_BYTES", str(1_500_000)))


def _github_headers() -> dict:
    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not available")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


# ── Tool Schemas ─────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "github_read",
            "description": "Read a file or list a directory from a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repo owner (default: nicholascannon560-ship-it)"},
                    "repo": {"type": "string", "description": "Repo name"},
                    "path": {"type": "string", "description": "File or directory path"},
                    "branch": {"type": "string", "description": "Branch or ref (default: repo default)"}
                },
                "required": ["repo", "path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "github_commit",
            "description": "Commit a single file to a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repo owner"},
                    "repo": {"type": "string", "description": "Repo name"},
                    "path": {"type": "string", "description": "File path inside repo"},
                    "content": {"type": "string", "description": "Raw file content"},
                    "message": {"type": "string", "description": "Git commit message"},
                    "branch": {"type": "string", "description": "Branch (default: repo default)"}
                },
                "required": ["repo", "path", "content", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "railway_redeploy",
            "description": "Redeploy a Railway service (triggers a new build + deploy).",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_id": {"type": "string", "description": "Railway service UUID (default: bridge service)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "railway_set_env",
            "description": "Set or update an environment variable on a Railway service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Variable name"},
                    "value": {"type": "string", "description": "Variable value"},
                    "service_id": {"type": "string", "description": "Service ID (default: bridge)"}
                },
                "required": ["name", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "railway_get_status",
            "description": "Get the latest deployment status of a Railway service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_id": {"type": "string", "description": "Service ID (default: bridge)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "railway_get_logs",
            "description": "Get recent logs for a specific Railway deployment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "deployment_id": {"type": "string", "description": "Deployment UUID"},
                    "limit": {"type": "integer", "description": "Max lines", "default": 100}
                },
                "required": ["deployment_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "llm_chat",
            "description": "Ask a sub-question to an LLM. Use for analysis, planning, code review, or breaking down complex tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The question or task"},
                    "provider": {"type": "string", "default": "moonshot"},
                    "model": {"type": "string", "default": "kimi-k3"},
                    "max_tokens": {"type": "integer", "default": 2048},
                    "reasoning_effort": {"type": "string", "default": "low", "description": "low | high | max"}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_memory",
            "description": "Write a reflection, lesson learned, or decision rule to the agent's persistent memory. Use after successes, failures, or insights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entry": {"type": "string", "description": "The reflection text"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "e.g. [\"success\", \"kalshiml\", \"deployment\"]"}
                },
                "required": ["entry"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "Read recent agent memory entries to inform current decisions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
                    "tag": {"type": "string", "description": "Filter by tag"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "github_read_issue",
            "description": "Read comments from a GitHub issue (used for project change-logs).",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repo owner"},
                    "repo": {"type": "string", "description": "Repo name"},
                    "issue_number": {"type": "integer", "description": "Issue number"},
                    "per_page": {"type": "integer", "default": 30}
                },
                "required": ["repo", "issue_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": (
                "Sandboxed HTTP GET. Currently allow-listed only to the KalshiML production dashboard "
                "(kalshiml-production.up.railway.app). Use this to read the full live data surface."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL (must be https://kalshiml-production.up.railway.app/...)"
                    },
                    "timeout_sec": {
                        "type": "integer",
                        "default": 25,
                        "description": "Request timeout in seconds (max 60)"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_research",
            "description": (
                "Use a real browser (Browser Use + Browserbase) to research a topic, extract information, "
                "or complete limited multi-step web tasks. Creates a fresh Browserbase session on each call. "
                "Supports read_only (default) and elevated modes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Clear natural language instruction for what to research or extract."
                    },
                    "start_url": {
                        "type": "string",
                        "description": "Optional starting URL."
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["read_only", "elevated"],
                        "default": "read_only",
                        "description": "read_only = browse + extract only. elevated = limited interaction allowed."
                    },
                    "max_steps": {
                        "type": "integer",
                        "default": 12
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "default": 180
                    }
                },
                "required": ["task"]
            }
        }
    },
]


DEFAULT_TOOLS = TOOL_SCHEMAS


# ── Tool Handlers ────────────────────────────────────────────────────────────

async def run_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if not BRIDGE_MODE:
        return {"error": "Agent tools require bridge environment"}

    handler = _TOOL_HANDLERS.get(name)
    if not handler:
        return {"error": f"Unknown tool: {name}"}

    try:
        return await handler(arguments)
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()[-800:]}


async def _tool_github_read(args: Dict) -> Dict:
    import httpx
    owner = args.get("owner", OWNER)
    repo = args["repo"]
    path = args["path"]
    branch = args.get("branch")

    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": branch} if branch else None

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, headers=_github_headers(), params=params)

    if resp.status_code != 200:
        return {"error": f"GitHub API {resp.status_code}", "detail": resp.text[:300]}

    data = resp.json()
    if isinstance(data, list):
        return {"type": "directory", "entries": [{"name": e["name"], "type": e["type"]} for e in data]}

    content = ""
    if data.get("encoding") == "base64" and data.get("content"):
        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")

    return {
        "type": "file",
        "path": data["path"],
        "sha": data.get("sha"),
        "size": data.get("size"),
        "content": content[:8000]
    }


async def _tool_github_commit(args: Dict) -> Dict:
    import httpx
    owner = args.get("owner", OWNER)
    repo = args["repo"]
    path = args["path"]
    content = args["content"]
    message = args["message"]
    branch = args.get("branch")

    contents_path = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"

    existing_sha = None
    params = {"ref": branch} if branch else None
    async with httpx.AsyncClient(timeout=20) as client:
        lookup = await client.get(contents_path, headers=_github_headers(), params=params)
    if lookup.status_code == 200:
        existing_sha = lookup.json().get("sha")

    payload = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii")
    }
    if branch:
        payload["branch"] = branch
    if existing_sha:
        payload["sha"] = existing_sha

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.put(contents_path, headers=_github_headers(), json=payload)

    if resp.status_code not in (200, 201):
        return {"error": f"GitHub API {resp.status_code}", "detail": resp.text[:300]}

    data = resp.json()
    return {
        "committed": True,
        "path": path,
        "commit_sha": data.get("commit", {}).get("sha"),
        "updated": existing_sha is not None
    }


async def _tool_railway_redeploy(args: Dict) -> Dict:
    sid = args.get("service_id", BRIDGE_SERVICE_ID)
    result = redeploy_service(sid, environment="production")
    return {"redeployed": True, "service_id": sid, "result": result}


async def _tool_railway_set_env(args: Dict) -> Dict:
    name = args["name"]
    value = args["value"]
    sid = args.get("service_id")
    result = set_service_variable(name, value, service_id=sid, environment_name="production")
    return {"set": True, "name": name, "result": result}


async def _tool_railway_get_status(args: Dict) -> Dict:
    sid = args.get("service_id") or BRIDGE_SERVICE_ID
    return get_service_status(sid)


async def _tool_railway_get_logs(args: Dict) -> Dict:
    did = args["deployment_id"]
    limit = args.get("limit", 100)
    return get_logs(did, limit=limit)


async def _tool_llm_chat(args: Dict) -> Dict:
    router = get_router()
    chat_req = ChatRequest(
        provider=args.get("provider", "moonshot"),
        model=args.get("model", "kimi-k3"),
        messages=[ChatMessage(role="user", content=args["prompt"])],
        max_tokens=args.get("max_tokens", 2048),
        temperature=1.0,
        reasoning_effort=args.get("reasoning_effort", "low")
    )
    resp = await router.chat(chat_req)
    return {
        "content": resp.content,
        "provider": resp.provider,
        "model": resp.model,
        "usage": resp.usage,
        "cost_cents": resp.cost_cents,
        "finish_reason": resp.finish_reason
    }


async def _tool_write_memory(args: Dict) -> Dict:
    from .memory import MemoryStore
    store = MemoryStore()
    entry = store.append(args["entry"], tags=args.get("tags", []))
    return {"written": True, "entry_id": entry.get("id")}


async def _tool_read_memory(args: Dict) -> Dict:
    from .memory import MemoryStore
    store = MemoryStore()
    entries = store.read(limit=args.get("limit", 10), tag=args.get("tag"))
    return {"entries": entries, "count": len(entries)}


async def _tool_github_read_issue(args: Dict) -> Dict:
    import httpx
    owner = args.get("owner", OWNER)
    repo = args["repo"]
    issue_num = args["issue_number"]
    per_page = args.get("per_page", 30)

    url = f"{GITHUB_API}/repos/{owner}/{repo}/issues/{issue_num}/comments"
    params = {"per_page": per_page}

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, headers=_github_headers(), params=params)

    if resp.status_code != 200:
        return {"error": f"GitHub API {resp.status_code}", "detail": resp.text[:300]}

    comments = resp.json()
    return {
        "issue_number": issue_num,
        "count": len(comments),
        "comments": [
            {
                "id": c.get("id"),
                "author": (c.get("user") or {}).get("login"),
                "created_at": c.get("created_at"),
                "body": c.get("body", "")[:500]
            }
            for c in comments
        ]
    }


async def _tool_http_get(args: Dict) -> Dict:
    import httpx

    url = (args.get("url") or "").strip()
    if not url:
        return {"error": "url is required"}

    try:
        parsed = urlparse(url)
    except Exception:
        return {"error": "invalid url"}

    if parsed.scheme != "https":
        return {"error": "only https is allowed"}
    host = (parsed.hostname or "").lower()
    if host not in HTTP_ALLOW_HOSTS:
        return {
            "error": f"host '{host}' is not allow-listed",
            "allowed": sorted(HTTP_ALLOW_HOSTS),
        }

    timeout = min(max(int(args.get("timeout_sec") or 25), 5), 60)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            resp = await client.get(url)
    except httpx.TimeoutException:
        return {"error": f"request timed out after {timeout}s", "url": url}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "url": url}

    content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    raw = resp.content or b""
    truncated = False
    if len(raw) > HTTP_MAX_BYTES:
        raw = raw[:HTTP_MAX_BYTES]
        truncated = True

    text = raw.decode("utf-8", errors="replace")

    body: Any = text
    if "json" in content_type or text.lstrip().startswith(("{", "[")):
        try:
            body = json.loads(text)
        except Exception:
            body = text

    return {
        "ok": 200 <= resp.status_code < 300,
        "status_code": resp.status_code,
        "url": str(resp.url),
        "content_type": content_type,
        "bytes": len(raw),
        "truncated": truncated,
        "body": body,
    }


async def _tool_browser_research(args: Dict) -> Dict:
    """Browser research using Browser Use + Browserbase.

    Creates a fresh Browserbase session on every call using BROWSERBASE_API_KEY.
    Falls back to local Chromium if no API key is set.
    """
    task = (args.get("task") or "").strip()
    if not task:
        return {"success": False, "error": "task is required"}

    start_url = args.get("start_url")
    mode = args.get("mode", "read_only")
    max_steps = int(args.get("max_steps") or 12)
    timeout_seconds = int(args.get("timeout_seconds") or 180)

    if mode not in ("read_only", "elevated"):
        mode = "read_only"

    started_at = datetime.now(timezone.utc).isoformat()
    session_id = None

    # Lazy imports
    try:
        from browser_use import Agent, Browser
        from browser_use.browser.browser import BrowserConfig
    except ImportError as e:
        return {
            "success": False,
            "error": f"browser-use not installed: {e}",
            "mode_used": mode,
            "started_at": started_at,
        }

    browser = None
    runtime = "local"

    try:
        api_key = os.getenv("BROWSERBASE_API_KEY")

        if api_key:
            try:
                from browserbase import Browserbase
                bb = Browserbase(api_key=api_key)
                session = bb.sessions.create()
                session_id = session.id
                cdp_url = session.connect_url
                browser = Browser(cdp_url=cdp_url)
                runtime = "browserbase"
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to create Browserbase session: {e}",
                    "mode_used": mode,
                    "started_at": started_at,
                }
        else:
            # Fallback to local Chromium
            browser = Browser(config=BrowserConfig(headless=True))
            runtime = "local"

        # Permission constraints
        if mode == "read_only":
            constraints = (
                "STRICT RULES — READ ONLY MODE:\n"
                "- You may navigate, scroll, click links, and extract information.\n"
                "- You must NOT fill forms, submit anything, log in, create accounts, "
                "post content, or perform any write/action that changes state.\n"
                "- If the task requires interaction beyond reading, stop and report what you found."
            )
        else:
            constraints = (
                "ELEVATED MODE (limited interaction allowed):\n"
                "- You may click buttons and fill simple forms if necessary.\n"
                "- You must NOT create accounts, make purchases, post publicly, "
                "or perform irreversible actions.\n"
                "- Prefer the least interactive path that still answers the task."
            )

        full_task = f"{task}\n\n{constraints}"
        if start_url:
            full_task = f"Start by going to: {start_url}\n\n{full_task}"

        router = get_router()

        class _BridgeLLM:
            def __init__(self, router):
                self.router = router

            async def ainvoke(self, messages, **kwargs):
                chat_messages = []
                for m in messages:
                    role = getattr(m, "type", None) or getattr(m, "role", "user")
                    content = getattr(m, "content", str(m))
                    if role in ("human", "user"):
                        chat_messages.append(ChatMessage(role="user", content=content))
                    elif role in ("ai", "assistant"):
                        chat_messages.append(ChatMessage(role="assistant", content=content))
                    else:
                        chat_messages.append(ChatMessage(role="user", content=content))

                req = ChatRequest(
                    provider="moonshot",
                    model="kimi-k3",
                    messages=chat_messages,
                    max_tokens=2048,
                    temperature=1.0,
                    reasoning_effort="low",
                )
                resp = await self.router.chat(req)
                class _Resp:
                    content = resp.content
                return _Resp()

        llm = _BridgeLLM(router)

        agent = Agent(
            task=full_task,
            llm=llm,
            browser=browser,
            max_actions_per_step=3,
        )

        import asyncio
        try:
            history = await asyncio.wait_for(
                agent.run(max_steps=max_steps),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Timed out after {timeout_seconds}s",
                "mode_used": mode,
                "runtime": runtime,
                "session_id": session_id,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "task": task,
                "start_url": start_url,
            }

        final_result = None
        if hasattr(history, "final_result"):
            final_result = history.final_result()
        if not final_result:
            final_result = str(history) if history else "No result returned"

        urls_visited = []
        try:
            for h in getattr(history, "history", []):
                state = getattr(h, "state", None)
                if state and getattr(state, "url", None):
                    urls_visited.append(state.url)
        except Exception:
            pass

        result = {
            "success": True,
            "final_result": final_result,
            "steps_taken": len(getattr(history, "history", [])),
            "urls_visited": urls_visited,
            "mode_used": mode,
            "runtime": runtime,
            "session_id": session_id,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "start_url": start_url,
        }
        if session_id:
            result["replay_url"] = f"https://www.browserbase.com/sessions/{session_id}"
        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()[-600:],
            "mode_used": mode,
            "runtime": runtime,
            "session_id": session_id,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "start_url": start_url,
        }
    finally:
        if browser:
            try:
                await browser.close()
            except Exception:
                pass


_TOOL_HANDLERS = {
    "github_read": _tool_github_read,
    "github_commit": _tool_github_commit,
    "railway_redeploy": _tool_railway_redeploy,
    "railway_set_env": _tool_railway_set_env,
    "railway_get_status": _tool_railway_get_status,
    "railway_get_logs": _tool_railway_get_logs,
    "llm_chat": _tool_llm_chat,
    "write_memory": _tool_write_memory,
    "read_memory": _tool_read_memory,
    "github_read_issue": _tool_github_read_issue,
    "http_get": _tool_http_get,
    "browser_research": _tool_browser_research,
}
