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
    }
,
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": (
                "Fetch an allowlisted https URL and return its body as text. Allowed hosts come "
                "from FETCH_ALLOWED_HOSTS and currently cover the KalshiML dashboard, the bridge "
                "itself, api.elections.kalshi.com, Open-Meteo, IEM (mesonet.agron.iastate.edu) and "
                "api.weather.gov. A refusal naming FETCH_ALLOWED_HOSTS means that HOST is not "
                "permitted -- it does NOT mean this tool is unavailable, so do not conclude you "
                "have no HTTP access and fall back to guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full https URL"},
                    "max_bytes": {"type": "integer", "description": "Response cap", "default": 8000}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kml_data_read",
            "description": (
                "Read a live state/JSON file from the running KalshiML instance -- CURRENT state, "
                "not repo content, and the two disagree often. Useful paths: exec_scorecard.json "
                "(live fill rates, n_graded), exec_backfill_scorecard.json (maker-vs-taker "
                "backfill), maker_policy_promote.json (promote gates), ml_status.json, "
                "exec_grade_checkpoint.json."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Filename, e.g. exec_scorecard.json"},
                    "host": {"type": "string", "description": "Service host to read from. Defaults to the KalshiML dashboard. Any allowlisted host with an /api/file endpoint works."},
                    "max_bytes": {"type": "integer", "description": "Response cap", "default": 8000}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kml_app_logs",
            "description": (
                "Read the running KalshiML process's in-memory log tail, each line stamped with its "
                "own UTC time. ALWAYS pass `contains` -- the buffer holds ~8000 lines and an "
                "unfiltered pull is mostly noise. Useful filters: 'SCAN' (candidate counts and "
                "reject reasons), '429', '[ENS-GATE]', '[evidence]', '[exec-grade]'. The buffer does "
                "NOT survive a process restart, so after a deploy it only covers time since restart."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "contains": {"type": "string", "description": "Substring filter"},
                    "n": {"type": "integer", "description": "Lines to search, max 8000", "default": 2000}
                },
                "required": []
            }
        }
    }
,
    {
        "type": "function",
        "function": {
            "name": "github_list_repos",
            "description": (
                "List every GitHub repo the bridge token can reach, newest-pushed first. Use this "
                "before github_read when you do not already know the exact repo name -- do not "
                "guess names, and note a 404 from github_read usually means a wrong name or a "
                "wrong branch, not a missing file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max repos", "default": 50}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "railway_list",
            "description": (
                "List Railway projects, and the services + environments inside one. Call with no "
                "arguments to see all projects, then pass project_id for its services. Service and "
                "project NAMES have been changed before while IDs stayed constant -- always work "
                "from IDs. Feed the service_id into railway_get_status or railway_get_logs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Omit to list all projects"}
                },
                "required": []
            }
        }
    }
]

# ── Tool sets and the separation rule ────────────────────────────────────────
#
# browser_research reads attacker-controllable text into the model's context.
# It must never share a loop with a tool that writes code, env vars, or the
# memory file that is replayed into later runs. That is enforced here, not
# left to the caller's discipline.

try:
    from .browser import BROWSER_TOOL_SCHEMA, browser_research as _tool_browser_research
    BROWSER_TOOL_AVAILABLE = True
except Exception as _browser_import_err:  # pragma: no cover
    BROWSER_TOOL_AVAILABLE = False
    print(f"[agent_loop] browser tool unavailable: {_browser_import_err}", flush=True)

WRITE_TOOL_NAMES = {
    "github_commit",
    "railway_set_env",
    "railway_redeploy",
    "write_memory",
}

UNTRUSTED_INPUT_TOOL_NAMES = {"browser_research"}

READ_ONLY_TOOL_NAMES = [
    "github_read",
    "github_list_repos",
    "railway_get_status",
    "railway_get_logs",
    "railway_list",
    "llm_chat",
    "read_memory",
    "http_get",
    "kml_data_read",
    "kml_app_logs",
]


def _by_name(names) -> list:
    wanted = set(names)
    return [t for t in TOOL_SCHEMAS if t["function"]["name"] in wanted]


# Everything the bridge can do, minus the browser. This is the default.
BUILD_TOOLS = list(TOOL_SCHEMAS)

# Read + browse, no writes anywhere. write_memory is deliberately excluded:
# a page that talks the agent into writing a "lesson" would be planting an
# instruction for every later run.
RESEARCH_TOOLS = _by_name(READ_ONLY_TOOL_NAMES) + (
    [BROWSER_TOOL_SCHEMA] if BROWSER_TOOL_AVAILABLE else []
)

TOOL_SETS = {"build": BUILD_TOOLS, "research": RESEARCH_TOOLS}


def assert_tool_set_safe(tools) -> None:
    """Refuse a tool set that hands a browsing agent write access."""
    names = {t.get("function", {}).get("name") for t in (tools or [])}
    untrusted = names & UNTRUSTED_INPUT_TOOL_NAMES
    writes = names & WRITE_TOOL_NAMES
    if untrusted and writes:
        raise ValueError(
            "unsafe tool set: "
            f"{sorted(untrusted)} reads untrusted web content and cannot be combined with "
            f"{sorted(writes)}. Use tool_set='research' to browse, then hand the findings to a "
            "separate build run."
        )


def resolve_tools(tools=None, tool_set: str = None) -> list:
    """Pick a tool set by name or validate an explicit list."""
    if tools:
        assert_tool_set_safe(tools)
        return tools
    chosen = TOOL_SETS.get((tool_set or "build").lower())
    if chosen is None:
        raise ValueError(f"unknown tool_set {tool_set!r}; expected one of {sorted(TOOL_SETS)}")
    assert_tool_set_safe(chosen)
    return chosen


# Convenience alias — unchanged meaning: the full build-capable set, no browser.
DEFAULT_TOOLS = BUILD_TOOLS


# ── Tool Handlers ────────────────────────────────────────────────────────────

async def run_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool by name with the given arguments."""
    if not BRIDGE_MODE:
        return {"error": "Agent tools require bridge environment (llm_gateway + railway_extension not importable)"}

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
        "content": content[:8000]  # Truncate very large files
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


# Variables the agent must never be able to write. Self-granting browser
# permissions, rotating the bridge key out from under the operator, or
# swapping a token are all one set_env away otherwise.
PROTECTED_ENV_PREFIXES = ("BROWSER_", "BRIDGE_")
PROTECTED_ENV_SUBSTRINGS = ("TOKEN", "API_KEY", "SECRET", "PASSWORD", "CREDENTIAL")


def env_name_is_protected(name: str) -> bool:
    upper = (name or "").upper()
    return upper.startswith(PROTECTED_ENV_PREFIXES) or any(
        s in upper for s in PROTECTED_ENV_SUBSTRINGS
    )


async def _tool_railway_set_env(args: Dict) -> Dict:
    name = args["name"]
    value = args["value"]
    if env_name_is_protected(name):
        return {
            "error": f"{name} is operator-only and cannot be set by an agent",
            "hint": "permission grants and secrets are changed by hand in the Railway dashboard",
        }
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


KML_DASHBOARD = os.getenv("KML_DASHBOARD_URL", "https://kalshiml-production.up.railway.app")


async def _do_fetch(url: str, max_bytes: int = 8000) -> Dict:
    """Reuse the bridge's own allowlisted fetch path rather than opening a second
    unchecked egress. Host policy stays in exactly one place (fetch_routes)."""
    from fetch_routes import FetchRequest, fetch
    try:
        resp = await fetch(FetchRequest(url=url, max_bytes=max_bytes))
    except Exception as e:
        detail = getattr(e, "detail", None) or str(e)
        return {"error": str(detail),
                "hint": ("If this names FETCH_ALLOWED_HOSTS, the HOST is not permitted. "
                         "The tool works -- pick an allowlisted host.")}
    return resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()


async def _tool_http_get(args: Dict) -> Dict:
    url = (args.get("url") or "").strip()
    if not url:
        return {"error": "url is required"}
    return await _do_fetch(url, int(args.get("max_bytes") or 8000))


async def _tool_kml_data_read(args: Dict) -> Dict:
    from urllib.parse import quote
    path = (args.get("path") or "").strip().lstrip("/")
    if not path:
        return {"error": "path is required"}
    host = (args.get("host") or "").strip().rstrip("/")
    base = host if host.startswith("https://") else (f"https://{host}" if host else KML_DASHBOARD)
    return await _do_fetch(f"{base}/api/file?path={quote(path)}",
                           int(args.get("max_bytes") or 8000))


async def _tool_kml_app_logs(args: Dict) -> Dict:
    from urllib.parse import quote
    n = max(1, min(int(args.get("n") or 2000), 8000))
    url = f"{KML_DASHBOARD}/api/logs?n={n}"
    contains = (args.get("contains") or "").strip()
    if contains:
        url += f"&contains={quote(contains)}"
    return await _do_fetch(url, 16000)


async def _tool_github_list_repos(args: Dict) -> Dict:
    import httpx
    limit = max(1, min(int(args.get("limit") or 50), 100))
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(f"{GITHUB_API}/user/repos",
                                headers=_github_headers(),
                                params={"per_page": limit, "sort": "pushed"})
    if resp.status_code != 200:
        return {"error": f"GitHub API {resp.status_code}", "detail": resp.text[:300]}
    return {"repos": [{"full_name": r["full_name"],
                       "default_branch": r.get("default_branch"),
                       "private": r.get("private"),
                       "pushed_at": r.get("pushed_at")} for r in resp.json()]}


async def _tool_railway_list(args: Dict) -> Dict:
    pid = (args.get("project_id") or "").strip()
    if pid:
        return {"project_id": pid, "services": await list_services(pid)}
    return {"projects": await list_projects()}


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
    "http_get": _tool_http_get,
    "kml_data_read": _tool_kml_data_read,
    "kml_app_logs": _tool_kml_app_logs,
    "github_list_repos": _tool_github_list_repos,
    "railway_list": _tool_railway_list,
}

if BROWSER_TOOL_AVAILABLE:
    _TOOL_HANDLERS["browser_research"] = _tool_browser_research
