"""Tool registry for the autonomous agent harness.

Each tool has:
  - An OpenAI-compatible function schema (for the LLM)
  - An async execution handler (for the bridge to run)

All handlers assume they are running inside the bridge process where
llm_gateway and railway_extension are importable.
"""

import asyncio
import base64
import io
import json
import os
import re
import time
import traceback
import zipfile
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
        get_service_domains,
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
            "description": (
                "Read a file or list a directory from a GitHub repository. Large files are "
                "returned in windows: the reply carries total_chars, returned_chars, truncated "
                "and next_offset. If truncated is true you have NOT seen the whole file — call "
                "again with offset=next_offset until truncated is false. Never judge a file you "
                "have only read one window of."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repo owner (default: nicholascannon560-ship-it)"},
                    "repo": {"type": "string", "description": "Repo name"},
                    "path": {"type": "string", "description": "File or directory path"},
                    "branch": {"type": "string", "description": "Branch or ref (default: repo default)"},
                    "offset": {"type": "integer", "description": "Character offset to start reading from (default 0)"},
                    "max_chars": {"type": "integer", "description": "Characters to return this call (default 8000, max 60000)"}
                },
                "required": ["repo", "path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "repo_search",
            "description": (
                "Search a repository's file CONTENTS for a regex and get back the matching "
                "lines with their path and line number — so you can jump straight to the right "
                "spot instead of reading whole files to find it. Branch-aware: it searches the "
                "actual branch you name (unlike GitHub's code search, which only sees the "
                "default branch). ALWAYS use this to LOCATE code before github_read, then "
                "github_read a tight window around the reported line. Narrow the scan with "
                "path= (path prefix) and extensions= whenever you can. Returns matches[], "
                "files_scanned, and truncated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repo owner (default: nicholascannon560-ship-it)"},
                    "repo": {"type": "string", "description": "Repo name"},
                    "pattern": {"type": "string", "description": "Python regex, matched per line"},
                    "branch": {"type": "string", "description": "Branch or ref to search (default: repo default branch)"},
                    "path": {"type": "string", "description": "Only search files whose path starts with this prefix (optional)"},
                    "extensions": {"type": "array", "items": {"type": "string"}, "description": "Only these extensions, e.g. [\"py\",\"md\"] (optional)"},
                    "ignore_case": {"type": "boolean", "description": "Case-insensitive match (default false)"},
                    "context": {"type": "integer", "description": "Extra lines of context around each match (default 0, max 5)"},
                    "max_results": {"type": "integer", "description": "Max matching lines to return (default 40, max 200)"}
                },
                "required": ["repo", "pattern"]
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
            "name": "github_patch",
            "description": (
                "Change part of an existing file by replacing an exact snippet, without "
                "resending the whole file. PREFER THIS OVER github_commit for any edit to a "
                "file that already exists — github_commit re-sends every byte of the file and "
                "is far more expensive. old_string must be unique in the file: include a few "
                "surrounding lines if the snippet appears more than once. Only the first match "
                "is replaced. To create a new file, use github_commit instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repo owner"},
                    "repo": {"type": "string", "description": "Repo name"},
                    "path": {"type": "string", "description": "File path inside repo"},
                    "old_string": {"type": "string", "description": "Exact text to replace, copied from the file"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                    "message": {"type": "string", "description": "Git commit message"},
                    "branch": {"type": "string", "description": "Branch (default: repo default)"}
                },
                "required": ["repo", "path", "old_string", "new_string", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "github_create_repo",
            "description": (
                "Create a NEW GitHub repository under the operator's account (owner is fixed). "
                "Use this to start a new project. After it succeeds, commit files into it with "
                "github_commit. Defaults to private with an initial commit on 'main' so the repo "
                "is immediately writable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Repo name: letters, digits, -, _, . only"},
                    "description": {"type": "string", "description": "Short description"},
                    "private": {"type": "boolean", "description": "Default true"},
                    "gitignore_template": {"type": "string", "description": "e.g. 'Python', 'Node' (optional)"},
                    "readme": {"type": "string", "description": "Optional README.md content for the initial commit"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "github_create_branch",
            "description": (
                "Create a new branch in a repository, from a given base branch (default: the "
                "repo's default branch). Safe to call when the branch already exists — it then "
                "just reports the existing head."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repo owner (default: operator account)"},
                    "repo": {"type": "string", "description": "Repo name"},
                    "branch": {"type": "string", "description": "New branch name"},
                    "from_branch": {"type": "string", "description": "Base branch/ref (default: repo default branch)"}
                },
                "required": ["repo", "branch"]
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
            "name": "railway_get_domains",
            "description": (
                "Get the public domains of a Railway service by ID, as bare hostnames and as "
                "https:// URLs. Use this to find a service's base URL before calling its "
                "endpoints (/health and the like) with http_get. Works by ID, so it still "
                "works when railway_list returns no projects -- a project-scoped token can "
                "act on its own resources by UUID while appearing in no listing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service_id": {"type": "string", "description": "Service UUID (default: bridge)"}
                },
                "required": []
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
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Execute a shell command (pytest, python, linters) in a clean throwaway "
                "GitHub Actions container and return its exit status and output. This is "
                "the ONLY way to actually run code — you have no shell otherwise, so never "
                "claim code works because it looks correct. Commit what you want to test to "
                "the sandbox repo FIRST (github_commit, repo='agent-sandbox'), then call "
                "this. The sandbox holds no secrets and cannot reach other private repos, so "
                "copy in any module under test alongside its tests. Takes 1-3 minutes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to run, e.g. 'pytest -q test_journal.py'"},
                    "setup": {"type": "string", "description": "Optional command run first, e.g. 'pip install pytest httpx'"},
                    "workdir": {"type": "string", "description": "Directory relative to repo root (default '.')"},
                    "timeout_sec": {"type": "integer", "description": "Max seconds to wait for completion (default 420, max 900)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "s3_status",
            "description": (
                "Summary of the S3 archive bucket: object count, total bytes, "
                "newest key and its timestamp. Use to check whether the nightly "
                "offload is still writing. Read-only."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "s3_list",
            "description": "List object keys in the archive bucket under a prefix. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prefix": {"type": "string", "description": "key prefix"},
                    "max_keys": {"type": "integer", "description": "1..1000, default 200"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "s3_get",
            "description": (
                "Read one object from the archive bucket as text, capped at 1 MB "
                "by default. Read-only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "object key"},
                    "max_bytes": {"type": "integer", "description": "cap, hard max 5242880"},
                },
                "required": ["key"],
            },
        },
    },
]

# ── Tool sets and the separation rule ────────────────────────────────────────
#
# browser_research reads attacker-controllable text into the model's context.
# It must never share a loop with a tool that writes code, env vars, or the
# memory file that is replayed into later runs. That is enforced here, not
# left to the caller's discipline.

try:
    from .browser import (
        BROWSER_TOOL_SCHEMAS,
        browser_read as _tool_browser_read,
        browser_research as _tool_browser_research,
    )
    BROWSER_TOOL_AVAILABLE = True
except Exception as _browser_import_err:  # pragma: no cover
    BROWSER_TOOL_AVAILABLE = False
    print(f"[agent_loop] browser tool unavailable: {_browser_import_err}", flush=True)

WRITE_TOOL_NAMES = {
    "github_commit",
    "github_patch",
    "github_create_repo",
    "github_create_branch",
    "run_tests",
    "railway_set_env",
    "railway_redeploy",
    "write_memory",
}

UNTRUSTED_INPUT_TOOL_NAMES = {"browser_research", "browser_read"}

READ_ONLY_TOOL_NAMES = [
    "github_read",
    "repo_search",
    "github_list_repos",
    "railway_get_status",
    "railway_get_logs",
    "railway_get_domains",
    "railway_list",
    "llm_chat",
    "read_memory",
    "http_get",
    "kml_data_read",
    "kml_app_logs",
    "s3_status",
    "s3_list",
    "s3_get",
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
    list(BROWSER_TOOL_SCHEMAS) if BROWSER_TOOL_AVAILABLE else []
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


GITHUB_READ_DEFAULT_CHARS = int(os.environ.get("GITHUB_READ_DEFAULT_CHARS", "8000"))
GITHUB_READ_MAX_CHARS = int(os.environ.get("GITHUB_READ_MAX_CHARS", "60000"))

# repo_search caps. Content search fetches candidate blobs server-side and greps
# them in-process, returning only the matching lines — so the model pays for
# snippets, not whole files. These bound the server-side work per call.
REPO_SEARCH_MAX_FILES = int(os.environ.get("REPO_SEARCH_MAX_FILES", "400"))
REPO_SEARCH_MAX_FILE_BYTES = int(os.environ.get("REPO_SEARCH_MAX_FILE_BYTES", "400000"))
REPO_SEARCH_CONCURRENCY = int(os.environ.get("REPO_SEARCH_CONCURRENCY", "8"))
REPO_SEARCH_DEFAULT_RESULTS = int(os.environ.get("REPO_SEARCH_DEFAULT_RESULTS", "40"))
REPO_SEARCH_MAX_RESULTS = int(os.environ.get("REPO_SEARCH_MAX_RESULTS", "200"))

# Extensions never worth grepping — binary or bulk data. Skipped unless the
# caller explicitly names extensions.
_BINARY_EXTS = {
    "png", "jpg", "jpeg", "gif", "webp", "ico", "pdf", "zip", "gz", "tar", "tgz",
    "bz2", "7z", "parquet", "xlsx", "xls", "db", "sqlite", "so", "dylib", "dll",
    "bin", "exe", "woff", "woff2", "ttf", "eot", "mp4", "mov", "mp3", "wav",
    "pkl", "npy", "npz",
}


def _ext_of(path: str) -> str:
    base = path.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[-1].lower() if "." in base else ""


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

    # A flat 8k cut made every file over ~200 lines unreadable, silently: the
    # model saw a prefix and reasoned about it as if it were the whole file.
    # Window it instead and say plainly that more remains.
    offset = max(0, int(args.get("offset") or 0))
    max_chars = int(args.get("max_chars") or GITHUB_READ_DEFAULT_CHARS)
    max_chars = max(1000, min(max_chars, GITHUB_READ_MAX_CHARS))
    window = content[offset:offset + max_chars]
    end = offset + len(window)

    return {
        "type": "file",
        "path": data["path"],
        "sha": data.get("sha"),
        "size": data.get("size"),
        "offset": offset,
        "total_chars": len(content),
        "returned_chars": len(window),
        "truncated": end < len(content),
        "next_offset": end if end < len(content) else None,
        "content": window,
    }


async def _tool_repo_search(args: Dict) -> Dict:
    """Branch-aware content grep: tree -> fetch candidate blobs -> regex per line.

    Returns only matching lines (path + line number + text), so the model pays
    for snippets it can act on, not whole files it has to skim. Server-side work
    is bounded by the REPO_SEARCH_* caps; the model narrows with path/extensions.
    """
    import httpx

    owner = args.get("owner", OWNER)
    repo = args["repo"]
    pattern = args["pattern"]
    branch = args.get("branch")
    path_prefix = (args.get("path") or "").lstrip("/")
    exts = {str(e).lower().lstrip(".") for e in (args.get("extensions") or []) if isinstance(e, str)}
    flags = re.IGNORECASE if args.get("ignore_case") else 0
    ctx = max(0, min(int(args.get("context") or 0), 5))
    max_results = max(1, min(int(args.get("max_results") or REPO_SEARCH_DEFAULT_RESULTS),
                             REPO_SEARCH_MAX_RESULTS))

    try:
        rx = re.compile(pattern, flags)
    except re.error as e:
        return {"error": f"bad regex: {e}"}

    headers = _github_headers()
    async with httpx.AsyncClient(timeout=30) as client:
        ref = branch
        if not ref:
            r = await client.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=headers)
            if r.status_code != 200:
                return {"error": f"GitHub API {r.status_code} resolving repo", "detail": r.text[:200]}
            ref = r.json().get("default_branch", "main")

        # A branch name resolves fine as a tree-ish here; recursive lists all blobs.
        tr = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{ref}",
            headers=headers, params={"recursive": "1"},
        )
        if tr.status_code != 200:
            return {"error": f"GitHub API {tr.status_code} reading tree for ref {ref!r}",
                    "detail": tr.text[:200]}
        tree = tr.json()

        blobs = []
        for e in tree.get("tree", []):
            if e.get("type") != "blob":
                continue
            p = e.get("path", "")
            if path_prefix and not p.startswith(path_prefix):
                continue
            ext = _ext_of(p)
            if exts:
                if ext not in exts:
                    continue
            elif ext in _BINARY_EXTS:
                continue
            if int(e.get("size") or 0) > REPO_SEARCH_MAX_FILE_BYTES:
                continue
            blobs.append(e)

        tree_truncated = bool(tree.get("truncated"))
        overflow = len(blobs) > REPO_SEARCH_MAX_FILES
        blobs = blobs[:REPO_SEARCH_MAX_FILES]

        sem = asyncio.Semaphore(REPO_SEARCH_CONCURRENCY)
        lock = asyncio.Lock()
        matches: List[Dict[str, Any]] = []
        files_scanned = 0
        hit_cap = False

        async def _scan(blob):
            nonlocal files_scanned, hit_cap
            async with sem:
                if hit_cap:
                    return
                try:
                    br = await client.get(
                        f"{GITHUB_API}/repos/{owner}/{repo}/git/blobs/{blob['sha']}",
                        headers=headers,
                    )
                    if br.status_code != 200:
                        return
                    bj = br.json()
                    if bj.get("encoding") != "base64":
                        return
                    raw = base64.b64decode(bj.get("content") or "")
                    if b"\x00" in raw[:4096]:
                        return  # binary
                    text = raw.decode("utf-8", errors="replace")
                except Exception:
                    return
            lines = text.split("\n")
            local = []
            for i, line in enumerate(lines):
                if rx.search(line):
                    entry = {"path": blob["path"], "line": i + 1, "text": line[:400]}
                    if ctx:
                        lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
                        entry["context"] = "\n".join(lines[lo:hi])[:1200]
                    local.append(entry)
            async with lock:
                files_scanned += 1
                for m in local:
                    if len(matches) >= max_results:
                        hit_cap = True
                        break
                    matches.append(m)

        await asyncio.gather(*[_scan(b) for b in blobs])

    matches.sort(key=lambda m: (m["path"], m["line"]))
    hint = ""
    if hit_cap:
        hint = "Hit max_results — narrow with path= or extensions=, or raise max_results."
    elif overflow:
        hint = "Candidate files exceeded the scan cap — narrow with path= or extensions=."
    elif tree_truncated:
        hint = "Repo tree was truncated by GitHub; some files were not considered."
    return {
        "ref": ref,
        "pattern": pattern,
        "files_considered": len(blobs),
        "files_scanned": files_scanned,
        "match_count": len(matches),
        "truncated": bool(hit_cap or overflow or tree_truncated),
        "matches": matches,
        "hint": hint,
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


def _fuzzy_replace(text: str, old: str, new: str) -> str:
    """Fall back to whitespace-insensitive line matching, preserving the
    indentation actually found in the file. Mirrors patch_routes._fuzzy_replace."""
    old_lines = old.splitlines()
    text_lines = text.splitlines()

    for i in range(len(text_lines) - len(old_lines) + 1):
        window = text_lines[i:i + len(old_lines)]
        if all(w.strip() == o.strip() for w, o in zip(window, old_lines)):
            indent = len(text_lines[i]) - len(text_lines[i].lstrip())
            new_lines = []
            for idx, line in enumerate(new.splitlines()):
                if idx == 0:
                    new_lines.append((" " * indent + line.lstrip()) if line.strip() else line)
                else:
                    new_lines.append(line)
            return "\n".join(text_lines[:i] + new_lines + text_lines[i + len(old_lines):])

    raise ValueError("fuzzy match failed")


async def _tool_github_patch(args: Dict) -> Dict:
    """Replace an exact snippet in an existing file. Costs the size of the edit
    rather than the size of the file."""
    import httpx
    owner = args.get("owner", OWNER)
    repo = args["repo"]
    path = args["path"]
    old_string = args["old_string"]
    new_string = args["new_string"]
    message = args["message"]
    branch = args.get("branch")

    contents_path = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": branch} if branch else None

    async with httpx.AsyncClient(timeout=20) as client:
        lookup = await client.get(contents_path, headers=_github_headers(), params=params)

    if lookup.status_code != 200:
        return {"error": f"GitHub API {lookup.status_code}", "detail": lookup.text[:300]}

    data = lookup.json()
    if isinstance(data, list):
        return {"error": f"{path} is a directory, not a file"}

    existing_sha = data.get("sha")
    try:
        current = base64.b64decode(data.get("content", "")).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as e:
        return {"error": f"cannot patch binary or non-UTF8 file: {e}"}

    occurrences = current.count(old_string)
    if occurrences > 1:
        return {
            "error": "old_string is not unique",
            "occurrences": occurrences,
            "hint": "Include surrounding lines so the snippet appears exactly once.",
        }

    if occurrences == 1:
        updated = current.replace(old_string, new_string, 1)
    else:
        try:
            updated = _fuzzy_replace(current, old_string, new_string)
        except ValueError:
            return {
                "error": "old_string not found in file",
                "hint": "Use github_read to copy the exact current text, then retry.",
                "total_chars": len(current),
            }

    if updated == current:
        return {"error": "patch would not change the file", "path": path}

    payload = {
        "message": message,
        "content": base64.b64encode(updated.encode("utf-8")).decode("ascii"),
        "sha": existing_sha,
    }
    if branch:
        payload["branch"] = branch

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.put(contents_path, headers=_github_headers(), json=payload)

    if resp.status_code not in (200, 201):
        return {"error": f"GitHub API {resp.status_code}", "detail": resp.text[:300]}

    result = resp.json()
    return {
        "patched": True,
        "path": path,
        "commit_sha": result.get("commit", {}).get("sha"),
        "chars_before": len(current),
        "chars_after": len(updated),
    }


_REPO_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


async def _tool_github_create_repo(args: Dict) -> Dict:
    """Create a repo under the operator's account only. The owner is NOT a
    parameter — an agent must never be able to plant code under someone else's
    namespace."""
    import httpx

    name = (args.get("name") or "").strip()
    if not _REPO_NAME_RE.match(name) or name.startswith((".", "-")):
        return {"error": f"invalid repo name {name!r}: use letters, digits, -, _, . (not leading . or -)"}

    payload: Dict[str, Any] = {
        "name": name,
        "description": (args.get("description") or "")[:350],
        "private": bool(args.get("private", True)),
        # auto_init gives the repo an initial commit on 'main', which makes it
        # immediately writable through the contents API (empty repos reject PUTs).
        "auto_init": True,
    }
    if args.get("gitignore_template"):
        payload["gitignore_template"] = str(args["gitignore_template"])

    async with httpx.AsyncClient(timeout=25) as client:
        resp = await client.post(f"{GITHUB_API}/user/repos",
                                 headers=_github_headers(), json=payload)
        if resp.status_code == 422 and "name already exists" in resp.text:
            return {"error": f"repo {name!r} already exists",
                    "hint": "commit into it with github_commit, or pick another name"}
        if resp.status_code not in (200, 201):
            return {"error": f"GitHub API {resp.status_code}", "detail": resp.text[:300]}

        data = resp.json()
        out = {
            "created": True,
            "full_name": data.get("full_name"),
            "html_url": data.get("html_url"),
            "private": data.get("private"),
            "default_branch": data.get("default_branch"),
        }

        readme = args.get("readme")
        if readme:
            put = await client.put(
                f"{GITHUB_API}/repos/{OWNER}/{name}/contents/README.md",
                headers=_github_headers(),
                json={
                    "message": "Initial README",
                    "content": base64.b64encode(str(readme).encode("utf-8")).decode("ascii"),
                },
            )
            out["readme_committed"] = put.status_code in (200, 201)
        return out


async def _tool_github_create_branch(args: Dict) -> Dict:
    import httpx

    owner = args.get("owner", OWNER)
    repo = args["repo"]
    branch = (args.get("branch") or "").strip()
    if not branch or branch.startswith(("-", ".")) or ".." in branch or " " in branch:
        return {"error": f"invalid branch name {branch!r}"}
    from_branch = (args.get("from_branch") or "").strip() or None

    base = f"{GITHUB_API}/repos/{owner}/{repo}"
    async with httpx.AsyncClient(timeout=20) as client:
        if not from_branch:
            meta = await client.get(base, headers=_github_headers())
            if meta.status_code != 200:
                return {"error": f"GitHub API {meta.status_code}", "detail": meta.text[:300]}
            from_branch = meta.json().get("default_branch") or "main"

        head = await client.get(f"{base}/git/ref/heads/{from_branch}",
                                headers=_github_headers())
        if head.status_code != 200:
            return {"error": f"base branch {from_branch!r} not found (HTTP {head.status_code})",
                    "detail": head.text[:300]}
        sha = head.json()["object"]["sha"]

        create = await client.post(f"{base}/git/refs", headers=_github_headers(),
                                   json={"ref": f"refs/heads/{branch}", "sha": sha})
        if create.status_code == 422:
            # Already exists — report the current head rather than failing.
            existing = await client.get(f"{base}/git/ref/heads/{branch}",
                                        headers=_github_headers())
            if existing.status_code == 200:
                return {"created": False, "already_existed": True,
                        "branch": branch, "sha": existing.json()["object"]["sha"]}
            return {"error": f"GitHub API 422", "detail": create.text[:300]}
        if create.status_code not in (200, 201):
            return {"error": f"GitHub API {create.status_code}", "detail": create.text[:300]}

        return {"created": True, "branch": branch, "from": from_branch, "sha": sha}


async def _tool_railway_redeploy(args: Dict) -> Dict:
    sid = args.get("service_id", BRIDGE_SERVICE_ID)
    result = await asyncio.to_thread(redeploy_service, sid, environment="production")
    return {"redeployed": True, "service_id": sid, "result": result}


# Variables the agent must never be able to write. Self-granting browser
# permissions, rotating the bridge key out from under the operator, or
# swapping a token are all one set_env away otherwise.
PROTECTED_ENV_PREFIXES = ("BROWSER_", "BRIDGE_", "AWS_")
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
    result = await asyncio.to_thread(
        set_service_variable, name, value, service_id=sid, environment_name="production"
    )
    return {"set": True, "name": name, "result": result}


async def _tool_railway_get_status(args: Dict) -> Dict:
    sid = args.get("service_id") or BRIDGE_SERVICE_ID
    return await asyncio.to_thread(get_service_status, sid)


async def _tool_railway_get_domains(args: Dict) -> Dict:
    sid = args.get("service_id") or BRIDGE_SERVICE_ID
    return await asyncio.to_thread(get_service_domains, sid)


async def _tool_railway_get_logs(args: Dict) -> Dict:
    did = args["deployment_id"]
    limit = args.get("limit", 100)
    return await asyncio.to_thread(get_logs, did, limit=limit)


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
    # Every railway_extension helper is a plain sync function doing a blocking
    # requests.post() to Railway's GraphQL API. Never `await` one directly --
    # awaiting its dict return raises "object dict can't be used in 'await'
    # expression". Hand it to a worker thread so the event loop stays free
    # during the HTTP round-trip.
    pid = (args.get("project_id") or "").strip()
    if pid:
        return {"project_id": pid, "services": await asyncio.to_thread(list_services, pid)}
    return {"projects": await asyncio.to_thread(list_projects)}



SANDBOX_REPO = os.getenv("AGENT_SANDBOX_REPO", "agent-sandbox")
SANDBOX_WORKFLOW = os.getenv("AGENT_SANDBOX_WORKFLOW", "sandbox.yml")


async def _tool_run_tests(args: Dict) -> Dict:
    """Dispatch the sandbox workflow, poll for completion, return logs.

    Deliberately scoped to one hardcoded repo. The bridge's GITHUB_TOKEN has
    `workflow` scope, so a caller-supplied repo here would let an agent run
    arbitrary code in a repo that DOES hold secrets. The sandbox repo holds
    none, which is the entire safety property — do not parameterise it.
    """
    import httpx

    command = args["command"]
    setup = args.get("setup", "") or ""
    workdir = args.get("workdir", ".") or "."
    # 420 not 300: the sandbox workflow's worst case is ~5.5 min (2-min setup
    # step + 3-min command step + checkout/setup-python), and the caller must
    # outlive that or it abandons a run it is still paying for and never learns
    # the outcome — exactly what happened on 2026-08-06. Raise this only
    # alongside the workflow's step timeouts; the two numbers are a pair.
    timeout_sec = min(int(args.get("timeout_sec") or 420), 900)

    base = f"{GITHUB_API}/repos/{OWNER}/{SANDBOX_REPO}"
    started = time.time()

    async with httpx.AsyncClient(timeout=30) as client:
        # Record the newest existing run id so we can tell ours apart. GitHub's
        # dispatch endpoint returns 204 with no body -- it does not tell you
        # which run it created.
        before = await client.get(
            f"{base}/actions/workflows/{SANDBOX_WORKFLOW}/runs",
            headers=_github_headers(), params={"per_page": 1},
        )
        prior_id = 0
        if before.status_code == 200:
            runs = before.json().get("workflow_runs") or []
            if runs:
                prior_id = runs[0].get("id", 0)

        dispatch = await client.post(
            f"{base}/actions/workflows/{SANDBOX_WORKFLOW}/dispatches",
            headers=_github_headers(),
            json={"ref": "main", "inputs": {
                "command": command, "setup": setup, "workdir": workdir,
            }},
        )
        if dispatch.status_code not in (201, 204):
            return {"error": f"dispatch failed: {dispatch.status_code} {dispatch.text[:300]}"}

        run_id = None
        while time.time() - started < timeout_sec:
            await asyncio.sleep(5)
            listing = await client.get(
                f"{base}/actions/workflows/{SANDBOX_WORKFLOW}/runs",
                headers=_github_headers(), params={"per_page": 5},
            )
            if listing.status_code != 200:
                continue
            for run in listing.json().get("workflow_runs") or []:
                if run.get("id", 0) > prior_id:
                    run_id = run["id"]
                    break
            if run_id:
                break
        if not run_id:
            return {"error": "dispatched but no new run appeared", "waited_sec": round(time.time() - started)}

        conclusion = None
        while time.time() - started < timeout_sec:
            detail = await client.get(f"{base}/actions/runs/{run_id}", headers=_github_headers())
            if detail.status_code == 200:
                body = detail.json()
                if body.get("status") == "completed":
                    conclusion = body.get("conclusion")
                    break
            await asyncio.sleep(5)

        if conclusion is None:
            # Carry the URL and the queue/start timestamps: a timeout with
            # run_started_at far after created_at is GitHub queueing, not a
            # hanging command, and the two need different responses.
            meta = {}
            try:
                d = await client.get(f"{base}/actions/runs/{run_id}", headers=_github_headers())
                if d.status_code == 200:
                    b = d.json()
                    meta = {"html_url": b.get("html_url"), "run_status": b.get("status"),
                            "created_at": b.get("created_at"),
                            "run_started_at": b.get("run_started_at")}
            except Exception:
                pass
            return {
                "error": "timed out waiting for completion",
                "run_id": run_id, "waited_sec": round(time.time() - started),
                "hint": ("raise timeout_sec, or the command may be hanging. "
                         "GET /sandbox/run/<run_id> on the bridge shows queue vs run time."),
                **meta,
            }

        logs_text = ""
        logs = await client.get(f"{base}/actions/runs/{run_id}/logs",
                                headers=_github_headers(), follow_redirects=True)
        if logs.status_code == 200:
            try:
                with zipfile.ZipFile(io.BytesIO(logs.content)) as z:
                    parts = [z.read(n).decode("utf-8", "replace")
                             for n in sorted(z.namelist()) if n.endswith(".txt")]
                logs_text = "\n".join(parts)
            except Exception as exc:  # noqa: BLE001
                logs_text = f"(could not unpack logs: {exc})"

        if len(logs_text) > 40000:
            logs_text = logs_text[:20000] + "\n...[truncated]...\n" + logs_text[-20000:]

        return {
            "passed": conclusion == "success",
            "conclusion": conclusion,
            "run_id": run_id,
            "elapsed_sec": round(time.time() - started),
            "command": command,
            "logs": logs_text,
        }


# ── Read-only S3 ─────────────────────────────────────────────────────────────
#
# These delegate to aws_routes, which pins the bucket from env so a caller can
# never retarget them, and whose IAM identity is denied every write verb. No
# s3_put exists on purpose; if one is ever added it belongs in WRITE_TOOL_NAMES
# on the same commit.

async def _aws_call(coro):
    from fastapi import HTTPException
    try:
        return await coro
    except HTTPException as e:
        return {"error": f"{e.status_code}: {e.detail}"}


async def _tool_s3_status(args: Dict) -> Dict:
    from aws_routes import s3_status
    return await _aws_call(s3_status())


async def _tool_s3_list(args: Dict) -> Dict:
    from aws_routes import ListRequest, s3_list
    return await _aws_call(s3_list(ListRequest(
        prefix=(args.get("prefix") or None),
        max_keys=(int(args["max_keys"]) if args.get("max_keys") else None),
    )))


async def _tool_s3_get(args: Dict) -> Dict:
    from aws_routes import GetRequest, s3_get
    key = (args.get("key") or "").strip()
    if not key:
        return {"error": "key is required"}
    return await _aws_call(s3_get(GetRequest(
        key=key,
        max_bytes=(int(args["max_bytes"]) if args.get("max_bytes") else None),
    )))


_TOOL_HANDLERS = {
    "github_read": _tool_github_read,
    "repo_search": _tool_repo_search,
    "github_commit": _tool_github_commit,
    "github_patch": _tool_github_patch,
    "github_create_repo": _tool_github_create_repo,
    "github_create_branch": _tool_github_create_branch,
    "railway_redeploy": _tool_railway_redeploy,
    "railway_set_env": _tool_railway_set_env,
    "railway_get_status": _tool_railway_get_status,
    "railway_get_logs": _tool_railway_get_logs,
    "railway_get_domains": _tool_railway_get_domains,
    "llm_chat": _tool_llm_chat,
    "write_memory": _tool_write_memory,
    "read_memory": _tool_read_memory,
    "http_get": _tool_http_get,
    "kml_data_read": _tool_kml_data_read,
    "kml_app_logs": _tool_kml_app_logs,
    "github_list_repos": _tool_github_list_repos,
    "run_tests": _tool_run_tests,
    "railway_list": _tool_railway_list,
    "s3_status": _tool_s3_status,
    "s3_list": _tool_s3_list,
    "s3_get": _tool_s3_get,
}

if BROWSER_TOOL_AVAILABLE:
    _TOOL_HANDLERS["browser_read"] = _tool_browser_read
    _TOOL_HANDLERS["browser_research"] = _tool_browser_research
