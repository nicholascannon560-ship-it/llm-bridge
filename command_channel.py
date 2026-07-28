"""
GitHub-based command channel for the bridge.

I (the agent) write JSON command files into commands/pending/<id>.json.
The running service picks them up (on every /health check), executes them
using its own Railway + GitHub tokens, writes the result to
commands/results/<id>.json, and deletes the pending file.

IMPORTANT: /health is also the Railway liveness probe. Processing many or
slow commands (llm_chat can take 60s+) inside /health causes 503s when the
probe times out. We therefore process at most ONE pending command per
health tick, and cap Moonshot wait time.
"""

from __future__ import annotations

import base64
import json
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
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

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OWNER = os.getenv("GITHUB_OWNER", "nicholascannon560-ship-it")
REPO = os.getenv("GITHUB_REPO", "llm-bridge")
PENDING_PATH = "commands/pending"
RESULTS_PATH = "commands/results"

MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")
MOONSHOT_URL = "https://api.moonshot.ai/v1/chat/completions"
# Match the main gateway's MOONSHOT_TIMEOUT_SEC (120). Previous 55s default
# caused frequent ReadTimeouts on anything beyond short WAKE checks.
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "120"))
MAX_CMDS_PER_HEALTH = int(os.getenv("MAX_CMDS_PER_HEALTH", "1"))


def _github_headers() -> dict[str, str]:
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN not set")
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _list_pending() -> list[dict[str, Any]]:
    url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/contents/{PENDING_PATH}"
    with httpx.Client(timeout=20) as client:
        resp = client.get(url, headers=_github_headers())
    if resp.status_code == 404:
        return []
    if resp.status_code != 200:
        raise RuntimeError(f"list pending failed: {resp.status_code} {resp.text[:300]}")
    items = resp.json()
    if not isinstance(items, list):
        return []
    return [i for i in items if i.get("type") == "file" and i.get("name", "").endswith(".json")]


def _read_file(path: str) -> tuple[str, str]:
    url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/contents/{path}"
    with httpx.Client(timeout=20) as client:
        resp = client.get(url, headers=_github_headers())
    if resp.status_code != 200:
        raise RuntimeError(f"read {path} failed: {resp.status_code}")
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]


def _write_file(path: str, content: str, message: str) -> None:
    url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/contents/{path}"
    sha = None
    with httpx.Client(timeout=20) as client:
        lookup = client.get(url, headers=_github_headers())
        if lookup.status_code == 200:
            sha = lookup.json().get("sha")

    payload: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha

    with httpx.Client(timeout=20) as client:
        resp = client.put(url, headers=_github_headers(), json=payload)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"write {path} failed: {resp.status_code} {resp.text[:300]}")


def _delete_file(path: str, sha: str, message: str) -> None:
    url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/contents/{path}"
    payload = {"message": message, "sha": sha}
    headers = _github_headers()
    headers["Content-Type"] = "application/json"
    with httpx.Client(timeout=20) as client:
        resp = client.request(
            "DELETE",
            url,
            headers=headers,
            content=json.dumps(payload),
        )
    if resp.status_code not in (200, 204):
        print(f"warning: delete {path} failed: {resp.status_code} {resp.text[:200]}")


def _llm_chat(cmd: dict[str, Any]) -> dict[str, Any]:
    provider = cmd.get("provider", "moonshot")
    model = cmd.get("model", "kimi-k3")
    messages = cmd.get("messages")
    if not messages:
        raise ValueError("llm_chat requires 'messages'")

    temperature = float(cmd.get("temperature", 1.0))
    max_tokens = int(cmd.get("max_tokens", 900))
    # Moonshot defaults to reasoning_effort="max" if omitted, which makes
    # calls slow and timeout-prone. Forward the value when present; default
    # to "low" for mechanical advisor calls.
    reasoning_effort = cmd.get("reasoning_effort", "low")

    if provider != "moonshot":
        raise ValueError(f"llm_chat currently only supports provider=moonshot, got {provider}")

    if not MOONSHOT_API_KEY:
        raise RuntimeError("MOONSHOT_API_KEY not set on the service")

    # kimi-k3 only accepts temperature=1.0; ignore other values
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "reasoning_effort": reasoning_effort,
    }

    with httpx.Client(timeout=LLM_TIMEOUT_SEC) as client:
        resp = client.post(
            MOONSHOT_URL,
            headers={
                "Authorization": f"Bearer {MOONSHOT_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Moonshot API error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content") or msg.get("reasoning_content") or ""
    usage = data.get("usage", {})

    return {
        "provider": "moonshot",
        "model": model,
        "content": content,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        },
        "finish_reason": choice.get("finish_reason"),
        "requested_temperature": temperature,
        "requested_reasoning_effort": reasoning_effort,
    }


def _execute(cmd: dict[str, Any]) -> Any:
    action = cmd.get("action")
    if not action:
        raise ValueError("missing 'action'")

    if action == "llm_chat":
        return _llm_chat(cmd)

    if action == "railway_gql":
        query = cmd.get("query")
        if not query:
            raise ValueError("railway_gql requires 'query'")
        return railway_query(query, cmd.get("variables"))

    if action == "set_env":
        name = cmd.get("name")
        value = cmd.get("value")
        if not name or value is None:
            raise ValueError("set_env requires 'name' and 'value'")
        return set_service_variable(
            name=name,
            value=str(value),
            service_id=cmd.get("service_id"),
            environment_name=cmd.get("environment_name"),
        )

    if action == "get_status":
        sid = cmd.get("service_id") or BRIDGE_SERVICE_ID
        return get_service_status(sid)

    if action == "get_logs":
        did = cmd.get("deployment_id")
        if not did:
            raise ValueError("get_logs requires 'deployment_id'")
        return get_logs(did, limit=int(cmd.get("limit", 100)))

    if action == "redeploy":
        sid = cmd.get("service_id") or BRIDGE_SERVICE_ID
        return redeploy_service(sid, environment=cmd.get("environment", "production"))

    if action == "list_projects":
        return list_projects()

    if action == "list_services":
        pid = cmd.get("project_id")
        if not pid:
            raise ValueError("list_services requires 'project_id'")
        return list_services(pid)

    raise ValueError(f"unknown action: {action}")


def process_pending_commands() -> dict[str, Any]:
    """
    Main entry point. Called from /health.
    Processes at most MAX_CMDS_PER_HEALTH files so a probe never stacks
    multiple long Moonshot calls and trips Railway 503.
    """
    summary: dict[str, Any] = {"processed": 0, "errors": 0, "ids": [], "skipped": 0}

    try:
        pending = _list_pending()
    except Exception as e:
        return {"error": f"list failed: {e}", **summary}

    # Stable order: oldest first if GitHub returns by name; else as listed
    pending = [i for i in pending if i.get("name") != ".gitkeep"]
    if len(pending) > MAX_CMDS_PER_HEALTH:
        summary["skipped"] = len(pending) - MAX_CMDS_PER_HEALTH
        pending = pending[:MAX_CMDS_PER_HEALTH]

    for item in pending:
        name = item["name"]
        cmd_id = name.replace(".json", "")
        path = f"{PENDING_PATH}/{name}"

        try:
            raw, sha = _read_file(path)
            cmd = json.loads(raw)

            result_payload = {
                "id": cmd_id,
                "status": "ok",
                "action": cmd.get("action"),
                "result": None,
                "error": None,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }

            try:
                result_payload["result"] = _execute(cmd)
            except Exception as exec_err:
                result_payload["status"] = "error"
                result_payload["error"] = str(exec_err)
                result_payload["traceback"] = traceback.format_exc()[-1500:]
                summary["errors"] += 1

            result_path = f"{RESULTS_PATH}/{cmd_id}.json"
            _write_file(
                result_path,
                json.dumps(result_payload, indent=2, default=str),
                f"cmd result: {cmd_id}",
            )
            _delete_file(path, sha, f"cmd processed: {cmd_id}")

            summary["processed"] += 1
            summary["ids"].append(cmd_id)

        except Exception as e:
            summary["errors"] += 1
            try:
                err_payload = {
                    "id": cmd_id,
                    "status": "error",
                    "error": str(e),
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }
                _write_file(
                    f"{RESULTS_PATH}/{cmd_id}.json",
                    json.dumps(err_payload, indent=2),
                    f"cmd error: {cmd_id}",
                )
            except Exception:
                pass

    return summary
