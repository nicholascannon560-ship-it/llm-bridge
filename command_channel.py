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
import threading
import time
import traceback

# Agent loop (optional — only loaded when agent_run is used)
try:
    from agent_loop.harness import run_agent
    from agent_loop.tools import TOOL_SCHEMAS, env_name_is_protected
    AGENT_LOOP_AVAILABLE = True
except Exception as _agent_loop_err:
    AGENT_LOOP_AVAILABLE = False
    # Log the import failure so we know why agent_loop is disabled
    print(f'[agent_loop] import failed: {_agent_loop_err}', flush=True)
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
# Code deploys from one branch and the queue was read from the repo default
# branch — they drifted, and a command committed to the wrong one is invisible
# forever with no error. Pin both to one env var.
BRANCH = os.getenv("GITHUB_BRANCH", "").strip()
PENDING_PATH = "commands/pending"
RESULTS_PATH = "commands/results"
# In-progress state (agent_run placeholder + checkpoints) lives here, NOT in
# commands/results. Two writers on one path raced: the placeholder and the
# worker's final payload have no ordering, so a fast run either 409'd or — worse
# — had its real result overwritten by a late "started" placeholder and was
# frozen at that forever. commands/results now has exactly ONE writer per id:
# the final write. See llm-bridge issue #2.
RUNNING_PATH = "commands/running"
# One retry is enough for a stale-read 409; more just delays a real conflict.
WRITE_CONFLICT_RETRIES = int(os.getenv("WRITE_CONFLICT_RETRIES", "1"))
WRITE_CONFLICT_BACKOFF_SEC = float(os.getenv("WRITE_CONFLICT_BACKOFF_SEC", "0.5"))

MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")
MOONSHOT_URL = "https://api.moonshot.ai/v1/chat/completions"
# Match the main gateway's MOONSHOT_TIMEOUT_SEC (120). Previous 55s default
# caused frequent ReadTimeouts on anything beyond short WAKE checks.
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "120"))
MAX_CMDS_PER_HEALTH = int(os.getenv("MAX_CMDS_PER_HEALTH", "1"))
# A command that fails outside _execute (empty file, unreadable, bad JSON)
# used to stay in pending forever and retry on every health tick. Cap the
# attempts and quarantine it instead.
MAX_CMD_ATTEMPTS = int(os.getenv("MAX_CMD_ATTEMPTS", "3"))

# Kill switch. Set AGENT_LOOP_ENABLED=0 in Railway to stop all autonomous runs
# without redeploying code.
AGENT_LOOP_ENABLED = os.getenv("AGENT_LOOP_ENABLED", "1").lower() not in ("0", "false", "no")
# One agent run at a time, and never inside the health probe (see below).
_agent_slot = threading.BoundedSemaphore(1)


def _github_headers() -> dict[str, str]:
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN not set")
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _ref_params() -> dict[str, str]:
    return {"ref": BRANCH} if BRANCH else {}


def _list_pending() -> list[dict[str, Any]]:
    url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/contents/{PENDING_PATH}"
    with httpx.Client(timeout=20) as client:
        resp = client.get(url, headers=_github_headers(), params=_ref_params())
    if resp.status_code != 200 and resp.status_code != 404:
        raise RuntimeError(f"list pending failed: {resp.status_code} {resp.text[:300]}")
    if resp.status_code == 404:
        return []
    items = resp.json()
    if not isinstance(items, list):
        return []
    return [i for i in items if i.get("type") == "file" and i.get("name", "").endswith(".json")]


def _read_file(path: str) -> tuple[str, str]:
    url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/contents/{path}"
    with httpx.Client(timeout=20) as client:
        resp = client.get(url, headers=_github_headers(), params=_ref_params())
    if resp.status_code != 200:
        raise RuntimeError(f"read {path} failed: {resp.status_code}")
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]


def _write_file(path: str, content: str, message: str) -> None:
    """Commit `content` to `path`, re-resolving the blob sha on conflict.

    The sha lookup and the PUT are two round trips; anything that writes the
    same path in between makes the PUT 409. Retrying with a freshly read sha
    resolves the benign case (a stale read). This is a backstop, not the
    ordering fix — see RUNNING_PATH above for that.
    """
    url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/contents/{path}"
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")

    last_resp = None
    for attempt in range(WRITE_CONFLICT_RETRIES + 1):
        sha = None
        with httpx.Client(timeout=20) as client:
            lookup = client.get(url, headers=_github_headers(), params=_ref_params())
            if lookup.status_code == 200:
                sha = lookup.json().get("sha")

        payload: dict[str, Any] = {"message": message, "content": encoded}
        if BRANCH:
            payload["branch"] = BRANCH
        if sha:
            payload["sha"] = sha

        with httpx.Client(timeout=20) as client:
            resp = client.put(url, headers=_github_headers(), json=payload)
        if resp.status_code in (200, 201):
            return
        last_resp = resp
        if resp.status_code != 409:
            break
        if attempt < WRITE_CONFLICT_RETRIES:
            time.sleep(WRITE_CONFLICT_BACKOFF_SEC * (attempt + 1))

    raise RuntimeError(
        f"write {path} failed: {last_resp.status_code} {last_resp.text[:300]}"
    )


def _delete_file(path: str, sha: str, message: str) -> None:
    url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/contents/{path}"
    payload: dict[str, Any] = {"message": message, "sha": sha}
    if BRANCH:
        payload["branch"] = BRANCH
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
    # Forward tool schemas if provided (enables agent loops via command channel)
    if cmd.get("tools"):
        payload["tools"] = cmd["tools"]
    if cmd.get("tool_choice") is not None:
        payload["tool_choice"] = cmd["tool_choice"]

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


def _start_agent_run(cmd: dict[str, Any], cmd_id: str) -> None:
    """Run an agent on a background thread and write its result when done.

    Caller must already hold _agent_slot; this releases it.
    """
    import asyncio

    from agent_loop.browser import RunAuthorization

    checkpoint_every = int(os.getenv("AGENT_CHECKPOINT_EVERY_TURNS", "5"))

    def _checkpoint(partial: dict[str, Any]) -> None:
        """Write an in-progress snapshot so a run is observable while it runs.

        Rate-limited by AGENT_CHECKPOINT_EVERY_TURNS because every write here is
        a commit; set it to 0 to disable and rely on GET /agent/status instead.
        """
        try:
            _write_file(
                f"{RUNNING_PATH}/{cmd_id}.json",
                json.dumps({"id": cmd_id, "action": "agent_run", "status": "running",
                            "result": partial, "error": None,
                            "checkpoint_at": datetime.now(timezone.utc).isoformat()},
                           indent=2, default=str),
                f"agent checkpoint: {cmd_id} turn {partial.get('turns_used')}",
            )
        except Exception as e:
            print(f"[agent_loop] checkpoint failed for {cmd_id}: {e}", flush=True)

    def _worker() -> None:
        payload: dict[str, Any] = {
            "id": cmd_id,
            "action": "agent_run",
            "status": "ok",
            "result": None,
            "error": None,
        }
        try:
            payload["result"] = asyncio.run(
                run_agent(
                    task=cmd["task"],
                    tools=cmd.get("tools"),
                    tool_set=cmd.get("tool_set"),
                    max_turns=int(cmd.get("max_turns", 10)),
                    provider=cmd.get("provider", "moonshot"),
                    model=cmd.get("model", "kimi-k3"),
                    reasoning_effort=cmd.get("reasoning_effort", "low"),
                    max_tokens=(int(cmd["max_tokens"]) if cmd.get("max_tokens") else None),
                    on_checkpoint=(_checkpoint if checkpoint_every > 0 else None),
                    checkpoint_every=checkpoint_every,
                    task_id=cmd_id or cmd.get("id"),
                    browser_auth=RunAuthorization(),
                )
            )
        except Exception as exc:
            payload["status"] = "error"
            payload["error"] = str(exc)
            payload["traceback"] = traceback.format_exc()[-1500:]
        finally:
            _agent_slot.release()

        payload["processed_at"] = datetime.now(timezone.utc).isoformat()
        try:
            _write_file(
                f"{RESULTS_PATH}/{cmd_id}.json",
                json.dumps(payload, indent=2, default=str),
                f"agent result: {cmd_id}",
            )
        except Exception as write_err:  # nothing useful left to do but say so
            print(f"[agent_loop] could not write result for {cmd_id}: {write_err}", flush=True)

        # Clear the in-progress marker last: while it exists the run is either
        # live or died without writing a result, and that distinction is the
        # whole point of the separate path. Failure here is cosmetic.
        for attempt in range(3):
            try:
                _, running_sha = _read_file(f"{RUNNING_PATH}/{cmd_id}.json")
                _delete_file(f"{RUNNING_PATH}/{cmd_id}.json", running_sha,
                             f"agent run finished: {cmd_id}")
                break
            except Exception:
                time.sleep(0.5 * (attempt + 1))

    threading.Thread(target=_worker, name=f"agent-{cmd_id or 'run'}", daemon=True).start()


# ── byte-exact repo file tools (2026-08-06) ─────────────────────────────────
# LLM transcription of large files is lossy (observed: a 143792-byte dashboard.py
# silently lost ~43KB in transit, caught only by blob-SHA check). These two
# actions move the bytes with the bridge's own GitHub token and let the harness
# write the results, so file content never passes through a model. patch_file
# additionally fail-safes every op (anchor must match exactly once) and
# compile-checks Python before committing, so a bad patch can never deploy.

def _read_repo_file(repo: str, path: str, branch: str) -> tuple[str, str]:
    """Read a file from ANY repo the bridge token can see. Returns (content, blob_sha)."""
    url = f"{GITHUB_API}/repos/{OWNER}/{repo}/contents/{path}"
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, headers=_github_headers(), params={"ref": branch})
    if resp.status_code != 200:
        raise RuntimeError(f"read {repo}/{path} failed: {resp.status_code}")
    data = resp.json()
    return base64.b64decode(data["content"]).decode("utf-8"), data["sha"]


def _read_file_window(cmd: dict[str, Any]) -> dict[str, Any]:
    repo = cmd.get("repo") or REPO
    branch = cmd.get("branch") or BRANCH or "main"
    path = cmd.get("path")
    if not path:
        raise ValueError("read_file_window requires 'path'")
    content, sha = _read_repo_file(repo, path, branch)
    if cmd.get("find") is not None:
        needle = str(cmd["find"])
        offs = []
        start = 0
        while len(offs) < 50:
            i = content.find(needle, start)
            if i < 0:
                break
            offs.append(i)
            start = i + 1
        return {"repo": repo, "path": path, "branch": branch, "sha": sha,
                "total_chars": len(content), "find": needle, "offsets": offs}
    offset = max(0, int(cmd.get("offset", 0)))
    max_chars = min(40000, max(1, int(cmd.get("max_chars", 8000))))
    window = content[offset:offset + max_chars]
    return {"repo": repo, "path": path, "branch": branch, "sha": sha,
            "total_chars": len(content), "offset": offset,
            "returned_chars": len(window),
            "truncated": (offset + len(window)) < len(content),
            "next_offset": offset + len(window),
            "content": window}


def _patch_file(cmd: dict[str, Any]) -> dict[str, Any]:
    repo = cmd.get("repo") or REPO
    branch = cmd.get("branch") or BRANCH or "main"
    path = cmd.get("path")
    ops = cmd.get("ops") or []
    message = cmd.get("message") or f"patch_file: {path}"
    if not path or not ops:
        raise ValueError("patch_file requires 'path' and non-empty 'ops'")
    content, sha = _read_repo_file(repo, path, branch)
    applied = []
    for i, op in enumerate(ops):
        if "replace" in op:
            old = op["replace"]["old"]
            new = op["replace"]["new"]
            n = content.count(old)
            if n != 1:
                raise ValueError(f"op {i}: replace anchor occurs {n}x (need exactly 1); nothing committed")
            content = content.replace(old, new, 1)
            applied.append({"op": i, "type": "replace"})
        elif "insert_before" in op:
            anchor = op["insert_before"]["anchor"]
            text = op["insert_before"]["text"]
            n = content.count(anchor)
            if n != 1:
                raise ValueError(f"op {i}: insert_before anchor occurs {n}x (need exactly 1); nothing committed")
            content = content.replace(anchor, text + anchor, 1)
            applied.append({"op": i, "type": "insert_before"})
        elif "insert_after" in op:
            anchor = op["insert_after"]["anchor"]
            text = op["insert_after"]["text"]
            n = content.count(anchor)
            if n != 1:
                raise ValueError(f"op {i}: insert_after anchor occurs {n}x (need exactly 1); nothing committed")
            content = content.replace(anchor, anchor + text, 1)
            applied.append({"op": i, "type": "insert_after"})
        else:
            raise ValueError(f"op {i}: unknown op shape; nothing committed")
    if path.endswith(".py"):
        compile(content, path, "exec")  # raises BEFORE commit on any syntax error
    url = f"{GITHUB_API}/repos/{OWNER}/{repo}/contents/{path}"
    payload: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": branch,
        "sha": sha,
    }
    with httpx.Client(timeout=30) as client:
        w = client.put(url, headers=_github_headers(), json=payload)
    if w.status_code not in (200, 201):
        raise RuntimeError(f"commit failed: {w.status_code} {w.text[:300]}")
    return {"repo": repo, "path": path, "branch": branch,
            "old_blob_sha": sha,
            "new_commit": (w.json().get("commit") or {}).get("sha"),
            "chars_after": len(content), "ops_applied": applied}


def _execute(cmd: dict[str, Any], cmd_id: str = "") -> Any:
    action = cmd.get("action")
    if not action:
        raise ValueError("missing 'action'")

    if action == "llm_chat":
        return _llm_chat(cmd)

    if action == "read_file_window":
        return _read_file_window(cmd)

    if action == "patch_file":
        return _patch_file(cmd)

    if action == "railway_gql":
        query = cmd.get("query")
        if not query:
            raise ValueError("railway_gql requires 'query'")
        return railway_query(query, cmd.get("variables"))

    if action == "set_env":
        name = cmd.get("name")
        value = cmd.get("value")
        # Support base64-encoded values to bypass GitHub secret scanning
        if value is None and "value_b64" in cmd:
            import base64
            value = base64.b64decode(cmd["value_b64"]).decode("utf-8")
        if not name or value is None:
            raise ValueError("set_env requires 'name' and 'value' (or 'value_b64')")
        if AGENT_LOOP_AVAILABLE and env_name_is_protected(name):
            raise ValueError(
                f"{name} is operator-only — permission grants and secrets are set by hand "
                "in the Railway dashboard, not through the command channel"
            )
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

    if action == "agent_run":
        if not AGENT_LOOP_AVAILABLE:
            raise RuntimeError("agent_loop module not available — check import")
        if not AGENT_LOOP_ENABLED:
            raise RuntimeError("agent runs are disabled (AGENT_LOOP_ENABLED=0)")
        task = cmd.get("task")
        if not task:
            raise ValueError("agent_run requires 'task'")
        if not _agent_slot.acquire(blocking=False):
            raise RuntimeError("an agent run is already in progress; try again when it finishes")

        # An agent run takes minutes. _execute is called from GET /health,
        # which is Railway's liveness probe, so it must NOT block: a long run
        # here means failed probes, a restarted container, and a half-finished
        # run with no result. Start it on a background thread and return
        # immediately; the thread overwrites the result file when it is done.
        started = {
            "id": cmd_id,
            "action": "agent_run",
            "status": "running",
            "result": {"status": "started", "task_id": cmd_id},
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        # Write the marker BEFORE the thread exists. Writing it afterwards is a
        # race the worker wins on a fast crash: it looks for a marker that has
        # not been written yet, finds nothing to clear, and the marker lands
        # after the run is already over and outlives it. Observed live on the
        # first deploy of this fix.
        try:
            _write_file(
                f"{RUNNING_PATH}/{cmd_id}.json",
                json.dumps(started, indent=2, default=str),
                f"agent started: {cmd_id}",
            )
        except Exception as e:
            print(f"[agent_loop] could not write running marker for {cmd_id}: {e}", flush=True)

        try:
            _start_agent_run(cmd, cmd_id)
        except Exception:
            _agent_slot.release()
            raise
        return {
            "status": "started",
            "task_id": cmd_id,
            "note": "running in the background; this result file is overwritten on completion",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

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
                result_payload["result"] = _execute(cmd, cmd_id)
            except Exception as exec_err:
                result_payload["status"] = "error"
                result_payload["error"] = str(exec_err)
                result_payload["traceback"] = traceback.format_exc()[-1500:]
                summary["errors"] += 1

            # A backgrounded run (agent_run) returns immediately with a
            # "started" stub while its worker thread keeps going. Writing that
            # stub to commands/results would race the worker's final payload —
            # send it to commands/running instead so results stays single-writer.
            res = result_payload.get("result")
            backgrounded = (
                result_payload["status"] == "ok"
                and isinstance(res, dict)
                and res.get("status") == "started"
                and res.get("task_id")
            )
            # A backgrounded run already wrote its own marker to
            # commands/running before the worker started; writing anything here
            # would either duplicate it or race the worker's final payload.
            if not backgrounded:
                _write_file(
                    f"{RESULTS_PATH}/{cmd_id}.json",
                    json.dumps(result_payload, indent=2, default=str),
                    f"cmd result: {cmd_id}",
                )
            _delete_file(path, sha, f"cmd processed: {cmd_id}")

            summary["processed"] += 1
            summary["ids"].append(cmd_id)

        except Exception as e:
            summary["errors"] += 1

            # Count prior failures for this id. The result file is the only
            # durable state we have, so the attempt counter lives there.
            attempts = 1
            try:
                prior_raw, _ = _read_file(f"{RESULTS_PATH}/{cmd_id}.json")
                attempts = int(json.loads(prior_raw).get("attempts", 0)) + 1
            except Exception:
                pass

            give_up = attempts >= MAX_CMD_ATTEMPTS
            status = "quarantined" if give_up else "error"

            try:
                err_payload = {
                    "id": cmd_id,
                    "status": status,
                    "error": str(e),
                    "attempts": attempts,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }
                _write_file(
                    f"{RESULTS_PATH}/{cmd_id}.json",
                    json.dumps(err_payload, indent=2),
                    f"cmd {status}: {cmd_id}",
                )
            except Exception:
                pass

            # Remove the poison file so the next tick can make progress. The
            # sha comes from the directory listing, so this works even when
            # the file itself could not be read or parsed.
            if give_up:
                sha = item.get("sha")
                if sha:
                    try:
                        _delete_file(path, sha, f"cmd quarantined: {cmd_id}")
                    except Exception:
                        pass
                summary["quarantined"] = summary.get("quarantined", 0) + 1

    return summary
