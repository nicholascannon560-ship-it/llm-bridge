"""Auto mode — operator toggle for commit-without-asking agent behavior.

Default OFF. When ON, agent runs are told (via an extra system-prompt block)
to commit their work directly instead of pausing to ask the operator for
confirmation. This is a prompt-level policy change, not a guard removal:
protected Railway env names, secret handling and delete guards stay exactly
as they are.

Three layers:
  - AGENT_AUTO_MODE env var sets the boot default (read again on restart).
  - POST /agent/auto_mode {"enabled": true|false} flips it live, no redeploy.
  - An agent_run command may carry "auto_mode": true/false to override for
    that one run without touching the global switch.

State lives in memory: a container restart falls back to the env default,
which is the conservative choice.
"""
import os
import threading

_lock = threading.Lock()


def _env_default() -> bool:
    return os.getenv("AGENT_AUTO_MODE", "0").strip().lower() in ("1", "true", "yes", "on")


_state = {"on": _env_default()}


def is_auto() -> bool:
    with _lock:
        return _state["on"]


def set_auto(enabled: bool) -> dict:
    """Flip the global toggle. Returns the new state for the endpoint."""
    with _lock:
        _state["on"] = bool(enabled)
        return {
            "auto_mode": _state["on"],
            "set_by": "runtime",
            "boot_default_env": os.getenv("AGENT_AUTO_MODE", "0"),
            "note": (
                "Runtime toggle resets to the AGENT_AUTO_MODE env value on "
                "container restart."
            ),
        }


# Injected into the agent's system prompt when auto mode is active.
AUTO_PROMPT_BLOCK = """

AUTO MODE IS ON (operator-enabled). Additional standing rules:
11. Commit your work directly with github_commit / github_patch as you complete
    each coherent unit. Do NOT pause to ask the operator for permission before
    committing, and do not end the task with uncommitted work sitting in your
    last message.
12. For edits to existing files prefer github_patch over github_commit.
13. Auto mode removes the commit-confirmation step ONLY. Still forbidden
    without an explicit instruction this run: touching secrets or tokens,
    railway_set_env on protected variable names, DELETE endpoints, repo or
    file deletion, and anything destructive.
14. List every commit you made (repo, path, branch) in your final summary so
    the operator can audit the run afterwards."""
