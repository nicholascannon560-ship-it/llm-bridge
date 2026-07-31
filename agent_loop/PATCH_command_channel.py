"""Patch script: adds agent_run support to command_channel.py.

Run this from the repo root:
    python agent_loop/PATCH_command_channel.py

It modifies command_channel.py in-place (creates a .bak backup).
"""

import shutil
from pathlib import Path

TARGET = Path("command_channel.py")
BACKUP = Path("command_channel.py.bak")

if not TARGET.exists():
    print("ERROR: command_channel.py not found. Run from repo root.")
    exit(1)

with open(TARGET, "r") as f:
    original = f.read()

if "agent_run" in original:
    print("command_channel.py already has agent_run support. Nothing to do.")
    exit(0)

shutil.copy2(TARGET, BACKUP)
print(f"Backup created: {BACKUP}")

# Patch 1: Add agent_loop import
import_section = """import base64
import json
import os
import traceback"""

new_import_section = """import base64
import json
import os
import traceback

# Agent loop (optional — only loaded when agent_run is used)
try:
    from agent_loop.harness import run_agent
    from agent_loop.tools import TOOL_SCHEMAS
    AGENT_LOOP_AVAILABLE = True
except ImportError:
    AGENT_LOOP_AVAILABLE = False"""

patched = original.replace(import_section, new_import_section)

# Patch 2: Extend _llm_chat to forward tools/tool_choice
old_block = '    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "reasoning_effort": reasoning_effort,
    }'

new_block = '    payload: dict[str, Any] = {
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
        payload["tool_choice"] = cmd["tool_choice"]'

patched = patched.replace(old_block, new_block)

# Patch 3: Add agent_run to _execute
old_end = '    if action == "list_services":
        pid = cmd.get("project_id")
        if not pid:
            raise ValueError("list_services requires 'project_id'")
        return list_services(pid)

    raise ValueError(f"unknown action: {action}")'

new_end = '    if action == "list_services":
        pid = cmd.get("project_id")
        if not pid:
            raise ValueError("list_services requires 'project_id'")
        return list_services(pid)

    if action == "agent_run":
        if not AGENT_LOOP_AVAILABLE:
            raise RuntimeError("agent_loop module not available — check import")
        task = cmd.get("task")
        if not task:
            raise ValueError("agent_run requires 'task'")
        import asyncio
        return asyncio.run(run_agent(
            task=task,
            tools=cmd.get("tools"),
            max_turns=int(cmd.get("max_turns", 10)),
            provider=cmd.get("provider", "moonshot"),
            model=cmd.get("model", "kimi-k3"),
            reasoning_effort=cmd.get("reasoning_effort", "low"),
            task_id=cmd.get("id"),
        ))

    raise ValueError(f"unknown action: {action}")'

patched = patched.replace(old_end, new_end)

with open(TARGET, "w") as f:
    f.write(patched)

print("command_channel.py patched successfully.")
print("Changes made:")
print("  1. Added agent_loop import (with graceful fallback)")
print("  2. Extended _llm_chat to forward tools/tool_choice")
print("  3. Added agent_run action to _execute")
print("\nNext steps:")
print("  1. Review the diff: git diff command_channel.py")
print("  2. Commit: git add command_channel.py && git commit -m 'Add agent_run support'")
print("  3. Redeploy the bridge service")
