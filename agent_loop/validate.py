"""Validation test for agent_loop module.

Run this inside the bridge container (Railway shell) or locally with the
bridge's Python environment to verify agent_loop loads and behaves correctly
BEFORE patching command_channel.py.

Usage:
    python agent_loop/validate.py

What it checks:
  1. All agent_loop modules import without error
  2. Tool schemas are valid OpenAI function definitions
  3. MemoryStore works (append, read, search)
  4. AgentHarness instantiates correctly
  5. Tool handlers are registered
  6. No syntax errors in any file
"""

import ast
import json
import os
import sys
from pathlib import Path

ERRORS = []
WARNINGS = []

def error(msg):
    ERRORS.append(msg)
    print(f"  [FAIL] {msg}")

def warn(msg):
    WARNINGS.append(msg)
    print(f"  [WARN] {msg}")

def ok(msg):
    print(f"  [OK]   {msg}")

print("=" * 60)
print("AGENT_LOOP VALIDATION")
print("=" * 60)

# ── 1. Syntax check all .py files ──────────────────────────────────────────
print("\n1. Syntax check all .py files")
py_files = [
    "agent_loop/__init__.py",
    "agent_loop/harness.py",
    "agent_loop/tools.py",
    "agent_loop/memory.py",
    "agent_loop/PATCH_command_channel.py",
]
for fpath in py_files:
    try:
        with open(fpath, "r") as f:
            source = f.read()
        ast.parse(source)
        ok(f"{fpath} — syntax valid")
    except SyntaxError as e:
        error(f"{fpath} — SyntaxError at line {e.lineno}: {e.msg}")
    except FileNotFoundError:
        error(f"{fpath} — file not found")

# ── 2. Module imports ──────────────────────────────────────────────────────
print("\n2. Module imports")
try:
    from agent_loop import run_agent, DEFAULT_TOOLS, TOOL_SCHEMAS, MemoryStore
    ok("agent_loop imports successfully")
except Exception as e:
    error(f"agent_loop import failed: {e}")
    print("\nCannot continue without imports. Fix and rerun.")
    sys.exit(1)

try:
    from agent_loop.harness import AgentHarness
    ok("AgentHarness imports")
except Exception as e:
    error(f"AgentHarness import failed: {e}")

try:
    from agent_loop.tools import run_tool, _TOOL_HANDLERS
    ok("Tool handlers import")
except Exception as e:
    error(f"Tool handlers import failed: {e}")

# ── 3. Tool schema validation ─────────────────────────────────────────────
print("\n3. Tool schema validation")
required_keys = {"type", "function"}
func_required = {"name", "description", "parameters"}
for tool in TOOL_SCHEMAS:
    name = tool.get("function", {}).get("name", "<unknown>")
    if not required_keys.issubset(tool.keys()):
        error(f"Tool {name}: missing top-level keys")
    func = tool.get("function", {})
    if not func_required.issubset(func.keys()):
        error(f"Tool {name}: missing function keys")
    params = func.get("parameters", {})
    if params.get("type") != "object":
        warn(f"Tool {name}: parameters.type is not 'object'")
    if "properties" not in params:
        warn(f"Tool {name}: parameters missing 'properties'")
    if "required" not in params:
        warn(f"Tool {name}: parameters missing 'required' list")
ok(f"All {len(TOOL_SCHEMAS)} tool schemas validated")

# ── 4. Tool handler registration ───────────────────────────────────────────
print("\n4. Tool handler registration")
expected_tools = [
    "github_read", "github_commit", "railway_redeploy",
    "railway_set_env", "railway_get_status", "railway_get_logs",
    "llm_chat", "write_memory", "read_memory",
]
for tname in expected_tools:
    if tname in _TOOL_HANDLERS:
        ok(f"Handler registered: {tname}")
    else:
        error(f"Handler missing: {tname}")

# ── 5. MemoryStore functionality ────────────────────────────────────────────
print("\n5. MemoryStore functionality")
test_path = "/tmp/agent_test_memory.jsonl"
if os.path.exists(test_path):
    os.remove(test_path)

try:
    store = MemoryStore(path=test_path)
    entry = store.append("Test reflection", tags=["test", "validation"])
    ok(f"Memory append: id={entry['id']}")

    entries = store.read(limit=5)
    if len(entries) == 1 and entries[0]["entry"] == "Test reflection":
        ok("Memory read back correct")
    else:
        error("Memory read returned wrong data")

    results = store.search("reflection")
    if len(results) == 1:
        ok("Memory search works")
    else:
        error("Memory search failed")

    os.remove(test_path)
    ok("Test memory cleaned up")
except Exception as e:
    error(f"MemoryStore test failed: {e}")

# ── 6. AgentHarness instantiation (no LLM call) ─────────────────────────────
print("\n6. AgentHarness instantiation")
try:
    harness = AgentHarness(
        task="Validate the agent loop module",
        max_turns=3,
        provider="moonshot",
        model="kimi-k3",
    )
    ok(f"AgentHarness created: task_id={harness.task_id}")
    ok(f"System prompt built: {len(harness._build_system_prompt())} chars")
    ok(f"Default tools loaded: {len(harness.tools)}")
except Exception as e:
    error(f"AgentHarness instantiation failed: {e}")

# ── 7. Check bridge environment compatibility ──────────────────────────────
print("\n7. Bridge environment compatibility")
try:
    from llm_gateway import ChatRequest, ChatMessage, get_router
    ok("llm_gateway imports (bridge env confirmed)")
    router = get_router()
    providers = router.available_providers()
    ok(f"Available providers: {providers}")
except ImportError:
    warn("llm_gateway not importable — expected outside bridge container")
except Exception as e:
    error(f"llm_gateway check failed: {e}")

try:
    from railway_extension import BRIDGE_SERVICE_ID
    ok(f"railway_extension imports (service_id: {BRIDGE_SERVICE_ID[:8]}...)")
except ImportError:
    warn("railway_extension not importable — expected outside bridge container")
except Exception as e:
    error(f"railway_extension check failed: {e}")

# ── 8. Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Errors:   {len(ERRORS)}")
print(f"Warnings: {len(WARNINGS)}")
if ERRORS:
    print("\nFAILED — fix errors before patching command_channel.py")
    sys.exit(1)
else:
    print("\nPASSED — agent_loop module is healthy.")
    print("Safe to proceed: create branch, apply PATCH_command_channel.py, test deploy.")
    sys.exit(0)
