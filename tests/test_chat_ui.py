"""Console tests: cookie auth, cache-prefix stability, chat cannot use tools."""
import os
import sys
import time

os.environ.setdefault("BRIDGE_API_KEY", "testkey")
os.environ["UI_PASSWORD"] = "hunter2"
os.environ["UI_SESSION_SECRET"] = "test-secret"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chat_ui

fails = []


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        fails.append(label)


print("1. cookie signing")
good = chat_ui.mint_cookie()
check(chat_ui.valid_cookie(good), "freshly minted cookie is valid")
check(not chat_ui.valid_cookie(None), "missing cookie rejected")
check(not chat_ui.valid_cookie("garbage"), "malformed cookie rejected")
check(not chat_ui.valid_cookie("999.deadbeef"), "bad signature rejected")
stamp, _, sig = good.partition(".")
check(
    not chat_ui.valid_cookie(f"{int(stamp) + 9999}.{sig}"),
    "extending expiry invalidates the signature",
)
expired = int(time.time()) - 10
check(
    not chat_ui.valid_cookie(f"{expired}.{chat_ui._sign(expired)}"),
    "correctly signed but expired cookie rejected",
)
# A cookie must not be forgeable by someone who guesses a different secret.
os.environ["UI_SESSION_SECRET"] = "other-secret"
check(not chat_ui.valid_cookie(good), "cookie from a different secret rejected")
os.environ["UI_SESSION_SECRET"] = "test-secret"
check(chat_ui.valid_cookie(good), "original secret validates again")

print("2. system prompt is cache-stable")
p1 = chat_ui.build_system_prompt()
p2 = chat_ui.build_system_prompt()
check(p1 == p2, "identical across calls (byte-for-byte)")
check("task_id" not in p1, "carries no per-run task_id")
s = chat_ui.Session("sess1", p1)
s.messages.append({"role": "user", "content": "hi"})
check(s.system_prompt == p1, "session pins its prompt for its whole life")

print("3. session store")
chat_ui.SESSION_DIR = __import__("pathlib").Path(
    __import__("tempfile").mkdtemp()
)
a = chat_ui.get_session(None)
check(a.id in chat_ui.SESSIONS, "new session registered")
check(chat_ui.get_session(a.id) is a, "same id returns the same object")
a.messages.append({"role": "user", "content": "remember me"})
a.persist()
del chat_ui.SESSIONS[a.id]
restored = chat_ui.get_session(a.id)
check(restored.messages == a.messages, "history survives a reload from disk")
check(restored.system_prompt == a.system_prompt, "prompt survives reload (cache intact)")

print("4. trim keeps the message list valid")
chat_ui.MAX_HISTORY_MESSAGES = 4
t = chat_ui.Session("trim", "sys")
t.messages = [
    {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "a1"},
    {"role": "tool", "content": "orphan1"},
    {"role": "tool", "content": "orphan2"},
    {"role": "user", "content": "u2"},
    {"role": "assistant", "content": "a2"},
]
t.trim()
check(len(t.messages) <= 4, "trimmed to the cap")
check(
    t.messages[0]["role"] != "tool",
    "never leaves a leading orphan tool result (provider would 400)",
)

print("5. one mode: the console has a single send path")
src = open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chat_ui.py")
).read()
# The chat/executor split is gone, and with it the router call, the
# confirmation card, and the handoff brief that only existed because the two
# halves ran on different models.
for gone in ("async def ui_chat", "async def ui_turn", "async def ui_auto",
             "async def _classify", "_build_handoff_brief", "_ROUTER_PROMPT"):
    check(gone not in src, f"removed: {gone}")
check("async def ui_do" in src, "/ui/do is the single send path")
check("async def ui_approve" in src, "commit approval endpoint exists")

print("6. writing to a repo is gated; reading is not")
from agent_loop import harness as _h
check("github_commit" in _h.APPROVAL_TOOLS, "github_commit requires approval")
check("github_patch" in _h.APPROVAL_TOOLS, "github_patch requires approval")
check("github_read" not in _h.APPROVAL_TOOLS, "reads are never gated")
check("repo_search" not in _h.APPROVAL_TOOLS, "searches are never gated")

print("7. spend warns, it does not cut the run off")
check(_h.SPEND_WARN_CENTS > 0, "a spend warning threshold is set")
check("cost_budget_reached" not in src, "console no longer stops runs on cost")
loop_src = open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "agent_loop", "harness.py")
).read()
check("spend_warning" in loop_src, "loop emits a spend warning event")
check(loop_src.count('status = "cost_budget_reached"') == 0,
      "loop has no cost-based stop condition left")

print()
if fails:
    print(f"FAILED ({len(fails)}): " + "; ".join(fails))
    sys.exit(1)
print("ALL PASS")
