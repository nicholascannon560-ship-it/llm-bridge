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

print("5. chat mode forbids tools")
src = open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chat_ui.py")
).read()
chat_fn = src[src.index("async def ui_chat") : src.index("_DO_SLOT")]
check('tool_choice="none"' in chat_fn, "chat sends tool_choice='none'")
check("_tool_schemas()" in chat_fn, "chat still sends schemas (shared cache prefix)")
check("run_tool" not in chat_fn, "chat never executes a tool")

print("6. budget is per-request and bounded")
body = chat_ui.DoRequestBody(message="x", budget_usd=5)
check(body.budget_usd == 5, "budget_usd accepted")
check(body.max_turns is None, "max_turns defaults to None -> harness default")
try:
    chat_ui.DoRequestBody(message="x", budget_usd=0)
    check(False, "zero budget rejected")
except Exception:
    check(True, "zero budget rejected")
try:
    chat_ui.DoRequestBody(message="x", budget_usd=999)
    check(False, "absurd budget rejected")
except Exception:
    check(True, "absurd budget rejected")

print()
if fails:
    print(f"FAILED ({len(fails)}): " + "; ".join(fails))
    sys.exit(1)
print("ALL PASS")
