"""The console's auto-commit switch, end to end through the request body.

The API half of auto mode shipped once with no way to reach it from the
console: /agent/auto_mode sits behind the X-Bridge-Key middleware, which the
browser does not have, so there was nothing an operator could actually click.
These tests pin the three links in the chain that fix that — the control is in
the served page, the request body carries the flag, and ui_do hands it to
run_agent.
"""
import ast
import os
import sys

os.environ.setdefault("BRIDGE_API_KEY", "testkey")
os.environ.setdefault("UI_PASSWORD", "hunter2")
os.environ.setdefault("UI_SESSION_SECRET", "test-secret")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chat_ui  # noqa: E402

_SRC = open(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "chat_ui.py")
).read()


def test_request_body_accepts_auto_mode():
    """None means 'defer to the global toggle' — it must not coerce to False."""
    assert chat_ui.DoRequestBody(message="x").auto_mode is None
    assert chat_ui.DoRequestBody(message="x", auto_mode=True).auto_mode is True
    assert chat_ui.DoRequestBody(message="x", auto_mode=False).auto_mode is False


def test_ui_do_forwards_auto_mode_to_run_agent():
    tree = ast.parse(_SRC)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "ui_do")
    call = next(n for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "run_agent")
    assert "auto_mode" in {kw.arg for kw in call.keywords}


def test_console_renders_the_switch():
    """The operator-facing half: a control that exists and is off by default."""
    assert 'class="seg" id="automode"' in _SRC
    assert 'data-v="off"' in _SRC and 'data-v="on"' in _SRC
    assert 'automode: "off"' in _SRC          # default is ask-first
    assert 'auto_mode: CFG.automode === "on"' in _SRC   # reaches the wire
    assert "function paintAuto()" in _SRC     # state is visible on the chip
