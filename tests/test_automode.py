"""Tests for the auto-mode toggle (agent_loop/automode.py + harness wiring).

The sandbox has none of the bridge's third-party deps, so bridge modules are
stubbed in sys.modules before importing agent_loop — the same name-binding
path used in the real bridge process.
"""
import asyncio  # noqa: F401  (kept for parity with other test modules)
import inspect
import os
import sys
import types

sys.path.insert(0, ".")


def _real_modules_importable() -> bool:
    """True when the bridge's own deps are present (httpx, fastapi, ...).

    Only stub when they are NOT. Installing fakes into sys.modules
    unconditionally leaks into every test collected after this one — it broke
    test_cached_usage, which needs the real llm_gateway.UsageRecord.
    """
    try:
        import llm_gateway  # noqa: F401
        import railway_extension  # noqa: F401
        return True
    except Exception:
        return False


def _install_bridge_stubs():
    if "llm_gateway" in sys.modules or _real_modules_importable():
        return

    gw = types.ModuleType("llm_gateway")

    class ChatRequest(dict):
        pass

    class ChatMessage(dict):
        pass

    gw.ChatRequest = ChatRequest
    gw.ChatMessage = ChatMessage
    gw.get_router = lambda: None
    sys.modules["llm_gateway"] = gw

    rx = types.ModuleType("railway_extension")
    rx.railway_query = lambda query, variables=None: {}
    rx.set_service_variable = lambda *a, **k: {}
    rx.list_projects = lambda: {"projects": {"edges": []}}
    rx.list_services = lambda pid: {"project": {"services": {"edges": []}}}
    rx.get_service_status = lambda service_id=None: {}
    rx.get_logs = lambda deployment_id, limit=100: {}
    rx.redeploy_service = lambda service_id, environment=None: {}
    rx.BRIDGE_SERVICE_ID = "stub-service"
    sys.modules["railway_extension"] = rx


_install_bridge_stubs()

os.environ.pop("AGENT_AUTO_MODE", None)

from agent_loop import automode  # noqa: E402
from agent_loop.harness import AgentHarness, run_agent  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_toggle():
    automode.set_auto(False)
    yield
    automode.set_auto(False)


def test_default_is_off():
    assert automode.is_auto() is False


def test_set_auto_flips_global_toggle():
    out = automode.set_auto(True)
    assert automode.is_auto() is True
    assert out["auto_mode"] is True
    assert out["boot_default_env"] == "0"
    automode.set_auto(False)
    assert automode.is_auto() is False


def test_harness_defaults_to_global_toggle():
    h_off = AgentHarness(task="t")
    assert h_off.auto_mode is False
    automode.set_auto(True)
    h_on = AgentHarness(task="t")
    assert h_on.auto_mode is True


def test_harness_per_run_override_wins():
    automode.set_auto(False)
    assert AgentHarness(task="t", auto_mode=True).auto_mode is True
    automode.set_auto(True)
    assert AgentHarness(task="t", auto_mode=False).auto_mode is False


def test_prompt_carries_auto_block_only_when_enabled():
    marker = "AUTO MODE IS ON"
    off = AgentHarness(task="t", auto_mode=False)
    assert marker not in off._build_system_prompt()
    on = AgentHarness(task="t", auto_mode=True)
    assert marker in on._build_system_prompt()


def test_summary_records_auto_mode():
    h = AgentHarness(task="t", auto_mode=True)
    s = h._summary("complete", "done")
    assert s["auto_mode"] is True


def test_run_agent_accepts_auto_mode_kwarg():
    sig = inspect.signature(run_agent)
    assert "auto_mode" in sig.parameters
    assert sig.parameters["auto_mode"].default is None


def test_env_var_sets_boot_default(monkeypatch):
    monkeypatch.setattr(automode, "_state", {"on": automode._env_default.__wrapped__() if hasattr(automode._env_default, "__wrapped__") else False})
    # Direct check of the env parsing helper instead:
    monkeypatch.setenv("AGENT_AUTO_MODE", "true")
    assert automode._env_default() is True
    monkeypatch.setenv("AGENT_AUTO_MODE", "0")
    assert automode._env_default() is False


# ── wiring guards ───────────────────────────────────────────────────────────
# This feature shipped half-wired once: the harness half went to the deploy
# branch while automode.py, the endpoint and the command pass-through stayed on
# a feature branch. These tests fail loudly if any half goes missing again.


@pytest.fixture()
def routes_mod():
    """The real agent_loop.routes, regardless of collection order.

    test_delete_guard and test_result_race install a file-less stub
    `agent_loop` package into sys.modules and never remove it. Same escape
    hatch test_stream_and_rates uses: drop any file-less stub, re-import.
    """
    import importlib

    for name in [n for n in list(sys.modules)
                 if n == "agent_loop" or n.startswith("agent_loop.")]:
        if not getattr(sys.modules[name], "__file__", None):
            del sys.modules[name]
    return importlib.import_module("agent_loop.routes")


def test_auto_mode_endpoints_are_registered(routes_mod):
    agent_router = routes_mod.agent_router

    paths = {r.path for r in agent_router.routes}
    assert "/agent/auto_mode" in paths
    methods = {m for r in agent_router.routes if r.path == "/agent/auto_mode"
               for m in r.methods}
    assert {"GET", "POST"} <= methods


def test_post_auto_mode_flips_the_toggle(routes_mod):
    AutoModeRequest = routes_mod.AutoModeRequest
    get_auto_mode, post_auto_mode = routes_mod.get_auto_mode, routes_mod.post_auto_mode

    # Resolve automode through the same import state the fixture normalized, so
    # this is the exact module instance the endpoint and the harness share.
    import importlib
    am = importlib.import_module("agent_loop.automode")

    try:
        out = asyncio.run(post_auto_mode(AutoModeRequest(enabled=True)))
        assert out["auto_mode"] is True
        assert am.is_auto() is True          # endpoint writes the shared toggle
        assert asyncio.run(get_auto_mode())["auto_mode"] is True

        out = asyncio.run(post_auto_mode(AutoModeRequest(enabled=False)))
        assert out["auto_mode"] is False
        assert am.is_auto() is False
    finally:
        am.set_auto(False)


def test_command_channel_forwards_auto_mode_to_run_agent():
    """The agent_run command must pass auto_mode through to run_agent.

    Asserted against the source rather than a live run: _start_agent_run needs
    GitHub credentials and a real event loop, but the regression that bit us was
    purely a missing kwarg at this call site.
    """
    import ast

    tree = ast.parse(open("command_channel.py").read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_start_agent_run")
    call = next(n for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "run_agent")
    assert "auto_mode" in {kw.arg for kw in call.keywords}
    # and the journal writer arg it was once truncated over is still intact
    assert "on_turn" in {kw.arg for kw in call.keywords}
