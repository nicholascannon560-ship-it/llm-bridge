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


def _install_bridge_stubs():
    if "llm_gateway" in sys.modules:
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
