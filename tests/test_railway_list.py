"""Regression test: railway_list tool must call sync helpers without await.

railway_extension.list_projects() / list_services() are plain sync
functions returning dicts. The tool handler used to `await` them,
raising "TypeError: object dict can't be used in 'await' expression"
on every call, on both code paths (with and without project_id).

The sandbox has no railway_extension, so we install stub bridge modules
before importing agent_loop.tools — the same name-binding path used in
the real bridge process.
"""
import asyncio
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
    rx.list_services = lambda pid: {"services": {"edges": []}}
    rx.get_service_status = lambda service_id=None: {}
    rx.get_logs = lambda deployment_id, limit=100: {}
    rx.redeploy_service = lambda service_id, environment=None: {}
    rx.BRIDGE_SERVICE_ID = "stub-service"
    sys.modules["railway_extension"] = rx


_install_bridge_stubs()

from agent_loop import tools  # noqa: E402


def test_source_has_no_bogus_awaits():
    src = open("agent_loop/tools.py", "r").read()
    assert "await list_projects()" not in src
    assert "await list_services(" not in src


def test_railway_list_without_project_id(monkeypatch):
    calls = {}

    def fake_list_projects():
        calls["projects"] = True
        return {"projects": {"edges": [{"node": {"id": "p1", "name": "demo"}}]}}

    def fake_list_services(pid):
        raise AssertionError("list_services must not be called without project_id")

    monkeypatch.setattr(tools, "list_projects", fake_list_projects)
    monkeypatch.setattr(tools, "list_services", fake_list_services)

    out = asyncio.run(tools._tool_railway_list({}))
    assert calls.get("projects") is True
    assert out["projects"]["edges"][0]["node"]["name"] == "demo"


def test_railway_list_with_project_id(monkeypatch):
    calls = {}

    def fake_list_projects():
        raise AssertionError("list_projects must not be called with project_id")

    def fake_list_services(pid):
        calls["pid"] = pid
        return {"services": {"edges": [{"node": {"id": "s1"}}]}}

    monkeypatch.setattr(tools, "list_projects", fake_list_projects)
    monkeypatch.setattr(tools, "list_services", fake_list_services)

    out = asyncio.run(tools._tool_railway_list({"project_id": "abc123"}))
    assert calls.get("pid") == "abc123"
    assert out["project_id"] == "abc123"
    assert "services" in out


def test_railway_list_handler_registered():
    assert "railway_list" in tools._TOOL_HANDLERS
    assert tools._TOOL_HANDLERS["railway_list"] is tools._tool_railway_list
