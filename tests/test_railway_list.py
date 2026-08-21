"""Regression test: railway_list tool must call sync helpers without await.

railway_extension.list_projects() / list_services() are plain sync
functions returning dicts. The tool handler used to `await` them,
raising "TypeError: object dict can't be used in 'await' expression"
on every call, on both code paths (with and without project_id).
"""
import asyncio
import sys

sys.path.insert(0, ".")

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
    assert "projects" in out
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
