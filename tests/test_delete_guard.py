"""Guard tests for DELETE /contents.

The bridge token can reach every repo it can see, so the only thing standing
between an agent and `rm -rf` on source is _check_deletable. Test it directly:
the route is thin, the guard is the whole safety story.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest


def load_main():
    for name in ("railway_extension", "llm_routes", "command_channel", "kml_watchdog",
                 "agent_loop", "agent_loop.harness", "agent_loop.tools",
                 "agent_loop.browser", "agent_loop.routes", "skills_routes", "fetch_routes"):
        sys.modules.pop(name, None)

    from fastapi import APIRouter

    rw = types.ModuleType("railway_extension")
    rw.router = APIRouter()
    rw.set_service_variable = lambda *a, **k: {}
    rw.BRIDGE_SERVICE_ID = "svc"
    sys.modules["railway_extension"] = rw

    lr = types.ModuleType("llm_routes"); lr.llm_router = APIRouter()
    sys.modules["llm_routes"] = lr

    cc = types.ModuleType("command_channel")
    cc.process_pending_commands = lambda: {"processed": 0}
    sys.modules["command_channel"] = cc

    wd = types.ModuleType("kml_watchdog")
    wd.watchdog_router = APIRouter()
    async def _worker(*a, **k): return None
    wd.watchdog_worker = _worker
    sys.modules["kml_watchdog"] = wd

    sr = types.ModuleType("skills_routes"); sr.skills_router = APIRouter()
    sys.modules["skills_routes"] = sr

    fr = types.ModuleType("fetch_routes"); fr.fetch_router = APIRouter()
    sys.modules["fetch_routes"] = fr

    pkg = types.ModuleType("agent_loop"); pkg.__path__ = []
    routes = types.ModuleType("agent_loop.routes"); routes.agent_router = APIRouter()
    sys.modules.update({"agent_loop": pkg, "agent_loop.routes": routes})

    os.environ.setdefault("GITHUB_TOKEN", "fake")
    spec = importlib.util.spec_from_file_location("main_under_test", "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["main_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


main = load_main()
HTTPException = __import__("fastapi").HTTPException


ALLOWED = [
    "commands/pending/stuck-command.json",
    "commands/results/old-run.json",
    "commands/running/crashtest-issue2-20260806.json",
]

BLOCKED = [
    # source — the whole point of the guard
    "main.py",
    "command_channel.py",
    "agent_loop/harness.py",
    ".github/workflows/sandbox.yml",
    "MAP.md",
    # near misses on the prefix
    "commands/pending",
    "commandsX/pending/a.json",
    "docs/commands/results/a.json",
    # traversal
    "commands/pending/../../main.py",
    "/etc/passwd",
    # directory markers that keep the queue alive
    "commands/pending/.gitkeep",
    "commands/running/.gitkeep",
    # empty
    "",
]


@pytest.mark.parametrize("path", ALLOWED)
def test_allowed_paths_pass(path):
    main._check_deletable(path)


@pytest.mark.parametrize("path", BLOCKED)
def test_blocked_paths_raise(path):
    with pytest.raises(HTTPException) as exc:
        main._check_deletable(path)
    assert exc.value.status_code in (400, 403)


def test_route_is_registered():
    routes = [
        (r.path, sorted(r.methods))
        for r in main.app.routes
        if getattr(r, "path", "") == "/contents/{owner}/{repo}/{path:path}"
    ]
    methods = {m for _, ms in routes for m in ms}
    assert "DELETE" in methods, routes
    assert "GET" in methods, "the read route must survive"


def test_prefixes_are_configurable_but_default_to_commands():
    assert all(p.startswith("commands/") for p in main.DELETE_ALLOWED_PREFIXES)
