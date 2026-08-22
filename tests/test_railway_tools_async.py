"""Regression test: railway_* tool handlers must not await sync functions.

Every helper in railway_extension.py is a plain synchronous function doing a
blocking requests.post() to Railway's GraphQL API. A stray `await` on one of
them raises "TypeError: object dict can't be used in 'await' expression" --
which is exactly what killed railway_list's project-discovery branch. These
tests run each handler against sync stubs, so a re-introduced `await foo()`
fails here instead of in production.
"""
import asyncio
import contextlib
import os
import sys
import threading

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MAIN_THREAD = threading.current_thread().ident


def _agent_loop_modules():
    return [m for m in list(sys.modules) if m == "agent_loop" or m.startswith("agent_loop.")]


@contextlib.contextmanager
def _fresh_tools():
    """Import agent_loop.tools into a throwaway slot in sys.modules.

    Two reasons this is not a plain module-level import:

    - test_delete_guard installs a fake `agent_loop` package (with an empty
      __path__) into sys.modules and never removes it, so a later real import
      fails depending on collection order.
    - Replacing those entries for good would hand every module imported before
      us a stale reference. pytest imports all test modules during collection
      and runs them afterwards, so that breaks tests that ran fine alone.

    Drop whatever is there, import the real package, then put the originals
    back exactly as they were.
    """
    saved = {name: sys.modules[name] for name in _agent_loop_modules()}
    for name in saved:
        del sys.modules[name]
    try:
        import agent_loop.tools as tools
        yield tools
    finally:
        for name in _agent_loop_modules():
            del sys.modules[name]
        sys.modules.update(saved)


def _sync(calls, name, ret):
    """A stand-in for a blocking railway_extension helper."""
    def fn(*args, **kwargs):
        calls.append((name, args, kwargs, threading.current_thread().ident))
        return ret
    return fn


def _install_stubs(tools, calls):
    tools.BRIDGE_SERVICE_ID = "svc-default"
    tools.list_projects = _sync(calls, "list_projects", {"edges": [{"node": {"id": "p1"}}]})
    tools.list_services = _sync(calls, "list_services", {"edges": [{"node": {"id": "s1"}}]})
    tools.get_service_status = _sync(calls, "get_service_status", {"status": "SUCCESS"})
    tools.get_logs = _sync(calls, "get_logs", {"lines": []})
    tools.redeploy_service = _sync(calls, "redeploy_service", {"ok": True})
    tools.set_service_variable = _sync(calls, "set_service_variable", {"ok": True})


def _run(tools, name, args):
    return asyncio.run(tools._TOOL_HANDLERS[name](args))


def test_railway_list_discovers_projects_and_services():
    """The branch that raised: object dict can't be used in 'await' expression."""
    calls = []
    with _fresh_tools() as tools:
        _install_stubs(tools, calls)

        out = _run(tools, "railway_list", {})
        assert out == {"projects": {"edges": [{"node": {"id": "p1"}}]}}, out
        assert calls[-1][0] == "list_projects"

        out = _run(tools, "railway_list", {"project_id": "  p1  "})
        assert out == {"project_id": "p1", "services": {"edges": [{"node": {"id": "s1"}}]}}, out
        assert calls[-1][:2] == ("list_services", ("p1",)), calls[-1][:3]

    assert not [c for c in calls if c[3] == MAIN_THREAD], "blocked the event loop thread"


def test_other_railway_handlers_still_work():
    calls = []
    with _fresh_tools() as tools:
        _install_stubs(tools, calls)

        out = _run(tools, "railway_get_status", {})
        assert out == {"status": "SUCCESS"}, out
        assert calls[-1][1] == ("svc-default",), "should default to BRIDGE_SERVICE_ID"

        out = _run(tools, "railway_get_logs", {"deployment_id": "d1", "limit": 5})
        assert out == {"lines": []}, out
        assert calls[-1][1:3] == (("d1",), {"limit": 5}), calls[-1][1:3]

        out = _run(tools, "railway_redeploy", {"service_id": "svc-x"})
        assert out == {"redeployed": True, "service_id": "svc-x", "result": {"ok": True}}, out

        out = _run(tools, "railway_set_env", {"name": "FOO", "value": "bar"})
        assert out["set"] is True, out

    assert not [c for c in calls if c[3] == MAIN_THREAD], "blocked the event loop thread"


def test_protected_env_still_refused_before_any_railway_call():
    calls = []
    with _fresh_tools() as tools:
        _install_stubs(tools, calls)
        out = _run(tools, "railway_set_env", {"name": "BRIDGE_API_KEY", "value": "x"})

    assert "error" in out, out
    assert not calls, "protected env must not reach set_service_variable"


if __name__ == "__main__":
    test_railway_list_discovers_projects_and_services()
    test_other_railway_handlers_still_work()
    test_protected_env_still_refused_before_any_railway_call()
    print("All railway tool handler tests pass.")
