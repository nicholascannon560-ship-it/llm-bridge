"""Regression test: /approvals/{id}/approve must not await sync railway helpers.

`_execute_approved_action` dispatched the redeploy / set_env / railway_gql
actions with `await` on plain synchronous railway_extension functions, which
raises "TypeError: object dict can't be used in 'await' expression" -- the same
class of bug that broke railway_list. The set_env branch also passed its
arguments in the wrong order, and the railway_gql branch imported a name
(`railway_gql_query`) that does not exist in railway_extension.

These tests stub railway_extension with sync functions matching the real
signatures, so any of those three regressions fails here.
"""
import asyncio
import os
import sys
import threading
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MAIN_THREAD = threading.current_thread().ident
CALLS = []


def _record(name):
    def fn(*args, **kwargs):
        CALLS.append((name, args, kwargs, threading.current_thread().ident))
        return {"ok": name}
    return fn


def _stub_railway_extension():
    """railway_extension with the REAL signatures, so a wrong argument order
    raises TypeError instead of silently passing."""
    stub = types.ModuleType("railway_extension")

    def redeploy_service(service_id: str, environment: str = "production"):
        return _record("redeploy_service")(service_id, environment)

    def set_service_variable(name: str, value: str, service_id=None, environment_name=None):
        return _record("set_service_variable")(
            name, value, service_id=service_id, environment_name=environment_name
        )

    def railway_query(query: str, variables: dict = None):
        return _record("railway_query")(query, variables)

    stub.redeploy_service = redeploy_service
    stub.set_service_variable = set_service_variable
    stub.railway_query = railway_query
    return stub


def _run(action, payload):
    """Dispatch one approved action against the stubbed railway_extension.

    The stub is installed only for the duration of the call so it does not
    leak into other test modules.
    """
    import approval_routes

    saved = sys.modules.get("railway_extension")
    sys.modules["railway_extension"] = _stub_railway_extension()
    try:
        return asyncio.run(
            approval_routes._execute_approved_action({"action": action, "payload": payload})
        )
    finally:
        if saved is not None:
            sys.modules["railway_extension"] = saved
        else:
            sys.modules.pop("railway_extension", None)


def test_redeploy_action():
    del CALLS[:]
    out = _run("redeploy", {"service_id": "svc-1", "environment": "production"})
    assert out == {"ok": "redeploy_service"}, out
    assert CALLS[-1][1] == ("svc-1", "production"), CALLS[-1][1]
    assert CALLS[-1][3] != MAIN_THREAD, "blocked the event loop thread"


def test_set_env_action_passes_arguments_in_signature_order():
    del CALLS[:]
    out = _run("set_env", {"service_id": "svc-1", "name": "FOO", "value": "bar"})
    assert out == {"ok": "set_service_variable"}, out
    assert CALLS[-1][1] == ("FOO", "bar"), f"name/value in the wrong order: {CALLS[-1][1:3]}"
    assert CALLS[-1][2]["service_id"] == "svc-1", CALLS[-1][2]
    assert CALLS[-1][3] != MAIN_THREAD, "blocked the event loop thread"


def test_railway_gql_action_uses_a_name_that_exists():
    del CALLS[:]
    out = _run("railway_gql", {"query": "{ me { id } }", "variables": {"a": 1}})
    assert out == {"ok": "railway_query"}, out
    assert CALLS[-1][1] == ("{ me { id } }", {"a": 1}), CALLS[-1][1]
    assert CALLS[-1][3] != MAIN_THREAD, "blocked the event loop thread"


def test_unknown_action_is_rejected():
    del CALLS[:]
    try:
        _run("nonsense", {})
    except Exception as e:
        assert "nonsense" in str(e), repr(e)
    else:
        raise AssertionError("unknown action should raise")


if __name__ == "__main__":
    test_redeploy_action()
    test_set_env_action_passes_arguments_in_signature_order()
    test_railway_gql_action_uses_a_name_that_exists()
    test_unknown_action_is_rejected()
    print("All approval railway action tests pass.")
