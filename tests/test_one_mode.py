"""One-mode console: commit approval, spend warnings, streamed tool calls.

The approval gate is the only thing standing between an autonomous loop and an
unreviewed write to a repo, so it is tested for the ways it could fail OPEN:
a timeout, a stopped run, a double answer, an unknown id.
"""
import asyncio
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_gateway as g  # noqa: E402


@pytest.fixture()
def harness():
    """The real agent_loop.harness (see test_stream_and_rates for why)."""
    import importlib

    for name in [n for n in list(sys.modules)
                 if n == "agent_loop" or n.startswith("agent_loop.")]:
        if not getattr(sys.modules[name], "__file__", None):
            del sys.modules[name]
    h = importlib.import_module("agent_loop.harness")
    h._clear_approvals()
    h.clear_stop()
    return h


def _bare_harness(h):
    """An AgentHarness shell with only what the approval path touches."""
    obj = h.AgentHarness.__new__(h.AgentHarness)
    obj.task_id = "task-test"
    return obj


# ── the gate ────────────────────────────────────────────────────────────────

def test_read_tools_are_never_gated(harness):
    obj = _bare_harness(harness)
    for tool in ("github_read", "repo_search", "railway_get_logs", "run_tests"):
        assert asyncio.run(obj._await_approval(tool, {})) is None, tool
    assert harness.pending_approvals() == []


def test_commit_blocks_until_approved(harness):
    obj = _bare_harness(harness)

    async def scenario():
        task = asyncio.ensure_future(
            obj._await_approval("github_commit", {"path": "a.py"}))
        # Give the gate a moment to register, then confirm it is actually waiting.
        await asyncio.sleep(0.3)
        assert not task.done(), "gate returned before anyone answered"
        pending = harness.pending_approvals()
        assert len(pending) == 1 and pending[0]["name"] == "github_commit"
        harness.resolve_approval(pending[0]["id"], True)
        return await asyncio.wait_for(task, timeout=5)

    assert asyncio.run(scenario()) is True


def test_commit_denial_returns_false(harness):
    obj = _bare_harness(harness)

    async def scenario():
        task = asyncio.ensure_future(obj._await_approval("github_patch", {}))
        await asyncio.sleep(0.3)
        harness.resolve_approval(harness.pending_approvals()[0]["id"], False)
        return await asyncio.wait_for(task, timeout=5)

    assert asyncio.run(scenario()) is False


def test_stop_resolves_pending_commit_as_denied(harness):
    """Pressing Stop must not let a queued commit through."""
    obj = _bare_harness(harness)

    async def scenario():
        task = asyncio.ensure_future(obj._await_approval("github_commit", {}))
        await asyncio.sleep(0.3)
        harness.request_stop("task-test")
        return await asyncio.wait_for(task, timeout=5)

    assert asyncio.run(scenario()) is False


def test_timeout_fails_closed(harness, monkeypatch):
    """A closed browser must read as 'no', never as 'yes'."""
    monkeypatch.setattr(harness, "APPROVAL_TIMEOUT_SEC", 0.5)
    obj = _bare_harness(harness)
    assert asyncio.run(obj._await_approval("github_commit", {})) is False


def test_double_answer_and_unknown_id_are_rejected(harness):
    obj = _bare_harness(harness)

    async def scenario():
        task = asyncio.ensure_future(obj._await_approval("github_commit", {}))
        await asyncio.sleep(0.3)
        ap_id = harness.pending_approvals()[0]["id"]
        first = harness.resolve_approval(ap_id, False)
        second = harness.resolve_approval(ap_id, True)   # must not flip it
        unknown = harness.resolve_approval("ap-nope", True)
        await asyncio.wait_for(task, timeout=5)
        return first, second, unknown

    first, second, unknown = asyncio.run(scenario())
    assert first["ok"] is True
    assert second["ok"] is False, "a second answer overwrote the first"
    assert unknown["ok"] is False


def test_declined_commit_comes_back_as_a_tool_result(harness):
    """A refusal must not kill the run — the model needs to read and continue."""
    obj = _bare_harness(harness)
    obj.messages = []

    async def scenario():
        calls = [{"id": "t1", "type": "function",
                  "function": {"name": "github_commit",
                               "arguments": json.dumps({"path": "x.py"})}}]
        task = asyncio.ensure_future(obj._execute_tools(calls))
        await asyncio.sleep(0.3)
        harness.resolve_approval(harness.pending_approvals()[0]["id"], False)
        return await asyncio.wait_for(task, timeout=5)

    results = asyncio.run(scenario())
    assert len(results) == 1 and "error" in results[0]
    assert "declined" in results[0]["error"].lower()
    assert obj.messages[-1]["role"] == "tool", "refusal must be a tool result"


# ── spend ───────────────────────────────────────────────────────────────────

def test_spend_warns_once_per_threshold_and_never_stops(harness, monkeypatch):
    monkeypatch.setattr(harness, "SPEND_WARN_CENTS", 200.0)
    obj = _bare_harness(harness)
    obj._spend_warn_level = 0
    harness.reset_events()

    for cents in (50, 150, 199, 201, 250, 399, 401, 900):
        obj.total_cost_cents = cents
        obj._warn_on_spend()

    warns = [e for e in harness.run_events(0)["events"]
             if e["kind"] == "spend_warning"]
    # $2, $4, $6, $8 crossed — one each, no repeats within a band.
    assert [w["threshold_cents"] for w in warns] == [200.0, 400.0, 800.0]


def test_loop_has_no_cost_stop_condition(harness):
    src = open(harness.__file__).read()
    assert 'status = "cost_budget_reached"' not in src


# ── streamed tool calls ─────────────────────────────────────────────────────

class _FakeStream:
    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200
        self.request = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def aiter_lines(self):
        for l in self._lines:
            yield l


class _FakeClient:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def stream(self, *a, **kw):
        return _FakeStream(self._lines)


def _run_stream(provider, monkeypatch, lines):
    monkeypatch.setattr(g.httpx, "AsyncClient", lambda **kw: _FakeClient(lines))
    req = g.ChatRequest(provider="x", model="m",
                        messages=[g.ChatMessage(role="user", content="hi")],
                        max_tokens=64)

    async def collect():
        return [e async for e in provider.chat_stream_events(req)]

    return asyncio.run(collect())


def test_anthropic_stream_yields_text_and_tool_calls(monkeypatch):
    lines = [
        'data: ' + json.dumps({"type": "message_start", "message": {"usage": {
            "input_tokens": 10, "cache_read_input_tokens": 90,
            "cache_creation_input_tokens": 0}}}),
        'data: ' + json.dumps({"type": "content_block_delta", "index": 0,
                               "delta": {"type": "text_delta", "text": "Looking"}}),
        'data: ' + json.dumps({"type": "content_block_start", "index": 1,
                               "content_block": {"type": "tool_use",
                                                 "id": "toolu_1",
                                                 "name": "repo_search"}}),
        'data: ' + json.dumps({"type": "content_block_delta", "index": 1,
                               "delta": {"type": "input_json_delta",
                                         "partial_json": '{"q":'}}),
        'data: ' + json.dumps({"type": "content_block_delta", "index": 1,
                               "delta": {"type": "input_json_delta",
                                         "partial_json": '"cache"}'}}),
        'data: ' + json.dumps({"type": "message_delta",
                               "delta": {"stop_reason": "tool_use"},
                               "usage": {"output_tokens": 25}}),
    ]
    events = _run_stream(g.AnthropicProvider("k"), monkeypatch, lines)
    assert events[0] == {"type": "delta", "text": "Looking"}
    done = events[-1]
    assert done["type"] == "done"
    assert done["finish_reason"] == "tool_calls"
    assert done["tool_calls"] == [{
        "id": "toolu_1", "type": "function",
        "function": {"name": "repo_search", "arguments": '{"q":"cache"}'},
    }]
    # Fragments must reassemble into valid JSON or the loop cannot call the tool.
    json.loads(done["tool_calls"][0]["function"]["arguments"])
    assert done["usage"]["cached_tokens"] == 90
    assert done["usage"]["prompt_tokens"] == 100


def test_moonshot_stream_yields_text_and_tool_calls(monkeypatch):
    lines = [
        'data: ' + json.dumps({"choices": [{"delta": {"content": "Checking"}}]}),
        'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1",
             "function": {"name": "github_read", "arguments": '{"p'}}]}}]}),
        'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'ath":"a.py"}'}}]}}]}),
        'data: ' + json.dumps({"choices": [], "usage": {
            "prompt_tokens": 100, "completion_tokens": 20, "cached_tokens": 60}}),
        'data: [DONE]',
    ]
    events = _run_stream(g.MoonshotProvider("k"), monkeypatch, lines)
    assert events[0] == {"type": "delta", "text": "Checking"}
    done = events[-1]
    assert done["tool_calls"][0]["function"]["name"] == "github_read"
    assert json.loads(
        done["tool_calls"][0]["function"]["arguments"]) == {"path": "a.py"}
    assert done["finish_reason"] == "tool_calls"
    assert done["usage"]["cached_tokens"] == 60
