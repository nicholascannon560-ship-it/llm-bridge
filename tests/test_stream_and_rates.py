"""Regression tests for the console's streaming path, run feed, and rate table.

The rate tests exist because Opus 5 was entered in $/Mtok into a table whose
unit is cents/Ktok — a 10x overcharge that also made the agent loop's
cost_budget_cents cut Opus runs off at a tenth of their real budget. Anything
that silently re-breaks the unit should fail here.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_gateway as g  # noqa: E402


@pytest.fixture()
def harness():
    """The real agent_loop.harness, regardless of collection order.

    test_delete_guard and test_result_race install a stub `agent_loop` package
    (an empty __path__, no __file__) into sys.modules and never remove it, so a
    plain module-level import here resolves to the stub whenever those files are
    collected first. Drop any file-less stub, then import for real.
    """
    import importlib

    for name in [n for n in list(sys.modules)
                 if n == "agent_loop" or n.startswith("agent_loop.")]:
        if not getattr(sys.modules[name], "__file__", None):
            del sys.modules[name]
    return importlib.import_module("agent_loop.harness")


# ── rate table ──────────────────────────────────────────────────────────────

# Published list prices, $ per million tokens. Source of truth for the unit.
PUBLISHED_USD_PER_MTOK = {
    ("anthropic", "claude-opus-5"): (5.00, 25.00),
    ("anthropic", "claude-sonnet-5"): (3.00, 15.00),
    ("anthropic", "claude-haiku-4-5-20251001"): (1.00, 5.00),
    ("moonshot", "kimi-k3"): (3.00, 15.00),
    ("moonshot", "kimi-k2.6"): (0.95, 4.00),
}


@pytest.mark.parametrize("key,expected", sorted(PUBLISHED_USD_PER_MTOK.items()))
def test_rates_match_published_dollars_per_mtok(key, expected):
    provider, model = key
    rates = g.public_rates(provider, model)
    assert rates is not None, f"{provider}/{model} missing from COST_TABLE"
    assert (rates["input"], rates["output"]) == expected


def test_cost_table_unit_is_cents_per_1k_tokens():
    """One million Opus input tokens must cost $5.00 — i.e. 500 cents."""
    cost_cents = g.AnthropicProvider("k").estimate_cost(
        "claude-opus-5", prompt_tokens=1_000_000, completion_tokens=0
    )
    assert cost_cents == pytest.approx(500.0)


def test_opus_is_not_ten_times_sonnet_on_input():
    """Opus is 5/3 of Sonnet per token, not 50/3.

    The bug made Opus price 10x high, which reads as "Opus is 16x Sonnet".
    """
    opus = g.public_rates("anthropic", "claude-opus-5")["input"]
    sonnet = g.public_rates("anthropic", "claude-sonnet-5")["input"]
    assert opus / sonnet == pytest.approx(5.0 / 3.0, rel=1e-6)


def test_cached_input_is_discounted_not_free():
    for provider, model in PUBLISHED_USD_PER_MTOK:
        r = g.public_rates(provider, model)
        assert 0 < r["cached_input"] < r["input"], f"{model} cache rate looks wrong"


def test_cache_hit_reduces_cost():
    p = g.AnthropicProvider("k")
    cold = p.estimate_cost("claude-opus-5", 100_000, 1_000, cached_tokens=0)
    warm = p.estimate_cost("claude-opus-5", 100_000, 1_000, cached_tokens=90_000)
    assert warm < cold
    # 90% of the prompt at 10% of the rate.
    assert warm == pytest.approx((10_000 / 1000) * 0.50
                                 + (90_000 / 1000) * 0.05
                                 + (1_000 / 1000) * 2.50)


def test_public_rates_unknown_model_is_none():
    assert g.public_rates("anthropic", "no-such-model") is None


def test_every_picker_model_is_priced_and_servable():
    """Every row the console offers must resolve a rate.

    The picker is built from chat_ui.MODEL_CATALOG and priced through
    public_rates, so an id missing from COST_TABLE shows a blank rate — and an
    id that drifted from what the provider serves is worse than blank: the
    "stealth/ox-alpha" row outlived its preview slug and 404'd at OpenRouter
    for anyone who picked it. Pinning the catalog to the biller catches both.
    """
    os.environ.setdefault("BRIDGE_API_KEY", "testkey")
    os.environ.setdefault("UI_PASSWORD", "123456")
    os.environ.setdefault("UI_SESSION_SECRET", "test-secret")
    import chat_ui

    for m in chat_ui.MODEL_CATALOG:
        assert g.public_rates(m["provider"], m["id"]) is not None, (
            f'{m["provider"]}/{m["id"]} is in the picker but not in COST_TABLE'
        )


# ── streaming ───────────────────────────────────────────────────────────────

def test_streaming_payload_matches_blocking_payload():
    """Streaming must send a byte-identical prefix, or it gets its own cache
    entry and silently doubles the prompt cost of the conversation."""
    req = g.ChatRequest(
        provider="anthropic", model="claude-opus-5",
        messages=[g.ChatMessage(role="system", content="stable " * 50),
                  g.ChatMessage(role="user", content="hello")],
        max_tokens=512, temperature=1.0,
        tools=[{"type": "function",
                "function": {"name": "t", "description": "d", "parameters": {}}}],
        tool_choice="none",
    )
    provider = g.AnthropicProvider("k")
    blocking = provider._build_payload(req)
    streaming = dict(provider._build_payload(req))
    streaming["stream"] = True
    assert {k: v for k, v in streaming.items() if k != "stream"} == blocking


def test_build_payload_sets_cache_breakpoints():
    req = g.ChatRequest(
        provider="anthropic", model="claude-opus-5",
        messages=[g.ChatMessage(role="system", content="sys"),
                  g.ChatMessage(role="user", content="hi")],
        max_tokens=64,
    )
    payload = g.AnthropicProvider("k")._build_payload(req)
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_chat_stream_events_default_falls_back_to_chat():
    """A provider with no streaming implementation still satisfies the contract."""
    class Dummy(g.LLMProvider):
        async def chat(self, req):
            return g.ChatResponse(
                provider="dummy", model="m", content="hello there",
                usage={"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 0},
                cost_cents=1.25, latency_ms=1.0, timestamp="t",
            )

    events = [e async for e in Dummy("k").chat_stream_events(
        g.ChatRequest(provider="dummy", messages=[]))]
    assert [e["type"] for e in events] == ["delta", "done"]
    assert events[0]["text"] == "hello there"
    assert events[-1]["content"] == "hello there"
    assert events[-1]["cost_cents"] == 1.25


# ── agent run event feed ────────────────────────────────────────────────────

def test_run_events_are_incremental_and_monotonic(harness):
    reset_events, run_events, _emit = harness.reset_events, harness.run_events, harness._emit

    reset_events()
    _emit("run_start", task="demo")
    _emit("tool_call", name="repo_search", args="{}")
    first = run_events(0)
    assert [e["kind"] for e in first["events"]] == ["run_start", "tool_call"]
    assert first["cursor"] == 2

    # Polling from the cursor returns nothing until something new happens.
    assert run_events(first["cursor"])["events"] == []
    _emit("tool_result", name="repo_search", status="ok", preview="found")
    nxt = run_events(first["cursor"])
    assert [e["kind"] for e in nxt["events"]] == ["tool_result"]
    assert nxt["cursor"] == 3


def test_event_buffer_is_bounded(harness):
    harness.reset_events()
    for i in range(harness.EVENT_BUFFER_MAX + 60):
        harness._emit("tool_call", name=f"t{i}")
    assert len(harness.RUN_EVENTS) == harness.EVENT_BUFFER_MAX
    # The tail is kept, so a watcher always sees the most recent activity.
    assert harness.RUN_EVENTS[-1]["name"] == f"t{harness.EVENT_BUFFER_MAX + 59}"


# ── feed gaps ───────────────────────────────────────────────────────────────
# The console reads the feed by polling a bounded buffer. One event per streamed
# token piece overran it whenever a poll landed late — a throttled hidden tab, a
# sleeping phone — and the survivors were then spliced together as if nothing had
# been lost. That is what "the loop gaps out" looked like from the outside.


def test_stream_deltas_are_coalesced_into_chunks(harness):
    harness.reset_events()
    pieces = [f"tok{i} " for i in range(400)]
    for p in pieces:
        harness._emit_delta(p)
    harness.flush_deltas()

    feed = harness.run_events(0)["events"]
    assert all(e["kind"] == "assistant_delta" for e in feed)
    # Same text, far fewer events: ~2.8k chars at a 240-char chunk is well under
    # twenty events, not four hundred.
    assert "".join(e["text"] for e in feed) == "".join(pieces)
    assert len(feed) < len(pieces) / 10


def test_prose_is_flushed_before_the_event_that_follows_it(harness):
    """Text written before a tool call must not surface after its card."""
    harness.reset_events()
    harness._emit_delta("I'll search the repo. ")
    harness._emit("tool_call", name="repo_search", args="{}")

    assert [e["kind"] for e in harness.run_events(0)["events"]] == [
        "assistant_delta", "tool_call"]


def test_run_events_reports_how_many_it_dropped(harness):
    """A poller that missed events is told so, instead of being handed a
    seamless-looking splice of whatever survived."""
    harness.reset_events()
    for _ in range(harness.EVENT_BUFFER_MAX + 50):
        harness._emit_delta("x" * harness.DELTA_COALESCE_CHARS)
    harness.flush_deltas()

    feed = harness.run_events(0)
    assert len(feed["events"]) == harness.EVENT_BUFFER_MAX
    assert feed["dropped"] == 50
    # A caller that is up to date is never told it missed anything.
    assert harness.run_events(feed["cursor"])["dropped"] == 0


def test_tool_and_approval_events_outlive_the_prose_around_them(harness):
    """A dropped tool_call orphans its own result card, and a dropped
    approval_request loses the prompt the run is blocked on. Prose goes first."""
    harness.reset_events()
    harness._emit("tool_call", name="repo_search", args="{}")
    harness._emit("approval_request", id="ap-1", name="github_commit", args="{}")
    for _ in range(harness.EVENT_BUFFER_MAX + 50):
        harness._emit_delta("y" * harness.DELTA_COALESCE_CHARS)
    harness.flush_deltas()

    kinds = [e["kind"] for e in harness.RUN_EVENTS]
    assert kinds[:2] == ["tool_call", "approval_request"]
    assert kinds.count("assistant_delta") == harness.EVENT_BUFFER_MAX - 2


def test_a_stale_cursor_from_a_previous_run_is_not_reported_as_a_gap(harness):
    """reset_events() starts a new run's feed; the events below it were never
    this run's to miss."""
    harness.reset_events()
    for _ in range(20):
        harness._emit("tool_call", name="noise")
    harness.reset_events()
    harness._emit("run_start", task="demo")

    feed = harness.run_events(0)
    assert [e["kind"] for e in feed["events"]] == ["run_start"]
    assert feed["dropped"] == 0


def test_stop_is_scoped_to_one_task(harness):
    """A stop must not leak onto the next run that happens to start after it."""
    clear_stop, request_stop = harness.clear_stop, harness.request_stop
    _stop_is_requested = harness._stop_is_requested

    clear_stop()
    request_stop("task-A")
    assert _stop_is_requested("task-A")
    assert not _stop_is_requested("task-B")
    clear_stop()
    assert not _stop_is_requested("task-A")
