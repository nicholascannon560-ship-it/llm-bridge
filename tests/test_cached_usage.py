"""Proves cached_tokens survives log -> /llm/usage summary."""
import asyncio, json, os, tempfile, sys
os.environ.setdefault("BRIDGE_API_KEY", "x")
import llm_gateway
from llm_gateway import UsageRecord, ChatResponse, LLMRouter
from dataclasses import asdict

tmp = tempfile.mkdtemp()
llm_gateway.USAGE_LOG_PATH = os.path.join(tmp, "usage.jsonl")

# 1. gateway writes cached_tokens
r = LLMRouter.__new__(LLMRouter)
resp = ChatResponse(
    content="hi", provider="moonshot", model="kimi-k3",
    usage={"prompt_tokens": 100000, "completion_tokens": 500, "cached_tokens": 93000},
    cost_cents=5.0, latency_ms=10.0, timestamp="2026-08-05T12:00:00+00:00",
    finish_reason="stop",
)
LLMRouter._log_usage(r, resp)
rec = json.loads(open(llm_gateway.USAGE_LOG_PATH).read().strip())
assert rec["cached_tokens"] == 93000, rec
print("1. logged cached_tokens:", rec["cached_tokens"])

# 2. old records without the field don't crash the summary
with open(llm_gateway.USAGE_LOG_PATH, "a") as f:
    f.write(json.dumps({"timestamp": "2026-08-05T12:00:00+00:00", "provider": "moonshot",
                        "model": "kimi-k3", "prompt_tokens": 1000,
                        "completion_tokens": 10, "cost_cents": 0.3}) + "\n")

import llm_routes
out = asyncio.run(llm_routes.get_usage(days=3650, x_bridge_key=None))
print("2. summary:", out.total_prompt_tokens, out.total_cached_tokens, out.cache_hit_rate)
assert out.total_prompt_tokens == 101000
assert out.total_cached_tokens == 93000
assert out.cache_hit_rate == round(93000/101000, 4)
assert out.by_provider["moonshot"]["cached_tokens"] == 93000

# 3. empty log -> no ZeroDivision
llm_gateway.USAGE_LOG_PATH = os.path.join(tmp, "nope.jsonl")
out2 = asyncio.run(llm_routes.get_usage(days=1, x_bridge_key=None))
assert out2.cache_hit_rate == 0.0 and out2.total_requests == 0
print("3. empty log ok, hit_rate", out2.cache_hit_rate)
print("PASS")
