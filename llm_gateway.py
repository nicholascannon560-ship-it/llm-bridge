"""
llm_gateway.py — Multi-LLM provider gateway for the bridge.

Lets you route chat requests to Anthropic, Moonshot, OpenAI, or local models
through a single endpoint, using your own API keys at token-level pricing.

Use case: run a cheap model locally (or via cheap API) as your daily driver,
but escalate to Claude/Kimi/GPT for hard problems — all through one interface,
all at API rates instead of subscription/chat-app rates.
"""

import os
import json
import time
from typing import Dict, List, Optional, AsyncGenerator, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import httpx

# ── configuration ─────────────────────────────────────────────────────────────

# API keys read from env (set in Railway dashboard)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MOONSHOT_API_KEY = os.environ.get("MOONSHOT_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Default models per provider
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "moonshot": "kimi-k3",
    "openai": "gpt-4o-mini",
    "local": "llama3.1",
}

# Cost in CENTS per 1K tokens (input / output / cached_input).
#
# UNIT WARNING: these are cents-per-1K, NOT dollars-per-million. To convert a
# published $/Mtok rate: divide by 10.  ($3.00/Mtok -> 0.30 cents/1K)
# The previous version of this table pasted $/Mtok figures straight in, which
# overstated every cost by exactly 10x and corrupted auto_select()/budget_cents
# routing and /estimate_cost. Verified July 2026.
#
# "cached_input" is the prompt-cache-hit rate where the provider publishes one.
COST_TABLE = {
    "anthropic": {
        # Sonnet 5: $3.00 / $15.00 per Mtok
        "claude-sonnet-5": {"input": 0.30, "output": 1.50, "cached_input": 0.03},
        # Haiku 4.5: $1.00 / $5.00 per Mtok.
        # (Old value 0.25/1.25 was legacy Haiku 3 pricing — stale on two counts.)
        "claude-haiku-4-5-20251001": {"input": 0.10, "output": 0.50, "cached_input": 0.01},
    },
    "moonshot": {
        # Kimi K3: $3.00 / $15.00 per Mtok, cache-hit input $0.30/Mtok (90% off)
        "kimi-k3": {"input": 0.30, "output": 1.50, "cached_input": 0.03},
        # Kimi K2.6: $0.95 / $4.00 per Mtok — the cheap tier for routine work
        "kimi-k2.6": {"input": 0.095, "output": 0.40, "cached_input": 0.016},
        # moonshot-v1-8k is EOL; rate unverified, retained for back-compat only
        "moonshot-v1-8k": {"input": 0.05, "output": 0.20},
    },
    "openai": {
        "gpt-4o": {"input": 0.50, "output": 1.50},
        "gpt-4o-mini": {"input": 0.015, "output": 0.060},
    },
    "local": {
        "llama3.1": {"input": 0.0, "output": 0.0},
    },
}

USAGE_LOG_PATH = os.environ.get("LLM_USAGE_LOG", "llm_usage.jsonl")

# Agent-loop turns at reasoning_effort="max" measured 20-36s; tool round trips
# push higher. 60s was the old hardcoded ceiling and is too tight.
MOONSHOT_TIMEOUT_SEC = float(os.environ.get("MOONSHOT_TIMEOUT_SEC", "120"))


# ── data models ───────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    role: str  # "user", "assistant", "system", "tool"
    content: Optional[str] = None
    # Agent-loop plumbing. An assistant turn that called tools carries
    # tool_calls; the matching tool-result turn carries tool_call_id.
    # Both must survive the round trip or multi-turn tool use breaks.
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_wire(self) -> dict:
        """Serialize to OpenAI-compatible message format."""
        m: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            m["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            m["tool_call_id"] = self.tool_call_id
        if self.name:
            m["name"] = self.name
        return m


@dataclass
class ChatRequest:
    provider: str  # "anthropic", "moonshot", "openai", "local", "auto"
    model: Optional[str] = None
    messages: List[ChatMessage] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    # Auto-routing hints
    complexity: Optional[str] = None  # "low", "medium", "high" — for auto-routing
    budget_cents: Optional[float] = None  # max cost willing to pay
    # Tool calling (OpenAI-compatible schema). Without these the model can
    # only emit prose, which is why the bridge previously could not host an
    # agent loop.
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Any] = None  # "auto" | "none" | "required" | {...}
    # kimi-k3 top-level reasoning control: "low" | "high" | "max".
    # Moonshot defaults to "max", which is why K3 calls run long and
    # expensive; "low" is usually right for mechanical agent-loop steps.
    reasoning_effort: Optional[str] = None


@dataclass
class ChatResponse:
    provider: str
    model: str
    content: str
    usage: Dict[str, int]  # {"prompt_tokens": N, "completion_tokens": M, "cached_tokens": C}
    cost_cents: float
    latency_ms: float
    timestamp: str
    # Populated when the model requested tool execution. Callers must run the
    # tools and send results back as role="tool" messages to continue the loop.
    tool_calls: Optional[List[dict]] = None
    # "stop" | "tool_calls" | "length" — "length" means truncated output.
    finish_reason: Optional[str] = None


@dataclass
class UsageRecord:
    timestamp: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_cents: float
    latency_ms: float
    bridge_key_hash: str  # hashed key for audit, not the key itself


# ── provider clients ──────────────────────────────────────────────────────────

class LLMProvider:
    """Base class for LLM provider clients."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def chat(self, req: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    async def chat_stream(self, req: ChatRequest) -> AsyncGenerator[str, None]:
        raise NotImplementedError

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int,
                      cached_tokens: int = 0) -> float:
        """Estimate cost in cents.

        cached_tokens are billed at the provider's cache-hit rate when one is
        published (K3: $0.30/Mtok vs $3.00 fresh — a 90% discount). They are a
        SUBSET of prompt_tokens, so they're deducted before applying the fresh
        input rate rather than added on top.
        """
        provider = self.__class__.__name__.lower().replace("provider", "")
        rates = COST_TABLE.get(provider, {}).get(model)
        if rates is None:
            # A model missing from COST_TABLE used to silently price at zero,
            # which makes budget_cents routing believe the call is free and
            # hides typo'd model strings entirely. Warn loudly instead.
            print(
                f"WARNING: no COST_TABLE entry for {provider}/{model} — "
                f"billing this call as 0. Add it to COST_TABLE.",
                flush=True,
            )
            rates = {"input": 0.0, "output": 0.0}
        cached_rate = rates.get("cached_input", rates["input"])
        cached = max(0, min(cached_tokens, prompt_tokens))
        fresh = prompt_tokens - cached
        return (
            (fresh / 1000.0) * rates["input"]
            + (cached / 1000.0) * cached_rate
            + (completion_tokens / 1000.0) * rates["output"]
        )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude client."""

    BASE_URL = "https://api.anthropic.com/v1/messages"

    async def chat(self, req: ChatRequest) -> ChatResponse:
        start = time.time()

        # Convert messages to Anthropic format
        system_msg = ""
        user_assistant_msgs = []
        for m in req.messages:
            if m.role == "system":
                system_msg = m.content
            else:
                user_assistant_msgs.append({"role": m.role, "content": m.content})

        payload = {
            "model": req.model or DEFAULT_MODELS["anthropic"],
            "max_tokens": req.max_tokens,
            "messages": user_assistant_msgs,
        }
        # claude-sonnet-5 only accepts temperature=1.0; omit otherwise
        if req.temperature == 1.0:
            payload["temperature"] = 1.0
        if system_msg:
            payload["system"] = system_msg

        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                self.BASE_URL,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            r.raise_for_status()
            data = r.json()

        latency = (time.time() - start) * 1000
        content = data.get("content", [{}])[0].get("text", "") if data.get("content") else ""
        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        cost = self.estimate_cost(payload["model"], prompt_tokens, completion_tokens)

        return ChatResponse(
            provider="anthropic",
            model=payload["model"],
            content=content,
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            cost_cents=cost,
            latency_ms=latency,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def chat_stream(self, req: ChatRequest) -> AsyncGenerator[str, None]:
        # SSE streaming implementation
        system_msg = ""
        user_assistant_msgs = []
        for m in req.messages:
            if m.role == "system":
                system_msg = m.content
            else:
                user_assistant_msgs.append({"role": m.role, "content": m.content})

        payload = {
            "model": req.model or DEFAULT_MODELS["anthropic"],
            "max_tokens": req.max_tokens,
            "messages": user_assistant_msgs,
            "stream": True,
        }
        if req.temperature == 1.0:
            payload["temperature"] = 1.0
        if system_msg:
            payload["system"] = system_msg

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                self.BASE_URL,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            if event.get("type") == "content_block_delta":
                                yield event["delta"].get("text", "")
                        except json.JSONDecodeError:
                            pass


class MoonshotProvider(LLMProvider):
    """Moonshot AI (Kimi) client."""

    BASE_URL = "https://api.moonshot.ai/v1/chat/completions"

    async def chat(self, req: ChatRequest) -> ChatResponse:
        start = time.time()

        # to_wire() preserves tool_calls / tool_call_id, which a plain
        # {role, content} dict comprehension silently dropped — that break
        # made multi-turn tool loops impossible.
        messages = [m.to_wire() for m in req.messages]

        payload = {
            "model": req.model or DEFAULT_MODELS["moonshot"],
            "messages": messages,
            "max_tokens": req.max_tokens,
        }
        # Moonshot kimi-k3 only accepts temperature=1.0; omit otherwise
        if req.temperature == 1.0:
            payload["temperature"] = 1.0
        if req.tools:
            payload["tools"] = req.tools
            if req.tool_choice is not None:
                payload["tool_choice"] = req.tool_choice
        if req.reasoning_effort:
            payload["reasoning_effort"] = req.reasoning_effort

        # Tool loops with reasoning_effort="max" can exceed the old 60s ceiling.
        async with httpx.AsyncClient(timeout=MOONSHOT_TIMEOUT_SEC) as client:
            r = await client.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            r.raise_for_status()
            data = r.json()

        latency = (time.time() - start) * 1000
        choice = data["choices"][0] if data.get("choices") else {}
        msg = choice.get("message", {})
        tool_calls = msg.get("tool_calls")
        finish_reason = choice.get("finish_reason")
        # When the model calls a tool, content is legitimately empty — do NOT
        # fall back to reasoning_content in that case or the caller sees the
        # model's scratchpad instead of a clean empty turn.
        content = msg.get("content") or ""
        if not content and not tool_calls:
            content = msg.get("reasoning_content", "") or ""

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        # Moonshot reports cache hits either flat or nested, depending on route.
        cached_tokens = usage.get("cached_tokens") or (
            usage.get("prompt_tokens_details") or {}
        ).get("cached_tokens", 0) or 0
        cost = self.estimate_cost(
            payload["model"], prompt_tokens, completion_tokens, cached_tokens
        )

        return ChatResponse(
            provider="moonshot",
            model=payload["model"],
            content=content,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
            },
            cost_cents=cost,
            latency_ms=latency,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    async def chat_stream(self, req: ChatRequest) -> AsyncGenerator[str, None]:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        payload = {
            "model": req.model or DEFAULT_MODELS["moonshot"],
            "messages": messages,
            "max_tokens": req.max_tokens,
            "stream": True,
        }
        if req.temperature == 1.0:
            payload["temperature"] = 1.0

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            delta = event["choices"][0].get("delta", {})
                            yield delta.get("content", "")
                        except (json.JSONDecodeError, KeyError):
                            pass


class OpenAIProvider(LLMProvider):
    """OpenAI client (included for completeness)."""

    BASE_URL = "https://api.openai.com/v1/chat/completions"

    async def chat(self, req: ChatRequest) -> ChatResponse:
        start = time.time()

        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        payload = {
            "model": req.model or DEFAULT_MODELS["openai"],
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            r.raise_for_status()
            data = r.json()

        latency = (time.time() - start) * 1000
        choice = data["choices"][0] if data.get("choices") else {}
        msg = choice.get("message", {}); content = msg.get("content", "") or msg.get("reasoning_content", "")
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost = self.estimate_cost(payload["model"], prompt_tokens, completion_tokens)

        return ChatResponse(
            provider="openai",
            model=payload["model"],
            content=content,
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            cost_cents=cost,
            latency_ms=latency,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def chat_stream(self, req: ChatRequest) -> AsyncGenerator[str, None]:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        payload = {
            "model": req.model or DEFAULT_MODELS["openai"],
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            delta = event["choices"][0].get("delta", {})
                            yield delta.get("content", "")
                        except (json.JSONDecodeError, KeyError):
                            pass


# ── router / auto-selector ──────────────────────────────────────────────────

class LLMRouter:
    """Routes requests to the right provider. Can auto-select based on hints."""

    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        if ANTHROPIC_API_KEY:
            self.providers["anthropic"] = AnthropicProvider(ANTHROPIC_API_KEY)
        if MOONSHOT_API_KEY:
            self.providers["moonshot"] = MoonshotProvider(MOONSHOT_API_KEY)
        if OPENAI_API_KEY:
            self.providers["openai"] = OpenAIProvider(OPENAI_API_KEY)

    def available_providers(self) -> List[str]:
        return list(self.providers.keys())

    def auto_select(self, req: ChatRequest) -> str:
        """Pick the best provider based on hints and availability."""
        if req.provider != "auto" and req.provider in self.providers:
            return req.provider

        # Budget-based routing
        if req.budget_cents is not None:
            if req.budget_cents < 1.0 and "openai" in self.providers:
                return "openai"  # gpt-4o-mini is cheapest
            if req.budget_cents < 5.0 and "moonshot" in self.providers:
                return "moonshot"

        # Complexity-based routing
        if req.complexity == "high" and "anthropic" in self.providers:
            return "anthropic"  # Claude for hard problems
        if req.complexity == "low" and "openai" in self.providers:
            return "openai"  # cheap model for easy stuff

        # Default: use whatever is available, prefer moonshot for cost
        if "moonshot" in self.providers:
            return "moonshot"
        if "anthropic" in self.providers:
            return "anthropic"
        if "openai" in self.providers:
            return "openai"

        raise RuntimeError("No LLM providers configured. Set ANTHROPIC_API_KEY, MOONSHOT_API_KEY, or OPENAI_API_KEY.")

    async def chat(self, req: ChatRequest) -> ChatResponse:
        provider_name = self.auto_select(req)
        provider = self.providers[provider_name]
        resp = await provider.chat(req)
        self._log_usage(resp)
        return resp

    async def chat_stream(self, req: ChatRequest) -> AsyncGenerator[str, None]:
        provider_name = self.auto_select(req)
        provider = self.providers[provider_name]
        async for chunk in provider.chat_stream(req):
            yield chunk

    def _log_usage(self, resp: ChatResponse):
        """Append usage record to log file."""
        record = UsageRecord(
            timestamp=resp.timestamp,
            provider=resp.provider,
            model=resp.model,
            prompt_tokens=resp.usage.get("prompt_tokens", 0),
            completion_tokens=resp.usage.get("completion_tokens", 0),
            cost_cents=resp.cost_cents,
            latency_ms=resp.latency_ms,
            bridge_key_hash="",  # filled by route handler
        )
        try:
            with open(USAGE_LOG_PATH, "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        except Exception:
            pass


# ── singleton ─────────────────────────────────────────────────────────────────

_router: Optional[LLMRouter] = None


def get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
