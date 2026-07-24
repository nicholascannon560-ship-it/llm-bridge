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

# Cost per 1K tokens (input / output) — update as pricing changes
COST_TABLE = {
    "anthropic": {
        "claude-sonnet-5": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25},
    },
    "moonshot": {
        "kimi-k3": {"input": 3.0, "output": 15.0},
        "moonshot-v1-8k": {"input": 0.5, "output": 2.0},
    },
    "openai": {
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    },
    "local": {
        "llama3.1": {"input": 0.0, "output": 0.0},
    },
}

USAGE_LOG_PATH = os.environ.get("LLM_USAGE_LOG", "llm_usage.jsonl")


# ── data models ───────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    role: str  # "user", "assistant", "system"
    content: str


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


@dataclass
class ChatResponse:
    provider: str
    model: str
    content: str
    usage: Dict[str, int]  # {"prompt_tokens": N, "completion_tokens": M}
    cost_cents: float
    latency_ms: float
    timestamp: str


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

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in cents."""
        provider = self.__class__.__name__.lower().replace("provider", "")
        rates = COST_TABLE.get(provider, {}).get(model, {"input": 0.0, "output": 0.0})
        return (prompt_tokens / 1000.0) * rates["input"] + (completion_tokens / 1000.0) * rates["output"]


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

        messages = [{"role": m.role, "content": m.content} for m in req.messages]

        payload = {
            "model": req.model or DEFAULT_MODELS["moonshot"],
            "messages": messages,
            "max_tokens": req.max_tokens,
        }
        # Moonshot kimi-k3 only accepts temperature=1.0; omit otherwise
        if req.temperature == 1.0:
            payload["temperature"] = 1.0

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
            provider="moonshot",
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
