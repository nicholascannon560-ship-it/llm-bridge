"""
llm_gateway.py — Multi-LLM provider gateway for the bridge.

Lets you route chat requests to Anthropic, Moonshot, OpenAI, or local models
through a single endpoint, using your own API keys at token-level pricing.

Use case: run a cheap model locally (or via cheap API) as your daily driver,
but escalate to Claude/Kimi/GPT for hard problems — all through one interface,
all at API rates instead of subscription/chat-app rates.
"""

import asyncio
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
        # Opus 5: $5.00 / $25.00 per Mtok
        "claude-opus-5": {"input": 5.00, "output": 25.00, "cached_input": 0.50},
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

# A 429 used to kill an entire agent run on its first turn: raise_for_status
# fired, the exception propagated out of run_agent, and command_channel wrote the
# whole command off as failed. Rate limits are transient by definition and are
# the one error worth waiting out.
LLM_RETRY_ATTEMPTS = int(os.environ.get("LLM_RETRY_ATTEMPTS", "4"))
LLM_RETRY_BASE_SEC = float(os.environ.get("LLM_RETRY_BASE_SEC", "2"))
LLM_RETRY_MAX_SLEEP = float(os.environ.get("LLM_RETRY_MAX_SLEEP", "60"))
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _retry_after_sec(resp) -> float | None:
    """Honour Retry-After when the provider sends one — guessing a backoff when
    we have been told the exact wait is how you get rate-limited again."""
    raw = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))          # delta-seconds form
    except (TypeError, ValueError):
        pass
    try:                                     # HTTP-date form
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone as _tz
        when = parsedate_to_datetime(raw)
        if when.tzinfo is None:
            when = when.replace(tzinfo=_tz.utc)
        return max(0.0, (when - datetime.now(_tz.utc)).total_seconds())
    except Exception:
        return None


async def _post_with_retry(client, url: str, *, headers: dict, json: dict,
                           label: str = "llm"):
    """POST, retrying transient statuses with exponential backoff.

    Only retries 429 and 5xx. A 400/401/403 is a request or credential problem
    and retrying it just burns time and quota, so those return immediately and
    the caller's raise_for_status surfaces them unchanged.
    """
    last = None
    for attempt in range(LLM_RETRY_ATTEMPTS):
        resp = await client.post(url, headers=headers, json=json)
        if resp.status_code not in _RETRY_STATUS:
            return resp
        last = resp
        if attempt == LLM_RETRY_ATTEMPTS - 1:
            break
        sleep_s = _retry_after_sec(resp)
        if sleep_s is None:
            sleep_s = LLM_RETRY_BASE_SEC * (2 ** attempt)
        sleep_s = min(sleep_s, LLM_RETRY_MAX_SLEEP)
        print(f"[{label}] HTTP {resp.status_code} — retrying in {sleep_s:.1f}s "
              f"(attempt {attempt + 1}/{LLM_RETRY_ATTEMPTS})", flush=True)
        await asyncio.sleep(sleep_s)
    return last



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
    cached_tokens: int = 0  # prompt tokens served from the provider's prefix cache


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

    @staticmethod
    def _to_anthropic_messages(messages):
        """Flatten an OpenAI-style message list into Anthropic's format.

        Anthropic accepts only `user`/`assistant` turns (system is a separate
        field) and rejects a `tool` role outright. The agent loop's transcript
        is full of `tool` results and assistant turns that carry `tool_calls`
        with empty text — feeding those to Anthropic raw is a 400. This does
        NOT implement tool calling (that is a separate change); it makes a
        Claude model usable as a chat model or an advisor over a transcript it
        did not itself produce:
          - system  -> returned separately
          - assistant with tool_calls -> text plus a short note of what it called
          - tool     -> a user turn carrying the tool result
        Consecutive same-role turns are merged and the list is forced to start
        with `user`, both of which Anthropic requires.
        """
        system_parts = []
        flat = []  # list of (role, text)
        for m in messages:
            role = m.role
            content = m.content or ""
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            if role == "tool":
                flat.append(("user", f"[tool result] {content}"))
                continue
            if role == "assistant":
                note = ""
                if getattr(m, "tool_calls", None):
                    names = []
                    for tc in m.tool_calls:
                        fn = (tc or {}).get("function", {})
                        if fn.get("name"):
                            names.append(fn["name"])
                    if names:
                        note = f"[called tools: {', '.join(names)}]"
                text = (content + ("\n" + note if note else "")).strip()
                flat.append(("assistant", text or note or "[no output]"))
                continue
            # user or anything else
            flat.append(("user", content))

        # Drop leading assistant turns: Anthropic must start with user.
        while flat and flat[0][0] == "assistant":
            flat.pop(0)

        merged = []
        for role, text in flat:
            if not text:
                continue
            if merged and merged[-1]["role"] == role:
                merged[-1]["content"] += "\n\n" + text
            else:
                merged.append({"role": role, "content": text})

        if not merged:
            merged = [{"role": "user", "content": "Continue."}]
        return "\n\n".join(system_parts), merged

    @staticmethod
    def _to_anthropic_tools(tools):
        """OpenAI tool schema -> Anthropic tool schema.

        OpenAI wraps each tool as {"type":"function","function":{name,
        description, parameters}}. Anthropic wants {name, description,
        input_schema}. Tolerates an already-unwrapped {name,...} too.
        """
        out = []
        for t in tools or []:
            fn = t.get("function", t)
            out.append({
                "name": fn.get("name"),
                "description": fn.get("description", "") or "",
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        return out

    @staticmethod
    def _to_anthropic_tool_messages(messages):
        """Convert an OpenAI-style transcript to Anthropic content-block turns,
        PRESERVING tool calls so a Claude model can drive the agent loop.

        Mapping: system -> separate; assistant.content/tool_calls -> a single
        assistant turn with text + tool_use blocks; role="tool" -> a user turn
        with a tool_result block. Consecutive same-role turns are merged (a
        run's multiple tool results collapse into one user turn), which
        Anthropic requires.

        Then a repair pass makes the pairing valid no matter how the history
        was trimmed, because Anthropic 400s otherwise: tool_result blocks whose
        id was not declared by the immediately preceding assistant turn are
        dropped; assistant tool_use ids with no following result get a
        synthetic "(result unavailable)"; empty turns and leading assistant
        turns are removed.
        """
        system_parts = []
        out = []  # [{"role", "content":[blocks]}]

        def add(role, blocks):
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": list(blocks)})

        for m in messages:
            role = m.role
            content = m.content or ""
            if role == "system":
                if content:
                    system_parts.append(content)
            elif role == "tool":
                add("user", [{
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or "",
                    "content": content or "(no output)",
                }])
            elif role == "assistant":
                blocks = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in (m.tool_calls or []):
                    fn = (tc or {}).get("function", {})
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id") or "",
                        "name": fn.get("name", ""),
                        "input": args,
                    })
                add("assistant", blocks or [{"type": "text", "text": "(no output)"}])
            else:  # user or unknown
                add("user", [{"type": "text", "text": content or "(empty)"}])

        # Repair tool_use/tool_result pairing.
        for i, turn in enumerate(out):
            if turn["role"] != "assistant":
                continue
            use_ids = [b["id"] for b in turn["content"] if b.get("type") == "tool_use"]
            if not use_ids:
                continue
            nxt = out[i + 1] if i + 1 < len(out) and out[i + 1]["role"] == "user" else None
            seen = set()
            if nxt:
                kept = []
                for b in nxt["content"]:
                    if b.get("type") == "tool_result":
                        if b["tool_use_id"] in use_ids and b["tool_use_id"] not in seen:
                            seen.add(b["tool_use_id"])
                            kept.append(b)
                        # drop orphan / duplicate tool_result
                    else:
                        kept.append(b)
                nxt["content"] = kept
            missing = [uid for uid in use_ids if uid not in seen]
            if missing:
                synth = [{"type": "tool_result", "tool_use_id": uid,
                          "content": "(result unavailable)"} for uid in missing]
                if nxt:
                    nxt["content"] = synth + nxt["content"]
                else:
                    out.insert(i + 1, {"role": "user", "content": synth})

        # Drop any tool_result that still has no declaring assistant right before
        # it (e.g. a tool turn at the very start), then drop emptied turns and
        # leading assistant turns.
        for i, turn in enumerate(out):
            if turn["role"] == "user":
                prev = out[i - 1] if i > 0 else None
                prev_ids = ([b["id"] for b in prev["content"] if b.get("type") == "tool_use"]
                            if prev and prev["role"] == "assistant" else [])
                turn["content"] = [b for b in turn["content"]
                                   if b.get("type") != "tool_result" or b["tool_use_id"] in prev_ids]
        out = [t for t in out if t["content"]]
        while out and out[0]["role"] == "assistant":
            out.pop(0)
        if not out:
            out = [{"role": "user", "content": [{"type": "text", "text": "Continue."}]}]
        return "\n\n".join(system_parts), out

    @staticmethod
    def _inject_cache_control(payload: dict) -> None:
        """Add ephemeral prompt-cache breakpoints to an Anthropic payload.

        Anthropic caching is opt-in PER REQUEST: with no cache_control markers
        nothing is cached and usage.cache_read_input_tokens stays 0 — so a
        multi-call chat/agent loop re-bills the whole prefix at full rate every
        turn. Two breakpoints, matching Anthropic's tools->system->messages
        render order:
          - last system block  -> caches the tools+system prefix (both render
            before messages), i.e. the stable per-session part
          - last message block -> caches the conversation so far, so an
            appended tool loop re-bills only the newly added turns
        The minimum cacheable prefix is model-dependent (Haiku 4.5: 4096
        tokens); shorter prefixes silently don't cache, which is fine — the
        savings are meant to land once the transcript grows, which is exactly
        when a loop gets expensive.
        """
        cc = {"type": "ephemeral"}
        marked_prefix = False
        sys = payload.get("system")
        if isinstance(sys, str) and sys.strip():
            payload["system"] = [{"type": "text", "text": sys, "cache_control": cc}]
            marked_prefix = True
        elif isinstance(sys, list) and sys and isinstance(sys[-1], dict):
            sys[-1]["cache_control"] = cc
            marked_prefix = True
        if not marked_prefix:
            tools = payload.get("tools")
            if isinstance(tools, list) and tools and isinstance(tools[-1], dict):
                tools[-1]["cache_control"] = cc
        msgs = payload.get("messages")
        if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict):
            content = msgs[-1].get("content")
            if isinstance(content, str) and content:
                msgs[-1]["content"] = [
                    {"type": "text", "text": content, "cache_control": cc}
                ]
            elif isinstance(content, list) and content and isinstance(content[-1], dict):
                content[-1]["cache_control"] = cc

    async def chat(self, req: ChatRequest) -> ChatResponse:
        start = time.time()

        # Tools are honored only when the caller actually wants them. Chat and
        # advisor calls pass tool_choice="none" (schemas sent only to share a
        # cache prefix), and must stay on the robust text-flatten path.
        use_tools = bool(req.tools) and req.tool_choice != "none"

        payload = {
            "model": req.model or DEFAULT_MODELS["anthropic"],
            "max_tokens": req.max_tokens,
        }
        if req.temperature == 1.0:  # some Claude models only accept 1.0
            payload["temperature"] = 1.0

        if use_tools:
            system_msg, msgs = self._to_anthropic_tool_messages(req.messages)
            payload["messages"] = msgs
            payload["tools"] = self._to_anthropic_tools(req.tools)
            tc = req.tool_choice
            payload["tool_choice"] = ({"type": "any"} if tc in ("required", "any")
                                      else {"type": "auto"})
        else:
            system_msg, msgs = self._to_anthropic_messages(req.messages)
            payload["messages"] = msgs
        if system_msg:
            payload["system"] = system_msg

        # Opt into prompt caching (no-op if the prefix is too short to cache).
        self._inject_cache_control(payload)

        async with httpx.AsyncClient(timeout=120.0) as client:
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

        # Response content is a list of blocks: text and/or tool_use.
        text_parts, tool_calls = [], []
        for b in data.get("content") or []:
            if b.get("type") == "text":
                text_parts.append(b.get("text", ""))
            elif b.get("type") == "tool_use":
                tool_calls.append({
                    "id": b.get("id"),
                    "type": "function",
                    "function": {
                        "name": b.get("name"),
                        "arguments": json.dumps(b.get("input") or {}),
                    },
                })
        content = "".join(text_parts)

        stop = data.get("stop_reason")
        finish_reason = ("tool_calls" if stop == "tool_use"
                         else "length" if stop == "max_tokens"
                         else "stop")

        usage = data.get("usage", {})
        # With prompt caching on, Anthropic splits the prompt three ways:
        # input_tokens is the uncached remainder only, so the true prompt total
        # is input + cache_read + cache_creation. estimate_cost wants the FULL
        # prompt plus the cached portion, so reconstruct both here.
        input_tokens = usage.get("input_tokens", 0)
        cached_tokens = usage.get("cache_read_input_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        prompt_tokens = input_tokens + cached_tokens + cache_creation
        completion_tokens = usage.get("output_tokens", 0)
        cost = self.estimate_cost(
            payload["model"], prompt_tokens, completion_tokens, cached_tokens
        )

        return ChatResponse(
            provider="anthropic",
            model=payload["model"],
            content=content,
            usage={"prompt_tokens": prompt_tokens,
                   "completion_tokens": completion_tokens,
                   "cached_tokens": cached_tokens},
            cost_cents=cost,
            latency_ms=latency,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
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
            r = await _post_with_retry(
                client,
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                label="moonshot",
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
            cached_tokens=resp.usage.get("cached_tokens", 0),
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
