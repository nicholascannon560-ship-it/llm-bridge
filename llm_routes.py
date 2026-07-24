"""
llm_routes.py — FastAPI routes for the LLM gateway.

Add these routes to the bridge app to expose multi-LLM chat, streaming,
provider listing, and usage tracking.

Integration: import this module in the bridge's main.py and call
`app.include_router(llm_router, prefix="/llm")`
"""

from fastapi import APIRouter, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

import llm_gateway
from llm_gateway import ChatRequest, ChatMessage, get_router

llm_router = APIRouter(prefix="/llm", tags=["llm"])


# ── request/response models ───────────────────────────────────────────────────

class ChatRequestModel(BaseModel):
    provider: str = Field("auto", description="anthropic, moonshot, openai, local, or auto")
    model: Optional[str] = Field(None, description="Specific model name, or default for provider")
    messages: List[dict] = Field(..., description="Chat messages: [{role, content}, ...]")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1, le=128000)
    stream: bool = Field(False)
    complexity: Optional[str] = Field(None, description="low, medium, high — for auto-routing")
    budget_cents: Optional[float] = Field(None, description="Max cost willing to pay")


class ChatResponseModel(BaseModel):
    provider: str
    model: str
    content: str
    usage: Dict[str, int]
    cost_cents: float
    latency_ms: float
    timestamp: str


class ProviderInfo(BaseModel):
    name: str
    available: bool
    default_model: str
    models: List[str]


class UsageSummary(BaseModel):
    total_requests: int
    total_cost_cents: float
    total_prompt_tokens: int
    total_completion_tokens: int
    by_provider: Dict[str, dict]


# ── routes ────────────────────────────────────────────────────────────────────

@llm_router.get("/providers")
async def list_providers():
    """List available LLM providers and their configured status."""
    router = get_router()
    available = router.available_providers()

    providers = []
    for name in ["anthropic", "moonshot", "openai", "local"]:
        is_available = name in available
        providers.append(ProviderInfo(
            name=name,
            available=is_available,
            default_model=llm_gateway.DEFAULT_MODELS.get(name, ""),
            models=list(llm_gateway.COST_TABLE.get(name, {}).keys()),
        ))

    return {"providers": providers}


@llm_router.post("/chat", response_model=ChatResponseModel)
async def chat(req: ChatRequestModel, x_bridge_key: str = Header(None, alias="X-Bridge-Key")):
    """Send a chat request to the selected LLM provider.

    If provider="auto", the router picks based on complexity and budget hints.
    """
    router = get_router()

    if not router.available_providers():
        raise HTTPException(503, "No LLM providers configured. Set API keys in Railway env.")

    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in req.messages]

    chat_req = ChatRequest(
        provider=req.provider,
        model=req.model,
        messages=messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stream=False,
        complexity=req.complexity,
        budget_cents=req.budget_cents,
    )

    try:
        resp = await router.chat(chat_req)
        return ChatResponseModel(
            provider=resp.provider,
            model=resp.model,
            content=resp.content,
            usage=resp.usage,
            cost_cents=resp.cost_cents,
            latency_ms=resp.latency_ms,
            timestamp=resp.timestamp,
        )
    except Exception as e:
        raise HTTPException(502, f"LLM provider error: {e}")


@llm_router.post("/chat/stream")
async def chat_stream(req: ChatRequestModel, x_bridge_key: str = Header(None, alias="X-Bridge-Key")):
    """Stream a chat response from the selected LLM provider (SSE)."""
    router = get_router()

    if not router.available_providers():
        raise HTTPException(503, "No LLM providers configured.")

    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in req.messages]
    chat_req = ChatRequest(
        provider=req.provider,
        model=req.model,
        messages=messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stream=True,
        complexity=req.complexity,
        budget_cents=req.budget_cents,
    )

    async def event_generator():
        try:
            async for chunk in router.chat_stream(chat_req):
                if chunk:
                    yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@llm_router.post("/estimate_cost")
async def estimate_cost(req: ChatRequestModel):
    """Estimate the cost of a chat request without actually sending it."""
    router = get_router()
    provider_name = router.auto_select(ChatRequest(
        provider=req.provider,
        model=req.model,
        messages=[],
        complexity=req.complexity,
        budget_cents=req.budget_cents,
    ))

    # Rough token estimate: ~4 chars per token
    total_chars = sum(len(m["content"]) for m in req.messages)
    estimated_prompt_tokens = total_chars // 4
    estimated_completion_tokens = req.max_tokens

    provider = router.providers.get(provider_name)
    if not provider:
        raise HTTPException(503, "Provider not available")

    model = req.model or llm_gateway.DEFAULT_MODELS.get(provider_name, "")
    cost = provider.estimate_cost(model, estimated_prompt_tokens, estimated_completion_tokens)

    return {
        "provider": provider_name,
        "model": model,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "estimated_completion_tokens": estimated_completion_tokens,
        "estimated_cost_cents": round(cost, 4),
    }


@llm_router.get("/usage")
async def get_usage(days: int = 7, x_bridge_key: str = Header(None, alias="X-Bridge-Key")):
    """Get usage summary for the last N days."""
    import json
    from datetime import datetime, timezone, timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    total_requests = 0
    total_cost = 0.0
    total_prompt = 0
    total_completion = 0
    by_provider = {}

    try:
        with open(llm_gateway.USAGE_LOG_PATH) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    ts = datetime.fromisoformat(record["timestamp"])
                    if ts >= cutoff:
                        total_requests += 1
                        total_cost += record.get("cost_cents", 0)
                        total_prompt += record.get("prompt_tokens", 0)
                        total_completion += record.get("completion_tokens", 0)

                        prov = record.get("provider", "unknown")
                        by_provider.setdefault(prov, {
                            "requests": 0, "cost_cents": 0.0,
                            "prompt_tokens": 0, "completion_tokens": 0,
                        })
                        by_provider[prov]["requests"] += 1
                        by_provider[prov]["cost_cents"] += record.get("cost_cents", 0)
                        by_provider[prov]["prompt_tokens"] += record.get("prompt_tokens", 0)
                        by_provider[prov]["completion_tokens"] += record.get("completion_tokens", 0)
                except Exception:
                    continue
    except FileNotFoundError:
        pass

    return UsageSummary(
        total_requests=total_requests,
        total_cost_cents=round(total_cost, 4),
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        by_provider={k: {**v, "cost_cents": round(v["cost_cents"], 4)} for k, v in by_provider.items()},
    )
