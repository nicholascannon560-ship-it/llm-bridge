"""Core agent harness — multi-turn tool-calling loop.

Runs an autonomous agent that:
  1. Receives a task + available tools
  2. Calls Kimi (or another provider) with tool schemas
  3. Executes any tool_calls returned
  4. Feeds results back as role="tool" messages
  5. Repeats until finish_reason="stop" or max_turns reached
  6. Writes reflections to memory
  7. Returns full transcript + final answer

Designed to be called from:
  - command_channel.py (via `agent_run` action)
  - A standalone FastAPI route (future)
  - A scheduled worker (future)
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Bridge imports (fail gracefully outside bridge)
try:
    from llm_gateway import ChatRequest, ChatMessage, get_router
    BRIDGE_MODE = True
except ImportError:
    BRIDGE_MODE = False

from .tools import run_tool, TOOL_SCHEMAS
from .memory import MemoryStore


class AgentHarness:
    """Stateful agent that runs a task to completion using tool calls."""

    def __init__(
        self,
        task: str,
        tools: Optional[List[Dict]] = None,
        max_turns: int = 10,
        provider: str = "moonshot",
        model: str = "kimi-k3",
        reasoning_effort: str = "low",
        memory_path: Optional[str] = None,
        task_id: Optional[str] = None,
    ):
        self.task = task
        self.tools = tools or TOOL_SCHEMAS
        self.max_turns = max(max_turns, 1)
        self.provider = provider
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.memory = MemoryStore(path=memory_path)
        self.task_id = task_id or f"agent-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        self.messages: List[Dict[str, Any]] = []
        self.transcript: List[Dict[str, Any]] = []
        self.total_cost_cents = 0.0
        self.total_tokens = {"prompt": 0, "completion": 0}

    def _build_system_prompt(self) -> str:
        tool_names = [t["function"]["name"] for t in self.tools]
        tool_summaries = []
        for t in self.tools:
            name = t["function"]["name"]
            desc = t["function"].get("description", "")
            tool_summaries.append(f"  - {name}: {desc}")

        recent_memories = self.memory.read(limit=5)
        memory_block = ""
        if recent_memories:
            memory_block = "\n\nRecent lessons learned:\n"
            for m in recent_memories:
                memory_block += f"  - [{m.get('tags', [])}] {m.get('entry', '')[:200]}\n"

        return f"""You are an autonomous agent running inside the llm-bridge.
Your task_id is: {self.task_id}

You have access to the following tools:
{chr(10).join(tool_summaries)}

Rules:
1. Think step by step. Break complex tasks into smaller steps.
2. Call tools using the OpenAI function-calling format.
3. Wait for tool results before calling the next tool.
4. If a tool returns an error, try an alternative approach.
5. When the task is complete, respond with a final summary. Do NOT call more tools.
6. Be concise — each turn costs money and time.
7. Always verify deployments succeeded before declaring success.
8. After completing or failing a task, call write_memory to record what you learned.{memory_block}
"""

    async def run(self) -> Dict[str, Any]:
        if not BRIDGE_MODE:
            return {
                "status": "error",
                "error": "AgentHarness requires bridge environment (llm_gateway not importable)",
                "task_id": self.task_id,
            }

        self.messages.append({"role": "system", "content": self._build_system_prompt()})
        self.messages.append({
            "role": "user",
            "content": f"Task: {self.task}\n\nExecute this task using the available tools. "
                       f"You have up to {self.max_turns} turns. Start now."
        })

        final_answer = ""
        status = "incomplete"

        for turn in range(1, self.max_turns + 1):
            turn_record = {"turn": turn, "timestamp": datetime.now(timezone.utc).isoformat()}

            llm_result = await self._call_llm()
            turn_record["llm"] = {
                "content_preview": llm_result.get("content", "")[:200],
                "tool_calls_count": len(llm_result.get("tool_calls") or []),
                "cost_cents": llm_result.get("cost_cents", 0),
                "usage": llm_result.get("usage", {}),
                "finish_reason": llm_result.get("finish_reason"),
            }
            self.total_cost_cents += llm_result.get("cost_cents", 0)
            self.total_tokens["prompt"] += llm_result.get("usage", {}).get("prompt_tokens", 0)
            self.total_tokens["completion"] += llm_result.get("usage", {}).get("completion_tokens", 0)

            tool_calls = llm_result.get("tool_calls")

            if not tool_calls:
                final_answer = llm_result.get("content", "")
                status = "complete"
                turn_record["type"] = "final"
                turn_record["final_answer"] = final_answer[:500]
                self.transcript.append(turn_record)
                break

            tool_results = await self._execute_tools(tool_calls)
            turn_record["type"] = "tool_loop"
            turn_record["tool_calls"] = [
                {"name": tc["function"]["name"], "args_preview": tc["function"]["arguments"][:200]}
                for tc in tool_calls
            ]
            turn_record["tool_results"] = [
                {"status": "ok" if "error" not in r else "error", "preview": str(r)[:300]}
                for r in tool_results
            ]
            self.transcript.append(turn_record)

        else:
            status = "max_turns_reached"
            final_answer = "Max turns reached without a final answer."

        summary = {
            "status": status,
            "task_id": self.task_id,
            "task": self.task,
            "final_answer": final_answer,
            "transcript": self.transcript,
            "turns_used": len(self.transcript),
            "max_turns": self.max_turns,
            "total_cost_cents": round(self.total_cost_cents, 4),
            "total_tokens": self.total_tokens,
            "provider": self.provider,
            "model": self.model,
        }

        try:
            mem_entry = (
                f"Task [{self.task_id}]: {self.task[:120]} — "
                f"Status: {status}, Turns: {len(self.transcript)}, "
                f"Cost: {self.total_cost_cents:.2f}c"
            )
            self.memory.append(mem_entry, tags=["agent_run", status, self.provider])
        except Exception:
            pass

        return summary

    async def _call_llm(self) -> Dict[str, Any]:
        router = get_router()
        chat_messages = []
        for m in self.messages:
            cm = ChatMessage(
                role=m["role"],
                content=m.get("content"),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            )
            chat_messages.append(cm)

        chat_req = ChatRequest(
            provider=self.provider,
            model=self.model,
            messages=chat_messages,
            max_tokens=4096,
            temperature=1.0,
            tools=self.tools,
            tool_choice="auto",
            reasoning_effort=self.reasoning_effort,
        )

        resp = await router.chat(chat_req)

        self.messages.append({
            "role": "assistant",
            "content": resp.content,
            "tool_calls": resp.tool_calls,
        })

        return {
            "content": resp.content,
            "tool_calls": resp.tool_calls,
            "finish_reason": resp.finish_reason,
            "usage": resp.usage,
            "cost_cents": resp.cost_cents,
        }

    async def _execute_tools(self, tool_calls: List[Dict]) -> List[Dict]:
        results = []
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            tool_id = tc["id"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}
                result = {"error": "Invalid JSON in tool arguments"}
            else:
                result = await run_tool(tool_name, args)

            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": json.dumps(result),
            })
            results.append(result)

        return results


async def run_agent(
    task: str,
    tools: Optional[List[Dict]] = None,
    max_turns: int = 10,
    provider: str = "moonshot",
    model: str = "kimi-k3",
    reasoning_effort: str = "low",
    memory_path: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """One-shot agent run. Creates a harness, runs it, returns result."""
    harness = AgentHarness(
        task=task,
        tools=tools,
        max_turns=max_turns,
        provider=provider,
        model=model,
        reasoning_effort=reasoning_effort,
        memory_path=memory_path,
        task_id=task_id,
    )
    return await harness.run()
