"""Autonomous agent loop for llm-bridge.

Provides self-running agents that can call tools, maintain state,
write memory, and iterate until a task is complete.

Usage (inside bridge):
    from agent_loop import run_agent, DEFAULT_TOOLS
    result = await run_agent(
        task="Optimize KalshiML weather model",
        tools=DEFAULT_TOOLS,
        max_turns=10
    )
"""
from .harness import AgentHarness, run_agent
from .tools import DEFAULT_TOOLS, TOOL_SCHEMAS, run_tool
from .memory import MemoryStore

__all__ = ["AgentHarness", "run_agent", "DEFAULT_TOOLS", "TOOL_SCHEMAS", "run_tool", "MemoryStore"]
