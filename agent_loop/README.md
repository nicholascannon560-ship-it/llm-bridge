# Agent Loop for llm-bridge

Autonomous agent harness that runs multi-turn tool-calling loops via Kimi (or any bridge-supported LLM).

## What it does

- Receives a **task** (natural language)
- Calls the LLM with **tool schemas** (GitHub, Railway, memory, sub-LLM)
- **Executes** any tool_calls returned
- **Feeds results back** as `role="tool"` messages
- **Repeats** until the model says it is done or `max_turns` is hit
- **Writes reflections** to a JSONL memory log for future runs

## Files

| File | Purpose |
|------|---------|
| `harness.py` | Core `AgentHarness` class + `run_agent()` convenience function |
| `tools.py` | Tool schemas (OpenAI format) + async execution handlers |
| `memory.py` | JSONL memory store for agent reflections |
| `README.md` | This file |
| `PATCH_command_channel.py` | Script to patch `command_channel.py` for integration |

## Tools Available

| Tool | What it does |
|------|-------------|
| `github_read` | Read files / list dirs from any repo |
| `github_commit` | Commit a single file to any repo |
| `railway_redeploy` | Redeploy a Railway service |
| `railway_set_env` | Set/update a Railway env var |
| `railway_get_status` | Get latest deployment status |
| `railway_get_logs` | Fetch logs for a deployment |
| `llm_chat` | Recursive LLM call for sub-analysis |
| `write_memory` | Append a reflection to `agent_memory.jsonl` |
| `read_memory` | Read recent reflections |

## Integration

### Option A: Patch command_channel.py (recommended)

Run the patch script after reviewing it:

```bash
python agent_loop/PATCH_command_channel.py
```

This adds:
- `agent_run` action to the command channel
- Tool-aware `llm_chat` (forwards `tools`/`tool_choice`)

Then redeploy the bridge.

### Option B: Standalone FastAPI route (future)

Mount `agent_router` in `main.py` to expose `POST /agent/run`.

## Usage via Command Channel

Drop a JSON command into `commands/pending/`:

```json
{
  "action": "agent_run",
  "task": "Read KalshiML/MAP.md, check for stale slices, commit fixes if any",
  "max_turns": 8,
  "provider": "moonshot",
  "model": "kimi-k3",
  "reasoning_effort": "low"
}
```

The bridge processes it on the next health tick and writes the full transcript to `commands/results/<id>.json`.

## Self-Learning Loop

The agent automatically writes a memory entry after every run summarizing:
- Task description
- Final status (complete / max_turns_reached / error)
- Turns used
- Total cost

Future agent runs load the 5 most recent memories into the system prompt, so the agent learns from its own history.

## Cost Control

- `max_turns` caps the loop (default 10)
- `reasoning_effort="low"` keeps Kimi fast and cheap for mechanical steps
- Each turn's cost is tracked in the transcript
- Total cost is returned in the result

## Safety

- All file commits go through GitHub API (no local filesystem writes outside the container)
- Railway operations use the bridge's existing token (no new secrets)
- Memory is append-only JSONL (no destructive updates)
- Agent cannot delete repos, branches, or files — only read and commit
