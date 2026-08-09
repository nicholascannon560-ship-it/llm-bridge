# MAP: llm-bridge

**purpose:** Universal FastAPI bridge that gives agents controlled access to GitHub, Railway, multi-provider LLMs (Anthropic + Moonshot/Kimi), and a skills engine.  
**status:** active  
**last_verified:** 2026-07-28  
**token budget:** ~1500

## Entry Points
- **Main app:** `main.py` (FastAPI app assembly)
- **LLM gateway:** `llm_gateway.py` + `llm_routes.py`
- **Railway control:** `railway_extension.py`
- **Command channel (GitHub-based):** `command_channel.py` + `commands/` folder
- **Skills:** `skills/` + `skills_routes.py`
- **Deploy:** Railway auto-deploys from this repo. Health endpoint: `/health`

## Task Slices (load only the relevant one)

| Slice | When to use | Key files | Verified |
|-------|-------------|-----------|-------|
| **command-channel** | Sending/receiving Kimi or agent commands via GitHub | `command_channel.py`, `commands/pending/`, `commands/results/` | verified: — |
| **llm-gateway** | Provider routing, timeouts, reasoning_effort, cost table | `llm_gateway.py`, `llm_routes.py` | verified: — |
| **railway** | Deploy, status, logs, env vars | `railway_extension.py` | verified: — |
| **skills** | Skill definitions and /skills endpoints | `skills/`, `skills_routes.py` | verified: — |
| **app-wiring** | Routes, middleware, health, startup | `main.py`, `Procfile`, `railway.json` | verified: 2026-07-28 |

**Rule:** Match the task to a slice and open only those files first. Load the full tree only for cross-cutting work.

## Tree (depth ≤ 3, annotated)
```
.
├── main.py                 # FastAPI app entry + route mounting
├── llm_gateway.py          # Multi-provider LLM routing (core)
├── llm_routes.py           # /llm/* endpoints
├── railway_extension.py    # Railway GraphQL + control helpers
├── command_channel.py      # Processes commands/pending → results
├── skills_routes.py        # /skills/* endpoints
├── skills/                 # Skill definitions (markdown)
│   └── kimi-advisor-bridge.md
├── commands/
│   ├── pending/            # Drop JSON commands here
│   └── results/            # Results appear here after processing
├── Procfile                # Railway process definition
├── railway.json
├── requirements.txt
└── .env.example
```

## Key Files
| Path | Why it matters |
|------|----------------|
| `main.py` | App wiring, middleware, route inclusion |
| `llm_gateway.py` | Actual provider calls + routing logic |
| `command_channel.py` | How Grok and other agents talk to the bridge without direct HTTP |
| `skills/kimi-advisor-bridge.md` | The skill used for hierarchical planning with Kimi |
| `commands/pending/` | Primary way sandboxed agents issue work |

## Current State
- Live on Railway (`kalshiml-production-b2e9.up.railway.app`)
- Command channel is the primary transport for Grok (no outbound internet in sandbox)
- Skills system operational
- Multi-provider LLM gateway working (Moonshot/Kimi + Anthropic)
- `reasoning_effort` now forwarded; timeout raised to 120s

## Relationships
- **Central control plane** for almost every other project
- Consumed by: KalshiML, tali, resolume-bridge, change-log workflows
- Holds the tokens: `GITHUB_TOKEN`, `RAILWAY_API_TOKEN`, `MOONSHOT_API_KEY`, `ANTHROPIC_API_KEY`, `BRIDGE_API_KEY`

## Agent Start Instructions
1. Read this MAP.md first (or just the relevant Task Slice).
2. For most changes: edit the relevant `.py` file, then commit via the bridge itself or GitHub tools.
3. To talk to Kimi: drop a command JSON into `commands/pending/` (see existing examples in `commands/results/`).
4. After any meaningful change, **append a one-line delta** to Pending Map Updates below (do not rewrite the whole map unless structural).
5. **Do not touch** secrets or rotate keys unless explicitly asked.
6. Log the change to `change-log` issue **#2** (POST /issues/.../change-log/2/comments), using the Agent/What/When/Where/Verified/State/Next template.

## Deep Docs (L2 — load only if needed)
- Full endpoint list + architecture diagram → `README.md`
- Individual skill definitions → `skills/`
- Diff-update protocol → `change-log/MAP_UPDATE.md`

## Pending Map Updates
- 2026-08-09 | /ui redesign: compact composer bar (+ upload / model chip / gear chip / mic / send) replacing full-width segmented settings; effort/tools/executor moved into a popover; per-message Copy button; upload is UI-only (files not sent to model yet); REMOVED budget/turns controls (/ui/do omits budget_usd+max_turns -> server defaults, $1 cap / auto turns). Settings now in JS CFG behind unchanged setting() | files: chat_ui.py
- 2026-08-09 | chat: optional bounded read-only research loop (UI_CHAT_RESEARCH, default off) in /ui/chat — chat model does recon with the read-only `research` tool set (tool_choice=auto, assert_tool_set_safe blocks writes) before answering; only final text folds into thread; caps UI_CHAT_RESEARCH_MAX_STEPS(4) + forced tool_choice=none final | files: chat_ui.py
- 2026-08-09 | llm_gateway: ENABLE Anthropic prompt caching (was OFF — no cache_control markers, so cache_read_input_tokens always 0). AnthropicProvider._inject_cache_control adds ephemeral breakpoints (last system + last message block); token accounting reconstructs full prompt (input+cache_read+cache_creation). Verified live: 2nd identical Haiku call cached 8412/8415 | files: llm_gateway.py
- 2026-08-08 | prompts: tell the executor to BATCH independent tool calls into one turn (harness already runs all tool_calls per turn; llm_gateway returns all tool_use blocks) — flips the old "wait for a tool result before the next tool" rule; loop now only for real dependency chains | files: agent_loop/harness.py, chat_ui.py
- 2026-08-08 | chat_ui: brief-in/summary-out handoff (UI_HANDOFF_BRIEF, default on) so a chat<->executor model switch no longer re-bills the full transcript; chat distils a brief, only the final answer folds back | files: chat_ui.py
- 2026-08-08 | agent loop: repo_search tool (branch-aware content grep -> path:line, read-only) + read-narrow discipline added to both system prompts (locate with repo_search, then read a tight window) | files: agent_loop/tools.py, chat_ui.py, agent_loop/harness.py
- 2026-08-08 | agent loop: added github_patch (old_string/new_string edit) so the agent stops resending whole files on every edit; registered in WRITE_TOOL_NAMES so browse+write stays blocked | files: agent_loop/tools.py
- 2026-08-08 | RECOVERY: boot fixed — dropped deleted approval_routes import + repaired chat_ui try/except SyntaxError; patch_router still mounted; approval subsystem fully removed (/commit no longer takes require_approval/x-auto-approve, commits directly) | files: main.py
- 2026-08-08 | Phase 2: AnthropicProvider tool-calling -> Claude executors (Sonnet/Opus); UI Executor picker Sonnet/Opus/Kimi/Both | files: llm_gateway.py, chat_ui.py
- 2026-08-08 | console: single-flow /ui/turn confirm-gate; Haiku chat default; executor Kimi/Both | files: chat_ui.py
- 2026-08-08 | harness advisor hook (advisor_model+advise_every) injects 2nd-model review into loop | files: agent_loop/harness.py
- 2026-08-08 | AnthropicProvider.chat flattens tool/tool_calls turns so Claude works as chat/advisor | files: llm_gateway.py
- 2026-08-07 | chat console at /ui: Chat (tool_choice=none) vs Do it (agent+budget); cookie auth, no bridge key in browser | files: chat_ui.py, main.py
- 2026-08-07 | prefix caching: session-stable system prompt + verbatim history; harness takes history/system_prompt | files: agent_loop/harness.py, chat_ui.py
- 2026-08-07 | agent_run: per-task budget via budget_usd/cost_budget_cents; max_turns default 10->100 (AGENT_DEFAULT_MAX_TURNS) so spend governs | files: agent_loop/harness.py, command_channel.py
- 2026-08-03 | Browser tool via Browserbase REST (no browser-use: pydantic pin clash); free-tier budgets | files: agent_loop/browser.py
- 2026-08-03 | Tool-set split: browsing never shares a loop with commit/set_env/write_memory | files: agent_loop/tools.py, harness.py
- 2026-08-03 | agent_run now runs off the health probe on a background thread; AGENT_LOOP_ENABLED kill switch | files: command_channel.py
- 2026-08-03 | Added /agent/browser_budget and /agent/browser_read (operator-only) | files: agent_loop/routes.py, main.py
- 2026-08-02 | kml_watchdog.py: external KalshiML monitor, own asyncio task, Resend alerts on transitions, off by default | files: kml_watchdog.py, main.py
- 2026-08-02 | DEPLOY BRANCH IS Agent-loop, not main — commits to main never reach service 509d2aef | files: main.py
<!-- Agents: append one-line deltas here, MAX 120 CHARS.
     Format: - YYYY-MM-DD | short description | files: path1, path2
     Longer detail belongs in the change-log issue, not here. -->
- 2026-07-31 | Added agent_loop validation test (validate.py) | files: agent_loop/validate.py
- 2026-07-31 | Added agent_loop/ module: autonomous agent harness with tool calling + memory | files: agent_loop/*
- 2026-07-28 | Forwarded reasoning_effort + raised LLM timeout to 120s | files: command_channel.py
- 2026-07-28 | Added task slices section | files: MAP.md
- 2026-07-28 | Added /issues endpoints (list, create, list+add comments) so agents can close the change-log loop | files: main.py
- 2026-07-28 | Added GET /map_freshness: runs change-log's per-slice gate with the bridge's token | files: main.py
- 2026-08-06 | Railway deployments connection is newest-first: use first: not last: | files: railway_extension.py
- 2026-08-06 | llm-bridge-v2 service DELETED (was consuming queue commands) | files: none
- 2026-08-06 | run_tests tool: agent executes code in zero-secret agent-sandbox repo via Actions | files: agent_loop/tools.py, agent-sandbox/.github/workflows/sandbox.yml
- 2026-08-06 | agent_run `tools` takes full schema DICTS, not name strings | files: none (usage)
