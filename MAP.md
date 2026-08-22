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
- 2026-08-22 | railway_list awaited sync list_projects/list_services -> asyncio.to_thread; other railway_* offloaded too | files: agent_loop/tools.py
- 2026-08-22 | Railway deploys THIS branch (Agent-loop), not main -- main-only fixes never reach production | files: -
- 2026-08-22 | AST audit guards the railway sync/async boundary repo-wide | files: tests/test_railway_sync_boundary.py

- 2026-08-22 | auto-mode landed half on Agent-loop, half on feature/auto-mode: repaired | files: agent_loop/harness.py, agent_loop/automode.py, agent_loop/routes.py, command_channel.py
- 2026-08-22 | GOTCHA: main.py guards the agent_loop import, so a parse error there boots GREEN with /agent/* gone | files: main.py, agent_loop/harness.py
- 2026-08-12 | FIX 'ran a while then stopped, all output vanished' on a large pasted doc. THREE compounding bugs: (1) _trim_history_to_budget deleted msgs[1] = THE TASK first on a fresh session, so the agent lost its own instructions mid-run — task message is now pinned and never trimmed; (2) ui_do wrote the harness's TRIMMED transcript back over s.messages, permanently deleting the user's ask — now only adopted when the transcript still contains it (substring: the harness wraps it as 'Task: ...'); (3) followRun ended with load(), which does log.innerHTML='' — every finished run erased the tool cards/approvals/prose you just watched. Now refreshStats() only | files: agent_loop/harness.py, chat_ui.py, tests/test_one_mode.py
- 2026-08-12 | restored history renders tool activity ('used github_read, repo_search') so a reload of a tool-heavy run is not a blank gap; user bubbles strip the harness 'Task: .. Start now.' wrapper for display while keeping the bytes intact for cache matching | files: chat_ui.py
- 2026-08-12 | ONE MODE: chat/executor split removed. Deleted ui_chat, ui_stream, ui_turn, ui_auto, _classify + _ROUTER_PROMPT (a router LLM call on EVERY message), _build_handoff_brief/_brief_messages, and the whole UI_CHAT_RESEARCH subsystem. Every message now goes to /ui/do = the agent loop with tools; the model decides if it needs any. Brief is gone because one model = one cache: history passes to the loop verbatim and the harness transcript is kept as the session (warm prefix next turn) | files: chat_ui.py
- 2026-08-12 | commit approval replaces the per-task confirm card: harness APPROVAL_TOOLS (github_commit/patch/commit_tree/create_repo/create_branch, env AGENT_APPROVAL_TOOLS) pauses the loop mid-run and waits; POST /ui/approve + GET /ui/approvals. Reads/logs/tests/deploys run unattended. Denial returns a tool result (run continues, model explains) rather than killing the run. Stop and timeout (AGENT_APPROVAL_TIMEOUT_SEC=1800) both resolve DENIED — fails closed | files: agent_loop/harness.py, chat_ui.py
- 2026-08-12 | spend no longer stops a run: removed BOTH cost_budget_reached stops from the loop. AGENT_SPEND_WARN_CENTS (default 200 = $2) emits a spend_warning event per threshold crossed and keeps going. Runs now end only on completion, Stop, context ceiling, or error. NOTE the old console default was budget_usd=1.0 ($1/run) which, with the 10x Opus rate, meant Opus ran against ~$0.10 | files: agent_loop/harness.py, chat_ui.py
- 2026-08-12 | agent turns now STREAM: _call_llm uses chat_stream_events and emits assistant_delta per token. Required teaching both streaming providers to accumulate tool calls — Anthropic content_block_start/input_json_delta by index, Moonshot indexed delta.tool_calls fragments — so a streamed turn can still drive the tool loop. done event carries tool_calls + finish_reason | files: llm_gateway.py, agent_loop/harness.py
- 2026-08-12 | llm_gateway COST_TABLE unit bug: claude-opus-5 was entered in $/Mtok (5.00/25.00/0.50) into a cents/Ktok table = 10x overcharge; every Opus cost display AND the agent loop's cost_budget_cents gate were 10x, cutting Opus runs off at 1/10 of budget. Now 0.50/2.50/0.05. Regression tests in tests/test_stream_and_rates.py pin all 5 models to published $/Mtok | files: llm_gateway.py, tests/test_stream_and_rates.py
- 2026-08-12 | llm_gateway: chat_stream_events() — structured streaming (delta/done dicts w/ usage+cost) for Anthropic+Moonshot, default falls back to chat(). AnthropicProvider._build_payload extracted so streaming sends a byte-identical prefix to chat() (cache breakpoints preserved — the old chat_stream built its own payload and lost caching + all usage). Router logs usage on the done event. LLM_STREAM_TIMEOUT_SEC=300 | files: llm_gateway.py
- 2026-08-12 | agent_loop/harness: live run event feed (RUN_EVENTS ring buffer + run_events(since) cursor + _emit) publishing run_start/turn_start/assistant_text/tool_call/tool_result/advisor/run_end as they happen — previously the whole tool-by-tool trace was locked in self.transcript until the run ended and RUN_STATE exposed only last_tool. Plus cooperative stop: request_stop(task_id), checked at each turn boundary, scoped by task_id so a stop cannot leak onto the next run | files: agent_loop/harness.py
- 2026-08-12 | /ui rebuilt: streaming replies (POST /ui/stream SSE), live tool cards from GET /ui/events?since=, POST /ui/stop, GET /ui/models (rates from public_rates() so the picker cannot drift from the biller). Markdown now does tables/headings/blockquote/nested lists/fenced code w/ per-block copy. Sticky-only-if-near-bottom autoscroll, theme toggle, Esc=stop, Cmd-K=new. Gate toggle: 'Ask first' (default, unchanged safety) vs 'Run it'. FIX: a valid cookie now dismisses the PIN pad on reload — it was only ever shown, so returning users re-typed the PIN every load | files: chat_ui.py
- 2026-08-09 | /ui login is now a 6-digit PIN keypad (6 dots + 3x4 number pad, auto-submit on 6th digit, shake on wrong, physical-key input) replacing the password field; still POSTs digits as 'password' to /ui/login, so UI_PASSWORD is now the 6-digit PIN (changed via Railway env). No lockout/rate-limit added | files: chat_ui.py
- 2026-08-09 | chat UI: default model = Kimi 2.6; render() hides tool-call/tool-result internals + de-dupes consecutive identical assistant turns (kills code-dump + double-output); research prompt adds 'answer tight in prose, no code paste unless asked, say once'; LIVE progress — CHAT_PROGRESS + GET /ui/chat_progress + composer polls it to show 'searching the repo · step 1/4' etc | files: chat_ui.py
- 2026-08-09 | chat research: dedicated _research_system_prompt() (lists the ACTUAL read-only tools; hard-directs repo_search-first, read repo MAP.md slice, tight github_read on the working branch, answer only from what was read) replaces reusing the operator prompt; steps 3->4. Verified: Kimi 2.6 named the exact env var w/ line cites for 1.5c; Sonnet searched+located but honestly declined to guess (7c) | files: chat_ui.py
- 2026-08-09 | chat research is now model-selectable (research follows the model chip, default Sonnet); added Kimi 2.6 (kimi-k2.6, cheaper than Sonnet) to the picker; UI_CHAT_RESEARCH_MODEL/_PROVIDER default empty=use chip (set to pin). Loop wrapped so a provider error (Moonshot/Kimi 429 when balance dry) returns a readable note not a 500. NOTE: Moonshot 429-ing at time of writing | files: chat_ui.py
- 2026-08-09 | chat research loop cost fix: runs on Sonnet (UI_CHAT_RESEARCH_MODEL, overrides chat chip) not Haiku; tool results truncated (UI_CHAT_RESEARCH_TOOL_CHARS=4000); spend ceiling (UI_CHAT_RESEARCH_MAX_CENTS=8) then forced final; steps=3; forced-final appends 'answer from what you have' user turn to stop leaked tool-call-as-JSON. Measured: 1 grounded prose answer ~5.7c/3 steps vs prior Haiku session ~54c | files: chat_ui.py
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
