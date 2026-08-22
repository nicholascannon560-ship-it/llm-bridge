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

## Deploying (read before you merge)

Railway deploys the **`Agent-loop`** branch of this repo, not `main`. A fix on
`main` never reaches production. The two branches have diverged.

**Never assume a push deployed.** Deployments on this service are often created
and then marked **SKIPPED** within 3-5 seconds — one build log line
(`scheduling build on Metal builder`), no build steps, no container. Nothing
recovers a skipped deployment on its own. Cases where a skip appeared to
self-heal were a *second* `reason: deploy` for the same commit, i.e. someone
triggering a redeploy — not Railway retrying. PR #9 had nobody doing that and
sat undeployed for nearly two hours while both branches looked merged and green.

**The cause is not established.** Ruled out by measurement on 2026-08-22:

| Hypothesis | Refuted by |
|---|---|
| merge commits skip, single-parent deploy | `ea9dc0ac` — squashed, 1 parent, SKIPPED |
| a file path or directory is excluded | `agent_loop/harness.py` both SKIPPED (`670b0631`) and SUCCESS (`75d0c48b`) |
| root-level files don't match `/**` | `MAP.md` SKIPPED, `DEPLOY_TRIGGER.md` SUCCESS — both root |
| the watch-path negations are catching it | no skipped commit touched `commands/` or `revenue_agents/` |

One known-benign case: several commits pushed seconds apart, where earlier ones
are superseded. That does not explain the lone pushes that skipped.

**So, operationally — after every merge or push to `Agent-loop`:**

1. Check `list-deployments` for the service. Find your `commitHash`.
2. If its newest deployment is SKIPPED, or there is none, redeploy:
   `POST /railway/service/509d2aef-f1a5-4854-9a57-e44cf9c079a0/redeploy`.
3. Confirm the SUCCESS row reports **your** `commitHash`, then read the deploy
   logs. A green deploy means the container started, not that the code works.

A stale container that everyone believes is current is the failure mode that
has cost the most time in this repo. Verify the running commit, every time.

**Do not "fix" this by clearing the watch paths** (`/**`, `!/commands/**`,
`!/revenue_agents/**`). `commands/` has ~94 tracked files and the command
channel writes there constantly; without `!/commands/**` every agent command
would redeploy production. The negations are load-bearing.

`railway.json` claims `builder: NIXPACKS`; the live service uses `RAILPACK`.
The file is not being applied. Harmless, but do not trust it.

`AGENT_AUTO_MODE` is not set on the service, so auto mode boots OFF and any
runtime toggle via `/agent/auto_mode` is lost on the next restart. Set it as a
service variable if you want it to persist.

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
- 2026-08-22 | SKIPPED deploys diagnosed: watch paths reject merge commits (5/5); squash-merge or redeploy | files: MAP.md
- 2026-08-22 | AGENT_AUTO_MODE is not set on the service, so auto mode boots OFF and resets on every restart | files: -

- 2026-08-22 | railway_get_domains(service_id) tool: base URL by ID, works when projects listing is empty | files: railway_extension.py, agent_loop/tools.py
- 2026-08-22 | empty projects listing now returns _hint: token visibility != permission on Railway | files: railway_extension.py

- 2026-08-22 | railway_extension helpers are sync: call via asyncio.to_thread, never await | files: agent_loop/tools.py, approval_routes.py
- 2026-08-22 | approval set_env arg order fixed; railway_gql_query -> railway_query (name never existed) | files: approval_routes.py
- 2026-08-22 | AST audit guards the railway sync/async boundary repo-wide | files: tests/test_railway_sync_boundary.py

- 2026-08-06 | append-only journal commands/journal/<run_id>/<turn>.json; harness on_turn hook | files: command_channel.py, agent_loop/harness.py
- 2026-08-06 | MAP still has no agent_loop/ slice — harness.py now carries on_turn/_record | files: agent_loop/harness.py

- 2026-08-06 | DELETE /contents added (guarded to commands/* prefixes) | files: main.py, tests/test_delete_guard.py
- 2026-08-06 | OPERATING NOTE (superseded 2026-08-22, see ## Deploying): a commit may not auto-deploy | files: -

- 2026-08-06 | issue #2 fix drafted: in-progress state -> commands/running/, results single-writer | files: command_channel.py, tests/test_result_race.py
- 2026-08-06 | MAP has no agent_loop/ coverage (harness, tools, routes, browser, memory, validate) | files: agent_loop/*

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
