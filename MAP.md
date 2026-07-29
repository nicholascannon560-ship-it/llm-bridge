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

| Slice | When to use | Key files |
|-------|-------------|-----------|
| **command-channel** | Sending/receiving Kimi or agent commands via GitHub | `command_channel.py`, `commands/pending/`, `commands/results/` |
| **llm-gateway** | Provider routing, timeouts, reasoning_effort, cost table | `llm_gateway.py`, `llm_routes.py` |
| **railway** | Deploy, status, logs, env vars | `railway_extension.py` |
| **skills** | Skill definitions and /skills endpoints | `skills/`, `skills_routes.py` |
| **app-wiring** | Routes, middleware, health, startup | `main.py`, `Procfile`, `railway.json` |

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

## Deep Docs (L2 — load only if needed)
- Full endpoint list + architecture diagram → `README.md`
- Individual skill definitions → `skills/`
- Diff-update protocol → `change-log/MAP_UPDATE.md`

## Pending Map Updates
<!-- Agents: append one-line deltas here. Format: - YYYY-MM-DD | short description | files: path1, path2 -->
- 2026-07-28 | Forwarded reasoning_effort + raised LLM timeout to 120s | files: command_channel.py
- 2026-07-28 | Added task slices section | files: MAP.md
- 2026-07-28 | Added /issues endpoints (list, create, list+add comments) so agents can close the change-log loop | files: main.py
- 2026-07-28 | Added GET /map_freshness: runs change-log's per-slice gate with the bridge's token | files: main.py
