# MAP: llm-bridge

**purpose:** Universal FastAPI bridge that gives agents controlled access to GitHub, Railway, multi-provider LLMs (Anthropic + Moonshot/Kimi), and a skills engine.  
**status:** active  
**last_verified:** 2026-07-28  
**token budget:** ~1200

## Entry Points
- **Main app:** `main.py` (FastAPI app assembly)
- **LLM gateway:** `llm_gateway.py` + `llm_routes.py`
- **Railway control:** `railway_extension.py`
- **Command channel (GitHub-based):** `command_channel.py` + `commands/` folder
- **Skills:** `skills/` + `skills_routes.py`
- **Deploy:** Railway auto-deploys from this repo. Health endpoint: `/health`

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

## Relationships
- **Central control plane** for almost every other project
- Consumed by: KalshiML, tali, resolume-bridge, change-log workflows
- Holds the tokens: `GITHUB_TOKEN`, `RAILWAY_API_TOKEN`, `MOONSHOT_API_KEY`, `ANTHROPIC_API_KEY`, `BRIDGE_API_KEY`

## Agent Start Instructions
1. Read this MAP.md first.
2. For most changes: edit the relevant `.py` file, then commit via the bridge itself or GitHub tools.
3. To talk to Kimi: drop a command JSON into `commands/pending/` (see existing examples in `commands/results/`).
4. After any meaningful change, bump `last_verified` in this file and note it in the change-log.
5. **Do not touch** secrets or rotate keys unless explicitly asked.

## Deep Docs (L2 — load only if needed)
- Full endpoint list + architecture diagram → `README.md`
- Individual skill definitions → `skills/`
