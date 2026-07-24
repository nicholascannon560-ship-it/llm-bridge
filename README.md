# llm-bridge

A universal FastAPI bridge service that connects LLM agents to infrastructure:
- **GitHub control** — commit files, create repos, read contents
- **Railway control** — deploy services, read logs, manage env vars, rotate keys
- **LLM gateway** — route requests across providers (Anthropic, Moonshot/Kimi, OpenAI)
- **Skills engine** — server-side skill definitions with structured output parsing

Authentication uses environment variables (`GITHUB_TOKEN`, `RAILWAY_API_TOKEN`, `BRIDGE_API_KEY`).
Tokens are never returned in responses.

## Quick Start

Deploy to Railway from this repo. Required env vars:
- `GITHUB_TOKEN` — GitHub personal access token (repo scope)
- `RAILWAY_API_TOKEN` — Railway API token
- `BRIDGE_API_KEY` — self-managed auth key for bridge clients
- `ANTHROPIC_API_KEY` — for Claude access via gateway
- `MOONSHOT_API_KEY` — for Kimi access via gateway

## Core Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET`  | `/health` | Liveness check + key age |
| `GET`  | `/repos` | List GitHub repos |
| `POST` | `/repos` | Create a GitHub repo |
| `POST` | `/commit` | Commit a single file |
| `POST` | `/commit_tree` | Commit multiple files as one commit |
| `GET`  | `/contents/{owner}/{repo}/{path}` | Read file or list directory |
| `GET`  | `/skills` | List available skills |
| `GET`  | `/skills/{name}` | Read skill definition |
| `POST` | `/skills/{name}/run` | Execute a skill (builds prompt, calls LLM, parses output) |
| `GET`  | `/llm/providers` | List available LLM providers |
| `POST` | `/llm/chat` | Chat via LLM gateway |
| `GET`  | `/railway/projects` | List Railway projects |
| `GET`  | `/railway/service/{id}/status` | Get latest deployment status |
| `POST` | `/railway/service/{id}/redeploy` | Trigger redeploy |
| `POST` | `/railway/gql` | Generic Railway GraphQL proxy |
| `POST` | `/rotate_key` | Rotate the bridge API key |

See interactive docs at `/docs` when running.

## Architecture

```
┌─────────────┐     HTTP      ┌─────────────┐     ┌─────────────┐
│   Agent     │ ────────────→ │  llm-bridge │ ──→ │   GitHub    │
│  (any LLM)  │  X-Bridge-Key │   FastAPI   │     │   Railway   │
└─────────────┘               │   Gateway   │ ──→ │   LLM APIs  │
                              └─────────────┘     └─────────────┘
```

## Skills System

Skills are markdown files in `./skills/` with YAML frontmatter + mode instructions.
The bridge builds prompts from skill definitions, routes them through the LLM gateway,
and parses structured output into actionable sections.

## License

MIT
