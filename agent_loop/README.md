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

Tools come in **sets**, and the split is the point.

| Set | Tools | Can write? | Sees untrusted web text? |
|-----|-------|-----------|--------------------------|
| `build` (default) | github_read, github_commit, railway_* , llm_chat, read/write_memory, http_get, web_search, kml_* | yes | snippets only (web_search) |
| `research` | github_read, railway_get_*, llm_chat, read_memory, http_get, web_search, kml_*, **browser_read**, **browser_research** | no | yes |

`resolve_tools()` / `assert_tool_set_safe()` refuse any set that puts a
browsing tool next to `github_commit`, `railway_set_env`, `railway_redeploy`
or `write_memory`. A web page must not be able to cause a commit, change an
env var, or plant a "lesson" that gets replayed into later runs. Browse in one
run, act in a second one.

| Tool | What it does |
|------|-------------|
| `browser_read` | One page via Browserbase Fetch → markdown. Handles JS. Cheap: 1 of 1,000 monthly calls, no browser time. |
| `browser_research` | A real browser agent run (Browserbase Agents API) for multi-step tasks. **3 per month on the free plan.** |
| `github_read` / `github_commit` | Read / commit repo files |
| `railway_*` | Status, logs, redeploy, set_env (protected names refused) |
| `llm_chat` | Sub-question to another model |
| `read_memory` / `write_memory` | JSONL reflections (`write_memory` is build-only) |
| `http_get`, `kml_data_read`, `kml_app_logs` | Allowlisted fetch through `fetch_routes` |
| `web_search` | DuckDuckGo search via `search_routes` (upstream host hardcoded, no key). Returns titles/URLs/snippets; read a page via `http_get` (allowlisted) or `browser_read`. |

## Browsing: limits and permission

No browser-use, no playwright, no chromium in the container — browser-use
0.13.7 pins `pydantic==2.12.5` against the bridge's 2.10.4, and a browser
running *inside* this container could reach Railway's private network.
Everything goes over HTTPS to `api.browserbase.com`.

Free-tier ceilings, and what we set under them:

| | Free plan | Our default |
|---|---|---|
| Agent runs | 3 / month | 3 (1 per agent run) |
| Fetch calls | 1,000 / month | 900 (6 per agent run) |
| Browser time | 1 hour / month | 3,300 s |
| Concurrency | 3 | 1 |
| Session length | 15 min | 840 s |
| Proxy | 0 GB | never requested |

`GET /agent/browser_budget` shows current usage and the active grant. The
ledger is container-local and a deploy wipes it; set `BROWSERBASE_PROJECT_ID`
so it reconciles against Browserbase's own session list.

### Logging in — operator's order only

`mode="interactive"` is refused unless a grant is present:

```
BROWSER_INTERACTIVE_GRANT = domains=portal.acme.com;creds=ACME;until=2026-08-04T18:00Z
BROWSER_CRED_ACME_USERNAME = ...
BROWSER_CRED_ACME_PASSWORD = ...
```

Set it by hand in Railway. It is deliberately not a tool argument and not a
field in the command file — command files are committed to this repo, so a
secret there would live in git history. `railway_set_env` refuses to write any
`BROWSER_*` variable, so an agent cannot grant itself. Expiry is required;
keep it to hours. Credentials reach Browserbase as run `variables` and the
agent uses `%acme_password%` — our model never sees the values.

## Integration

`agent_run` through the command channel:

```json
{"action": "agent_run", "task": "...", "tool_set": "research", "max_turns": 8}
```

It returns immediately with `status: "started"`. The run happens on a
background thread — `_execute` is called from `GET /health`, which is
Railway's liveness probe, so a multi-minute run there means failed probes and
a restarted container. The result file is overwritten when the run finishes.
One run at a time; `AGENT_LOOP_ENABLED=0` stops all of them without a deploy.
