# Revenue agents — runbook

Lives at `revenue_agents/` in `llm-bridge`, branch **Agent-loop** (the branch
service `509d2aef` actually builds from — `main` never reaches the live bridge).

```
revenue_agents/
  assets.json          <- EDIT THE TODOs FIRST. Nothing works until you do.
  pitch.schema.json
  verdicts.jsonl       <- append-only feedback ledger
  prompts/00_runbook.md 01_scout.md 02_generator.md 03_redteam.md 04_ranker.md
```

These are **prompt/data files only**. No Python, no imports, nothing wired into
`main.py` — committing them cannot break the bridge or the watchdog.

## Run order — four separate agent_run commands

Drop each into `commands/pending/<id>.json` in `llm-bridge`. `GET /health`
ticks the queue; `agent_run` returns `status: started` and executes on a
background thread. **One run at a time** — wait for each result before the next.

```json
{"action":"agent_run","tool_set":"research","max_turns":10,
 "task":"Follow revenue_agents/prompts/01_scout.md in nicholascannon560-ship-it/llm-bridge branch Agent-loop. Read it first with github_read, then execute it exactly."}
```
Then `02_generator.md` with `"tool_set":"build"`, `03_redteam.md` with
`"research"`, `04_ranker.md` with `"build"`.

## Why the tool sets alternate

`resolve_tools()` refuses any set that puts a browsing tool next to
`github_commit` / `railway_set_env` / `write_memory`. So:

- **Scout** and **red team** browse and therefore cannot write. Each emits its
  JSON as its final message; the next build-stage run reads the result file and
  commits it.
- **Generator** and **ranker** can commit and therefore cannot browse. For the
  generator that is a feature, not a workaround — it should reason from the
  scout's evidence, not go shopping for validation.

Web text still reaches a run that can commit, one hop later. That is why every
build-stage prompt carries an UNTRUSTED INPUT clause and a whitelist of the
exact paths it may write. Keep both if you edit the prompts.

## Cadence
Weekly. The signal pool does not refresh nightly and a daily email trains you
to ignore it.

## verdicts.jsonl — the part that makes it improve
```json
{"id":"pitch_slug","verdict":"kill|park|test","reason":"your words","at":"ISO8601"}
```
The generator reads it every run and may not re-pitch a killed idea. Red-team
KILLs auto-append. Your own verdicts you append by hand (or reply-parse later).
Skip this and you get the same eight ideas forever.

## Before the first run
1. Fill the `operator` TODOs in `assets.json`. The hours and cash ceilings are
   what the generator binds against — left as TODO, its constraint checks are
   vacuous and you get slop.
2. Confirm `AGENT_LOOP_ENABLED` is not `0`.
3. Check `GET /agent/browser_budget` — `browser_research` is 3/month.
