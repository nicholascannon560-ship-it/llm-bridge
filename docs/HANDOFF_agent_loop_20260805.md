# Handoff — agent loop hardening + KalshiML audit

**Session:** 2026-08-05 evening EDT (2026-08-06 UTC)
**Agent:** Claude Opus 5, claude.ai chat + bridge
**Read this with:** `agent-map-system`, `github-railway-bridge`, `kimi-advisor-bridge` skills.
NOTE: KimiApiBridge- is ARCHIVED (read-only) as of 2026-08-06 — this handoff lives in llm-bridge/docs/ instead. Its HANDOFF.md is older than this file and documents v1.1.0; the live
service reports **1.5.0**. Prefer `GET /openapi.json` over any doc.

---

## 0. Facts that contradict the skill docs — check these first

| Thing | Doc says | Reality |
|---|---|---|
| llm-bridge deploy branch | `main` | **`Agent-loop`** (service `509d2aef`) |
| Command queue branch | unspecified (repo default) | **`main`**, now pinned by `GITHUB_BRANCH` |
| Bridge version | 1.4.2 | **1.5.0** |

Code deploys from `Agent-loop`; the queue lives on `main`. That split is now
deliberate — queue commits can't trigger a build that kills an in-flight run.
**Commit code to both branches** or the two drift again.

---

## 1. What shipped

### llm-bridge — `163adba` (Agent-loop) / `ef9ccb7` (main)
Made high/max reasoning actually usable. `reasoning_effort` was always a
per-command field; everything downstream of raising it was broken.

- `harness.py`: `max_tokens` was hardcoded 4096. Now a parameter defaulting by
  effort — low 4096 / high 16384 / max 32768, env-overridable
  (`AGENT_MAX_TOKENS_LOW|HIGH|MAX`). Moonshot bills reasoning tokens as
  completion tokens, so a low-effort ceiling truncates the moment effort rises.
- `harness.py`: any turn with no `tool_calls` was recorded `status: "complete"`,
  including `finish_reason: "length"`. Truncation now yields
  `status: "truncated"` with a diagnostic note. **This was the most misleading
  failure mode in the loop** — a truncated audit looked like a finished one.
- `harness.py`: run-level fault containment. A read timeout or context-overflow
  400 used to propagate out of `run()` and lose the entire transcript. Now
  `status: "llm_error"` with the transcript intact. Plus budget stops:
  `AGENT_CONTEXT_BUDGET_TOKENS` (180k) and `AGENT_COST_BUDGET_CENTS` (400).
- `tools.py`: `github_read` was `content[:8000]` with no offset — every file
  over ~200 lines was silently unreadable and the model audited a prefix as if
  it were the whole file. Now windowed: `offset` / `max_chars` (cap 60000) and
  `total_chars` / `truncated` / `next_offset` in the reply, with a description
  telling the model to page until `truncated` is false.
- `command_channel.py`: forwards `max_tokens` to `run_agent`.

### llm-bridge — `73cfe42` (Agent-loop) / `f5fed8e` (main)
Tier 0 of the reliability work.

- **`GET /agent/status`** — live run state from an in-memory dict: turn N of M,
  cost so far, last tool called, last prompt tokens. No GitHub call, no commit,
  safe to poll. Before this, a running agent and a dead one were
  indistinguishable from outside.
- **Per-turn checkpoints** — every `AGENT_CHECKPOINT_EVERY_TURNS` (5) turns the
  result file is overwritten with `status: "running"` plus transcript so far.
  Set 0 to disable. Cadence is capped because every checkpoint is a commit and
  unbounded commits is how the July 2,141-commit storm happened.
- **`GITHUB_BRANCH`** pins list / read / write / delete. Without it the queue
  followed the repo default and nothing enforced that it matched the deploy
  branch.

Tests (stubbed gateway, real loop):
```
checkpoints at turns [3, 6] | final state: complete turn 7 cost 7.0
truncation -> truncated     | max_tokens for effort=max: 32768
mid-run timeout -> llm_error | transcript preserved: 2 turns
/agent/status pre-run shape
```

### KalshiML — `b0ed775`
`adaptive_tree.recompute_tree()` plain-opened the live evidence file while
`learning.recompute_cell_states()` read through `evidence_archive.iter_evidence()`.
Identical only while rotation is off. Now reads the same corpus and gates on
`has_evidence()` instead of `os.path.exists()`. `import evidence_archive` is
inside the function — no new import cycle with `learning`.

`test_tree_evidence_corpus.py` proves the bug with real gzip chunks on disk:
```
old open():        5 rows / 5 distinct days
iter_evidence():  25 rows / 25 distinct days
```
Third case matters: on a volume where everything has rotated into chunks, the
live file doesn't exist and the old `os.path.exists()` guard would return `None`
and blank `cell_tree.json` — taking the system's only 1.0x license with it.

### Environment changes (no code)

`llm-bridge` (`509d2aef`): `MOONSHOT_TIMEOUT_SEC=300`,
`AGENT_CONTEXT_BUDGET_TOKENS=180000`, `AGENT_COST_BUDGET_CENTS=400`,
`AGENT_CHECKPOINT_EVERY_TURNS=5`, `GITHUB_BRANCH=main`.
Watch patterns: `["/**", "!/commands/**", "!/revenue_agents/**"]`.

`llm-bridge-v2` (`5cdb1036`): `AGENT_LOOP_ENABLED=0`,
`GITHUB_REPO=llm-bridge-v2-has-no-queue`.

Bridge key rotated at the start of the session (the old value had 30 minutes of
TTL left and had also landed in chat via the upload).

---

## 2. The v2 incident — read before touching either service

v2 has its own `GITHUB_TOKEN` and `MOONSHOT_API_KEY` and runs the same
queue-in-`/health` code. The first KalshiML audit run **never executed**: v2
dequeued the command, refused it (`AGENT_LOOP_ENABLED=0`), wrote an error
result, and deleted the pending file before the live service ticked.

`AGENT_LOOP_ENABLED=0` stops v2 *running* commands. It does not stop v2
*consuming* them. Pointing `GITHUB_REPO` at a nonexistent repo is what actually
fixed it (`_list_pending` 404s → returns `[]`, silent).

**Open decision for Nicholas: v2 has no known purpose. Deleting it removes a
whole class of failure.**

---

## 3. KalshiML audit — findings 2-6 still open

Run `audit-kalshiml-20260806b`: 23 turns, 9m40s, 95.6¢, `reasoning_effort=high`,
read-only tools. Full transcript in `commands/results/` on `main`.

**Finding 1 — FIXED** (`b0ed775` above).

**Finding 2 — HIGH. `record_error()` is called but defined nowhere.**
3 call sites in `learning.py`, 2 in `hourly_engine._one_pass`. Raises NameError
on the maker-abandoned and stuck-order alert paths — i.e. exactly when a cancel
succeeded but the cross failed — and takes out that pass's `live_signal()` on
the way down. Surfaces only as a generic "pass error" line.
*Needs a decision:* a writer in `config.py` appending to `errors.jsonl` on the
volume (dashboard can then surface it), or a two-line version printing to the
log buffer.

**Finding 3 — MEDIUM. Three byte-windowed day counters in `autopilot.py`,**
not one: `_evidence_field_coverage`, `_skill_history_days`, `_realized_coverage`,
all sharing `EVIDENCE_SCAN_BYTES`. Live state shows `pnl_aware_gate` at 2/7 days
and `hourly_wx_axes` at 2/14 while the same file reports 246 cells and 20 days on
the best cell. Raising the byte count buys days, not a fix — wants a monotonic
per-capability day counter persisted in `autopilot_state.json`.

**Finding 4 — MEDIUM.** `learning.gate_decision`'s tree path is
`except Exception: pass`, so a corrupt `cell_tree.json` or a throw from finding 1
silently drops sizing to the flat gate with no counter and no log.

**Finding 5 — MEDIUM.** Still no startup scan of exchange-side open orders, and
cross sizing still trusts `reduced_by` without a confirming fills query. Both are
the pre-`PLACE_REAL_ORDERS` blockers already documented in the `maker_orders.py`
docstring. `_hourly_already_open` and the circuit breaker both fail **open**.

**Finding 6 — LOW.** `hourly_engine.py` has a hardcoded `KALSHI_KEY_ID` literal
while `config.REQUIRED_SECRETS` and `daily_part_01.py` read it from env. Rotate
the Kalshi account and every signed request fails auth with no config error.

**Verified clean:** the realized-PnL identity (`edge_c + 100*(won - pred)`), the
day-blocked bootstrap (resamples day-means, not rows), and the 20-row / 5-day
floors are all implemented as specified and pinned by
`test_realized_pnl_gate.py`.

---

## 4. Tier 1 — reviewed design, not yet built

Replace GitHub-as-queue with Postgres + a dedicated Railway worker. Kimi
reviewed it at high effort: **approve with required changes.** Do not build the
naive version.

### Required before it can ship

1. **Resume-by-replay is unsafe with side-effecting tools.** Replaying
   `agent_turns` rebuilds what the model saw — fine for reads. The dangerous
   window is "tool executed, turn row not written": resume re-issues a
   `github_commit` or `railway_redeploy` that already happened.
   → `side_effects(run_id, turn, tool_call_id, tool, args, status, external_ref)`,
   **intent written before execution**, outcome after. On resume, an unresolved
   `intent` row is verified against external state (GitHub log, Railway
   deployment list); if unresolvable it **pauses for the operator** rather than
   auto-retrying. Embed `run_id:turn` in commit messages so dedup is a lookup.
2. **Fencing.** Heartbeat reclaim without it is double-execution with extra
   steps — worker A stalls past the timeout, B reclaims, A wakes and keeps
   committing. Add `claim_epoch`, `WHERE claim_epoch = $epoch` on every write,
   worker self-terminates on a zero-row update. Heartbeat needs a 30s timer
   during long LLM calls, not just per-turn, or one slow turn blows the window.
   *Kimi's stronger suggestion, which I agree with:* at single-worker scale make
   auto-reclaim opt-in and **manual requeue the default** — safest semantics for
   ~$1 runs with repeatable side effects.
3. **Poison messages survive the migration.** A malformed row that crashes the
   worker gets reclaimed forever. Add `attempts`, force `failed` at 3.
4. Cancel must be checked at the top of each turn loop. `CHECK` constraint on
   status. Index `(status, created_at)`. SIGTERM handling — the reclaim timeout
   is the per-run deploy downtime.
5. **Cursor polling (`/turns?after=N`) instead of SSE** — a 40-minute run will
   hit Railway's idle proxy timeout, and a cursor makes resume semantics
   explicit.

### Context/cost work — do last, and check the cheap thing first

The audit run burned **1.49M prompt tokens against 15k completion tokens** over
23 turns; the final turn alone re-sent 100k. Before building digest machinery:

1. Check whether Kimi K3 supports **prompt caching**. Cache hits likely beat
   digests with zero fidelity loss.
2. Then try plain truncation — drop old bulk-read results rather than
   summarizing; the model can re-read via tools.
3. Only then digests. Hard rule: **never digest results from `github_commit`,
   `railway_redeploy`, or anything returning an identifier a later turn
   references.** A digest that drops a commit SHA makes the model re-query or
   hallucinate one. Require structured digest output and assert every SHA / URL /
   number in the raw result survives into `identifiers`; fall back to raw on
   validation failure. Store digest text so the effective context stays
   auditable.

Postgres is justified **narrowly** — mostly because the bridge is already
multi-service on Railway. SQLite on a volume would be simpler and free, but
Railway volumes mount to a single service so the web tier couldn't read it.

Suggested build order: schema (`attempts`, `claim_epoch`, CHECK) → side_effects
with intent-before-execute → worker service with SIGTERM + manual requeue →
cursor-polling API → context work.

---

## 5. Operating the loop today

- Launch: commit `{"action":"agent_run","task":...,"tools":[...],"max_turns":N,
  "reasoning_effort":"high","max_tokens":16384}` to
  `commands/pending/<id>.json` **on `main`**.
- Watch: `GET /agent/status` (live) or `GET /llm/usage?days=1` (request count
  climbing = alive). Results land in `commands/results/<id>.json`.
- For read-only work pass an explicit `tools` list rather than
  `tool_set: "build"` (which includes `github_commit`). `tool_set: "research"`
  is read-only but includes browser tools — a prior commit message reports the
  research set hangs, unverified.
- High effort measured ~20-90s per turn. A 23-turn audit ran under 10 minutes
  and cost 95.6¢.
