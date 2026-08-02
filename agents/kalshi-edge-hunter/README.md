# Kalshi Edge Hunter

Autonomous money agent that lives on the llm-bridge.

## Goal
Find tradable edges in Kalshi NYC hourly temperature contracts (and related markets) that survive fees and have positive expected value over recent history.

## Data access (via `http_get`)
All live data is reached through the KalshiML production dashboard:

| Need | URL |
|------|-----|
| Learning brain snapshot | `https://kalshiml-production.up.railway.app/api/state` |
| Realized performance | `.../api/scorecard` |
| Every cell | `.../api/performance` |
| One cell deep-dive | `.../api/cell?cell=...` |
| Evidence / decisions | `.../api/evidence` |
| Live logs | `.../api/logs` |
| Raw file list | `.../api/files` |
| Specific file | `.../api/file?path=...` |
| Market-flow layer (~8 GB bulk) | `.../api/ml/status` (+ signals/tickers/train/flow) |
| Full catalog | `.../api` |

**Rule:** Prefer the summary endpoints. Never try to download multi-GB raw volumes.

## Where findings go

Every concrete finding, edge hypothesis, or failed idea is appended as one JSON line to:

```
agents/kalshi-edge-hunter/findings.jsonl
```

Schema for each line:
```json
{
  "ts": "ISO-8601 UTC",
  "type": "edge_hypothesis" | "performance_note" | "dead_end" | "risk_flag" | "action_proposal",
  "title": "short label",
  "summary": "1-3 sentence finding",
  "evidence": ["urls or key numbers"],
  "confidence": 0.0-1.0,
  "next": "what should happen next (optional)"
}
```

Short reflections also go into the shared agent memory via `write_memory` (tagged `kalshi-edge-hunter`).

High-value proposals can additionally be committed as a short markdown note or left for a human/executor to act on.

## Safety
- Budget-limited (see state file).
- Read-only against KalshiML (no order placement from this agent).
- Auditor / human review before any real-money change.
