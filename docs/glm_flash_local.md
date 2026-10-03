# Can GLM Flash run locally? (researched 2026-10-03)

**Short answer: No — the GLM "Flash" tier is API-only. Open-weights GLM siblings
can be self-hosted, and the bridge already has the wiring pattern to reach one.**

## What "glm flash" means in this bridge

The bridge's Flash model is `z-ai/glm-5.3-flash`, served via OpenRouter
(`OPENROUTER_API_KEY`). It is the cheap default (`DEFAULT_MODELS["openrouter"]`,
$0.075 in / $0.25 out per Mtok, 1M ctx). It never runs here: chat_ui.py lists it
last precisely because "it still leaves the bridge for a third party via
OpenRouter".

## Are Flash weights downloadable?

No. Both GLM-5.3-Flash and GLM-5.3 (asked directly through the gateway,
2026-10-03) report the Flash tier is a hosted serving tier on Z.ai /
bigmodel.cn with no published checkpoint on Hugging Face or ModelScope. No
independent web source was reachable from this environment to confirm — treat
"check Z.ai's HF org for a surprise Flash release" as the one open TODO.
Uncertainty flag: Flash-tier weights *could* have been open-sourced after these
models' training cutoffs.

## What GLM IS open weights (self-hostable)

Per the same two models (both MIT-licensed releases):

| Model | Size | Local footprint (Q4) |
|---|---|---|
| GLM-4-9B-Chat | 9B dense | ~6 GB file, runs in 8 GB VRAM or CPU RAM |
| GLM-4-32B-0414 / GLM-Z1-32B | 32B dense | ~20 GB |
| GLM-4.5-Air | 106B MoE / ~12B active | ~60-70 GB (64 GB+ RAM/VRAM) |
| GLM-4.5 / GLM-4.6 | 355B MoE | multi-GPU / >192 GB RAM |

Closest Flash-like local swap: **GLM-4-9B-Chat** for the size/cheapness slot,
**GLM-4.5-Air** for the capability slot (MoE with ~12B active params is
surprisingly fast per-token). GGUFs exist (bartowski / Unsloth HF repos);
llama.cpp, Ollama and vLLM all run them.

## How to wire a local GLM into this bridge (when wanted)

- Railway containers are CPU-only and hold no weights, so "local" means
  Nicholas's own hardware. The bridge can only reach it through a tunnel
  (Tailscale/tailscale funnel, ngrok, cloudflared) plus an auth token.
- Wiring shape: a provider class with a configurable base_url — exactly the
  pattern of `QwenProvider` on this branch's `main` (base_url + api_key over
  the OpenAI wire format). Ollama (`http://localhost:11434/v1/...`),
  llama.cpp-server and vLLM all speak that wire format.
- Gap today: `DEFAULT_MODELS` has a `"local"` slot and `llm_routes.py` lists
  "local" as a provider name, but `LLMRouter.__init__` registers only
  anthropic/moonshot/openai/openrouter. The local slot is a stub — a
  `provider="local"` request cannot be served until a `LocalProvider`
  (configurable base_url, e.g. `LOCAL_BASE_URL` + `LOCAL_API_KEY`) is added
  and registered when the env is set.

## Verification provenance

- Repo facts read from `llm_gateway.py` / `chat_ui.py` / `llm_routes.py`
  (branch `Agent-loop`, sha `765fd13b`; router wiring lines 1304-1310).
- Model self-descriptions fetched live via the gateway's `llm_chat` path:
  one z-ai/glm-5.3-flash call and one z-ai/glm-5.3 call, both answering the
  open-weights question directly. Moonshot was 429-rate-limited at the time,
  so no third-lab cross-check was possible.