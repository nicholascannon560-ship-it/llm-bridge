"""agent_loop/routes.py — operator-facing views of the browser tool.

These are for Nicholas, not for agents: they sit behind the X-Bridge-Key
middleware like everything else, and they let him check the budget and the
current interactive grant without starting an agent run. /agent/browser_read
is also the cheapest way to prove the Browserbase wiring works after a deploy
(one fetch call out of the monthly thousand, no browser time).
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from .browser import (
    API,
    FREE_TIER,
    _client,
    _reconcile_budget,
    browser_read,
    budget_status,
    parse_grant,
)

agent_router = APIRouter(tags=["agent"])


@agent_router.get("/agent/status")
async def get_agent_status():
    """Live view of the agent run in progress.

    The result file in commands/results is only written when a run finishes, so
    before this endpoint the only way to tell a working run from a dead one was
    to watch /llm/usage and guess. This reads an in-memory dict — no GitHub
    call, no commit, safe to poll every few seconds.
    """
    from .harness import current_run_state

    state = current_run_state()
    if not state.get("task_id"):
        return {"active": False, "note": "no agent run has started since this container booted"}
    return state


@agent_router.get("/agent/browser_budget")
async def get_browser_budget():
    # Reconcile against Browserbase before answering: the local ledger is wiped
    # by every deploy, so without this the number is only true until the next
    # commit. Needs BROWSERBASE_PROJECT_ID; no-ops without it.
    reconciled = False
    try:
        async with _client(20) as client:
            await _reconcile_budget(client)
        reconciled = True
    except Exception:
        pass

    grant = parse_grant()
    return {
        "budget": budget_status(),
        "reconciled_with_browserbase": reconciled,
        "free_tier": FREE_TIER,
        "interactive_grant": {
            "valid": grant["valid"],
            "domains": grant["domains"],
            "creds": grant["creds"],
            "until": grant["until"],
            "reason": grant["reason"],
        },
        "how_to_grant": (
            "Set BROWSER_INTERACTIVE_GRANT in Railway, e.g. "
            "domains=portal.acme.com;creds=ACME;until=2026-08-04T18:00Z — then the next agent run "
            "may log in on that domain until the expiry. Credentials come from "
            "BROWSER_CRED_<LABEL>_USERNAME / _PASSWORD."
        ),
    }


class BrowserReadRequest(BaseModel):
    url: str
    max_chars: int | None = None


@agent_router.post("/agent/browser_read")
async def post_browser_read(req: BrowserReadRequest):
    return await browser_read({"url": req.url, "max_chars": req.max_chars})


@agent_router.get("/agent/browserbase_projects")
async def get_browserbase_projects():
    """List Browserbase projects using the container's key.

    Only reason this exists: BROWSERBASE_PROJECT_ID is what lets the budget
    ledger reconcile after a deploy, and the project id is not printed
    anywhere the operator can copy without logging into the dashboard. The API
    key stays inside the container; only ids and names come back.
    """
    try:
        async with _client(30) as client:
            resp = await client.get(f"{API}/v1/projects")
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if resp.status_code != 200:
        return {"error": f"browserbase returned {resp.status_code}", "detail": resp.text[:200]}
    projects = resp.json() or []
    return {
        "projects": [
            {"id": p.get("id"), "name": p.get("name"), "createdAt": p.get("createdAt")}
            for p in projects
        ],
        "current_env_value": __import__("os").getenv("BROWSERBASE_PROJECT_ID") or None,
    }
