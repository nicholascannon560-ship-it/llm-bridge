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

from .browser import FREE_TIER, browser_read, budget_status, parse_grant

agent_router = APIRouter(tags=["agent"])


@agent_router.get("/agent/browser_budget")
async def get_browser_budget():
    grant = parse_grant()
    return {
        "budget": budget_status(),
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
