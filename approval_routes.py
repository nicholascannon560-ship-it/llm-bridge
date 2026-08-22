# approval_routes.py
import asyncio
import uuid
import time
from typing import Any, Optional
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

router = APIRouter(prefix="/approvals", tags=["approvals"])

_PENDING: dict[str, dict[str, Any]] = {}
_APPROVAL_TTL_SECONDS = 3600

class ProposedAction(BaseModel):
    action: str
    payload: dict[str, Any]
    requested_at: float = Field(default_factory=time.time)
    description: str

class ApprovalResponse(BaseModel):
    approval_id: str
    status: str
    description: str

def _expire_stale():
    now = time.time()
    stale = [k for k, v in _PENDING.items() if now - v["requested_at"] > _APPROVAL_TTL_SECONDS]
    for k in stale:
        _PENDING[k]["status"] = "expired"

def queue_for_approval(action: str, payload: dict, description: str) -> str:
    _expire_stale()
    approval_id = str(uuid.uuid4())[:8]
    _PENDING[approval_id] = {
        "approval_id": approval_id,
        "action": action,
        "payload": payload,
        "requested_at": time.time(),
        "status": "pending",
        "description": description,
    }
    return approval_id

def get_pending(approval_id: str) -> Optional[dict]:
    _expire_stale()
    return _PENDING.get(approval_id)

@router.post("/propose", response_model=ApprovalResponse)
async def propose_action(action: ProposedAction):
    aid = queue_for_approval(action.action, action.payload, action.description)
    return ApprovalResponse(approval_id=aid, status="pending", description=action.description)

@router.post("/{approval_id}/approve")
async def approve_action(approval_id: str):
    req = get_pending(approval_id)
    if not req:
        raise HTTPException(status_code=404, detail="approval not found or expired")
    if req["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"already {req['status']}")
    req["status"] = "approved"
    result = await _execute_approved_action(req)
    return {"approval_id": approval_id, "status": "executed", "result": result}

@router.post("/{approval_id}/reject")
async def reject_action(approval_id: str):
    req = get_pending(approval_id)
    if not req:
        raise HTTPException(status_code=404, detail="not found")
    req["status"] = "rejected"
    return {"approval_id": approval_id, "status": "rejected"}

@router.get("/pending")
async def list_pending():
    _expire_stale()
    return [v for v in _PENDING.values() if v["status"] == "pending"]

async def _execute_approved_action(req: dict) -> dict:
    action = req["action"]
    payload = req["payload"]

    if action == "commit":
        from main import _do_commit
        return await _do_commit(payload)
    # railway_extension helpers are all plain sync functions doing blocking
    # requests.post() calls. Awaiting one raises "object dict can't be used in
    # 'await' expression" -- offload to a thread instead so the event loop
    # stays free during the HTTP round-trip.
    elif action == "redeploy":
        from railway_extension import redeploy_service
        return await asyncio.to_thread(
            redeploy_service,
            payload["service_id"],
            payload.get("environment", "production"),
        )
    elif action == "set_env":
        from railway_extension import set_service_variable
        return await asyncio.to_thread(
            set_service_variable,
            payload["name"],
            payload["value"],
            service_id=payload.get("service_id"),
            environment_name=payload.get("environment", "production"),
        )
    elif action == "railway_gql":
        from railway_extension import railway_query
        return await asyncio.to_thread(
            railway_query, payload["query"], payload.get("variables", {})
        )
    else:
        raise HTTPException(status_code=400, detail=f"unknown action {action}")
