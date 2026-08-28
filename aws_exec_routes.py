"""
aws_exec_routes.py — run commands on the pinned instance via SSM Run Command.

WHY THIS EXISTS
  The box has no inbound ports and no SSH key by design, which is correct and
  also meant that after a launch there was no way to answer "is the engine
  actually running" without a console session. Run Command closes that gap
  without reopening the network: the SSM agent polls outbound, AWS brokers the
  command, and IAM decides who may send one. No key material anywhere.

WHAT THIS IS
  Remote code execution on the box that trades real money. That is the honest
  description and the gating follows from it:

  - AWS_EXEC_ENABLED must be "1". Off by default, same as compute and ssm.
  - Default mode is VERBS, not shell. The verb table below is code, not config,
    so a caller chooses from a fixed set rather than composing a command.
  - Raw shell is a SEPARATE flag, AWS_EXEC_RAW_ENABLED, off by default. Turning
    on exec does not turn on raw. Two deliberate acts, not one.
  - The instance is resolved from the pinned Name tag, exactly like
    aws_compute_routes. A caller cannot aim this at another instance by id.

WHY VERBS BEFORE RAW
  Everything routine — read logs, check the unit, restart, pull and redeploy,
  check disk — is a fixed command with no free text in it. Those cover the
  actual day-to-day, and none of them can be talked into something else. Raw
  exists because eventually something unforeseen needs doing, but it should be
  a decision you make in that moment rather than a standing capability.

WHAT IS DELIBERATELY NOT HERE
  No verb prints /etc/kalshiml.env values. `env` returns NAMES ONLY, for the
  same reason aws_ssm_routes has no GET: that file holds the Kalshi private key
  path, the API key id and the Anthropic key, and a value that reaches an HTTP
  response reaches logs and transcripts too.

CONFIG
  AWS_EXEC_ENABLED       "1" to allow verbs. Default off.
  AWS_EXEC_RAW_ENABLED   "1" to additionally allow /aws/exec/raw. Default off.
  AWS_COMPUTE_NAME_TAG   shared with aws_compute_routes; which box this targets.
  AWS_DEFAULT_REGION     required (shared with aws_routes).

IAM REQUIRED
  Bridge user: ssm:SendCommand on the instance and on
    arn:aws:ssm:*::document/AWS-RunShellScript, plus ssm:GetCommandInvocation
    and ssm:DescribeInstanceInformation.
  Instance role: AmazonSSMManagedInstanceCore (or the equivalent inline set) so
    the agent can register. Without it the box is invisible to Run Command and
    every call here returns "instance not registered".
"""
from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:
    from aws_routes import _aws_client, _boto_error
except Exception:  # pragma: no cover
    _aws_client = None
    _boto_error = None

try:
    from aws_compute_routes import _find_existing, _name_tag
except Exception:  # pragma: no cover
    _find_existing = None
    _name_tag = None

aws_exec_router = APIRouter(tags=["aws"])

SERVICE = "kalshiml"
REPO_DIR = "/opt/kalshiml"
ENV_FILE = "/etc/kalshiml.env"

DEFAULT_TIMEOUT_S = 60
MAX_TIMEOUT_S = 300
MAX_OUTPUT_CHARS = 24000

# The verb table is code on purpose. Adding a capability here is a commit and a
# deploy, which is the point — it means the set cannot drift at runtime.
VERBS: dict[str, str] = {
    "logs": f"journalctl -u {SERVICE} -n {{n}} --no-pager",
    "logs_follow_tail": f"journalctl -u {SERVICE} --since '10 min ago' --no-pager",
    "status": f"systemctl status {SERVICE} --no-pager || true",
    "is_active": f"systemctl is-active {SERVICE} || true",
    "restart": f"systemctl restart {SERVICE} && sleep 3 && systemctl is-active {SERVICE}",
    "start": f"systemctl start {SERVICE} && sleep 3 && systemctl is-active {SERVICE}",
    "stop": f"systemctl stop {SERVICE} && systemctl is-active {SERVICE} || true",
    "disk": "df -h / /var/lib 2>/dev/null; echo '--- largest ---'; "
            "du -sh /var/lib/kalshiml/* 2>/dev/null | sort -h | tail -15",
    "mem": "free -m; echo '--- top ---'; ps aux --sort=-%mem | head -8",
    # Names only. Never values — see module docstring.
    "env": f"cut -d= -f1 {ENV_FILE} 2>/dev/null | sort",
    "git": f"cd {REPO_DIR} && git log --oneline -5 && git status --short | head -20",
    "update": f"cd {REPO_DIR} && git pull --ff-only && systemctl restart {SERVICE} "
              f"&& sleep 3 && systemctl is-active {SERVICE}",
    "bootstrap_log": "tail -n {n} /var/log/kalshiml-bootstrap.log",
    "uptime": "uptime; echo '--- boot ---'; who -b",
}

VERBS_TAKING_N = {"logs", "bootstrap_log"}


# --------------------------------------------------------------------------- #
# Models


class VerbRequest(BaseModel):
    verb: str = Field(..., description=f"One of: {', '.join(sorted(VERBS))}")
    n: int = Field(50, ge=1, le=2000, description="Line count for log verbs.")
    timeout_s: int = Field(DEFAULT_TIMEOUT_S, ge=5, le=MAX_TIMEOUT_S)


class TerminateRequest(BaseModel):
    confirm_instance_id: str = Field(
        ..., description="Must equal the pinned instance id. Guard, not target.")


class RawRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=8000)
    timeout_s: int = Field(DEFAULT_TIMEOUT_S, ge=5, le=MAX_TIMEOUT_S)


# --------------------------------------------------------------------------- #
# Guards


def _enabled() -> None:
    if (os.getenv("AWS_EXEC_ENABLED") or "").strip() != "1":
        raise HTTPException(
            403,
            "remote exec is disabled — set AWS_EXEC_ENABLED=1, run what you "
            "need, then unset it.",
        )


def _raw_enabled() -> None:
    if (os.getenv("AWS_EXEC_RAW_ENABLED") or "").strip() != "1":
        raise HTTPException(
            403,
            "raw shell is disabled — set AWS_EXEC_RAW_ENABLED=1 for the one "
            "call, then unset it. AWS_EXEC_ENABLED alone does not grant this.",
        )


def _client(svc: str = "ssm"):
    if _aws_client is None:  # pragma: no cover
        raise HTTPException(503, "aws_routes failed to import; no AWS client")
    return _aws_client(svc)


def _fail(e: Exception) -> HTTPException:
    if _boto_error is None:  # pragma: no cover
        return HTTPException(502, f"aws error: {type(e).__name__}: {e}")
    return _boto_error(e)


def _target_instance_id() -> str:
    """Resolve the pinned box. Never accepts an id from the caller."""
    if _find_existing is None or _name_tag is None:  # pragma: no cover
        raise HTTPException(503, "aws_compute_routes failed to import")
    name = _name_tag()
    inst = _find_existing(_client("ec2"), name)
    if not inst:
        raise HTTPException(404, f"no running instance tagged Name={name}")
    return inst["InstanceId"]


def _registered(instance_id: str) -> bool:
    ssm = _client("ssm")
    try:
        resp = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [instance_id]}])
    except Exception as e:
        raise _fail(e)
    info = resp.get("InstanceInformationList") or []
    if not info:
        return False
    # Presence is not liveness. A terminated or starved box keeps its record
    # for a while with PingStatus ConnectionLost, and reporting that as
    # "registered" sent me chasing a healthy-looking box that was OOM-looping.
    return info[0].get("PingStatus") == "Online"


# --------------------------------------------------------------------------- #
# Execution


def _run(command: str, timeout_s: int) -> dict:
    instance_id = _target_instance_id()
    if not _registered(instance_id):
        raise HTTPException(
            409,
            f"{instance_id} is not registered with SSM. The agent needs an "
            "instance role carrying AmazonSSMManagedInstanceCore; until then "
            "Run Command cannot reach the box.",
        )
    ssm = _client("ssm")
    try:
        sent = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [command],
                        "executionTimeout": [str(timeout_s)]},
            TimeoutSeconds=timeout_s,
            Comment="llm-bridge /aws/exec",
        )
    except Exception as e:
        raise _fail(e)

    cmd_id = sent["Command"]["CommandId"]
    deadline = time.time() + timeout_s + 15
    inv: dict = {}
    while time.time() < deadline:
        time.sleep(2)
        try:
            inv = ssm.get_command_invocation(
                CommandId=cmd_id, InstanceId=instance_id)
        except Exception:
            # Invocation is not immediately queryable after send.
            continue
        if inv.get("Status") not in ("Pending", "InProgress", "Delayed"):
            break

    out = (inv.get("StandardOutputContent") or "")[:MAX_OUTPUT_CHARS]
    err = (inv.get("StandardErrorContent") or "")[:MAX_OUTPUT_CHARS]
    return {
        "instance_id": instance_id,
        "command_id": cmd_id,
        "status": inv.get("Status") or "Unknown",
        "exit_code": inv.get("ResponseCode"),
        "stdout": out,
        "stderr": err,
        "truncated": len(inv.get("StandardOutputContent") or "") > MAX_OUTPUT_CHARS,
    }


# --------------------------------------------------------------------------- #
# Routes


@aws_exec_router.get(
    "/aws/exec/probe",
    summary="Is the pinned box reachable by Run Command? (read-only)",
)
async def probe():
    """Deliberately ungated: answering 'can this work' reveals nothing and
    saves a round of guessing when it does not."""
    try:
        instance_id = _target_instance_id()
    except HTTPException as e:
        return {"ok": False, "stage": "instance", "detail": e.detail}
    try:
        registered = _registered(instance_id)
    except HTTPException as e:
        return {"ok": False, "stage": "describe", "instance_id": instance_id,
                "detail": e.detail}
    return {
        "ok": registered,
        "instance_id": instance_id,
        "ssm_registered": registered,
        "exec_enabled": (os.getenv("AWS_EXEC_ENABLED") or "").strip() == "1",
        "raw_enabled": (os.getenv("AWS_EXEC_RAW_ENABLED") or "").strip() == "1",
        "verbs": sorted(VERBS),
        "detail": None if registered else (
            "instance profile needs AmazonSSMManagedInstanceCore before the "
            "agent can register"),
    }


@aws_exec_router.post(
    "/aws/exec/run",
    summary="Run one allowlisted verb on the pinned box",
)
async def run_verb(req: VerbRequest):
    _enabled()
    template = VERBS.get(req.verb)
    if template is None:
        raise HTTPException(
            400,
            f"unknown verb '{req.verb}' — this list is code, not config: "
            f"{sorted(VERBS)}",
        )
    command = (template.format(n=req.n)
               if req.verb in VERBS_TAKING_N else template)
    result = _run(command, req.timeout_s)
    result["verb"] = req.verb
    return result


@aws_exec_router.post(
    "/aws/exec/reboot",
    summary="Reboot the pinned box (EC2-level, no agent required)",
)
async def reboot():
    """The escape hatch for the bootstrap paradox: if the SSM agent is wedged,
    every verb above is unreachable, and restarting the agent is exactly what
    is needed. This goes through the EC2 control plane instead, so it works
    when the agent does not. Gated by AWS_EXEC_ENABLED like the verbs."""
    _enabled()
    instance_id = _target_instance_id()
    ec2 = _client("ec2")
    try:
        ec2.reboot_instances(InstanceIds=[instance_id])
    except Exception as e:
        raise _fail(e)
    return {"instance_id": instance_id, "rebooting": True,
            "note": "give the agent 2-4 minutes, then re-check /aws/exec/probe"}


@aws_exec_router.post(
    "/aws/exec/raw",
    summary="Run an arbitrary shell command (second flag, operator only)",
)
async def run_raw(req: RawRequest):
    _enabled()
    _raw_enabled()
    result = _run(req.command, req.timeout_s)
    result["raw"] = True
    return result
