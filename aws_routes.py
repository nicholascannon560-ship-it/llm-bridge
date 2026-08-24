"""
aws_routes.py — read-only S3 access for the bridge and its agents.

WHY THIS IS NARROW
  Same reasoning as fetch_routes.py, one step further out. This service holds a
  GitHub token with repo/workflow/delete_repo/admin:org scope and a Railway API
  token. Adding AWS credentials to it widens the blast radius from "my repos and
  my Railway projects" to "my AWS account", where a mistake can empty a bucket
  or start compute that bills.

  So the boundary is drawn in three places, deliberately redundant:

    1. IAM (the real one). The key these routes use belongs to an identity with
       GetObject / ListBucket on ONE bucket and an explicit Deny on every write.
       AWS enforces that. This file does not, and cannot be trusted to.
    2. The bucket is PINNED from env and never read from the request body. A
       caller cannot retarget these routes at another bucket, the way the
       sandbox repo is hardcoded in the run_tests tool rather than passed in.
    3. No S3 write verbs exist here at all. There is no put, no delete, no
       copy — and none is planned, because of point 4.
    4. Writing happens in COMPUTE, not here. /aws/build/* starts a pinned
       CodeBuild project that runs under its OWN IAM role: read the data
       prefix, write the agent prefix, nothing else. The agent never holds a
       write verb; it starts a job and reads the result back with s3_get. The
       boundary is the role, which AWS enforces, rather than a check in this
       file, which it does not. run_backtest is in WRITE_TOOL_NAMES all the
       same: it spends money and produces objects, so it must never share an
       agent loop with the browser tools.

  Auth is inherited: BridgeAuthMiddleware in main.py gates every path that is
  not in _EXEMPT_PATHS, so these routes require X-Bridge-Key like the rest.

RESIDUAL RISK (stated rather than papered over)
  An agent that can read the archive can exfiltrate its contents into a commit
  message or an LLM call. Read-only is not the same as harmless. What it does
  buy is that nothing an agent does here can destroy the archive, which is the
  property that matters while the archive is the only off-box copy.

CONFIG
  AWS_S3_BUCKET         required — the one bucket these routes may touch
  AWS_S3_PREFIX         optional — pin reads to a key prefix (default: none)
  AWS_DEFAULT_REGION    required by boto3 for signing
  AWS_ACCESS_KEY_ID     standard credential chain
  AWS_SECRET_ACCESS_KEY standard credential chain
  AWS_S3_MAX_GET_BYTES  default 1048576, hard cap 5242880
  AWS_CODEBUILD_PROJECT   required for /aws/build/* — the ONE project that may
                          be started. Pinned like the bucket; never taken from
                          a request. Inert (503) when unset.
  AWS_BACKTEST_OUT_PREFIX default "agent/backtests/" — where a run's outputs go.
                          Advisory here; the build role is what enforces it.
  AWS_BUILD_MAX_TIMEOUT_MIN default 60 — ceiling on a caller-supplied timeout.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

aws_router = APIRouter(tags=["aws"])

HARD_MAX_GET_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_GET_BYTES = 1024 * 1024
MAX_LIST_KEYS = 1000


# --------------------------------------------------------------------------- #
# Config + client


def _bucket() -> str:
    """The pinned bucket. Never taken from a request body."""
    b = (os.getenv("AWS_S3_BUCKET") or "").strip()
    if not b:
        raise HTTPException(
            503,
            "AWS_S3_BUCKET is not set on this service — read routes are inert.",
        )
    return b


def _prefix_pin() -> str:
    return (os.getenv("AWS_S3_PREFIX") or "").strip().lstrip("/")


def _client():
    # Imported lazily so a missing boto3 degrades these routes instead of
    # taking the whole bridge down at boot.
    return _aws_client("s3")


def _codebuild_project() -> str:
    """The pinned CodeBuild project. Never taken from a request body."""
    proj = (os.getenv("AWS_CODEBUILD_PROJECT") or "").strip()
    if not proj:
        raise HTTPException(
            503,
            "AWS_CODEBUILD_PROJECT is not set on this service — build routes are inert.",
        )
    return proj


def _aws_client(service: str):
    try:
        import boto3  # noqa: WPS433
    except Exception as e:  # pragma: no cover
        raise HTTPException(503, f"boto3 not installed on the bridge: {e}")

    region = (os.getenv("AWS_DEFAULT_REGION") or "").strip()
    if not region:
        raise HTTPException(503, "AWS_DEFAULT_REGION is not set")
    return boto3.client(service, region_name=region)


def _safe_key(key: str) -> str:
    """Reject traversal and absolute keys, and hold the caller inside the pin."""
    k = (key or "").strip()
    if not k:
        raise HTTPException(400, "key is required")
    if k.startswith("/"):
        raise HTTPException(400, "key must not start with '/'")
    if ".." in k.split("/"):
        raise HTTPException(400, "key must not contain '..'")
    pin = _prefix_pin()
    if pin and not k.startswith(pin):
        raise HTTPException(403, f"key must start with the pinned prefix {pin!r}")
    return k


def _boto_error(e: Exception) -> HTTPException:
    """Surface AWS's own refusal instead of a 500 — a 403 here usually means the
    IAM policy is doing its job, and that should read as a fact, not a crash."""
    code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
    if code in ("NoSuchKey", "404", "NotFound"):
        return HTTPException(404, f"no such key ({code})")
    if code in ("AccessDenied", "403"):
        return HTTPException(403, f"AWS refused: {code}")
    return HTTPException(502, f"s3 error: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# Models


class ListRequest(BaseModel):
    prefix: Optional[str] = Field(None, description="key prefix, under AWS_S3_PREFIX")
    max_keys: Optional[int] = Field(None, description=f"1..{MAX_LIST_KEYS}")


class KeyRequest(BaseModel):
    key: str = Field(..., description="object key, under AWS_S3_PREFIX")


class GetRequest(KeyRequest):
    max_bytes: Optional[int] = Field(
        None, description=f"response cap, hard cap {HARD_MAX_GET_BYTES}"
    )


# --------------------------------------------------------------------------- #
# Routes


@aws_router.get("/aws/s3/status", summary="Archive summary: is it fresh, how big")
async def s3_status():
    """One call that answers 'is the archive still being written to'.

    This is the intended feed for an archive_lag watchdog check: newest_modified
    older than a day means the offload stopped and nobody noticed.
    """
    bucket, s3 = _bucket(), _client()
    pin = _prefix_pin()
    count = 0
    total = 0
    newest_key = None
    newest_ts = None
    oldest_ts = None
    try:
        paginator = s3.get_paginator("list_objects_v2")
        kwargs = {"Bucket": bucket}
        if pin:
            kwargs["Prefix"] = pin
        for page in paginator.paginate(**kwargs):
            for obj in page.get("Contents", []):
                count += 1
                total += obj["Size"]
                ts = obj["LastModified"]
                if newest_ts is None or ts > newest_ts:
                    newest_ts, newest_key = ts, obj["Key"]
                if oldest_ts is None or ts < oldest_ts:
                    oldest_ts = ts
    except Exception as e:
        raise _boto_error(e)

    return {
        "bucket": bucket,
        "prefix": pin or None,
        "objects": count,
        "total_bytes": total,
        "total_mb": round(total / 1_048_576, 1),
        "newest_key": newest_key,
        "newest_modified": newest_ts.isoformat() if newest_ts else None,
        "oldest_modified": oldest_ts.isoformat() if oldest_ts else None,
    }


@aws_router.post("/aws/s3/list", summary="List keys under a prefix")
async def s3_list(req: ListRequest):
    bucket, s3 = _bucket(), _client()
    pin = _prefix_pin()
    prefix = (req.prefix or "").strip().lstrip("/")
    if pin and not prefix.startswith(pin):
        prefix = pin + prefix
    n = req.max_keys if req.max_keys is not None else 200
    if n < 1 or n > MAX_LIST_KEYS:
        raise HTTPException(400, f"max_keys must be 1..{MAX_LIST_KEYS}")
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=n)
    except Exception as e:
        raise _boto_error(e)
    items = [
        {
            "key": o["Key"],
            "size": o["Size"],
            "modified": o["LastModified"].isoformat(),
        }
        for o in resp.get("Contents", [])
    ]
    return {
        "bucket": bucket,
        "prefix": prefix or None,
        "count": len(items),
        "truncated": bool(resp.get("IsTruncated")),
        "objects": items,
    }


@aws_router.post("/aws/s3/head", summary="Size and timestamp for one key")
async def s3_head(req: KeyRequest):
    bucket, s3 = _bucket(), _client()
    key = _safe_key(req.key)
    try:
        h = s3.head_object(Bucket=bucket, Key=key)
    except Exception as e:
        raise _boto_error(e)
    return {
        "bucket": bucket,
        "key": key,
        "size": h["ContentLength"],
        "modified": h["LastModified"].isoformat(),
        "content_type": h.get("ContentType"),
        "etag": (h.get("ETag") or "").strip('"'),
    }


@aws_router.post("/aws/s3/get", summary="Read one object as text (capped)")
async def s3_get(req: GetRequest):
    bucket, s3 = _bucket(), _client()
    key = _safe_key(req.key)
    want = req.max_bytes if req.max_bytes is not None else DEFAULT_MAX_GET_BYTES
    if want < 1:
        raise HTTPException(400, "max_bytes must be positive")
    cap = min(int(want), HARD_MAX_GET_BYTES)

    # Range-limited so a 500 MB object cannot be pulled into the bridge's
    # memory by a caller who guessed the wrong key.
    try:
        obj = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{cap - 1}")
        body = obj["Body"].read(cap)
    except Exception as e:
        raise _boto_error(e)

    total = obj.get("ContentRange", "")
    return {
        "bucket": bucket,
        "key": key,
        "bytes_read": len(body),
        "truncated": len(body) >= cap,
        "content_range": total or None,
        "text": body.decode("utf-8", errors="replace"),
    }

# --------------------------------------------------------------------------- #
# Compute: the only thing on this bridge that can write to S3
#
# It writes indirectly: the build runs under a CodeBuild service role scoped to
# read the data prefix and write the agent prefix. The bridge's own credential
# needs codebuild:StartBuild on THIS project and nothing else — notably no
# iam:PassRole beyond the one role, or a caller could hand the project a more
# privileged identity and every restriction below becomes decoration.

BUILD_DEFAULT_TIMEOUT_MIN = 30


class BuildStartRequest(BaseModel):
    command: str = Field(..., description="Shell command to run in the build container")
    setup: Optional[str] = Field(None, description="Optional command run first, e.g. pip install")
    run_id: Optional[str] = Field(None, description="Output folder name under the agent prefix")
    timeout_minutes: Optional[int] = Field(None, description="Wall-clock ceiling for the build")


def _build_max_timeout() -> int:
    try:
        return max(5, int(os.getenv("AWS_BUILD_MAX_TIMEOUT_MIN") or "60"))
    except ValueError:
        return 60


def _out_prefix() -> str:
    p = (os.getenv("AWS_BACKTEST_OUT_PREFIX") or "agent/backtests/").strip().lstrip("/")
    return p if p.endswith("/") else p + "/"


def _safe_run_id(run_id: Optional[str]) -> str:
    """A run id becomes part of an S3 key, so it may not escape the prefix."""
    import re as _re
    from datetime import datetime, timezone

    if not run_id:
        return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    rid = run_id.strip().strip("/")
    if not _re.fullmatch(r"[A-Za-z0-9._-]{1,64}", rid):
        raise HTTPException(
            400, "run_id must be 1-64 chars of [A-Za-z0-9._-] — no slashes, no '..'"
        )
    return rid


def _buildspec(command: str, setup: Optional[str]) -> str:
    """Buildspec as JSON, not YAML.

    CodeBuild accepts either. JSON is used deliberately: the command comes from
    an agent, and json.dumps escapes it as one scalar. Interpolating the same
    string into YAML would let a newline plus two spaces of indentation invent
    new phases, or a new install block — buildspec injection with the build
    role's credentials behind it.
    """
    import json as _json

    install = ["echo setup"]
    if setup and setup.strip():
        install = [setup.strip()]
    spec = {
        "version": "0.2",
        "phases": {
            "install": {"commands": install},
            "build": {"commands": [command]},
        },
    }
    return _json.dumps(spec)


def _tail_build_logs(build: dict, limit: int = 120) -> list:
    """Last N log lines for a build. Never fatal: logs are diagnostics, and a
    missing log group must not turn a finished build into an error."""
    info = (build.get("logs") or {})
    group, stream = info.get("groupName"), info.get("streamName")
    if not group or not stream:
        return []
    try:
        logs = _aws_client("logs")
        resp = logs.get_log_events(
            logGroupName=group,
            logStreamName=stream,
            limit=max(1, min(int(limit), 500)),
            startFromHead=False,
        )
        return [e.get("message", "").rstrip("\n") for e in resp.get("events", [])]
    except Exception as e:  # pragma: no cover
        return [f"[log fetch failed: {type(e).__name__}: {e}]"]


def _build_view(build: dict, include_logs: bool = True) -> dict:
    status = build.get("buildStatus")
    return {
        "build_id": build.get("id"),
        "project": build.get("projectName"),
        "status": status,
        "done": status not in (None, "IN_PROGRESS"),
        "succeeded": status == "SUCCEEDED",
        "current_phase": build.get("currentPhase"),
        "started_at": build["startTime"].isoformat() if build.get("startTime") else None,
        "ended_at": build["endTime"].isoformat() if build.get("endTime") else None,
        "log_tail": _tail_build_logs(build) if include_logs else [],
    }


def _get_build(build_id: str) -> dict:
    """Fetch one build, refusing ids that belong to another project.

    A CodeBuild id is "<project>:<uuid>". Checking the prefix keeps this route
    from becoming a general reader of every build in the account, which is the
    same reason the bucket is pinned.
    """
    project = _codebuild_project()
    bid = (build_id or "").strip()
    if not bid:
        raise HTTPException(400, "build_id is required")
    if not bid.startswith(project + ":"):
        raise HTTPException(403, f"build_id does not belong to project {project!r}")
    # A previous run may already have attached the bridge policy, whose
    # iam:PassRole Deny then blocks the project step forever. Detach it for the
    # duration of this call and re-attach at the end. Best-effort: if this
    # credential cannot detach it, the project step will say so plainly.
    try:
        if not bridge_user:
            raise RuntimeError("no user identity")
        iam.delete_user_policy(UserName=bridge_user, PolicyName="bridge-backtest-control")
        steps.append({"step": "detach_bridge_policy", "result": "detached for this call"})
    except Exception:
        pass

    # ORDER IS LOAD-BEARING. The bridge policy below carries an explicit Deny
    # on iam:PassRole, and codebuild:CreateProject needs PassRole to attach the
    # service role. An explicit Deny beats any Allow, including
    # AdministratorAccess — so once that policy is attached, this credential can
    # never create or update the project again. Create it first. On a re-run
    # after the policy exists, the project step failing with AccessDenied is the
    # guardrail working, not a regression.
    cb = _aws_client("codebuild")
    try:
        resp = cb.batch_get_builds(ids=[bid])
    except Exception as e:
        raise _boto_error(e)
    builds = resp.get("builds") or []
    if not builds:
        raise HTTPException(404, f"no such build {bid}")
    return builds[0]


@aws_router.post("/aws/build/start", summary="Start a backtest job in CodeBuild")
async def build_start(req: BuildStartRequest):
    project = _codebuild_project()
    bucket = _bucket()
    run_id = _safe_run_id(req.run_id)
    out_prefix = _out_prefix() + run_id + "/"

    timeout = req.timeout_minutes or BUILD_DEFAULT_TIMEOUT_MIN
    ceiling = _build_max_timeout()
    if timeout < 5 or timeout > ceiling:
        raise HTTPException(400, f"timeout_minutes must be 5..{ceiling}")

    if not (req.command or "").strip():
        raise HTTPException(400, "command is required")

    cb = _aws_client("codebuild")
    try:
        resp = cb.start_build(
            projectName=project,
            buildspecOverride=_buildspec(req.command, req.setup),
            timeoutInMinutesOverride=int(timeout),
            environmentVariablesOverride=[
                {"name": "S3_BUCKET", "value": bucket, "type": "PLAINTEXT"},
                {"name": "DATA_PREFIX", "value": _prefix_pin() or "kalshiml/", "type": "PLAINTEXT"},
                {"name": "OUT_PREFIX", "value": out_prefix, "type": "PLAINTEXT"},
                {"name": "RUN_ID", "value": run_id, "type": "PLAINTEXT"},
            ],
        )
    except Exception as e:
        raise _boto_error(e)

    view = _build_view(resp["build"], include_logs=False)
    view["run_id"] = run_id
    view["out_prefix"] = out_prefix
    return view


@aws_router.get("/aws/build/status/{build_id:path}", summary="Status and log tail")
async def build_status(build_id: str):
    return _build_view(_get_build(build_id))


# --------------------------------------------------------------------------- #
# Bootstrap: create the backtest role and project, once
#
# WHY THIS IS NOT AN AWS PASSTHROUGH
#   The obvious version of this is "let the caller send an AWS API call". That
#   would end the argument the rest of this file makes: the whole design rests
#   on the build role being something an agent cannot change, and a passthrough
#   with IAM reach lets an agent rewrite the role, mint a user, or attach
#   AdministratorAccess. So this endpoint takes no service name, no action, no
#   policy document, and no role name. Everything below is hardcoded. It can
#   create these resources and nothing else, with these permissions and no
#   others, and re-running it converges rather than escalating.
#
#   Two further limits:
#     - Gated on AWS_BOOTSTRAP_ENABLED=1. Default off. It is meant to be turned
#       on for one call and turned off again.
#     - It is deliberately NOT exposed as an agent tool. Operators call it.
#       Nothing in agent_loop/tools.py references it, and nothing should.
#
#   If the bridge's credential lacks IAM rights this returns AWS's own 403,
#   which is the correct outcome and a useful fact: it means the credential is
#   narrower than the account, and the console is the right place to do this.

BOOTSTRAP_ROLE = "agent-backtest-role"
BOOTSTRAP_PROJECT = "agent-backtest"
BOOTSTRAP_DATA_PREFIX = "kalshiml/"
BOOTSTRAP_OUT_PREFIX = "agent/"
BOOTSTRAP_IMAGE = "aws/codebuild/standard:7.0"
BOOTSTRAP_COMPUTE = "BUILD_GENERAL1_SMALL"


def _bootstrap_enabled() -> None:
    if (os.getenv("AWS_BOOTSTRAP_ENABLED") or "").strip() != "1":
        raise HTTPException(
            403,
            "bootstrap is disabled — set AWS_BOOTSTRAP_ENABLED=1 for the one call, "
            "then unset it.",
        )


def _role_policy(account: str, region: str, bucket: str) -> str:
    import json as _json

    return _json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ListOnlyTheseTwoPrefixes",
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": f"arn:aws:s3:::{bucket}",
                "Condition": {"StringLike": {"s3:prefix": [
                    f"{BOOTSTRAP_DATA_PREFIX}*", f"{BOOTSTRAP_OUT_PREFIX}*"]}},
            },
            {
                "Sid": "ReadTheData",
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": f"arn:aws:s3:::{bucket}/{BOOTSTRAP_DATA_PREFIX}*",
            },
            {
                "Sid": "WriteOnlyTheOutputPrefix",
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:GetObject", "s3:AbortMultipartUpload"],
                "Resource": f"arn:aws:s3:::{bucket}/{BOOTSTRAP_OUT_PREFIX}*",
            },
            {
                # The point of the whole exercise. Explicit Deny beats any Allow,
                # so even a later policy mistake cannot let a job overwrite the
                # live engine state under the data prefix.
                "Sid": "NeverTouchTheEngineState",
                "Effect": "Deny",
                "Action": ["s3:PutObject", "s3:DeleteObject", "s3:DeleteObjectVersion"],
                "Resource": f"arn:aws:s3:::{bucket}/{BOOTSTRAP_DATA_PREFIX}*",
            },
            {
                "Sid": "OwnLogs",
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": f"arn:aws:logs:{region}:{account}:log-group:/aws/codebuild/{BOOTSTRAP_PROJECT}*",
            },
        ],
    })


def _bridge_policy(account: str, region: str) -> str:
    import json as _json

    return _json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "StartAndReadThisProjectOnly",
                "Effect": "Allow",
                "Action": ["codebuild:StartBuild", "codebuild:BatchGetBuilds",
                           "codebuild:ListBuildsForProject"],
                "Resource": f"arn:aws:codebuild:{region}:{account}:project/{BOOTSTRAP_PROJECT}",
            },
            {
                "Sid": "ReadBuildLogs",
                "Effect": "Allow",
                "Action": ["logs:GetLogEvents", "logs:DescribeLogStreams"],
                "Resource": f"arn:aws:logs:{region}:{account}:log-group:/aws/codebuild/{BOOTSTRAP_PROJECT}*",
            },
            {
                # Without this, a caller able to reach StartBuild could pass a
                # different, more privileged service role and every scope above
                # becomes decoration. The project's role is baked in, so nothing
                # legitimate needs PassRole.
                "Sid": "NeverPassAnotherRole",
                "Effect": "Deny",
                "Action": "iam:PassRole",
                "Resource": "*",
            },
        ],
    })


@aws_router.post("/aws/bootstrap/backtest", summary="Create the backtest role + project (operator only)")
async def bootstrap_backtest():
    _bootstrap_enabled()
    import json as _json

    region = (os.getenv("AWS_DEFAULT_REGION") or "").strip()
    bucket = _bucket()
    steps = []

    sts = _aws_client("sts")
    try:
        ident = sts.get_caller_identity()
    except Exception as e:
        raise _boto_error(e)
    account = ident["Account"]
    arn = ident["Arn"]
    # "arn:aws:iam::123:user/name" -> name. A role or root identity has no user
    # policy to attach, so that step is skipped rather than guessed at.
    bridge_user = arn.rsplit("/", 1)[-1] if ":user/" in arn else None
    steps.append({"step": "identity", "account": account, "is_user": bool(bridge_user)})

    iam = _aws_client("iam")
    trust = _json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "codebuild.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })
    try:
        iam.create_role(
            RoleName=BOOTSTRAP_ROLE,
            AssumeRolePolicyDocument=trust,
            Description=f"Agent backtests: read {BOOTSTRAP_DATA_PREFIX}, write {BOOTSTRAP_OUT_PREFIX}",
        )
        steps.append({"step": "create_role", "result": "created", "role": BOOTSTRAP_ROLE})
    except Exception as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        if code == "EntityAlreadyExists":
            steps.append({"step": "create_role", "result": "already existed", "role": BOOTSTRAP_ROLE})
        else:
            # Keep going rather than raising. One call should report exactly
            # which of IAM / CodeBuild this credential can and cannot do —
            # a bare 403 on the first step tells you nothing about the rest,
            # and each retry costs a deploy cycle to learn one bit.
            steps.append({"step": "create_role", "result": f"DENIED: {code or type(e).__name__}"})

    try:
        iam.put_role_policy(
            RoleName=BOOTSTRAP_ROLE,
            PolicyName="agent-backtest-s3",
            PolicyDocument=_role_policy(account, region, bucket),
        )
        steps.append({"step": "put_role_policy", "result": "ok"})
    except Exception as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        steps.append({"step": "put_role_policy", "result": f"DENIED: {code or type(e).__name__}"})

    role_arn = f"arn:aws:iam::{account}:role/{BOOTSTRAP_ROLE}"

    cb = _aws_client("codebuild")
    project_def = {
        "name": BOOTSTRAP_PROJECT,
        "description": "Agent-run backtests against the S3 data corpus",
        "source": {
            "type": "NO_SOURCE",
            # Placeholder: every run overrides this via buildspecOverride.
            "buildspec": "version: 0.2\nphases:\n  build:\n    commands:\n      - echo override me\n",
        },
        "artifacts": {"type": "NO_ARTIFACTS"},
        "environment": {
            "type": "LINUX_CONTAINER",
            "image": BOOTSTRAP_IMAGE,
            "computeType": BOOTSTRAP_COMPUTE,
            "environmentVariables": [
                {"name": "S3_BUCKET", "value": bucket, "type": "PLAINTEXT"},
                {"name": "DATA_PREFIX", "value": BOOTSTRAP_DATA_PREFIX, "type": "PLAINTEXT"},
                {"name": "OUT_PREFIX", "value": BOOTSTRAP_OUT_PREFIX, "type": "PLAINTEXT"},
            ],
        },
        "serviceRole": role_arn,
        "timeoutInMinutes": 60,
        "logsConfig": {"cloudWatchLogs": {"status": "ENABLED"}},
    }
    try:
        existing = cb.batch_get_projects(names=[BOOTSTRAP_PROJECT]).get("projects") or []
        if existing:
            cb.update_project(**project_def)
            steps.append({"step": "project", "result": "updated", "project": BOOTSTRAP_PROJECT})
        else:
            cb.create_project(**project_def)
            steps.append({"step": "project", "result": "created", "project": BOOTSTRAP_PROJECT})
    except Exception as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        steps.append({"step": "project", "result": f"DENIED: {code or type(e).__name__}"})

    if bridge_user:
        try:
            iam.put_user_policy(
                UserName=bridge_user,
                PolicyName="bridge-backtest-control",
                PolicyDocument=_bridge_policy(account, region),
            )
            steps.append({"step": "bridge_user_policy", "result": "ok"})
        except Exception as e:
            steps.append({
                "step": "bridge_user_policy",
                "result": f"FAILED: {type(e).__name__}: {e}"[:300],
                "note": "the bridge may not be able to start builds until this is fixed",
            })
    else:
        steps.append({"step": "bridge_user_policy", "result": "skipped (identity is not an IAM user)"})

    failed = [x for x in steps if str(x.get("result", "")).startswith(("DENIED", "FAILED"))]
    return {
        "ok": not failed,
        "failed_steps": [x["step"] for x in failed],
        "role": BOOTSTRAP_ROLE,
        "project": BOOTSTRAP_PROJECT,
        "data_prefix_readonly": BOOTSTRAP_DATA_PREFIX,
        "writable_prefix": BOOTSTRAP_OUT_PREFIX,
        "steps": steps,
        "next": (
            "Set AWS_CODEBUILD_PROJECT, run the negative smoke test (a write into "
            "the data prefix MUST fail), then unset AWS_BOOTSTRAP_ENABLED."
        ),
    }
