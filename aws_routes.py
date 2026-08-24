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
    3. No write verbs exist here at all. There is no put, no delete, no copy.
       Adding one is a separate, deliberate decision — and if it happens, the
       tool that calls it belongs in WRITE_TOOL_NAMES so it can never share an
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
    try:
        import boto3  # noqa: WPS433
    except Exception as e:  # pragma: no cover
        raise HTTPException(503, f"boto3 not installed on the bridge: {e}")

    region = (os.getenv("AWS_DEFAULT_REGION") or "").strip()
    if not region:
        raise HTTPException(503, "AWS_DEFAULT_REGION is not set")
    return boto3.client("s3", region_name=region)


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
