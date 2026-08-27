"""
aws_ssm_routes.py — write secrets into ONE SSM prefix. Write-only, on purpose.

WHY THIS EXISTS
  The instance reads its credentials from /kalshiml/prod/* at boot, so those
  parameters have to be created somehow. The console is the right place to do
  it, but it is not always reachable, so this makes it an API call.

WHY THERE IS NO GET
  This is the load-bearing decision in the file. A write route puts a secret
  into AWS. A read route takes it back out — into an HTTP response, a log, a
  transcript, an LLM call. Those are not symmetric risks. The instance role
  reads these values; the bridge writes them and can never see them again.
  /aws/ssm/list exists and returns NAMES ONLY, so "did all four land" is
  answerable without any value leaving AWS.

  A caller who can write here can OVERWRITE a credential and break the trader,
  or plant a value the box will trust at next boot. That is real, and it is why
  the route is prefix-pinned, SecureString-only, and gated off by default. It
  is a smaller surface than read-back, which would hand the whole trading
  identity to anything holding a bridge key.

OTHER LIMITS
  - Prefix is pinned from env and validated; a caller cannot write to
    /prod/other-system/ or to another team's path.
  - Type is forced to SecureString. There is no plaintext option, so a
    mis-specified request cannot store a private key in the clear.
  - No delete verb. Removing a parameter is a console action.
  - Values are never echoed. The response carries a name, a version, and a
    SHA-256 prefix so a caller can confirm what landed by comparing digests.

CONFIG
  AWS_SSM_WRITE_ENABLED  "1" for the load, then unset. Default off.
  AWS_SSM_WRITE_PREFIX   default "/kalshiml/prod/". Trailing slash enforced.
  AWS_DEFAULT_REGION     required (shared with aws_routes).
"""
from __future__ import annotations

import hashlib
import os
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:
    from aws_routes import _aws_client, _boto_error
except Exception:  # pragma: no cover
    _aws_client = None
    _boto_error = None

aws_ssm_router = APIRouter(tags=["aws"])

DEFAULT_WRITE_PREFIX = "/kalshiml/prod/"
MAX_VALUE_BYTES = 8 * 1024  # advanced tier ceiling; standard is 4 KB
NAME_RE = re.compile(r"^[A-Za-z0-9_.\-/]+$")


def _enabled() -> None:
    if (os.getenv("AWS_SSM_WRITE_ENABLED") or "").strip() != "1":
        raise HTTPException(
            403,
            "SSM writes are disabled — set AWS_SSM_WRITE_ENABLED=1 for the "
            "load, then unset it.",
        )


def _client():
    if _aws_client is None:  # pragma: no cover
        raise HTTPException(503, "aws_routes failed to import; no AWS client")
    return _aws_client("ssm")


def _fail(e: Exception) -> HTTPException:
    if _boto_error is None:  # pragma: no cover
        return HTTPException(502, f"aws error: {type(e).__name__}: {e}")
    return _boto_error(e)


def _prefix() -> str:
    p = (os.getenv("AWS_SSM_WRITE_PREFIX") or DEFAULT_WRITE_PREFIX).strip()
    if not p.startswith("/"):
        p = "/" + p
    if not p.endswith("/"):
        p += "/"
    return p


def _safe_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        raise HTTPException(400, "name is required")
    if not n.startswith("/"):
        n = "/" + n
    if ".." in n.split("/"):
        raise HTTPException(400, "name must not contain '..'")
    if not NAME_RE.match(n):
        raise HTTPException(400, "name has characters outside [A-Za-z0-9_.-/]")
    pin = _prefix()
    if not n.startswith(pin):
        raise HTTPException(403, f"name must start with the pinned prefix {pin!r}")
    if n.endswith("/"):
        raise HTTPException(400, "name must not end with '/'")
    return n


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


class PutRequest(BaseModel):
    name: str = Field(..., description="Full parameter name, under the pinned prefix")
    value: str = Field(..., description="Secret value. Never echoed back.")
    overwrite: bool = Field(
        True, description="Replace an existing value. False fails if it exists."
    )


@aws_ssm_router.post(
    "/aws/ssm/put",
    summary="Write one SecureString under the pinned prefix (write-only)",
)
async def ssm_put(req: PutRequest):
    _enabled()
    name = _safe_name(req.name)
    value = req.value or ""
    if not value:
        raise HTTPException(400, "value is empty — refusing to store a blank secret")
    size = len(value.encode("utf-8"))
    if size > MAX_VALUE_BYTES:
        raise HTTPException(400, f"value is {size} bytes, over the {MAX_VALUE_BYTES} limit")

    ssm = _client()
    kwargs = {
        "Name": name,
        "Value": value,
        "Type": "SecureString",  # forced; no plaintext path exists
        "Overwrite": bool(req.overwrite),
    }
    if size > 4096:
        kwargs["Tier"] = "Advanced"
    try:
        resp = ssm.put_parameter(**kwargs)
    except Exception as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        if code == "ParameterAlreadyExists":
            raise HTTPException(409, f"{name} exists and overwrite was false")
        raise _fail(e)
    return {
        "name": name,
        "version": resp.get("Version"),
        "type": "SecureString",
        "bytes": size,
        "sha256_12": _digest(value),
        "note": "value stored; it cannot be read back through this bridge",
    }


@aws_ssm_router.get(
    "/aws/ssm/list",
    summary="Names under the pinned prefix — never values",
)
async def ssm_list():
    pin = _prefix()
    ssm = _client()
    try:
        resp = ssm.describe_parameters(ParameterFilters=[
            {"Key": "Name", "Option": "BeginsWith", "Values": [pin]},
        ], MaxResults=50)
    except Exception as e:
        raise _fail(e)
    out = []
    for p in resp.get("Parameters") or []:
        mod = p.get("LastModifiedDate")
        out.append({
            "name": p.get("Name"),
            "type": p.get("Type"),
            "version": p.get("Version"),
            "last_modified": mod.isoformat() if hasattr(mod, "isoformat") else mod,
        })
    return {"prefix": pin, "count": len(out), "parameters": out}
