"""
aws_compute_routes.py — provision ONE pinned EC2 instance, idempotently.

WHY THIS EXISTS
  AWS_DEPLOY.md's step 1 ("create the instance") was the last thing that
  required a human at a console. This makes it an API call.

WHY IT IS SHAPED LIKE THIS
  aws_routes.py already argues the case at length and this file inherits it.
  The short version: the caller supplies NOTHING that widens the blast radius.
  Not the instance type, not the image, not the region, not an IAM role, not a
  count. Those are pinned from env and validated against a hardcoded allowlist,
  so a compromised or looping agent can spend a bounded amount of money on one
  known-shaped box and nothing else.

  Specifically:
    1. MaxCount is the literal 1. There is no batch verb.
    2. Instance type must be in ALLOWED_INSTANCE_TYPES below — an env var alone
       cannot select an expensive box, so editing Railway config is not enough
       to turn this into a GPU fleet.
    3. Idempotent on a pinned Name tag. A retry, a duplicated command-channel
       job, or an agent in a loop returns the EXISTING instance rather than
       minting another. This is the single most important property here: the
       failure mode of a provisioning endpoint is not one wrong box, it is
       fifty right ones.
    4. No terminate, stop, or modify verb exists. Tearing down is a console or
       CLI action on purpose — an agent that cannot delete cannot lose data.
    5. Gated on AWS_COMPUTE_ENABLED=1, default off, same as
       /aws/bootstrap/backtest.
    6. IMDSv2 is required on the instance, so a stray SSRF on the box cannot
       read its instance-profile credentials.

  THE PINS HERE ARE THE WEAKER HALF. As aws_routes.py says of its own bucket
  pin: IAM is what AWS enforces and this file is not. With an admin-scoped
  credential behind the bridge, everything above is advisory. The matching
  policy should allow ec2:RunInstances only with a
  Condition on ec2:InstanceType, and deny iam:PassRole except for the one
  instance profile named below.

CONFIG
  AWS_COMPUTE_ENABLED        "1" to enable. Default off (403).
  AWS_DEFAULT_REGION         required (shared with aws_routes).
  AWS_COMPUTE_INSTANCE_TYPE  default "t3.small". Must be in ALLOWED_INSTANCE_TYPES.
  AWS_COMPUTE_NAME_TAG       default "kalshiml-engine". The idempotency key.
  AWS_COMPUTE_AMI_SSM_PARAM  default: Canonical's Ubuntu 24.04 amd64 pointer.
  AWS_COMPUTE_AMI_ID         optional explicit AMI; overrides the SSM lookup.
  AWS_COMPUTE_VOLUME_GB      default 20, hard ceiling MAX_VOLUME_GB.
  AWS_COMPUTE_SUBNET_ID      optional; default-VPC subnet if unset.
  AWS_COMPUTE_SECURITY_GROUP optional security group id.
  AWS_COMPUTE_IAM_PROFILE    optional instance profile NAME (how the box reads
                             its secrets from SSM without anyone pasting them).
  AWS_COMPUTE_KEY_NAME       optional EC2 keypair name for SSH.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:  # reuse the client factory + error mapping rather than forking them
    from aws_routes import _aws_client, _boto_error
except Exception:  # pragma: no cover - only if aws_routes fails to import
    _aws_client = None
    _boto_error = None

aws_compute_router = APIRouter(tags=["aws"])

# Hardcoded, not env. An operator who can set env can still not pick m5.24xlarge.
ALLOWED_INSTANCE_TYPES = frozenset({
    "t3.micro", "t3.small", "t3.medium",
    "t4g.micro", "t4g.small", "t4g.medium",
})
DEFAULT_INSTANCE_TYPE = "t3.small"
DEFAULT_NAME_TAG = "kalshiml-engine"
DEFAULT_AMI_SSM_PARAM = (
    "/aws/service/canonical/ubuntu/server/24.04/stable/current/"
    "amd64/hvm/ebs-gp3/ami-id"
)
DEFAULT_VOLUME_GB = 20
MAX_VOLUME_GB = 100
MAX_USER_DATA_BYTES = 16 * 1024  # EC2's own limit

# States that mean "a box already exists" for idempotency purposes. A
# terminated instance keeps its tag for a while, so it must NOT count.
LIVE_STATES = ("pending", "running", "stopping", "stopped")


# --------------------------------------------------------------------------- #
# Config


def _enabled() -> None:
    if (os.getenv("AWS_COMPUTE_ENABLED") or "").strip() != "1":
        raise HTTPException(
            403,
            "compute provisioning is disabled — set AWS_COMPUTE_ENABLED=1 for "
            "the one call, then unset it.",
        )


def _client(service: str):
    if _aws_client is None:  # pragma: no cover
        raise HTTPException(503, "aws_routes failed to import; no AWS client")
    return _aws_client(service)


def _fail(e: Exception) -> HTTPException:
    if _boto_error is None:  # pragma: no cover
        return HTTPException(502, f"aws error: {type(e).__name__}: {e}")
    return _boto_error(e)


def _instance_type() -> str:
    t = (os.getenv("AWS_COMPUTE_INSTANCE_TYPE") or DEFAULT_INSTANCE_TYPE).strip()
    if t not in ALLOWED_INSTANCE_TYPES:
        raise HTTPException(
            400,
            f"instance type {t!r} is not in the allowlist "
            f"{sorted(ALLOWED_INSTANCE_TYPES)} — this list is code, not config.",
        )
    return t


def _name_tag() -> str:
    return (os.getenv("AWS_COMPUTE_NAME_TAG") or DEFAULT_NAME_TAG).strip()


def _volume_gb() -> int:
    raw = (os.getenv("AWS_COMPUTE_VOLUME_GB") or "").strip()
    try:
        gb = int(raw) if raw else DEFAULT_VOLUME_GB
    except ValueError:
        raise HTTPException(400, f"AWS_COMPUTE_VOLUME_GB is not an integer: {raw!r}")
    if gb < 8 or gb > MAX_VOLUME_GB:
        raise HTTPException(400, f"volume must be 8..{MAX_VOLUME_GB} GB, got {gb}")
    return gb


def _resolve_ami() -> str:
    """Explicit AMI wins; otherwise read Canonical's public SSM pointer so the
    image is current rather than a hardcoded id that rots."""
    explicit = (os.getenv("AWS_COMPUTE_AMI_ID") or "").strip()
    if explicit:
        return explicit
    param = (os.getenv("AWS_COMPUTE_AMI_SSM_PARAM") or DEFAULT_AMI_SSM_PARAM).strip()
    ssm = _client("ssm")
    try:
        resp = ssm.get_parameter(Name=param)
    except Exception as e:
        raise _fail(e)
    value = (resp.get("Parameter") or {}).get("Value") or ""
    if not value.startswith("ami-"):
        raise HTTPException(502, f"SSM parameter {param} did not return an AMI id")
    return value


# --------------------------------------------------------------------------- #
# Models


class ProvisionRequest(BaseModel):
    user_data: Optional[str] = Field(
        None,
        description="First-boot script (cloud-init). Capped at 16 KB by EC2.",
    )
    dry_run: bool = Field(
        False,
        description="Ask EC2 to validate permissions without launching anything.",
    )


# --------------------------------------------------------------------------- #
# Lookup


def _find_existing(ec2, name: str) -> Optional[dict]:
    try:
        resp = ec2.describe_instances(Filters=[
            {"Name": "tag:Name", "Values": [name]},
            {"Name": "instance-state-name", "Values": list(LIVE_STATES)},
        ])
    except Exception as e:
        raise _fail(e)
    for res in resp.get("Reservations") or []:
        for inst in res.get("Instances") or []:
            return inst
    return None


def _view(inst: dict) -> dict:
    launched = inst.get("LaunchTime")
    return {
        "instance_id": inst.get("InstanceId"),
        "state": (inst.get("State") or {}).get("Name"),
        "instance_type": inst.get("InstanceType"),
        "public_ip": inst.get("PublicIpAddress"),
        "private_ip": inst.get("PrivateIpAddress"),
        "image_id": inst.get("ImageId"),
        "launch_time": launched.isoformat() if hasattr(launched, "isoformat") else launched,
    }


# --------------------------------------------------------------------------- #
# Routes


@aws_compute_router.post(
    "/aws/compute/provision",
    summary="Create the one pinned instance (idempotent, operator only)",
)
async def provision(req: ProvisionRequest):
    _enabled()
    name = _name_tag()
    itype = _instance_type()
    ec2 = _client("ec2")

    existing = _find_existing(ec2, name)
    if existing is not None:
        # The important branch. Never launch a second box behind the same tag.
        return {
            "created": False,
            "reason": f"an instance tagged {name!r} already exists",
            "instance": _view(existing),
        }

    user_data = req.user_data or ""
    if len(user_data.encode("utf-8")) > MAX_USER_DATA_BYTES:
        raise HTTPException(
            400, f"user_data exceeds {MAX_USER_DATA_BYTES} bytes (EC2's limit)"
        )

    kwargs = {
        "ImageId": _resolve_ami(),
        "InstanceType": itype,
        "MinCount": 1,
        "MaxCount": 1,  # literal, never derived from a request
        "BlockDeviceMappings": [{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": _volume_gb(),
                "VolumeType": "gp3",
                "DeleteOnTermination": True,
                "Encrypted": True,
            },
        }],
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": name},
                {"Key": "ManagedBy", "Value": "llm-bridge"},
            ],
        }],
    }
    if user_data:
        kwargs["UserData"] = user_data
    subnet = (os.getenv("AWS_COMPUTE_SUBNET_ID") or "").strip()
    if subnet:
        kwargs["SubnetId"] = subnet
    sg = (os.getenv("AWS_COMPUTE_SECURITY_GROUP") or "").strip()
    if sg:
        kwargs["SecurityGroupIds"] = [sg]
    profile = (os.getenv("AWS_COMPUTE_IAM_PROFILE") or "").strip()
    if profile:
        kwargs["IamInstanceProfile"] = {"Name": profile}
    key_name = (os.getenv("AWS_COMPUTE_KEY_NAME") or "").strip()
    if key_name:
        kwargs["KeyName"] = key_name

    if req.dry_run:
        kwargs["DryRun"] = True
        try:
            ec2.run_instances(**kwargs)
        except Exception as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if code == "DryRunOperation":
                # AWS's way of saying "you are allowed to do this".
                return {"created": False, "dry_run": True, "would_launch": {
                    "image_id": kwargs["ImageId"],
                    "instance_type": itype,
                    "name_tag": name,
                    "volume_gb": kwargs["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"],
                }}
            raise _fail(e)
        return {"created": False, "dry_run": True, "note": "no DryRunOperation raised"}

    try:
        resp = ec2.run_instances(**kwargs)
    except Exception as e:
        raise _fail(e)
    instances = resp.get("Instances") or []
    if not instances:
        raise HTTPException(502, "run_instances returned no instance")
    return {"created": True, "instance": _view(instances[0])}


@aws_compute_router.get(
    "/aws/compute/status",
    summary="State and address of the pinned instance",
)
async def status():
    name = _name_tag()
    ec2 = _client("ec2")
    inst = _find_existing(ec2, name)
    if inst is None:
        return {"exists": False, "name_tag": name}
    return {"exists": True, "name_tag": name, "instance": _view(inst)}
