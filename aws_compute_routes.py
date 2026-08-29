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

# --------------------------------------------------------------------------- #
# Targets
#
# WHY THIS TABLE EXISTS
#   Everything here used to resolve through ONE global, AWS_COMPUTE_NAME_TAG,
#   shared with aws_exec_routes. That was fine with one box and actively
#   dangerous with two: pointing the bridge at a second instance silently
#   retargeted every exec verb -- stop, restart, update -- away from the live
#   trading engine. A name tag is not a safe place to keep "which machine am I
#   operating".
#
#   So a target is now an explicit, named, code-defined profile. It is a
#   table and not config for the same reason ALLOWED_INSTANCE_TYPES is: an
#   operator who can set env still cannot invent a machine for the bridge to
#   drive, and a caller cannot pass an instance id.
#
#   The kalshiml entry keeps the UNSUFFIXED env vars so existing deployment
#   config keeps working untouched; every other target reads a _<TARGET>
#   suffixed variant. Omitting `target` anywhere resolves to kalshiml, so the
#   pre-existing behaviour of every route is byte-for-byte unchanged.
TARGETS: dict[str, dict] = {
    "kalshiml": {
        "name_tag": "kalshiml-engine",
        "service": "kalshiml",
        "repo_dir": "/opt/kalshiml",
        "env_file": "/etc/kalshiml.env",
        "data_dir": "/var/lib/kalshiml",
        "bootstrap_log": "/var/log/kalshiml-bootstrap.log",
    },
    "nowcaster": {
        "name_tag": "nowcaster-engine",
        "service": "nowcaster",
        "repo_dir": "/opt/nowcaster",
        "env_file": "/etc/nowcaster.env",
        "data_dir": "/var/lib/nowcaster",
        "bootstrap_log": "/var/log/nowcaster-bootstrap.log",
    },
}
DEFAULT_TARGET = "kalshiml"
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


def _target(name: Optional[str]) -> str:
    """Validate a target name against the table. None means the default."""
    t = (name or DEFAULT_TARGET).strip().lower()
    if t not in TARGETS:
        raise HTTPException(
            400,
            f"unknown target {t!r} — this list is code, not config: "
            f"{sorted(TARGETS)}",
        )
    return t


def target_profile(name: Optional[str]) -> dict:
    """Public accessor: aws_exec_routes builds its verb table from this."""
    return TARGETS[_target(name)]


def _env_for(target: str, base: str) -> str:
    """Read a per-target env var.

    The default target keeps the UNSUFFIXED name so nothing already deployed
    has to be re-set; anything else reads BASE_<TARGET>. Returns "" when unset
    so callers keep their existing `or DEFAULT` fallbacks.
    """
    key = base if target == DEFAULT_TARGET else f"{base}_{target.upper()}"
    return (os.getenv(key) or "").strip()


def _instance_type(target: Optional[str] = None) -> str:
    tgt = _target(target)
    t = _env_for(tgt, "AWS_COMPUTE_INSTANCE_TYPE") or DEFAULT_INSTANCE_TYPE
    if t not in ALLOWED_INSTANCE_TYPES:
        raise HTTPException(
            400,
            f"instance type {t!r} is not in the allowlist "
            f"{sorted(ALLOWED_INSTANCE_TYPES)} — this list is code, not config.",
        )
    return t


def _name_tag(target: Optional[str] = None) -> str:
    tgt = _target(target)
    return _env_for(tgt, "AWS_COMPUTE_NAME_TAG") or TARGETS[tgt]["name_tag"]


def _volume_gb(target: Optional[str] = None) -> int:
    tgt = _target(target)
    raw = _env_for(tgt, "AWS_COMPUTE_VOLUME_GB")
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
    target: Optional[str] = Field(
        None,
        description="Which machine profile to launch. Omit for the default "
                    "(kalshiml). A name from the TARGETS table, never an id.",
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


# --------------------------------------------------------------------------- #
# Bootstrap: create the instance role + profile, once
#
# WHY THIS IS NOT AN IAM PASSTHROUGH
#   Same argument as /aws/bootstrap/backtest in aws_routes.py, and for the same
#   reason: an endpoint that accepts a policy document, a role name, or an
#   action list is an endpoint that can attach AdministratorAccess to anything.
#   So this takes NO parameters. Role name, profile name, trust policy and
#   permission policy are all below in code. Re-running it converges instead of
#   escalating.
#
#   Deliberately absent: user creation and access-key issuance. Those emit a
#   long-lived secret, and a secret that arrives in an HTTP response body has
#   been written to a log, a transcript, and whatever read it. Narrowing the
#   bridge's own credential stays a console action on purpose.
#
#   The policy mirrors infra/policies/instance_role_permissions.json in the
#   KalshiML repo. Bucket and region come from the bridge's own env rather than
#   being hardcoded, so this cannot be pointed at someone else's bucket.
#
# CONFIG
#   AWS_COMPUTE_BOOTSTRAP_ENABLED  "1" for the one call, then unset.

# Overridable because the console's Create-role wizard is the only path that
# links a role to an instance profile by clicking, and it names them itself.
BOOTSTRAP_ROLE = (os.getenv("AWS_COMPUTE_ROLE_NAME") or "kalshiml-prod").strip()
BOOTSTRAP_PROFILE = (os.getenv("AWS_COMPUTE_IAM_PROFILE") or "kalshiml-prod").strip()
BOOTSTRAP_SSM_PATH = "kalshiml/prod"
BOOTSTRAP_S3_PREFIX = "kalshiml"


def _compute_bootstrap_enabled() -> None:
    if (os.getenv("AWS_COMPUTE_BOOTSTRAP_ENABLED") or "").strip() != "1":
        raise HTTPException(
            403,
            "compute bootstrap is disabled — set AWS_COMPUTE_BOOTSTRAP_ENABLED=1 "
            "for the one call, then unset it.",
        )


def _trust_policy() -> str:
    import json as _json
    return _json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })


def _instance_policy(region: str, bucket: str) -> str:
    import json as _json
    return _json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ReadOwnSecrets",
                "Effect": "Allow",
                "Action": ["ssm:GetParameter", "ssm:GetParameters",
                           "ssm:GetParametersByPath"],
                "Resource": f"arn:aws:ssm:{region}:*:parameter/{BOOTSTRAP_SSM_PATH}/*",
            },
            {
                "Sid": "DecryptOwnSecrets",
                "Effect": "Allow",
                "Action": ["kms:Decrypt"],
                "Resource": "*",
                "Condition": {"StringEquals": {
                    "kms:ViaService": f"ssm.{region}.amazonaws.com"}},
            },
            {
                "Sid": "BackupAndHeartbeat",
                "Effect": "Allow",
                "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": f"arn:aws:s3:::{bucket}",
            },
            {
                "Sid": "BackupReadWrite",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": f"arn:aws:s3:::{bucket}/{BOOTSTRAP_S3_PREFIX}/*",
            },
            {
                # Lets the SSM agent register so Run Command can reach the box.
                # This is what /aws/exec depends on. It grants the agent's own
                # channel actions only — no parameter or document writes, and
                # nothing that lets the box act on other instances.
                "Sid": "SsmAgentRegistration",
                "Effect": "Allow",
                "Action": [
                    "ssm:UpdateInstanceInformation",
                    "ssm:ListAssociations",
                    "ssm:ListInstanceAssociations",
                    "ssm:DescribeAssociation",
                    "ssm:GetDocument",
                    "ssm:DescribeDocument",
                    "ssm:GetManifest",
                    "ssm:PutInventory",
                    "ssm:UpdateAssociationStatus",
                    "ssm:UpdateInstanceAssociationStatus",
                    "ssmmessages:CreateControlChannel",
                    "ssmmessages:CreateDataChannel",
                    "ssmmessages:OpenControlChannel",
                    "ssmmessages:OpenDataChannel",
                    "ec2messages:AcknowledgeMessage",
                    "ec2messages:DeleteMessage",
                    "ec2messages:FailMessage",
                    "ec2messages:GetEndpoint",
                    "ec2messages:GetMessages",
                    "ec2messages:SendReply",
                ],
                "Resource": "*",
            },
            {
                # The box backs up. It never removes a backup — losing the only
                # off-box copy is the failure this exists to prevent.
                "Sid": "NeverDelete",
                "Effect": "Deny",
                "Action": ["s3:DeleteObject", "s3:DeleteObjectVersion",
                           "s3:PutBucketPolicy"],
                "Resource": "*",
            },
        ],
    })


@aws_compute_router.get(
    "/aws/whoami",
    summary="Which AWS principal the bridge is actually using",
)
async def whoami():
    """Read-only. Exists because 'the key has admin' and 'AWS said
    AccessDenied' cannot both be true, and guessing which one is wrong wastes
    more time than one STS call."""
    sts = _client("sts")
    try:
        ident = sts.get_caller_identity()
    except Exception as e:
        raise _fail(e)
    return {
        "account": ident.get("Account"),
        "arn": ident.get("Arn"),
        "user_id": ident.get("UserId"),
        "region": os.getenv("AWS_DEFAULT_REGION"),
    }


@aws_compute_router.get(
    "/aws/compute/iam_probe",
    summary="Report the RAW AWS error for each IAM call bootstrap needs",
)
async def iam_probe():
    """Calls each IAM action in a way that cannot mutate anything, and returns
    AWS's full message rather than a collapsed error code. AWS names the exact
    principal and action in its denial text, which is the fact needed to tell a
    missing permission apart from a different identity than expected."""
    iam = _client("iam")
    out = []

    def probe(label, fn):
        try:
            fn()
            out.append({"action": label, "result": "allowed"})
        except Exception as e:
            err = getattr(e, "response", {}).get("Error", {})
            out.append({
                "action": label,
                "code": err.get("Code"),
                "message": (err.get("Message") or str(e))[:400],
            })

    probe("iam:GetUser", lambda: iam.get_user())
    probe("iam:GetRole/kalshiml-prod", lambda: iam.get_role(RoleName=BOOTSTRAP_ROLE))
    probe("iam:GetInstanceProfile/kalshiml-prod",
          lambda: iam.get_instance_profile(InstanceProfileName=BOOTSTRAP_PROFILE))
    probe("iam:ListAttachedUserPolicies",
          lambda: iam.list_attached_user_policies(
              UserName=(iam.get_user().get("User") or {}).get("UserName", "")))

    # The reads that actually explain a NoCredentials box: does the profile
    # contain the role, and does the role carry any policy at all?
    detail = {}
    try:
        p = iam.get_instance_profile(InstanceProfileName=BOOTSTRAP_PROFILE)
        roles = [r.get("RoleName") for r in
                 (p.get("InstanceProfile") or {}).get("Roles") or []]
        detail["profile_roles"] = roles
    except Exception as e:
        detail["profile_roles"] = "ERR " + str(e)[:200]
    try:
        detail["role_inline_policies"] = iam.list_role_policies(
            RoleName=BOOTSTRAP_ROLE).get("PolicyNames")
    except Exception as e:
        detail["role_inline_policies"] = "ERR " + str(e)[:200]
    for probe_name in ("kml", BOOTSTRAP_PROFILE):
        try:
            p = iam.get_instance_profile(InstanceProfileName=probe_name)
            detail["profile_" + probe_name] = [
                r.get("RoleName") for r in
                (p.get("InstanceProfile") or {}).get("Roles") or []]
        except Exception as e:
            detail["profile_" + probe_name] = "ERR " + str(e)[:120]
    try:
        detail["role_attached_policies"] = [
            a.get("PolicyName") for a in iam.list_attached_role_policies(
                RoleName=BOOTSTRAP_ROLE).get("AttachedPolicies") or []]
    except Exception as e:
        detail["role_attached_policies"] = "ERR " + str(e)[:200]
    try:
        u = (iam.get_user().get("User") or {}).get("UserName", "")
        detail["user_inline_policies"] = iam.list_user_policies(
            UserName=u).get("PolicyNames")
    except Exception as e:
        detail["user_inline_policies"] = "ERR " + str(e)[:200]

    try:
        u = (iam.get_user().get("User") or {}).get("UserName", "")
        docs = {}
        for pn in iam.list_user_policies(UserName=u).get("PolicyNames") or []:
            docs[pn] = iam.get_user_policy(UserName=u, PolicyName=pn).get(
                "PolicyDocument")
        detail["user_policy_docs"] = docs
    except Exception as e:
        detail["user_policy_docs"] = "ERR " + str(e)[:200]
    return {"probes": out, "detail": detail}


@aws_compute_router.post(
    "/aws/bootstrap/compute",
    summary="Create the instance role + profile (operator only, idempotent)",
)
async def bootstrap_compute():
    _compute_bootstrap_enabled()
    region = (os.getenv("AWS_DEFAULT_REGION") or "").strip()
    if not region:
        raise HTTPException(503, "AWS_DEFAULT_REGION is not set")
    bucket = (os.getenv("AWS_S3_BUCKET") or "").strip()
    if not bucket:
        raise HTTPException(503, "AWS_S3_BUCKET is not set")

    iam = _client("iam")
    steps = []

    def _exists(e) -> bool:
        return getattr(e, "response", {}).get("Error", {}).get(
            "Code", "") == "EntityAlreadyExists"

    try:
        iam.create_role(RoleName=BOOTSTRAP_ROLE,
                        AssumeRolePolicyDocument=_trust_policy(),
                        Description="KalshiML production instance role")
        steps.append({"step": "create_role", "result": "created"})
    except Exception as e:
        if not _exists(e):
            err = getattr(e, "response", {}).get("Error", {})
            # AWS names the principal and the action in its denial text. Keep it.
            raise HTTPException(502, "create_role failed: %s: %s" % (
                err.get("Code"), (err.get("Message") or str(e))[:500]))
        steps.append({"step": "create_role", "result": "already exists"})

    try:
        iam.put_role_policy(RoleName=BOOTSTRAP_ROLE,
                            PolicyName="kalshiml-prod-instance",
                            PolicyDocument=_instance_policy(region, bucket))
        steps.append({"step": "put_role_policy", "result": "written"})
    except Exception as e:
        raise _fail(e)

    try:
        iam.create_instance_profile(InstanceProfileName=BOOTSTRAP_PROFILE)
        steps.append({"step": "create_instance_profile", "result": "created"})
    except Exception as e:
        if not _exists(e):
            raise _fail(e)
        steps.append({"step": "create_instance_profile", "result": "already exists"})

    # Check the link before trying to make it. AddRoleToInstanceProfile is the
    # one action this credential is denied, so when the console wizard has
    # already linked them, calling it anyway turns a finished job into a 502.
    already = False
    try:
        p = iam.get_instance_profile(InstanceProfileName=BOOTSTRAP_PROFILE)
        linked = [r.get("RoleName") for r in
                  (p.get("InstanceProfile") or {}).get("Roles") or []]
        already = BOOTSTRAP_ROLE in linked
        if linked and not already:
            raise HTTPException(409, "profile %s already holds role %s, not %s"
                                % (BOOTSTRAP_PROFILE, linked[0], BOOTSTRAP_ROLE))
    except HTTPException:
        raise
    except Exception:
        pass
    if already:
        steps.append({"step": "add_role_to_profile", "result": "already linked"})
    else:
        try:
            iam.add_role_to_instance_profile(InstanceProfileName=BOOTSTRAP_PROFILE,
                                             RoleName=BOOTSTRAP_ROLE)
            steps.append({"step": "add_role_to_profile", "result": "attached"})
        except Exception as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if code not in ("LimitExceeded", "EntityAlreadyExists"):
                raise _fail(e)
            steps.append({"step": "add_role_to_profile", "result": "already attached"})

    return {
        "role": BOOTSTRAP_ROLE,
        "instance_profile": BOOTSTRAP_PROFILE,
        "region": region,
        "bucket": bucket,
        "steps": steps,
        "next": ("set AWS_COMPUTE_IAM_PROFILE=%s, unset "
                 "AWS_COMPUTE_BOOTSTRAP_ENABLED, then POST /aws/compute/provision "
                 "with dry_run=true" % BOOTSTRAP_PROFILE),
    }


# --------------------------------------------------------------------------- #
# Console output — the diagnostic path that should have shipped with provision.
#
# Launching a box with no way to read its boot log means a failed bootstrap is
# invisible: no SSH key, no console, nothing but "running" and silence. This
# reads the instance's serial console, which needs no key, no network path to
# the box, and no agent installed on it. Pinned to the same tag as everything
# else, so it cannot be aimed at an arbitrary instance id.


@aws_compute_router.get(
    "/aws/compute/console",
    summary="Serial console output for the pinned instance (boot log)",
)
async def console(tail: int = 20000):
    import base64

    name = _name_tag()
    ec2 = _client("ec2")
    inst = _find_existing(ec2, name)
    if inst is None:
        raise HTTPException(404, f"no instance tagged {name!r}")
    iid = inst.get("InstanceId")
    try:
        resp = ec2.get_console_output(InstanceId=iid, Latest=True)
    except Exception as e:
        raise _fail(e)
    raw = resp.get("Output") or ""
    try:
        text = base64.b64decode(raw).decode("utf-8", "replace")
    except Exception:
        # Some responses come back already decoded; fall back rather than 500.
        text = raw
    tail = max(1000, min(int(tail), 200000))
    truncated = len(text) > tail
    return {
        "instance_id": iid,
        "state": (inst.get("State") or {}).get("Name"),
        "timestamp": str(resp.get("Timestamp") or ""),
        "truncated": truncated,
        "bytes": len(text),
        "output": text[-tail:],
    }

# deploy-trigger: watchPatterns fixed 2026-08-27
