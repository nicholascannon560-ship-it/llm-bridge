"""Tests for aws_compute_routes — no AWS calls, a fake client stands in.

These cover the properties that matter: the gate, the allowlist, idempotency
(the failure mode is fifty boxes, not one), MaxCount, and the dry-run path.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import HTTPException  # noqa: E402

import aws_compute_routes as acr  # noqa: E402


class FakeError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeEC2:
    def __init__(self, existing=None, raise_code=None):
        self.existing = existing or []
        self.raise_code = raise_code
        self.run_kwargs = None

    def describe_instances(self, Filters=None):
        self.last_filters = Filters
        if not self.existing:
            return {"Reservations": []}
        return {"Reservations": [{"Instances": self.existing}]}

    def run_instances(self, **kwargs):
        self.run_kwargs = kwargs
        if self.raise_code:
            raise FakeError(self.raise_code)
        return {"Instances": [{
            "InstanceId": "i-new", "State": {"Name": "pending"},
            "InstanceType": kwargs["InstanceType"], "ImageId": kwargs["ImageId"],
        }]}


class FakeSSM:
    def get_parameter(self, Name=None):
        return {"Parameter": {"Value": "ami-abc123"}}


def run(coro):
    import asyncio
    return asyncio.get_event_loop().run_until_complete(coro)


def patch_clients(ec2, ssm=None):
    ssm = ssm or FakeSSM()
    return mock.patch.object(
        acr, "_client", lambda svc: ec2 if svc == "ec2" else ssm
    )


BASE_ENV = {"AWS_COMPUTE_ENABLED": "1", "AWS_DEFAULT_REGION": "us-east-2"}


class GateTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(HTTPException) as cm:
                run(acr.provision(acr.ProvisionRequest()))
            self.assertEqual(cm.exception.status_code, 403)

    def test_status_needs_no_gate(self):
        ec2 = FakeEC2()
        with mock.patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-east-2"}, clear=True):
            with patch_clients(ec2):
                out = run(acr.status())
        self.assertFalse(out["exists"])


class AllowlistTests(unittest.TestCase):
    def test_rejects_type_outside_allowlist(self):
        env = dict(BASE_ENV, AWS_COMPUTE_INSTANCE_TYPE="m5.24xlarge")
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(HTTPException) as cm:
                acr._instance_type()
            self.assertEqual(cm.exception.status_code, 400)

    def test_accepts_allowlisted_type(self):
        env = dict(BASE_ENV, AWS_COMPUTE_INSTANCE_TYPE="t3.micro")
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(acr._instance_type(), "t3.micro")

    def test_volume_ceiling(self):
        env = dict(BASE_ENV, AWS_COMPUTE_VOLUME_GB="500")
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(HTTPException):
                acr._volume_gb()


class IdempotencyTests(unittest.TestCase):
    def test_existing_instance_is_returned_not_relaunched(self):
        ec2 = FakeEC2(existing=[{
            "InstanceId": "i-old", "State": {"Name": "running"},
            "InstanceType": "t3.small", "PublicIpAddress": "1.2.3.4",
        }])
        with mock.patch.dict(os.environ, BASE_ENV, clear=True):
            with patch_clients(ec2):
                out = run(acr.provision(acr.ProvisionRequest()))
        self.assertFalse(out["created"])
        self.assertEqual(out["instance"]["instance_id"], "i-old")
        self.assertIsNone(ec2.run_kwargs, "run_instances must not be called")

    def test_terminated_does_not_count_as_existing(self):
        # The filter must exclude terminated, or a rebuild is impossible.
        ec2 = FakeEC2()
        with mock.patch.dict(os.environ, BASE_ENV, clear=True):
            with patch_clients(ec2):
                run(acr.provision(acr.ProvisionRequest()))
        states = [f for f in ec2.last_filters if f["Name"] == "instance-state-name"][0]
        self.assertNotIn("terminated", states["Values"])
        self.assertIn("running", states["Values"])


class LaunchTests(unittest.TestCase):
    def test_launch_pins_count_and_tag(self):
        ec2 = FakeEC2()
        with mock.patch.dict(os.environ, BASE_ENV, clear=True):
            with patch_clients(ec2):
                out = run(acr.provision(acr.ProvisionRequest(user_data="#!/bin/bash\necho hi")))
        self.assertTrue(out["created"])
        self.assertEqual(ec2.run_kwargs["MinCount"], 1)
        self.assertEqual(ec2.run_kwargs["MaxCount"], 1)
        self.assertEqual(ec2.run_kwargs["ImageId"], "ami-abc123")
        self.assertEqual(ec2.run_kwargs["MetadataOptions"]["HttpTokens"], "required")
        tags = ec2.run_kwargs["TagSpecifications"][0]["Tags"]
        self.assertIn({"Key": "Name", "Value": "kalshiml-engine"}, tags)

    def test_oversized_user_data_rejected(self):
        ec2 = FakeEC2()
        with mock.patch.dict(os.environ, BASE_ENV, clear=True):
            with patch_clients(ec2):
                with self.assertRaises(HTTPException) as cm:
                    run(acr.provision(acr.ProvisionRequest(user_data="x" * 20000)))
        self.assertEqual(cm.exception.status_code, 400)

    def test_optional_env_is_omitted_when_unset(self):
        ec2 = FakeEC2()
        with mock.patch.dict(os.environ, BASE_ENV, clear=True):
            with patch_clients(ec2):
                run(acr.provision(acr.ProvisionRequest()))
        for k in ("SubnetId", "SecurityGroupIds", "IamInstanceProfile", "KeyName"):
            self.assertNotIn(k, ec2.run_kwargs)

    def test_dry_run_reports_permission_ok(self):
        ec2 = FakeEC2(raise_code="DryRunOperation")
        with mock.patch.dict(os.environ, BASE_ENV, clear=True):
            with patch_clients(ec2):
                out = run(acr.provision(acr.ProvisionRequest(dry_run=True)))
        self.assertTrue(out["dry_run"])
        self.assertEqual(out["would_launch"]["instance_type"], "t3.small")
        self.assertTrue(ec2.run_kwargs["DryRun"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
