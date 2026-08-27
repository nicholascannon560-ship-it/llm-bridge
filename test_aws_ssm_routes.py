"""Tests for aws_ssm_routes — fake SSM client, no AWS calls."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import HTTPException  # noqa: E402

import aws_ssm_routes as asr  # noqa: E402


class FakeError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeSSM:
    def __init__(self, raise_code=None, params=None):
        self.raise_code = raise_code
        self.kwargs = None
        self.params = params or []

    def put_parameter(self, **kwargs):
        self.kwargs = kwargs
        if self.raise_code:
            raise FakeError(self.raise_code)
        return {"Version": 3}

    def describe_parameters(self, **kwargs):
        self.kwargs = kwargs
        return {"Parameters": self.params}


def run(coro):
    import asyncio
    return asyncio.new_event_loop().run_until_complete(coro)


ENV = {"AWS_SSM_WRITE_ENABLED": "1", "AWS_DEFAULT_REGION": "us-east-2"}


class GateTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(HTTPException) as cm:
                run(asr.ssm_put(asr.PutRequest(name="/kalshiml/prod/X", value="v")))
        self.assertEqual(cm.exception.status_code, 403)


class NamePinTests(unittest.TestCase):
    def test_rejects_name_outside_prefix(self):
        ssm = FakeSSM()
        with mock.patch.dict(os.environ, ENV, clear=True):
            with mock.patch.object(asr, "_client", lambda: ssm):
                with self.assertRaises(HTTPException) as cm:
                    run(asr.ssm_put(asr.PutRequest(name="/other/system/KEY", value="v")))
        self.assertEqual(cm.exception.status_code, 403)
        self.assertIsNone(ssm.kwargs)

    def test_rejects_traversal(self):
        with mock.patch.dict(os.environ, ENV, clear=True):
            with self.assertRaises(HTTPException):
                asr._safe_name("/kalshiml/prod/../../etc/KEY")

    def test_accepts_pinned_name(self):
        with mock.patch.dict(os.environ, ENV, clear=True):
            self.assertEqual(
                asr._safe_name("/kalshiml/prod/KALSHI_API_KEY_ID"),
                "/kalshiml/prod/KALSHI_API_KEY_ID",
            )


class PutTests(unittest.TestCase):
    def test_type_is_forced_securestring(self):
        ssm = FakeSSM()
        with mock.patch.dict(os.environ, ENV, clear=True):
            with mock.patch.object(asr, "_client", lambda: ssm):
                out = run(asr.ssm_put(asr.PutRequest(
                    name="/kalshiml/prod/ANTHROPIC_API_KEY", value="sk-test")))
        self.assertEqual(ssm.kwargs["Type"], "SecureString")
        self.assertEqual(out["version"], 3)

    def test_value_is_never_echoed(self):
        ssm = FakeSSM()
        secret = "super-secret-value"
        with mock.patch.dict(os.environ, ENV, clear=True):
            with mock.patch.object(asr, "_client", lambda: ssm):
                out = run(asr.ssm_put(asr.PutRequest(
                    name="/kalshiml/prod/K", value=secret)))
        self.assertNotIn(secret, str(out))
        self.assertEqual(len(out["sha256_12"]), 12)

    def test_empty_value_refused(self):
        ssm = FakeSSM()
        with mock.patch.dict(os.environ, ENV, clear=True):
            with mock.patch.object(asr, "_client", lambda: ssm):
                with self.assertRaises(HTTPException):
                    run(asr.ssm_put(asr.PutRequest(name="/kalshiml/prod/K", value="")))

    def test_large_pem_uses_advanced_tier(self):
        ssm = FakeSSM()
        with mock.patch.dict(os.environ, ENV, clear=True):
            with mock.patch.object(asr, "_client", lambda: ssm):
                run(asr.ssm_put(asr.PutRequest(
                    name="/kalshiml/prod/KALSHI_PRIVATE_KEY", value="x" * 5000)))
        self.assertEqual(ssm.kwargs["Tier"], "Advanced")

    def test_oversized_refused(self):
        ssm = FakeSSM()
        with mock.patch.dict(os.environ, ENV, clear=True):
            with mock.patch.object(asr, "_client", lambda: ssm):
                with self.assertRaises(HTTPException):
                    run(asr.ssm_put(asr.PutRequest(
                        name="/kalshiml/prod/K", value="x" * 9000)))

    def test_no_overwrite_conflict_is_409(self):
        ssm = FakeSSM(raise_code="ParameterAlreadyExists")
        with mock.patch.dict(os.environ, ENV, clear=True):
            with mock.patch.object(asr, "_client", lambda: ssm):
                with self.assertRaises(HTTPException) as cm:
                    run(asr.ssm_put(asr.PutRequest(
                        name="/kalshiml/prod/K", value="v", overwrite=False)))
        self.assertEqual(cm.exception.status_code, 409)


class ListTests(unittest.TestCase):
    def test_list_returns_names_not_values(self):
        ssm = FakeSSM(params=[
            {"Name": "/kalshiml/prod/A", "Type": "SecureString", "Version": 1},
        ])
        with mock.patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-east-2"}, clear=True):
            with mock.patch.object(asr, "_client", lambda: ssm):
                out = run(asr.ssm_list())
        self.assertEqual(out["count"], 1)
        self.assertNotIn("Value", str(out))
        self.assertEqual(out["parameters"][0]["name"], "/kalshiml/prod/A")

    def test_no_get_route_exists(self):
        # The design decision, pinned as a test: writing is allowed, reading back is not.
        paths = {r.path for r in asr.aws_ssm_router.routes}
        self.assertNotIn("/aws/ssm/get", paths)


if __name__ == "__main__":
    unittest.main(verbosity=2)
