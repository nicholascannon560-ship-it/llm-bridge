"""Regression test for llm-bridge issue #2 — agent results lost on fast runs.

Loads command_channel.py with a fake in-memory GitHub (sha-checked, like the
real contents API) and a fake agent that crashes instantly. The crash payload
must survive in commands/results/<id>.json.

Run against the pre-patch file too (MODULE_PATH env) — it is expected to FAIL
there. A test that passes on both proves nothing.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
import types

MODULE_PATH = os.getenv("MODULE_PATH", "command_channel.py")


# ── fake GitHub contents API ────────────────────────────────────────────────
class FakeGitHub:
    def __init__(self):
        self.files: dict[str, str] = {}
        self.lock = threading.Lock()
        self.conflicts = 0
        self.stale_sha_once: set[str] = set()   # paths that serve one stale read
        self.slow_paths: tuple[str, ...] = ()
        self.slow_sec = 0.0

    @staticmethod
    def _sha(content: str) -> str:
        return hashlib.sha1(content.encode()).hexdigest()

    def _delay(self, path: str):
        if self.slow_sec and any(p in path for p in self.slow_paths):
            time.sleep(self.slow_sec)

    def get(self, path: str):
        self._delay(path)
        with self.lock:
            if path not in self.files:
                # directory listing, as the contents API returns it
                prefix = path.rstrip("/") + "/"
                entries = [
                    {"name": p[len(prefix):], "path": p, "type": "file",
                     "sha": self._sha(c)}
                    for p, c in self.files.items()
                    if p.startswith(prefix) and "/" not in p[len(prefix):]
                ]
                if entries:
                    return 200, entries
                return 404, {}
            content = self.files[path]
            sha = self._sha(content)
            if path in self.stale_sha_once:
                self.stale_sha_once.discard(path)
                sha = self._sha("stale-" + content)
            return 200, {
                "content": base64.b64encode(content.encode()).decode(),
                "sha": sha,
                "encoding": "base64",
            }

    def put(self, path: str, payload: dict):
        self._delay(path)
        with self.lock:
            content = base64.b64decode(payload["content"]).decode()
            if path in self.files:
                if payload.get("sha") != self._sha(self.files[path]):
                    self.conflicts += 1
                    return 409, {"message": "does not match"}
            elif payload.get("sha"):
                self.conflicts += 1
                return 409, {"message": "no file to update"}
            self.files[path] = content
            return 200, {"content": {"sha": self._sha(content)}}

    def delete(self, path: str, payload: dict):
        with self.lock:
            if path not in self.files:
                return 404, {}
            if payload.get("sha") != self._sha(self.files[path]):
                return 409, {"message": "sha mismatch"}
            del self.files[path]
            return 200, {}


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeClient:
    """Stands in for httpx.Client. Only the contents API is modelled."""

    def __init__(self, hub, *a, **k):
        self.hub = hub

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @staticmethod
    def _path(url: str) -> str:
        return url.split("/contents/", 1)[1]

    def get(self, url, **k):
        return FakeResponse(*self.hub.get(self._path(url)))

    def put(self, url, json=None, **k):
        return FakeResponse(*self.hub.put(self._path(url), json))

    def request(self, method, url, content=None, **k):
        import json as _json
        return FakeResponse(*self.hub.delete(self._path(url), _json.loads(content)))


# ── module loading with stubbed dependencies ────────────────────────────────
def load_module(hub, run_agent_impl):
    for name in ("railway_extension", "agent_loop", "agent_loop.harness",
                 "agent_loop.tools", "agent_loop.browser"):
        sys.modules.pop(name, None)

    rw = types.ModuleType("railway_extension")
    for fn in ("railway_query", "set_service_variable", "list_projects",
               "list_services", "get_service_status", "get_logs", "redeploy_service"):
        setattr(rw, fn, lambda *a, **k: {})
    rw.BRIDGE_SERVICE_ID = "svc"
    sys.modules["railway_extension"] = rw

    pkg = types.ModuleType("agent_loop"); pkg.__path__ = []
    harness = types.ModuleType("agent_loop.harness"); harness.run_agent = run_agent_impl
    tools = types.ModuleType("agent_loop.tools")
    tools.TOOL_SCHEMAS = {}
    tools.env_name_is_protected = lambda n: False
    browser = types.ModuleType("agent_loop.browser")
    browser.RunAuthorization = lambda *a, **k: object()
    sys.modules.update({"agent_loop": pkg, "agent_loop.harness": harness,
                        "agent_loop.tools": tools, "agent_loop.browser": browser})

    os.environ.setdefault("GITHUB_TOKEN", "fake")
    spec = importlib.util.spec_from_file_location("cc_under_test", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cc_under_test"] = mod
    spec.loader.exec_module(mod)

    mod.httpx = types.SimpleNamespace(Client=lambda *a, **k: FakeClient(hub))
    mod.GITHUB_TOKEN = "fake"
    return mod


def _join_agent_threads(timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        live = [t for t in threading.enumerate() if t.name.startswith("agent-")]
        if not live:
            return True
        for t in live:
            t.join(timeout=0.2)
    return False


def _run_one(hub, mod, cmd_id):
    hub.files[f"commands/pending/{cmd_id}.json"] = json.dumps(
        {"action": "agent_run", "task": "boom", "max_turns": 2}
    )
    summary = mod.process_pending_commands()
    assert _join_agent_threads(), "agent worker thread never finished"
    time.sleep(0.3)
    return summary


# ── tests ───────────────────────────────────────────────────────────────────
def test_fast_crash_result_survives():
    """The killer ordering: the run dies before the 'started' stub is written."""
    hub = FakeGitHub()
    # Make writes under commands/ slow enough that the stub write and the
    # worker's final write reliably interleave.
    hub.slow_paths = ("commands/",)
    hub.slow_sec = 0.05

    async def exploding_agent(**kwargs):
        raise ValueError("tools must be a list of schema dicts, got ['run_tests']")

    mod = load_module(hub, exploding_agent)
    _run_one(hub, mod, "fastcrash")

    result_raw = hub.files.get("commands/results/fastcrash.json")
    assert result_raw is not None, "no result file was written at all"
    result = json.loads(result_raw)

    assert result.get("status") == "error", f"result frozen at: {result.get('status')}"
    assert "tools must be a list" in (result.get("error") or ""), result
    assert result.get("traceback"), "traceback was lost"
    assert "commands/pending/fastcrash.json" not in hub.files


def test_results_path_has_one_writer():
    """Checkpoints and the stub must not land in commands/results."""
    hub = FakeGitHub()

    async def checkpointing_agent(**kwargs):
        cb = kwargs.get("on_checkpoint")
        for turn in (1, 2):
            if cb:
                cb({"turns_used": turn})
            time.sleep(0.05)
        return {"turns_used": 2, "final": "done"}

    mod = load_module(hub, checkpointing_agent)
    _run_one(hub, mod, "chkpt")

    result = json.loads(hub.files["commands/results/chkpt.json"])
    assert result["status"] == "ok", result
    assert result["result"]["final"] == "done", "final payload was overwritten"
    assert "commands/running/chkpt.json" not in hub.files, "running marker not cleared"


def test_write_retries_on_stale_sha_409():
    """A stale sha read must not lose the write."""
    hub = FakeGitHub()

    async def noop_agent(**kwargs):
        return {}

    mod = load_module(hub, noop_agent)
    hub.files["commands/results/x.json"] = json.dumps({"status": "old"})
    hub.stale_sha_once.add("commands/results/x.json")

    mod._write_file("commands/results/x.json", json.dumps({"status": "new"}), "msg")

    assert hub.conflicts == 1, "expected the fake to reject one stale write"
    assert json.loads(hub.files["commands/results/x.json"])["status"] == "new"
