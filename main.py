"""
kimi-github-bridge
==================

A small FastAPI service that exposes a handful of GitHub operations over a
clean HTTP API, so an agent (or anything that can speak HTTP) can commit
files, create repositories, list repositories and read file contents
without embedding GitHub client logic of its own.

Authentication uses a single ``GITHUB_TOKEN`` environment variable which is
forwarded to the GitHub REST API as a Bearer token. The token is never
returned in a response.
"""

from __future__ import annotations

import asyncio
import base64
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from railway_extension import router as railway_router, set_service_variable
from llm_routes import llm_router
from command_channel import process_pending_commands

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REQUEST_TIMEOUT = float(os.getenv("GITHUB_TIMEOUT_SECONDS", "30"))

# How often the background command worker polls pending files (seconds).
COMMAND_POLL_INTERVAL_SEC = float(os.getenv("COMMAND_POLL_INTERVAL_SEC", "10"))

# Last command-channel summary (updated by the background worker).
_LAST_CMD_SUMMARY: dict[str, Any] = {
    "processed": 0,
    "errors": 0,
    "ids": [],
    "skipped": 0,
    "last_tick_at": None,
}


async def _command_worker() -> None:
    """Background loop: process pending GitHub commands without blocking /health."""
    loop = asyncio.get_running_loop()
    while True:
        try:
            summary = await loop.run_in_executor(None, process_pending_commands)
            if isinstance(summary, dict):
                summary = dict(summary)
                summary["last_tick_at"] = time.time()
                _LAST_CMD_SUMMARY.clear()
                _LAST_CMD_SUMMARY.update(summary)
        except Exception as e:
            _LAST_CMD_SUMMARY["error"] = str(e)
            _LAST_CMD_SUMMARY["last_tick_at"] = time.time()
        await asyncio.sleep(COMMAND_POLL_INTERVAL_SEC)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_command_worker())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="kimi-github-bridge",
    description=(
        "A thin HTTP bridge over the GitHub REST API: commit files, create "
        "repos, list repos and read file contents using a single "
        "GITHUB_TOKEN."
    ),
    version="1.4.2",
    lifespan=lifespan,
)

# Include Railway extension router
app.include_router(railway_router)

# Include LLM gateway router
app.include_router(llm_router)

# Include skills router
from skills_routes import skills_router
app.include_router(skills_router)


# --------------------------------------------------------------------------- #
# Bridge key auth
# --------------------------------------------------------------------------- #
BRIDGE_KEY_TTL_SECONDS = float(os.getenv("BRIDGE_KEY_TTL_HOURS", "24")) * 3600
BRIDGE_KEY_GRACE_SECONDS = float(os.getenv("BRIDGE_KEY_GRACE_HOURS", "2")) * 3600
_EXEMPT_PATHS = {"/health"}


class _KeyState:
    def __init__(self) -> None:
        self.key: Optional[str] = os.getenv("BRIDGE_API_KEY")
        self.issued_at: float = time.time()


KEY_STATE = _KeyState()


def _auth_enabled() -> bool:
    return bool(KEY_STATE.key)


class BridgeAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not _auth_enabled() or request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        supplied = request.headers.get("x-bridge-key")
        if supplied != KEY_STATE.key:
            return JSONResponse(
                {"detail": "missing or invalid X-Bridge-Key header"},
                status_code=401,
            )

        age = time.time() - KEY_STATE.issued_at
        if age > BRIDGE_KEY_TTL_SECONDS + BRIDGE_KEY_GRACE_SECONDS:
            return JSONResponse(
                {
                    "detail": (
                        "key expired beyond grace period — set BRIDGE_API_KEY "
                        "by hand in the Railway dashboard and redeploy"
                    )
                },
                status_code=401,
            )
        if age > BRIDGE_KEY_TTL_SECONDS and request.url.path != "/rotate_key":
            return JSONResponse(
                {"detail": "key expired — call POST /rotate_key to renew"},
                status_code=401,
            )
        return await call_next(request)


app.add_middleware(BridgeAuthMiddleware)


class RotateKeyResponse(BaseModel):
    new_key: str
    issued_at: float
    ttl_hours: float


@app.post("/rotate_key", tags=["meta"], summary="Rotate the bridge API key")
async def rotate_key() -> RotateKeyResponse:
    if not _auth_enabled():
        raise HTTPException(
            status_code=400,
            detail=(
                "auth is not enabled yet — set BRIDGE_API_KEY in the Railway "
                "dashboard and redeploy before rotating"
            ),
        )
    new_key = secrets.token_urlsafe(32)
    set_service_variable("BRIDGE_API_KEY", new_key)
    KEY_STATE.key = new_key
    KEY_STATE.issued_at = time.time()
    return RotateKeyResponse(
        new_key=new_key,
        issued_at=KEY_STATE.issued_at,
        ttl_hours=BRIDGE_KEY_TTL_SECONDS / 3600,
    )


def _github_headers() -> dict[str, str]:
    if not GITHUB_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="GITHUB_TOKEN environment variable is not set on the server.",
        )
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def github_request(
    method: str, path: str, *, json: Any = None, params: Any = None
) -> httpx.Response:
    url = path if path.startswith("http") else f"{GITHUB_API}{path}"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.request(
                method, url, headers=_github_headers(), json=json, params=params
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502, detail=f"Error contacting GitHub: {exc}"
            ) from exc

    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)
    return response


def _diagnose_github_token() -> dict[str, Any]:
    """Test the GITHUB_TOKEN the process is actually using. Never returns the token."""
    if not GITHUB_TOKEN:
        return {"ok": False, "error": "GITHUB_TOKEN env var is empty or missing"}

    token_preview = f"{GITHUB_TOKEN[:4]}...{GITHUB_TOKEN[-4:]}" if len(GITHUB_TOKEN) > 8 else "(too short)"
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{GITHUB_API}/user",
                headers={
                    "Authorization": f"Bearer {GITHUB_TOKEN}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "ok": True,
                "login": data.get("login"),
                "token_preview": token_preview,
                "scopes": resp.headers.get("x-oauth-scopes", ""),
            }
        return {
            "ok": False,
            "status_code": resp.status_code,
            "error": resp.text[:300],
            "token_preview": token_preview,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "token_preview": token_preview}


@app.get("/health", tags=["meta"], summary="Liveness / readiness check")
async def health() -> dict[str, Any]:
    """Return service status. Command processing runs in a background worker (not here)."""
    info: dict[str, Any] = {
        "status": "ok",
        "service": "kimi-github-bridge",
        "version": app.version,
        "github_token_configured": bool(GITHUB_TOKEN),
        "github_token_diag": _diagnose_github_token(),
        "auth_enabled": _auth_enabled(),
        "commands": dict(_LAST_CMD_SUMMARY),
    }
    if _auth_enabled():
        age = time.time() - KEY_STATE.issued_at
        info["key_age_hours"] = round(age / 3600, 2)
        info["key_ttl_hours"] = BRIDGE_KEY_TTL_SECONDS / 3600
        info["key_expired"] = age > BRIDGE_KEY_TTL_SECONDS
    return info


# --------------------------------------------------------------------------- #
# The rest of the endpoints are unchanged
# --------------------------------------------------------------------------- #

class CommitRequest(BaseModel):
    owner: str = Field(..., description="Repository owner (user or org).")
    repo: str = Field(..., description="Repository name.")
    path: str = Field(..., description="Path to the file inside the repo.")
    content: str = Field(..., description="Raw (unencoded) file content.")
    message: str = Field(..., description="Commit message.")
    branch: Optional[str] = Field(
        None, description="Branch to commit to. Defaults to the repo default."
    )


class CreateRepoRequest(BaseModel):
    name: str = Field(..., description="Name of the repository to create.")
    description: str = Field("", description="Repository description.")
    private: bool = Field(True, description="Whether the repo is private.")
    auto_init: bool = Field(
        True, description="Initialize with an empty README so it has a branch."
    )
    organization: Optional[str] = Field(
        None,
        description="Create the repo inside this org instead of the user account.",
    )


class RenameRepoRequest(BaseModel):
    name: str = Field(..., description="New name for the repository.")
    description: str = Field("", description="Updated description (optional).")


@app.get("/repos", tags=["repos"], summary="List repositories")
async def list_repos(
    visibility: str = Query("all", description="Filter by visibility: all, public, or private."),
    per_page: int = Query(30, ge=1, le=100, description="Results per page."),
    page: int = Query(1, ge=1, description="Page number."),
) -> list[dict[str, Any]]:
    response = await github_request(
        "GET",
        "/user/repos",
        params={"visibility": visibility, "per_page": per_page, "page": page, "sort": "updated"},
    )
    repos = response.json()
    return [
        {
            "name": r["name"],
            "full_name": r["full_name"],
            "private": r["private"],
            "html_url": r["html_url"],
            "default_branch": r["default_branch"],
            "description": r.get("description"),
        }
        for r in repos
    ]


@app.post("/repos", tags=["repos"], summary="Create a repository", status_code=201)
async def create_repo(req: CreateRepoRequest) -> dict[str, Any]:
    path = f"/orgs/{req.organization}/repos" if req.organization else "/user/repos"
    response = await github_request(
        "POST",
        path,
        json={
            "name": req.name,
            "description": req.description,
            "private": req.private,
            "auto_init": req.auto_init,
        },
    )
    repo = response.json()
    return {
        "name": repo["name"],
        "full_name": repo["full_name"],
        "private": repo["private"],
        "html_url": repo["html_url"],
        "default_branch": repo["default_branch"],
    }


@app.patch("/repos/{owner}/{repo}", tags=["repos"], summary="Rename a repository")
async def rename_repo(
    owner: str = Path(..., description="Repository owner."),
    repo: str = Path(..., description="Current repository name."),
    req: RenameRepoRequest = ...,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": req.name}
    if req.description:
        payload["description"] = req.description
    response = await github_request("PATCH", f"/repos/{owner}/{repo}", json=payload)
    data = response.json()
    return {
        "old_name": repo,
        "new_name": data["name"],
        "full_name": data["full_name"],
        "html_url": data["html_url"],
        "description": data.get("description", ""),
    }


@app.post("/commit", tags=["files"], summary="Create or update a file")
async def commit_file(req: CommitRequest) -> dict[str, Any]:
    contents_path = f"/repos/{req.owner}/{req.repo}/contents/{req.path}"

    existing_sha: Optional[str] = None
    params = {"ref": req.branch} if req.branch else None
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        lookup = await client.get(
            f"{GITHUB_API}{contents_path}",
            headers=_github_headers(),
            params=params,
        )
    if lookup.status_code == 200:
        existing_sha = lookup.json().get("sha")
    elif lookup.status_code not in (404,):
        try:
            detail = lookup.json()
        except ValueError:
            detail = lookup.text
        raise HTTPException(status_code=lookup.status_code, detail=detail)

    payload: dict[str, Any] = {
        "message": req.message,
        "content": base64.b64encode(req.content.encode("utf-8")).decode("ascii"),
    }
    if req.branch:
        payload["branch"] = req.branch
    if existing_sha:
        payload["sha"] = existing_sha

    response = await github_request("PUT", contents_path, json=payload)
    data = response.json()
    return {
        "committed": True,
        "path": req.path,
        "commit_sha": data.get("commit", {}).get("sha"),
        "content_sha": data.get("content", {}).get("sha"),
        "html_url": data.get("content", {}).get("html_url"),
        "updated": existing_sha is not None,
    }


@app.get(
    "/contents/{owner}/{repo}/{path:path}",
    tags=["files"],
    summary="Read file contents",
)
async def read_contents(
    owner: str = Path(..., description="Repository owner."),
    repo: str = Path(..., description="Repository name."),
    path: str = Path(..., description="Path to the file inside the repo."),
    ref: Optional[str] = Query(None, description="Branch, tag or commit sha to read from."),
) -> dict[str, Any]:
    params = {"ref": ref} if ref else None
    response = await github_request(
        "GET", f"/repos/{owner}/{repo}/contents/{path}", params=params
    )
    data = response.json()

    if isinstance(data, list):
        return {
            "type": "directory",
            "path": path,
            "entries": [
                {"name": e["name"], "path": e["path"], "type": e["type"]}
                for e in data
            ],
        }

    decoded: Optional[str] = None
    if data.get("encoding") == "base64" and data.get("content"):
        try:
            decoded = base64.b64decode(data["content"]).decode("utf-8")
        except UnicodeDecodeError:
            decoded = None

    return {
        "type": "file",
        "path": data["path"],
        "sha": data["sha"],
        "size": data["size"],
        "encoding": data.get("encoding"),
        "content": decoded,
        "html_url": data.get("html_url"),
    }


class TreeFile(BaseModel):
    path: str = Field(..., description="Path of the file inside the repo.")
    content: str = Field(..., description="Raw (unencoded) UTF-8 file content.")


class CommitTreeRequest(BaseModel):
    owner: str = Field(..., description="Repository owner (user or org).")
    repo: str = Field(..., description="Repository name.")
    files: list[TreeFile] = Field(..., description="Files to write in one commit.")
    message: str = Field(..., description="Commit message.")
    branch: Optional[str] = Field(
        None, description="Branch to commit to. Defaults to the repo default."
    )


@app.post("/commit_tree")
async def commit_tree(req: CommitTreeRequest) -> dict[str, Any]:
    if not req.files:
        raise HTTPException(status_code=422, detail="files must be non-empty")

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        headers = _github_headers()

        branch = req.branch
        if not branch:
            repo_resp = await client.get(
                f"{GITHUB_API}/repos/{req.owner}/{req.repo}", headers=headers
            )
            if repo_resp.status_code != 200:
                raise HTTPException(status_code=repo_resp.status_code, detail=repo_resp.json())
            branch = repo_resp.json().get("default_branch", "main")

        ref_resp = await client.get(
            f"{GITHUB_API}/repos/{req.owner}/{req.repo}/git/ref/heads/{branch}",
            headers=headers,
        )
        if ref_resp.status_code != 200:
            raise HTTPException(
                status_code=ref_resp.status_code,
                detail=f"branch '{branch}' not found: {ref_resp.text[:300]}",
            )
        base_commit_sha = ref_resp.json()["object"]["sha"]

        commit_resp = await client.get(
            f"{GITHUB_API}/repos/{req.owner}/{req.repo}/git/commits/{base_commit_sha}",
            headers=headers,
        )
        if commit_resp.status_code != 200:
            raise HTTPException(status_code=commit_resp.status_code, detail=commit_resp.json())
        base_tree_sha = commit_resp.json()["tree"]["sha"]

        tree_items = [
            {"path": f.path, "mode": "100644", "type": "blob", "content": f.content}
            for f in req.files
        ]
        tree_resp = await client.post(
            f"{GITHUB_API}/repos/{req.owner}/{req.repo}/git/trees",
            headers=headers,
            json={"base_tree": base_tree_sha, "tree": tree_items},
        )
        if tree_resp.status_code not in (200, 201):
            raise HTTPException(status_code=tree_resp.status_code, detail=tree_resp.text[:500])
        new_tree_sha = tree_resp.json()["sha"]

        new_commit_resp = await client.post(
            f"{GITHUB_API}/repos/{req.owner}/{req.repo}/git/commits",
            headers=headers,
            json={
                "message": req.message,
                "tree": new_tree_sha,
                "parents": [base_commit_sha],
            },
        )
        if new_commit_resp.status_code not in (200, 201):
            raise HTTPException(status_code=new_commit_resp.status_code, detail=new_commit_resp.text[:500])
        new_commit_sha = new_commit_resp.json()["sha"]

        update_resp = await client.patch(
            f"{GITHUB_API}/repos/{req.owner}/{req.repo}/git/refs/heads/{branch}",
            headers=headers,
            json={"sha": new_commit_sha, "force": False},
        )
        if update_resp.status_code != 200:
            raise HTTPException(
                status_code=update_resp.status_code,
                detail=f"commit created ({new_commit_sha}) but ref update failed: {update_resp.text[:300]}",
            )

    return {
        "commit_sha": new_commit_sha,
        "branch": branch,
        "files_committed": len(req.files),
        "html_url": f"https://github.com/{req.owner}/{req.repo}/commit/{new_commit_sha}",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
