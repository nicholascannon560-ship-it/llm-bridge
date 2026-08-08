# patch_routes.py
import difflib
from typing import Optional
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["github"])

class PatchRequest(BaseModel):
    owner: str
    repo: str
    path: str
    old_string: str
    new_string: str
    message: str
    branch: str = "main"

class PatchResponse(BaseModel):
    sha: str
    commit_sha: str
    patch_applied: bool
    diff_preview: str

async def _read_file_raw(owner: str, repo: str, path: str, branch: str = "main") -> Optional[str]:
    from main import github_request
    resp = await github_request("GET", f"/repos/{owner}/{repo}/contents/{path}?ref={branch}")
    data = resp.json()
    import base64
    content = data.get("content", "").replace("\n", "")
    return base64.b64decode(content).decode("utf-8") if content else ""

async def _get_file_sha(owner: str, repo: str, path: str, branch: str = "main") -> Optional[str]:
    from main import github_request
    resp = await github_request("GET", f"/repos/{owner}/{repo}/contents/{path}?ref={branch}")
    return resp.json().get("sha")

@router.post("/patch", response_model=PatchResponse)
async def patch_file(req: PatchRequest):
    current = await _read_file_raw(req.owner, req.repo, req.path, req.branch)
    if current is None:
        raise HTTPException(status_code=404, detail="file not found")

    if req.old_string in current:
        new_content = current.replace(req.old_string, req.new_string, 1)
    else:
        try:
            new_content = _fuzzy_replace(current, req.old_string, req.new_string)
        except ValueError:
            diff = "\n".join(difflib.unified_diff(
                current.splitlines(),
                (current + "\n" + req.new_string).splitlines(),
                lineterm=""
            ))
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "old_string not found in file",
                    "hint": "Use GET /contents to read the exact current text.",
                    "diff_preview": diff[:2000],
                }
            )

    file_sha = await _get_file_sha(req.owner, req.repo, req.path, req.branch)

    from main import _do_commit
    commit_result = await _do_commit({
        "owner": req.owner,
        "repo": req.repo,
        "path": req.path,
        "content": new_content,
        "message": req.message,
        "branch": req.branch,
        "sha": file_sha,
    })

    diff_preview = "\n".join(difflib.unified_diff(
        current.splitlines(),
        new_content.splitlines(),
        fromfile=f"a/{req.path}",
        tofile=f"b/{req.path}",
        lineterm=""
    ))

    return PatchResponse(
        sha=commit_result.get("content_sha", ""),
        commit_sha=commit_result.get("commit_sha", ""),
        patch_applied=True,
        diff_preview=diff_preview[:3000]
    )

def _fuzzy_replace(text: str, old: str, new: str) -> str:
    old_lines = old.splitlines()
    text_lines = text.splitlines()

    for i in range(len(text_lines) - len(old_lines) + 1):
        window = text_lines[i:i + len(old_lines)]
        match = all(w.strip() == o.strip() for w, o in zip(window, old_lines))
        if match:
            indent = len(text_lines[i]) - len(text_lines[i].lstrip())
            new_lines = []
            for idx, line in enumerate(new.splitlines()):
                if idx == 0:
                    new_lines.append((" " * indent + line.lstrip()) if line.strip() else line)
                else:
                    new_lines.append(line)
            return "\n".join(text_lines[:i] + new_lines + text_lines[i + len(old_lines):])

    raise ValueError("fuzzy match failed")
