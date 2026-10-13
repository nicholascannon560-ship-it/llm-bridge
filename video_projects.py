"""Projects for Video Studio: folders that hold an account's videos, pictures and edits.

Each video (job), picture and timeline carries a "project" field (None = Unsorted). Where new
items land is decided in video_app (_stamp / _index_image): the project named for the request,
else the project of the clip it was made from, else the account's active project.

The active project lives on the account record (u["project"]) and is shared by the app and the
Claude connector, so whatever Claude is working on is what the app opens to.

Routes (signed in):
  GET  /video/projects                      projects with counts, plus the active one
  POST /video/projects                      {name} -> new project (becomes active)
  POST /video/projects/active               {id | null}
  POST /video/projects/{pid}/rename         {name}
  POST /video/projects/{pid}/delete         items move to Unsorted; files are kept
  POST /video/projects/move                 {kind: video|picture|edit, ids: [...], project: id|null}
  GET  /video/pictures?project=<id|none>    this account's pictures in a project
"""
from __future__ import annotations

import json
import secrets
import time

from fastapi import APIRouter, Request

import video_app as va

projects_router = APIRouter()
PROJECTS_KEY = f"{va.PREFIX}projects.json"
_projects: list[dict] | None = None
NAME_MAX = 60


async def all_projects() -> list[dict]:
    global _projects
    if _projects is None:
        raw = await va._get(PROJECTS_KEY)
        try:
            _projects = json.loads(raw) if raw else []
        except Exception:
            _projects = []
    return _projects


async def _save() -> None:
    await va._put(PROJECTS_KEY, json.dumps(await all_projects(), separators=(",", ":")).encode(), "application/json")


async def mine(uid: str) -> list[dict]:
    return [p for p in await all_projects() if p["user"] == uid]


async def resolve(uid: str, ref) -> tuple[bool, dict | None]:
    """Find a project by id or name (case-insensitive). (True, None) means Unsorted;
    (False, None) means no match."""
    s = str(ref or "").strip()
    if s.lower() in ("", "none", "unsorted", "null"):
        return True, None
    rows = await mine(uid)
    hit = next((p for p in rows if p["id"] == s), None) or \
        next((p for p in rows if p["name"].strip().lower() == s.lower()), None)
    return (hit is not None), hit


async def create(uid: str, name: str) -> dict:
    name = " ".join(str(name or "").split())[:NAME_MAX]
    if not name:
        raise ValueError("Give the project a name")
    if any(p["name"].lower() == name.lower() for p in await mine(uid)):
        raise ValueError(f"You already have a project called {name}")
    p = {"id": "p_" + secrets.token_urlsafe(9).replace("-", "x").replace("_", "y"), "user": uid,
         "name": name, "created": time.time()}
    rows = await all_projects()
    async with va._lock:
        rows.append(p)
    await _save()
    await set_active(uid, p["id"])
    return p


async def _user_rec(uid: str) -> dict | None:
    return await (va._owner_rec() if uid == "owner" else va._user_by_id(uid))


async def active(uid: str) -> str | None:
    u = await _user_rec(uid)
    return (u or {}).get("project")


async def set_active(uid: str, pid: str | None) -> None:
    u = await _user_rec(uid)
    if u is not None and u.get("project") != pid:
        u["project"] = pid
        await va._save_users()


async def name_of(uid: str, pid: str | None) -> str:
    if not pid:
        return "Unsorted"
    p = next((p for p in await mine(uid) if p["id"] == pid), None)
    return p["name"] if p else "Unsorted"


def is_edited(j: dict) -> bool:
    """Stitches, loops, zips and editor renders go in the Edited tab."""
    return j.get("model") == "editor"


async def counts(uid: str) -> dict:
    out: dict = {}
    for j in await va._all_jobs():
        if va._owns(uid, j) and not j.get("superseded"):
            c = out.setdefault(j.get("project"), {"videos": 0, "edited": 0, "pictures": 0})
            c["edited" if is_edited(j) else "videos"] += 1
    for im in await va._all_images():
        if im.get("user") == uid:
            out.setdefault(im.get("project"), {"videos": 0, "edited": 0, "pictures": 0})["pictures"] += 1
    return out


async def move(uid: str, kind: str, ids: list[str], pid: str | None) -> int:
    rows = {"video": await va._all_jobs(), "picture": await va._all_images(), "edit": await va._all_edits()}.get(kind)
    if rows is None:
        raise ValueError("kind must be video, picture or edit")
    want, n = set(ids), 0
    for r in rows:
        if r.get("id") in want and (r.get("user") or "owner") == uid:
            r["project"] = pid
            n += 1
    if n:
        await {"video": va._save_jobs, "picture": va._save_images, "edit": va._save_edits}[kind]()
    return n


async def overview(uid: str) -> dict:
    c, act = await counts(uid), await active(uid)
    zero = {"videos": 0, "edited": 0, "pictures": 0}
    rows = [{"id": p["id"], "name": p["name"], "created": p["created"], **c.get(p["id"], zero)}
            for p in sorted(await mine(uid), key=lambda p: p["created"])]
    return {"projects": rows, "unsorted": c.get(None, zero), "active": act,
            "active_name": await name_of(uid, act)}


async def pictures(uid: str, pid: str | None, limit: int = 200) -> list[dict]:
    rows = [im for im in await va._all_images() if im.get("user") == uid and im.get("project") == pid]
    return [{"id": im["id"], "url": f"/video/img/{im['id']}", "kind": im.get("kind"), "note": im.get("note") or "",
             "created": im.get("t"), "project": im.get("project")} for im in reversed(rows[-limit:])]


# --------------------------------------------------------------------------- routes
def _bad(msg: str, code: int = 400):
    return va._err(msg, code)


@projects_router.get("/video/projects")
async def list_projects(request: Request):
    if (d := await va._deny(request)):
        return d
    return await overview(va._uid(request))


@projects_router.post("/video/projects")
async def new_project(request: Request):
    if (d := await va._deny(request)):
        return d
    try:
        p = await create(va._uid(request), (await va._body(request)).get("name"))
    except ValueError as e:
        return _bad(str(e))
    return {"project": {"id": p["id"], "name": p["name"]}, **await overview(va._uid(request))}


@projects_router.post("/video/projects/active")
async def choose_project(request: Request):
    if (d := await va._deny(request)):
        return d
    uid = va._uid(request)
    ok, p = await resolve(uid, (await va._body(request)).get("id"))
    if not ok:
        return _bad("No such project", 404)
    await set_active(uid, p["id"] if p else None)
    return await overview(uid)


@projects_router.post("/video/projects/move")
async def move_items(request: Request):
    if (d := await va._deny(request)):
        return d
    uid, b = va._uid(request), await va._body(request)
    ok, p = await resolve(uid, b.get("project"))
    if not ok:
        return _bad("No such project", 404)
    try:
        n = await move(uid, str(b.get("kind") or ""), [str(x) for x in (b.get("ids") or [])][:200],
                       p["id"] if p else None)
    except ValueError as e:
        return _bad(str(e))
    return {"moved": n}


@projects_router.post("/video/projects/{pid}/rename")
async def rename_project(pid: str, request: Request):
    if (d := await va._deny(request)):
        return d
    uid = va._uid(request)
    p = next((p for p in await mine(uid) if p["id"] == pid), None)
    name = " ".join(str((await va._body(request)).get("name") or "").split())[:NAME_MAX]
    if not p:
        return _bad("No such project", 404)
    if not name:
        return _bad("Give the project a name")
    if any(x["name"].lower() == name.lower() and x["id"] != pid for x in await mine(uid)):
        return _bad(f"You already have a project called {name}")
    p["name"] = name
    await _save()
    return await overview(uid)


@projects_router.post("/video/projects/{pid}/delete")
async def delete_project(pid: str, request: Request):
    """Removes the folder only: its videos, pictures and edits move to Unsorted."""
    if (d := await va._deny(request)):
        return d
    uid = va._uid(request)
    rows = await all_projects()
    if not any(p["id"] == pid and p["user"] == uid for p in rows):
        return _bad("No such project", 404)
    for kind, src in (("video", await va._all_jobs()), ("picture", await va._all_images()),
                      ("edit", await va._all_edits())):
        ids = [r["id"] for r in src if r.get("project") == pid and (r.get("user") or "owner") == uid]
        if ids:
            await move(uid, kind, ids, None)
    async with va._lock:
        rows[:] = [p for p in rows if not (p["id"] == pid and p["user"] == uid)]
    await _save()
    if await active(uid) == pid:
        await set_active(uid, None)
    return await overview(uid)


@projects_router.get("/video/pictures")
async def list_pictures(request: Request):
    if (d := await va._deny(request)):
        return d
    uid = va._uid(request)
    ok, p = await resolve(uid, request.query_params.get("project"))
    if not ok:
        return _bad("No such project", 404)
    return {"pictures": await pictures(uid, p["id"] if p else None)}
