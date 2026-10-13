"""Video Studio as its own Railway service.

Only the /video app is mounted here: none of the bridge's GitHub, Railway or AWS
routes exist in this process, so customer traffic never shares a server with them.
Start: uvicorn video_server:app --host 0.0.0.0 --port $PORT
Data lives under VIDEO_S3_PREFIX (set to studio/ on this service).
"""
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from video_app import VIDEO_APP_VERSION, video_router
from video_mcp import mcp_router
from video_demo import demo_router
from video_oauth import oauth_router
from video_projects import projects_router
from video_routing import routing_router

app = FastAPI(title="Video Studio", docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(video_router)
app.include_router(projects_router)  # project folders (videos, pictures, edited)
app.include_router(routing_router)  # auto model routing (/video/route, /video/routing)
app.include_router(oauth_router)  # sign-in for the customer connector (OAuth 2.1 + PKCE)
app.include_router(mcp_router)  # Claude connector: /mcp (signed-in accounts) and /mcp/{VIDEO_MCP_SECRET} (owner)
app.include_router(demo_router)  # free public demo at /video/demo (embedded on North River Labs)


@app.get("/")
async def root():
    return RedirectResponse("/video")


@app.get("/health")
async def health():
    return {"ok": True, "version": VIDEO_APP_VERSION}
