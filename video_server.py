"""Video Studio as its own Railway service.

Only the /video app is mounted here: none of the bridge's GitHub, Railway or AWS
routes exist in this process, so customer traffic never shares a server with them.
Start: uvicorn video_server:app --host 0.0.0.0 --port $PORT
Data lives under VIDEO_S3_PREFIX (set to studio/ on this service).
"""
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from video_app import VIDEO_APP_VERSION, video_router

app = FastAPI(title="Video Studio", docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(video_router)


@app.get("/")
async def root():
    return RedirectResponse("/video")


@app.get("/health")
async def health():
    return {"ok": True, "version": VIDEO_APP_VERSION}
