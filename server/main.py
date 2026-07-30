"""FastAPI entry point. Serves the built dashboard and the capture control API on localhost."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import audio, config

DIST = config.ROOT / "dashboard" / "dist"

state: dict = {"recorder": None, "cfg": config.load()}


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if rec := state["recorder"]:
        rec.stop()


app = FastAPI(title="MeetTranslate", lifespan=lifespan)

# The Vite dev server runs on its own port; the packaged app is same-origin so this is dev-only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:2886", "http://127.0.0.1:2886"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/devices")
def devices() -> dict:
    """Input devices, plus which one the current config resolves to.

    `selected` being null with a non-empty `configured` never happens — resolve_device raises
    instead — so a null here means "system default", not "lookup failed".
    """
    cfg = state["cfg"]
    try:
        selected = audio.resolve_device(cfg.input_device)
        error = None
    except audio.DeviceNotFound as exc:
        selected, error = None, str(exc)

    return {"devices": audio.list_input_devices(), "configured": cfg.input_device, "selected": selected, "error": error}


@app.get("/api/config")
def get_config() -> dict:
    cfg = state["cfg"]
    return {"languages": cfg.languages, "inputDevice": cfg.input_device}


@app.put("/api/config")
def put_config(body: dict) -> dict:
    cfg = state["cfg"]
    if "languages" in body:
        langs = [str(s) for s in body["languages"]]
        if not 2 <= len(langs) <= 3:
            raise HTTPException(400, "languages must contain 2 or 3 entries")
        # A duplicate would mean translating a line into its own source language.
        if len(set(langs)) != len(langs):
            raise HTTPException(400, "languages must be distinct")
        cfg.languages = langs
    if "inputDevice" in body:
        cfg.input_device = str(body["inputDevice"])
    cfg.save()
    return get_config()


@app.post("/api/recording/start")
def start_recording() -> dict:
    if state["recorder"]:
        raise HTTPException(409, "already recording")

    cfg = state["cfg"]
    try:
        device = audio.resolve_device(cfg.input_device)
    except audio.DeviceNotFound as exc:
        raise HTTPException(400, str(exc)) from exc

    rec = audio.Recorder(device)
    rec.start(audio.new_session_path())
    state["recorder"] = rec
    return recording_status()


@app.post("/api/recording/stop")
def stop_recording() -> dict:
    rec = state["recorder"]
    if not rec:
        raise HTTPException(409, "not recording")
    path = rec.stop()
    state["recorder"] = None
    return {"recording": False, "path": str(path) if path else None}


@app.get("/api/recording/status")
def recording_status() -> dict:
    rec = state["recorder"]
    if not rec:
        return {"recording": False, "path": None, "seconds": 0.0, "peak": 0.0, "droppedBlocks": 0}
    s = rec.status()
    return {
        "recording": s.recording,
        "path": s.path,
        "seconds": round(s.seconds, 2),
        "peak": round(s.peak, 4),
        "droppedBlocks": s.dropped_blocks,
    }


if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str) -> FileResponse:
        """Client-side routing: unknown paths return index.html, real files are served as-is."""
        # Without this an unknown /api/* path would return the HTML shell with status 200, which
        # surfaces as a confusing JSON parse error in the dashboard instead of a plain 404.
        if path.startswith("api/"):
            raise HTTPException(404, "Not Found")
        candidate = DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST / "index.html")
