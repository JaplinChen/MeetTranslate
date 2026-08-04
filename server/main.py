"""FastAPI entry point. Serves the built dashboard and the capture API on localhost.

The endpoints live in the `routes_*` modules; what stays here is the app, the process-wide
singletons they read through, and the handful of helpers that touch more than one of them.
Routers reach those through this module at call time rather than importing the objects, so
swapping `main.store` or `main.postprocess` under a running app takes effect everywhere.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, jobs, llm, postprocess, translate
from .hub import Hub
from .store import Store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("meettranslate")

DIST = config.ROOT / "dashboard" / "dist"
CLIP_SECONDS = 4

state: dict = {"recorder": None, "pipeline": None, "session": None, "gpu": False,
               "cfg": config.load(), "llm": llm.load_llm()}
store = Store()
keys = llm.KeyStore()
hub = Hub()


def _api_key_present() -> bool:
    """Whether translation can run. Separate from _api_key so a status check never burns a rotation
    slot — next_key() advances the cursor and increments the key's request count."""
    cfg: llm.LlmConfig = state["llm"]
    pooled = any(k["provider"] == cfg.provider for k in keys.list())
    return pooled or bool(cfg.api_key) or bool(os.environ.get("ANTHROPIC_API_KEY"))


def _api_key() -> str:
    """Key precedence: rotation pool, then the LLM settings page, then the environment."""
    cfg: llm.LlmConfig = state["llm"]
    return keys.next_key(cfg.provider) or cfg.api_key or os.environ.get("ANTHROPIC_API_KEY", "")


def _make_translator() -> translate.Translator | None:
    """No key configured is a supported mode: transcription still runs, translations stay empty."""
    key = _api_key()
    if not key:
        log.warning("no API key configured — transcribing without translation")
        return None
    return translate.Translator(key, model=state["llm"].model or "claude-opus-5")


def _refine(session_id: int, wav: Path) -> None:
    """Queue the post-meeting pass for a session that just ended.

    This used to be something a person had to remember. An imported recording was refined on the
    way in (see `import_recording`) while a meeting the room actually captured was not, so the same
    audio produced a better transcript when uploaded through the dashboard than when recorded in
    the room it was built for. Whether a transcript got the large model, offline clustering and the
    per-speaker language pass came down to whether anyone clicked a button.
    """
    def run(cancel: threading.Event) -> None:
        postprocess.rewrite_session(store, session_id, wav, state["cfg"], _make_translator(),
                                    should_stop=cancel.is_set)

    if not jobs.schedule(session_id, run):
        log.info("session %d is already being refined", session_id)


def _stop_capture(refine: bool = True) -> dict:
    rec, pipe = state["recorder"], state["pipeline"]
    session_id, holds_gpu = state["session"], state["gpu"]
    path = rec.stop() if rec else None
    if pipe:
        pipe.join()
    if session_id is not None:
        store.end_session(session_id, time.strftime("%Y-%m-%dT%H:%M:%S"))
    state.update(recorder=None, pipeline=None, session=None, gpu=False)
    # Released before scheduling, or the pass would wait on a gate this thread still holds.
    if holds_gpu:
        jobs.release_gpu()
    # The session row, not the recorder's return value: a stop that fails to hand back a path would
    # otherwise skip the refine silently, which is the exact failure this whole change removes.
    if refine and session_id is not None:
        session = store.session(session_id)
        if session and Path(session["wav_path"]).exists():
            _refine(session_id, Path(session["wav_path"]))
        else:
            log.warning("session %d has no recording on disk, not refining", session_id)
    return {"recording": False, "path": str(path) if path else None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.bind(asyncio.get_running_loop())
    yield
    # No refine on the way out: the worker is a daemon thread, so scheduling one here would either
    # be killed halfway or hold the process open past the point the user asked it to stop.
    _stop_capture(refine=False)
    jobs.cancel_all(wait=2.0)
    store.close()


app = FastAPI(title="MeetTranslate", lifespan=lifespan)

# The Vite dev server runs on its own port; the packaged app is same-origin so this is dev-only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:2886", "http://127.0.0.1:2886"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Imported here, not at the top: each router reads `main.store` and friends, so they have to exist
# before the modules are loaded.
from . import (  # noqa: E402
    routes_capture, routes_core, routes_glossary, routes_llm, routes_sessions, routes_speakers,
)

for _r in (routes_core, routes_glossary, routes_llm, routes_sessions, routes_speakers,
           routes_capture):
    app.include_router(_r.router)

# Tests and older callers reach these through main.
_transcript = routes_sessions._transcript
RERUN_MAX_SECONDS = routes_sessions.RERUN_MAX_SECONDS


# ── static dashboard ────────────────────────────────────────────────────

if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def spa(path: str) -> FileResponse:
        """Client-side routing: unknown paths return index.html, real files are served as-is."""
        # Without this an unknown /api/* path would return the HTML shell with status 200, which
        # surfaces as a confusing JSON parse error in the dashboard instead of a plain 404.
        #
        # Every method, not just GET: a POST to an endpoint this build does not have used to fall
        # through to a GET-only route and come back as a bare 405, which reads as "wrong method"
        # when the truth is "no such endpoint" — the shape a stale server takes.
        if path.startswith("api/"):
            raise HTTPException(404, "Not Found")
        candidate = DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST / "index.html")
