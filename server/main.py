"""FastAPI entry point. Serves the built dashboard and the capture API on localhost."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import audio, config, correct, llm, postprocess, translate
from .hub import Hub
from .pipeline import Pipeline
from .store import TERM_MODES, Store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("meettranslate")

DIST = config.ROOT / "dashboard" / "dist"

state: dict = {"recorder": None, "pipeline": None, "session": None,
               "cfg": config.load(), "llm": llm.load_llm()}
store = Store()
keys = llm.KeyStore()
hub = Hub()


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.bind(asyncio.get_running_loop())
    yield
    _stop_capture()
    store.close()


app = FastAPI(title="MeetTranslate", lifespan=lifespan)

# The Vite dev server runs on its own port; the packaged app is same-origin so this is dev-only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:2886", "http://127.0.0.1:2886"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── health, devices, config ─────────────────────────────────────────────


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/devices")
def devices() -> dict:
    cfg = state["cfg"]
    try:
        selected = audio.resolve_device(cfg.input_device)
        error = None
    except audio.DeviceNotFound as exc:
        selected, error = None, str(exc)
    return {"devices": audio.list_input_devices(), "configured": cfg.input_device,
            "selected": selected, "error": error}


@app.get("/api/config")
def get_config() -> dict:
    cfg = state["cfg"]
    return {
        "languages": cfg.languages,
        "inputDevice": cfg.input_device,
        "whisperModel": cfg.whisper_model,
        "availableModels": config.available_whisper_models(),
        "pinnedLanguages": cfg.pinned_languages,
        "translatorReady": bool(_api_key_present()),
        "display": asdict(cfg.display),
    }


@app.put("/api/config")
def put_config(body: dict) -> dict:
    """Validate everything first, then apply.

    A partially applied update is worse than a rejected one: the caller sees a 400 and assumes
    nothing changed, while the running config has silently drifted.
    """
    cfg = state["cfg"]
    pending: dict = {}

    if "languages" in body:
        langs = [str(s) for s in body["languages"]]
        if not 2 <= len(langs) <= 3:
            raise HTTPException(400, "languages must contain 2 or 3 entries")
        # A duplicate would mean translating a line into its own source language.
        if len(set(langs)) != len(langs):
            raise HTTPException(400, "languages must be distinct")
        pending["languages"] = langs

    if "display" in body:
        patch = {k: v for k, v in dict(body["display"]).items() if k in config.Display.__dataclass_fields__}
        try:
            candidate = config.Display(**{**asdict(cfg.display), **patch})
            candidate.font_size = int(candidate.font_size)
            candidate.lines = int(candidate.lines)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"invalid display settings: {exc}") from exc
        if candidate.show_source not in ("top", "bottom", "hidden"):
            raise HTTPException(400, "show_source must be top, bottom or hidden")
        if not 1 <= candidate.lines <= 20:
            raise HTTPException(400, "lines must be between 1 and 20")
        if not 12 <= candidate.font_size <= 200:
            raise HTTPException(400, "font_size must be between 12 and 200")
        if candidate.theme not in ("dark", "light"):
            raise HTTPException(400, "theme must be dark or light")
        pending["display"] = candidate

    if "inputDevice" in body:
        pending["input_device"] = str(body["inputDevice"])
    if "whisperModel" in body:
        pending["whisper_model"] = str(body["whisperModel"])
    if "pinnedLanguages" in body:
        pending["pinned_languages"] = {str(k): str(v) for k, v in dict(body["pinnedLanguages"]).items()}

    for key, value in pending.items():
        setattr(cfg, key, value)
    cfg.save()
    # The subtitle page applies display changes live rather than needing a reload on the TV.
    hub.publish({"type": "config", "languages": cfg.languages, "display": asdict(cfg.display)})
    return get_config()


# ── glossary ────────────────────────────────────────────────────────────


def _term_json(t) -> dict:
    return {"id": t.id, "source": t.source, "lang": t.lang, "mode": t.mode,
            "category": t.category, "targets": t.targets}


@app.get("/api/glossary")
def get_glossary() -> list[dict]:
    return [_term_json(t) for t in store.glossary()]


@app.post("/api/glossary")
def post_glossary(body: dict) -> list[dict]:
    mode = str(body.get("mode", "translate"))
    if mode not in TERM_MODES:
        raise HTTPException(400, f"mode must be one of {TERM_MODES}")
    try:
        store.add_term(
            source=str(body.get("source", "")),
            targets={str(k): str(v) for k, v in dict(body.get("targets", {})).items()},
            lang=str(body.get("lang", "")),
            mode=mode,
            category=str(body.get("category", "")),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_glossary()


@app.get("/api/glossary/collisions")
def get_collisions(source: str) -> dict:
    """What adding this term would overwrite in the meetings already recorded.

    Called before the term is added, because afterwards is too late to be useful: adding 料號
    rewrote 料耗 — a real term of the trade — forty-two times, silently.
    """
    source = source.strip()
    if not source:
        raise HTTPException(400, "source required")
    known = {t.source for t in store.glossary()} | {source}
    hits = correct.collisions(source, store.transcript_text(), known)
    return {"source": source,
            "collisions": [{"text": w, "count": n} for w, n in
                           sorted(hits.items(), key=lambda kv: -kv[1])]}


@app.delete("/api/glossary")
def delete_glossary(source: str, lang: str = "") -> list[dict]:
    store.remove_term(source, lang)
    return get_glossary()


# ── LLM settings and key rotation ───────────────────────────────────────


@app.get("/api/translate/config")
def get_llm_config() -> dict:
    return state["llm"].to_json()


@app.put("/api/translate/config")
def put_llm_config(body: dict) -> dict:
    cfg: llm.LlmConfig = state["llm"]
    cfg.apply(body)
    cfg.save()
    return cfg.to_json()


@app.get("/api/keyproxy/keys")
def get_keys() -> list[dict]:
    return keys.list()


@app.post("/api/keyproxy/keys")
def post_key(body: dict) -> list[dict]:
    try:
        return keys.add(str(body.get("provider", "")), str(body.get("apiKey", "")), str(body.get("account", "")))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/keyproxy/keys/{provider}/{index}")
def delete_key(provider: str, index: int) -> list[dict]:
    try:
        return keys.remove(provider, index)
    except IndexError as exc:
        raise HTTPException(404, str(exc)) from exc


# ── sessions ────────────────────────────────────────────────────────────


@app.get("/api/sessions")
def get_sessions() -> list[dict]:
    return store.sessions()


@app.get("/api/sessions/{session_id}/lines")
def get_lines(session_id: int) -> dict:
    return {"lines": store.lines(session_id), "speakers": store.speaker_names(session_id)}


@app.put("/api/sessions/{session_id}/speakers")
def put_speaker_names(session_id: int, body: dict) -> dict:
    for code, name in body.items():
        code, name = str(code), str(name).strip()
        store.set_speaker_name(session_id, code, name)
        # Naming a speaker is the only labelled data this system ever gets. Attaching it to the
        # voiceprint is what stops the next meeting asking the same question.
        if name and (centroid := store.voiceprint(session_id, code)):
            store.remember_speaker(name, centroid)
    return store.speaker_names(session_id)


@app.get("/api/speakers/known")
def get_known_speakers() -> list[dict]:
    return [{"name": name} for name, _ in store.known_speakers()]


@app.delete("/api/speakers/known/{name}")
def delete_known_speaker(name: str) -> list[dict]:
    store.forget_speaker(name)
    return get_known_speakers()


@app.put("/api/sessions/{session_id}/lines/{line_id}")
def put_line(session_id: int, line_id: int, body: dict) -> dict:
    """Correct one transcript line, and learn the pair.

    The edit is the only ground truth this system ever sees — someone who was in the room saying
    what was actually said. Storing the before/after means the same mistake is fixed automatically
    everywhere it appears next time, live as well as after the fact.
    """
    source = str(body.get("source", "")).strip()
    if not source:
        raise HTTPException(400, "source required")

    before = next((l for l in store.lines(session_id) if l["id"] == line_id), None)
    if before is None:
        raise HTTPException(404, "no such line in this session")

    store.update_line(line_id, source, before["translations"])
    for wrong, right in correct.diff_terms(before["source"], source):
        store.add_correction(wrong, right, before["lang"])
    return {"lines": store.lines(session_id), "speakers": store.speaker_names(session_id)}


@app.get("/api/corrections")
def get_corrections() -> list[dict]:
    return [{"wrong": w, "right": r} for w, r in store.corrections().items()]


@app.delete("/api/corrections/{wrong}")
def delete_correction(wrong: str) -> list[dict]:
    store.forget_correction(wrong)
    return get_corrections()


@app.post("/api/sessions/{session_id}/reprocess")
def reprocess(session_id: int) -> dict:
    """Re-derive the transcript from the recording with the largest model and offline clustering."""
    session = store.session(session_id)
    if not session:
        raise HTTPException(404, "no such session")
    if state["session"] == session_id:
        raise HTTPException(409, "session is still recording")

    wav = Path(session["wav_path"])
    if not wav.is_file():
        raise HTTPException(404, f"recording not found: {wav}")

    try:
        utterances = postprocess.rewrite_session(store, session_id, wav, state["cfg"], _make_translator())
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"utterances": len(utterances), "lines": len(store.lines(session_id))}


@app.get("/api/sessions/{session_id}/markdown")
def session_markdown(session_id: int) -> PlainTextResponse:
    if not store.session(session_id):
        raise HTTPException(404, "no such session")
    return PlainTextResponse(postprocess.to_markdown(store, session_id), media_type="text/markdown")


# ── capture ─────────────────────────────────────────────────────────────


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


def _stop_capture() -> dict:
    rec, pipe = state["recorder"], state["pipeline"]
    path = rec.stop() if rec else None
    if pipe:
        pipe.join()
    if state["session"] is not None:
        store.end_session(state["session"], time.strftime("%Y-%m-%dT%H:%M:%S"))
    state.update(recorder=None, pipeline=None, session=None)
    return {"recording": False, "path": str(path) if path else None}


@app.post("/api/recording/start")
def start_recording() -> dict:
    if state["recorder"]:
        raise HTTPException(409, "already recording")

    cfg = state["cfg"]
    try:
        candidates = audio.candidate_devices(cfg.input_device)
    except audio.DeviceNotFound as exc:
        raise HTTPException(400, str(exc)) from exc

    path = audio.new_session_path()
    session_id = store.start_session(time.strftime("%Y-%m-%dT%H:%M:%S"), str(path))

    pipe = Pipeline(cfg, store, session_id, _make_translator(), hub.publish)
    rec = audio.Recorder(candidates, tap=pipe.tap)
    try:
        rec.start(path)
    except RuntimeError as exc:
        store.end_session(session_id, time.strftime("%Y-%m-%dT%H:%M:%S"))
        raise HTTPException(400, str(exc)) from exc
    pipe.start()
    log.info("capturing from device %s at %s", rec.device, rec.native_format)

    state.update(recorder=rec, pipeline=pipe, session=session_id)
    return recording_status()


@app.post("/api/recording/stop")
def stop_recording() -> dict:
    if not state["recorder"]:
        raise HTTPException(409, "not recording")
    return _stop_capture()


@app.get("/api/recording/status")
def recording_status() -> dict:
    rec, pipe = state["recorder"], state["pipeline"]
    if not rec:
        return {"recording": False, "path": None, "seconds": 0.0, "peak": 0.0,
                "droppedBlocks": 0, "sessionId": None, "backlog": 0, "errors": 0}
    s = rec.status()
    return {
        "recording": s.recording,
        "path": s.path,
        "seconds": round(s.seconds, 2),
        "peak": round(s.peak, 4),
        "droppedBlocks": s.dropped_blocks,
        "sessionId": state["session"],
        "backlog": pipe.tap.qsize() if pipe else 0,
        "errors": pipe.errors if pipe else 0,
    }


@app.websocket("/ws/live")
async def live(ws: WebSocket) -> None:
    await ws.accept()
    queue_ = hub.subscribe()
    try:
        cfg = state["cfg"]
        await ws.send_json({"type": "config", "languages": cfg.languages, "display": asdict(cfg.display)})
        while True:
            await ws.send_json(await queue_.get())
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe(queue_)


# ── static dashboard ────────────────────────────────────────────────────

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
