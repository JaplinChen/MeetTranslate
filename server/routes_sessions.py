"""Recorded sessions: transcripts, corrections, re-derivation and import."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import soundfile as sf
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

from . import asr, asr_gpu, config, correct, ingest, jobs, main, translate

log = logging.getLogger("meettranslate")

router = APIRouter()

# Longest single utterance a rerun will decode. VAD cuts at 20 s, so anything past this is a row
# whose end_time is wrong rather than a real utterance, and decoding it would tie up the card.
RERUN_MAX_SECONDS = 60.0


@router.get("/api/sessions")
def get_sessions() -> list[dict]:
    """Sessions, each carrying where its post-meeting pass got to.

    Carried on the list rather than fetched per session so the page can show which meetings are
    still being refined before the user picks one, instead of after.
    """
    running = jobs.states()
    return [{**s, "refine": running.get(s["id"], {"state": "idle", "error": ""})}
            for s in main.store.sessions()]


@router.get("/api/sessions/{session_id}/lines")
def get_lines(session_id: int) -> dict:
    return {"lines": main.store.lines(session_id), "speakers": main.store.speaker_names(session_id)}


@router.put("/api/sessions/{session_id}/lines/{line_id}")
def put_line(session_id: int, line_id: int, body: dict) -> dict:
    """Correct one transcript line, and learn the pair.

    The edit is the only ground truth this system ever sees — someone who was in the room saying
    what was actually said. Storing the before/after means the same mistake is fixed automatically
    everywhere it appears next time, live as well as after the fact.
    """
    source = str(body.get("source", "")).strip()
    if not source:
        raise HTTPException(400, "source required")

    before = next((l for l in main.store.lines(session_id) if l["id"] == line_id), None)
    if before is None:
        raise HTTPException(404, "no such line in this session")

    main.store.update_line(line_id, source, before["translations"])
    for wrong, right in correct.diff_terms(before["source"], source):
        main.store.add_correction(wrong, right, before["lang"])
    return {"lines": main.store.lines(session_id), "speakers": main.store.speaker_names(session_id)}


@router.get("/api/corrections")
def get_corrections() -> list[dict]:
    return [{"wrong": w, "right": r} for w, r in main.store.corrections().items()]


@router.put("/api/corrections/{wrong}")
def put_correction(wrong: str, body: dict) -> list[dict]:
    try:
        main.store.edit_correction(wrong, str(body.get("wrong", wrong)), str(body.get("right", "")))
    except KeyError as exc:
        raise HTTPException(404, f"no correction for {wrong}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_corrections()


@router.delete("/api/corrections/{wrong}")
def delete_correction(wrong: str) -> list[dict]:
    main.store.forget_correction(wrong)
    return get_corrections()


@router.post("/api/sessions/{session_id}/reprocess")
def reprocess(session_id: int) -> dict:
    """Queue a re-derivation from the recording, with the largest model and offline clustering.

    Queued rather than run inline, and through the same gate as the automatic pass: two of these at
    once would put two Whisper models on one card, and the second would be rewriting a transcript
    the first is already rewriting.
    """
    session = main.store.session(session_id)
    if not session:
        raise HTTPException(404, "no such session")
    if main.state["session"] == session_id:
        raise HTTPException(409, "session is still recording")

    wav = Path(session["wav_path"])
    if not wav.is_file():
        raise HTTPException(404, f"recording not found: {wav}")

    if not jobs.schedule(session_id, lambda cancel: main.postprocess.rewrite_session(
            main.store, session_id, wav, main.state["cfg"], main._make_translator(),
            should_stop=cancel.is_set)):
        raise HTTPException(409, "already refining this session")
    return {"session": session_id, **(jobs.state(session_id) or {})}


def _transcript(session_id: int, status: str) -> dict:
    """The shape every transcript-mutating endpoint returns.

    One helper rather than a literal per exit: the two rerun outcomes drifted apart once already,
    one returning `line` where the other returned `lines`, which the page reads straight into
    state — so a failed rerun blanked the transcript it was supposed to be fixing.
    """
    return {"lines": main.store.lines(session_id), "speakers": main.store.speaker_names(session_id),
            "status": status}


@router.post("/api/sessions/{session_id}/lines/{line_id}/rerun")
def rerun_line(session_id: int, line_id: int) -> dict:
    """Decode and translate one line again from the recording.

    The audio comes from the session row, never from the request: the caller names a line, not a
    path or an offset, so there is nothing here to point at another file. The work is bounded by
    the line's own span and takes the same GPU gate as a full pass, because this endpoint has no
    authentication in front of it and a loop over it would otherwise starve a live meeting.
    """
    if main.state["session"] == session_id:
        raise HTTPException(409, "session is still recording")
    session = main.store.session(session_id)
    line = main.store.line(line_id)
    if not session or not line or line["session_id"] != session_id:
        raise HTTPException(404, "no such line")

    wav = Path(session["wav_path"])
    if not wav.is_file():
        raise HTTPException(404, f"recording not found: {wav}")

    start = float(line["start"])
    end = line["end_time"] if line["end_time"] is not None else start + RERUN_MAX_SECONDS
    seconds = min(max(float(end) - start, 0.0), RERUN_MAX_SECONDS)
    if seconds <= 0:
        raise HTTPException(400, "line has no duration to re-run")

    with jobs.borrow_gpu():
        try:
            # Only this line's span is read, not the whole meeting: a ninety-minute wav does not
            # belong in memory to re-decode four seconds of it.
            samples, rate = sf.read(str(wav), dtype="float32", start=int(start * config.SAMPLE_RATE),
                                    frames=int(seconds * config.SAMPLE_RATE), always_2d=False)
        except Exception as exc:
            raise HTTPException(400, f"could not read the recording: {exc}") from exc
        if rate != config.SAMPLE_RATE:
            raise HTTPException(400, f"{wav.name} is {rate} Hz, expected {config.SAMPLE_RATE}")
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)

        transcriber = (asr_gpu.maybe(main.state["cfg"].languages,
                                     asr_gpu.hotwords_from(main.store.glossary()))
                       or asr.Transcriber(model_dir=main.postprocess.best_model(), quantized=False,
                                          languages=main.state["cfg"].languages))
        text, used = transcriber.transcribe(samples, line["lang"] or "")

    if not text:
        main.store.replace_line(line_id, line["source"], line["lang"], {}, "asr_failed")
        return _transcript(session_id, "asr_failed")

    text = correct.Corrector(main.store.glossary(), main.store.corrections()).fix(text)
    translator = main._make_translator()
    targets = [c for c in main.state["cfg"].languages if c != (used or line["lang"])]
    translations, status = {}, "ok"
    if translator and targets:
        try:
            translations = translator.translate(
                translate.Line(text=text, lang=used or line["lang"], speaker=line["speaker"]),
                targets, terms=main.store.glossary()).translations
        except Exception:
            log.exception("rerun translation failed for line %d", line_id)
            status = "translate_failed"

    main.store.replace_line(line_id, text, used or line["lang"], translations, status)
    return _transcript(session_id, status)


@router.get("/api/sessions/{session_id}/refine")
def refine_state(session_id: int) -> dict:
    """Where the post-meeting pass got to. `idle` means there has not been one this run."""
    return {"session": session_id, **(jobs.state(session_id) or {"state": "idle", "error": ""})}


@router.post("/api/sessions/import")
async def import_recording(request: Request, filename: str = "upload") -> dict:
    """Learn from a meeting that was recorded somewhere else.

    The upload becomes an ordinary session, so everything the room learns from a live capture — a
    voice once someone names it, a correction once someone fixes a line — is learned from a file
    the same way. Nothing downstream is told it came from an upload.
    """
    # ponytail: raw body rather than multipart, so no python-multipart dependency. Streamed to
    # disk because a meeting recording does not belong in memory.
    stem = re.sub(r"[^\w.-]", "_", Path(filename).name).strip("._") or "upload"
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    # Two imports inside the same second would otherwise share a name, and the first session would
    # end up pointing at the second one's audio.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag, n = stamp, 1
    while (config.RECORDINGS_DIR / f"import-{tag}.wav").exists():
        tag, n = f"{stamp}-{n}", n + 1
    source = config.RECORDINGS_DIR / f"import-{tag}-{stem}"
    wav = config.RECORDINGS_DIR / f"import-{tag}.wav"

    written = 0
    with source.open("wb") as out:
        async for chunk in request.stream():
            written += out.write(chunk)
    if not written:
        source.unlink(missing_ok=True)
        raise HTTPException(400, "empty upload")

    try:
        ingest.extract_audio(source, wav)
    except ValueError as exc:
        # ffmpeg creates the output before it discovers it cannot read the input.
        wav.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    finally:
        # The original video is not evidence — every stage after this reads the wav.
        source.unlink(missing_ok=True)

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    session_id = main.store.start_session(now, str(wav))
    main.store.end_session(session_id, now)
    try:
        # ponytail: synchronous — a long recording holds the request open. Worth a job queue only
        # once someone imports something long enough to time out. It still queues for the card, so
        # an import during a meeting waits rather than loading a second model beside the live one.
        with jobs.borrow_gpu():
            main.postprocess.rewrite_session(main.store, session_id, wav, main.state["cfg"],
                                             main._make_translator())
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": session_id, "lines": len(main.store.lines(session_id))}


@router.get("/api/sessions/{session_id}/markdown")
def session_markdown(session_id: int) -> PlainTextResponse:
    if not main.store.session(session_id):
        raise HTTPException(404, "no such session")
    return PlainTextResponse(main.postprocess.to_markdown(main.store, session_id),
                             media_type="text/markdown")
