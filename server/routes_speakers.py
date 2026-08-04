"""Speaker names, per session and across the ones already recorded."""

from __future__ import annotations

import io
from pathlib import Path

import soundfile as sf
from fastapi import APIRouter, HTTPException, Response

from . import main

router = APIRouter()


@router.put("/api/sessions/{session_id}/speakers")
def put_speaker_names(session_id: int, body: dict) -> dict:
    for code, name in body.items():
        code, name = str(code), str(name).strip()
        main.store.set_speaker_name(session_id, code, name)
        # Naming a speaker is the only labelled data this system ever gets. Attaching it to the
        # voiceprint is what stops the next meeting asking the same question.
        if name and (centroid := main.store.voiceprint(session_id, code)):
            main.store.remember_speaker(name, centroid)
    return main.store.speaker_names(session_id)


@router.get("/api/speakers/known")
def get_known_speakers() -> list[dict]:
    counts = main.store.speaker_sessions()
    return [{"name": name, "sessions": counts.get(name, 0)} for name, _ in main.store.known_speakers()]


@router.get("/api/speakers/known/{name}/clip")
def get_speaker_clip(name: str) -> Response:
    """A few seconds of the voice behind the name, so a wrong match is audible rather than guessed."""
    sample = main.store.speaker_sample(name)
    if sample is None:
        raise HTTPException(404, "no recording for this voice")
    wav_path, start = sample
    if not Path(wav_path).is_file():
        raise HTTPException(404, f"recording not found: {wav_path}")

    with sf.SoundFile(wav_path) as f:
        f.seek(min(int(start * f.samplerate), max(len(f) - 1, 0)))
        block = f.read(main.CLIP_SECONDS * f.samplerate, dtype="int16")
        rate = f.samplerate
    buf = io.BytesIO()
    sf.write(buf, block, rate, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


@router.put("/api/speakers/known/{name}")
def rename_known_speaker(name: str, body: dict) -> list[dict]:
    new = str(body.get("name", "")).strip()
    if not new:
        raise HTTPException(400, "name required")
    main.store.rename_speaker(name, new)
    return get_known_speakers()


@router.delete("/api/speakers/known/{name}")
def delete_known_speaker(name: str) -> list[dict]:
    main.store.forget_speaker(name)
    return get_known_speakers()
