"""Which weights get loaded, and what the recogniser does with a language it was not given."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import asr, asr_gpu, config


def test_weight_selection_prefers_quantized_for_live(tmp: Path) -> None:
    """Live capture wants int8; postprocess wants float32 and falls back if it is absent."""
    d = config.MODELS_DIR / "sherpa-onnx-whisper-tiny"
    if not d.is_dir():
        print("  (skipped: whisper model not present)")
        return

    live_enc, _, _ = asr.Transcriber(model_dir=d)._paths()
    assert live_enc.endswith(".int8.onnx"), live_enc

    slow_enc, _, _ = asr.Transcriber(model_dir=d, quantized=False)._paths()
    assert slow_enc.endswith(".onnx") and not slow_enc.endswith(".int8.onnx"), slow_enc

    assert 2 <= asr.default_threads() <= 4


def test_gpu_backend_declines_cleanly_when_disabled(tmp: Path) -> None:
    """The GPU path must be optional: every caller falls back to sherpa-onnx when it says no."""
    original = os.environ.get("MEETTRANSLATE_NO_GPU")
    try:
        os.environ["MEETTRANSLATE_NO_GPU"] = "1"
        assert asr_gpu.maybe(["zh", "en"]) is None
    finally:
        os.environ.pop("MEETTRANSLATE_NO_GPU", None)
        if original is not None:
            os.environ["MEETTRANSLATE_NO_GPU"] = original


def test_autodetect_reports_the_language(tmp: Path) -> None:
    """Auto-detect must return which language it decoded in.

    Without this a speaker's language can never be established: every utterance would report ''
    and the pipeline would stay on auto-detect for the whole meeting, which is the exact fragility
    the per-speaker language design exists to avoid.
    """
    wav = config.MODELS_DIR / "sherpa-onnx-whisper-tiny" / "test_wavs" / "1.wav"
    if not wav.is_file():
        print("  (skipped: whisper test wav not present)")
        return

    import soundfile as sf

    audio, _ = sf.read(str(wav), dtype="float32")
    tr = asr.Transcriber(model_dir=config.MODELS_DIR / "sherpa-onnx-whisper-tiny")

    text, detected = tr.transcribe(audio, "")
    assert text
    assert detected == "en", f"auto-detect reported {detected!r}"


def test_long_utterance_is_not_truncated(tmp: Path) -> None:
    """Whisper drops everything past 30 s; the decoder must chunk rather than lose speech.

    Exercises `_decode` rather than `transcribe`, because the only test audio available is two
    short clips and anything long enough to need chunking has to repeat them — which is genuinely
    degenerate, and `transcribe` now refuses degenerate output. That refusal is right for a
    transcript and makes the fixture useless for measuring length, so the two are tested apart.
    """
    clips = [config.MODELS_DIR / "sherpa-onnx-whisper-tiny" / "test_wavs" / f"{n}.wav"
             for n in (0, 1)]
    if not all(c.is_file() for c in clips):
        print("  (skipped: whisper test wavs not present)")
        return

    import soundfile as sf

    audio = [sf.read(str(c), dtype="float32")[0] for c in clips]
    tr = asr.Transcriber(model_dir=config.MODELS_DIR / "sherpa-onnx-whisper-tiny")

    short, _ = tr._decode(audio[1], "en")
    # Past the 25 s the decoder allows per pass, so it must split and every part must contribute.
    long_text, _ = tr._decode(np.concatenate([audio[1], audio[0], audio[1]]), "en")

    assert short, "baseline transcription is empty"
    assert len(long_text) > len(short) * 1.5, (len(short), len(long_text))
