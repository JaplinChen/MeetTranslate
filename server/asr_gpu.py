"""CTranslate2 transcriber, interchangeable with the sherpa-onnx one in `asr`.

Measured on this meeting room's box (RTX 5060 Ti, 20 cores): sherpa-onnx running Whisper small on
the CPU reaches 0.57 realtime only by taking every core, which makes the machine unusable for
anything else. The same recording through CTranslate2 on the GPU runs large-v3 at 0.064 — a nine
times faster wall clock on a far better model, with the CPU free.

Only the recogniser changes. VAD and speaker embeddings stay on sherpa-onnx: they are cheap, and
they are what the live path's latency actually depends on.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

from . import asr, config

log = logging.getLogger("meettranslate.asr_gpu")


def _add_cuda_dlls() -> None:
    """Put the pip-installed CUDA runtime on PATH before CTranslate2 loads.

    `os.add_dll_directory` is not enough — CTranslate2 resolves cuBLAS and cuDNN through the
    default search order, which on Windows means PATH. Without this the model loads and then fails
    on the first encode with 'Library cublas64_12.dll is not found'.
    """
    nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia.is_dir():
        return
    dirs = [str(p) for p in nvidia.glob("*/bin") if p.is_dir()]
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])


def available() -> bool:
    """True when a CUDA device and the CTranslate2 runtime are both present."""
    try:
        _add_cuda_dlls()
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class Transcriber:
    """Same surface as `asr.Transcriber`: transcribe(samples, language) -> (text, language)."""

    def __init__(self, model: str | Path | None = None, device: str = "cuda",
                 compute_type: str = "float16", languages: list[str] | None = None,
                 hotwords: str = ""):
        _add_cuda_dlls()
        from faster_whisper import WhisperModel

        self._languages = list(languages or [])
        self._hotwords = hotwords
        name = str(model or config.gpu_model(self._languages))
        self._model = WhisperModel(name, device=device, compute_type=compute_type)
        log.info("ct2 model %s on %s/%s", name, device, compute_type)

    def transcribe(self, samples: np.ndarray, language: str) -> tuple[str, str]:
        segments, info = self._model.transcribe(
            samples.astype(np.float32),
            language=language or None,  # None means detect
            beam_size=5,
            # Hotwords are the biasing sherpa-onnx cannot do for Whisper at all.
            hotwords=self._hotwords or None,
            condition_on_previous_text=False,  # one VAD utterance at a time carries no history
        )
        text = "".join(s.text for s in segments).strip()
        detected = (info.language or language or "").strip()

        if asr.is_noise(text) or not self._allowed(detected):
            return "", detected
        return asr._post(text, detected), detected

    def _allowed(self, detected: str) -> bool:
        if not self._languages or not detected:
            return True
        base = detected.split("-")[0]
        return any(base == code.split("-")[0] for code in self._languages)


def maybe(languages: list[str], hotwords: str = "") -> Transcriber | None:
    """The GPU recogniser when this machine can run it, otherwise None so the caller falls back.

    Auto-enabled rather than configured: it is faster and more accurate on every axis measured, so
    a knob would only ever be turned one way. `MEETTRANSLATE_NO_GPU=1` exists for the case where
    the card is needed for something else.
    """
    if os.environ.get("MEETTRANSLATE_NO_GPU"):
        return None
    if not available():
        return None
    try:
        return Transcriber(languages=languages, hotwords=hotwords)
    except Exception:
        log.exception("GPU transcriber unavailable, falling back to CPU")
        return None


def hotwords_from(terms: list) -> str:
    """faster-whisper takes one string; the glossary is a list of terms."""
    return " ".join(t.source for t in terms)
