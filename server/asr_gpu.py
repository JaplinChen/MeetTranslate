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

# Utterances decoded together. Thirty-two fits in 16 GB beside a large-v3 in float16.
BATCH_SIZE = 32
# Silence inserted between utterances when they are laid end to end for batching. Every gap is
# real audio through the encoder, so it stays as short as the boundaries tolerate.
BATCH_GAP_SECONDS = 0.2
# Greedy decoding measurably dropped whole utterances, so this is 5 as that note anticipated.
# Alternating four runs over 6.7 minutes of a real morning meeting, 47 utterances:
#
#     beam 1   12.9s   2 utterances decoded to nothing   1528 characters
#     beam 5   13.2s   0                                 1669
#
# One of the two it silently dropped was a hundred characters of customer-visit detail. The cost
# is 2-4% wall clock: at batch 32 the GPU is not compute-bound, so the wider search rides along.
BEAM_SIZE = 5


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
        self._batched = None
        log.info("ct2 model %s on %s/%s", name, device, compute_type)

    def set_hotwords(self, hotwords: str) -> None:
        """Re-bias without reloading the model.

        A term added during a meeting used to bias nothing until the next one, because the string
        was baked in when the recogniser was built. Only the prompt text changes here; the weights
        are untouched, so this costs nothing and can run between utterances.
        """
        self._hotwords = hotwords

    def transcribe_many(self, clips: list[np.ndarray], language: str) -> list[tuple[str, str]]:
        """Decode many utterances in one pass, keeping every boundary.

        Whisper's encoder always processes a thirty-second window, so a five-second utterance
        costs the same as a thirty-second one — and a meeting is thousands of short utterances.
        Measured on this box: 0.186 realtime one at a time against 0.045 batched.

        The clips are laid end to end with a second of silence between them and handed over with
        `clip_timestamps`, which is what keeps the boundaries. Without it the batching pipeline
        applies its own VAD and returns five segments where there were twenty-one — and speaker
        identity, per-speaker language and the subtitle line are all pinned to our boundaries, so
        letting the model re-segment would take the transcript apart.
        """
        if not clips:
            return []

        gap = np.zeros(int(BATCH_GAP_SECONDS * config.SAMPLE_RATE), dtype=np.float32)
        spans, parts, at = [], [], 0.0
        for clip in clips:
            seconds = len(clip) / config.SAMPLE_RATE
            spans.append({"start": at, "end": at + seconds})
            parts += [clip.astype(np.float32), gap]
            at += seconds + BATCH_GAP_SECONDS

        from faster_whisper import BatchedInferencePipeline

        if self._batched is None:
            self._batched = BatchedInferencePipeline(model=self._model)
        segments, info = self._batched.transcribe(
            np.concatenate(parts), language=language or None, beam_size=BEAM_SIZE,
            batch_size=BATCH_SIZE, vad_filter=False, clip_timestamps=spans,
            hotwords=self._hotwords or None, condition_on_previous_text=False,
        )

        # Each segment is placed by its midpoint, so a decode that runs slightly over its clip
        # still lands on the utterance it came from.
        texts = ["" for _ in clips]
        for seg in segments:
            middle = (seg.start + seg.end) / 2
            for i, span in enumerate(spans):
                if span["start"] <= middle <= span["end"]:
                    texts[i] = (texts[i] + seg.text).strip()
                    break

        detected = (info.language or language or "").strip()
        return [self._judge(text, detected) for text in texts]

    def _judge(self, text: str, detected: str) -> tuple[str, str]:
        if (asr.is_noise(text) or asr.is_hallucination(text) or asr.is_degenerate(text)
                or not self._allowed(detected)):
            return "", detected
        return asr._post(text, detected), detected

    def transcribe(self, samples: np.ndarray, language: str) -> tuple[str, str]:
        segments, info = self._model.transcribe(
            samples.astype(np.float32),
            language=language or None,  # None means detect
            beam_size=BEAM_SIZE,
            # Hotwords are the biasing sherpa-onnx cannot do for Whisper at all.
            hotwords=self._hotwords or None,
            condition_on_previous_text=False,  # one VAD utterance at a time carries no history
        )
        text = "".join(s.text for s in segments).strip()
        detected = (info.language or language or "").strip()

        # Same three refusals as the sherpa path, including the collapse check: a first-pass
        # auto-detect that returns 產品 產品 產品 產品 must not have the language it invented for
        # that counted as evidence of what the speaker speaks.
        return self._judge(text, detected)

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


# Characters of glossary allowed into the decoder prompt. Whisper reserves half its 448-token
# context for prompt text, so hotwords past ~224 tokens are silently dropped — and it drops the
# tail, which is whichever terms happen to sort last. Budgeting in characters rather than tokens
# is deliberately pessimistic: one token per character is the worst case (Chinese), so 200 stays
# inside the window even for a glossary that is entirely CJK, and leaves room for the scaffolding
# faster-whisper wraps around it.
HOTWORD_BUDGET = 200


def hotwords_from(terms: list) -> str:
    """faster-whisper takes one string; the glossary is a list of terms.

    Two things are filtered out. `protect` terms, because that mode means "this word is real, do
    not rewrite it" — 才夠 is registered only to shield it from the corrector, and biasing the
    decoder toward an ordinary word would manufacture the very mistake the glossary entry exists to
    prevent. And anything past the budget, because the alternative is Whisper truncating it for us,
    without saying so.
    """
    usable = [t for t in terms if getattr(t, "mode", "") != "protect"]
    kept, used = [], 0
    for term in usable:
        cost = len(term.source) + 1  # the joining space
        if used + cost > HOTWORD_BUDGET:
            continue
        kept.append(term.source)
        used += cost
    if len(kept) < len(usable):
        # Named, not counted: knowing which terms lost their bias is the difference between
        # diagnosing a recognition complaint and guessing at it.
        dropped = [t.source for t in usable if t.source not in kept]
        log.warning("glossary exceeds the %d-character hotword budget; %d term(s) not biased: %s",
                    HOTWORD_BUDGET, len(dropped), ", ".join(dropped[:20]))
    return " ".join(kept)
