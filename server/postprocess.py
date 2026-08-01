"""Post-meeting pass over the recording.

The live pipeline trades accuracy for latency: a small model, online clustering that cannot see
what comes later, and one-shot refinement. None of those constraints apply once the meeting ends,
so this re-runs the whole thing from the wav with the largest model available and clusters over
every segment at once, then rewrites the stored transcript.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from . import asr, asr_gpu, config, correct, diarize, translate
from .store import Store

log = logging.getLogger("meettranslate.postprocess")


@dataclass
class Utterance:
    start: float
    samples: np.ndarray
    speaker: str = ""
    lang: str = ""
    text: str = ""


def best_model() -> Path:
    """Largest Whisper tier present on disk. Accuracy matters here, speed does not."""
    available = config.available_whisper_models()
    if not available:
        raise FileNotFoundError(f"no Whisper model found under {config.MODELS_DIR}")
    order = list(config.WHISPER_DIRS)
    return config.WHISPER_DIRS[max(available, key=order.index)]


def segment(wav: Path) -> list[Utterance]:
    audio, rate = sf.read(str(wav), dtype="float32")
    if rate != config.SAMPLE_RATE:
        raise ValueError(f"{wav} is {rate} Hz, expected {config.SAMPLE_RATE}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    vad = asr.Vad()
    out: list[Utterance] = []
    for i in range(0, len(audio), config.BLOCK_SIZE):
        out += [Utterance(s.start, s.samples) for s in vad.push(audio[i : i + config.BLOCK_SIZE])]
    out += [Utterance(s.start, s.samples) for s in vad.flush()]
    return out


def assign_speakers(utterances: list[Utterance], diarizer: diarize.Diarizer) -> None:
    """Cluster over the whole meeting at once.

    This is what the online pass could not do: a speaker whose first few seconds were atypical
    gets merged with their later segments instead of living on as a phantom second participant.
    """
    # Same rule the live path applies in Diarizer.assign: a clip too short to embed reliably
    # inherits the previous speaker. Clustering it instead mints a phantom participant per blip —
    # on a real recording the ten idle minutes before the meeting produced fourteen of them.
    long = [u for u in utterances if len(u.samples) / config.SAMPLE_RATE >= config.MIN_EMBED_SECONDS]
    if not long:
        for u in utterances:
            u.speaker = "S1"
        return

    labels = diarize.cluster_offline([diarizer.embed(u.samples) for u in long])
    for utterance, label in zip(long, labels):
        utterance.speaker = f"S{label + 1}"

    previous = f"S{labels[0] + 1}"
    for u in utterances:
        if u.speaker:
            previous = u.speaker
        else:
            u.speaker = previous


def dominant_languages(utterances: list[Utterance]) -> dict[str, str]:
    """Majority language per speaker, computed after clustering rather than as the meeting ran."""
    counts: dict[str, dict[str, int]] = {}
    for u in utterances:
        # Text-less utterances are dropped noise; their detected language is Whisper guessing at
        # static and must not vote.
        if u.lang and u.text:
            counts.setdefault(u.speaker, {})[u.lang] = counts.setdefault(u.speaker, {}).get(u.lang, 0) + 1
    return {code: max(langs, key=langs.get) for code, langs in counts.items() if langs}


def transcribe_all(utterances: list[Utterance], transcriber: asr.Transcriber,
                   progress: Callable[[Utterance, int, int], None] | None = None) -> None:
    """Two passes: detect each speaker's language, then re-transcribe anyone who was decoded
    under a language that disagrees with their majority.

    `progress` is called after each first-pass decode. A ninety-minute recording spends most of an
    hour in that first loop, and a caller with somewhere to put partial results should not have to
    wait for the whole thing to survive an interruption.
    """
    for i, u in enumerate(utterances, 1):
        u.text, u.lang = transcriber.transcribe(u.samples, "")
        if progress:
            progress(u, i, len(utterances))

    dominant = dominant_languages(utterances)
    for u in utterances:
        want = dominant.get(u.speaker, "")
        if want and want != u.lang:
            text, used = transcriber.transcribe(u.samples, want)
            # Empty means the speaker's own language decoded this as a noise annotation, which is
            # what static sounds like to Whisper. The stray foreign-language first pass was the
            # hallucination, so drop it rather than keep it as a phantom line.
            u.text, u.lang = (text, used) if text else ("", u.lang)


def rewrite_session(store: Store, session_id: int, wav: Path, cfg: config.Config,
                    translator: translate.Translator | None = None) -> list[Utterance]:
    """Re-derive the transcript and replace the stored lines for this session."""
    # GPU first. The CPU fallback keeps float32 weights and every core: this runs after the
    # meeting, so accuracy is the only concern — but it also makes the machine unusable while it
    # runs, which is the other reason the GPU path exists.
    transcriber = asr_gpu.maybe(cfg.languages, asr_gpu.hotwords_from(store.glossary()))         or asr.Transcriber(model_dir=best_model(), quantized=False, num_threads=os.cpu_count() or 4,
                           languages=cfg.languages)
    diarizer = diarize.Diarizer(cfg=cfg)

    utterances = segment(wav)
    log.info("%d utterances from %s", len(utterances), wav.name)
    if not utterances:
        return []

    assign_speakers(utterances, diarizer)
    transcribe_all(utterances, transcriber)

    store.clear_lines(session_id)
    terms = store.glossary()
    corrector = correct.Corrector(terms, store.corrections())
    context: list[translate.Line] = []

    for u in utterances:
        if not u.text:
            continue
        u.text = corrector.fix(u.text)
        line = translate.Line(text=u.text, lang=u.lang, speaker=u.speaker)
        targets = [c for c in cfg.languages if c != u.lang]
        translations: dict[str, str] = {}
        if translator and targets:
            try:
                translations = translator.translate(line, targets, context=context[-3:], terms=terms).translations
            except Exception:
                log.exception("translation failed at %.2fs", u.start)
        store.add_line(session_id, u.start, u.speaker, u.lang, u.text, translations)
        context.append(line)

    return utterances


def to_markdown(store: Store, session_id: int) -> str:
    """Speaker-attributed transcript with every language stacked under each turn."""
    lines = store.lines(session_id)
    names = store.speaker_names(session_id)
    if not lines:
        return "# 會議紀錄\n\n（無內容）\n"

    out = ["# 會議紀錄", ""]
    speakers = sorted({l["speaker"] for l in lines})
    out += ["## 發言者", ""]
    out += [f"- **{names.get(code, code)}**" + ("" if code in names else "（未命名）") for code in speakers]
    out += ["", "## 逐字稿", ""]

    for line in lines:
        stamp = f"{int(line['start']) // 60}:{int(line['start']) % 60:02d}"
        who = names.get(line["speaker"], line["speaker"])
        out.append(f"**[{stamp}] {who}**")
        out.append(f"> {line['source']}")
        for lang, text in line["translations"].items():
            out.append(f"> _{lang}_ {text}")
        out.append("")

    return "\n".join(out) + "\n"
