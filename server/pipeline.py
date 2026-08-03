"""Wires capture into subtitles: VAD -> speaker -> transcribe -> translate -> emit.

Order matters and is not interchangeable. The speaker must be identified *before* transcription
because Whisper's language is chosen per recognizer, and forcing the wrong one does not degrade —
it collapses into repeated filler. Speaker embeddings need only the waveform, so putting
clustering first costs no extra latency.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import asr, asr_gpu, config, correct, diarize, translate
from .store import Store

log = logging.getLogger("meettranslate.pipeline")

# Utterances of context sent with each translation request.
CONTEXT_LINES = 3
# Blocks the pipeline may fall behind before it starts dropping audio. 600 blocks = 60 s.
TAP_CAPACITY = 600
# Warn once the backlog passes this; a sustained backlog means the realtime factor is above 1
# and subtitles will drift further behind for the rest of the meeting.
BACKLOG_WARN = 100


@dataclass
class Emitted:
    """One subtitle line as the browser sees it."""

    id: int
    start: float
    speaker: str
    lang: str
    source: str
    translations: dict[str, str] = field(default_factory=dict)
    refined: bool = False

    def event(self, kind: str) -> dict:
        return {
            "type": kind,
            "line": {
                "id": self.id,
                "start": round(self.start, 2),
                "speaker": self.speaker,
                "lang": self.lang,
                "source": self.source,
                "translations": self.translations,
                "refined": self.refined,
            },
        }


class Pipeline:
    """Consumes audio blocks on a worker thread and emits subtitle events."""

    def __init__(self, cfg: config.Config, store: Store, session_id: int,
                 translator: translate.Translator | None,
                 emit: Callable[[dict], None]):
        self._cfg = cfg
        self._store = store
        self._session = session_id
        self._translator = translator
        self._emit = emit

        self.tap: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=TAP_CAPACITY)
        self._vad = asr.Vad()
        # GPU first: measured on this box it is both faster and markedly more accurate. Hotwords
        # are read once here — a term added mid-meeting biases nothing until the next session, but
        # the post-decode corrector picks it up immediately, which is the half that matters.
        self._transcriber = (asr_gpu.maybe(cfg.languages,
                                           asr_gpu.hotwords_from(store.glossary()))
                             or asr.Transcriber(model_dir=cfg.whisper_dir(),
                                                languages=cfg.languages))
        self._diarizer = diarize.Diarizer(cfg=cfg, known=diarize.load_known(store))
        self._thread: threading.Thread | None = None

        self._context: list[translate.Line] = []
        # (line id, start seconds, line) of the utterance eligible for one refinement pass.
        self._previous: tuple[int, float, translate.Line] | None = None
        self.backlog_peak = 0
        self.errors = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()

    def join(self, timeout: float = 30) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        try:
            while (block := self.tap.get()) is not None:
                self.backlog_peak = max(self.backlog_peak, self.tap.qsize())
                if self.tap.qsize() == BACKLOG_WARN:
                    log.warning("pipeline backlog %d blocks — realtime factor above 1", self.tap.qsize())
                for segment in self._vad.push(block):
                    self._handle(segment)
            for segment in self._vad.flush():
                self._handle(segment)
        except Exception:  # a crashed pipeline must not take the recording with it
            log.exception("pipeline stopped")

    def _handle(self, segment: asr.Segment) -> None:
        try:
            speaker = self._diarizer.assign(segment.samples)
            # A voice the room already knows arrives named. The centroid is stored either way, so
            # naming an unknown speaker afterwards is enough to recognise them next time.
            self._store.save_voiceprint(self._session, speaker.code, speaker.centroid.tobytes())
            if name := self._diarizer.recognised.pop(speaker.code, ""):
                self._store.set_speaker_name(self._session, speaker.code, name)
            forced = self._diarizer.language_for(speaker)
            text, used = self._transcriber.transcribe(segment.samples, forced)
            if not text:
                return
            self._diarizer.observe_language(speaker, used)
            # The glossary is read per utterance so a term added mid-meeting takes effect at once,
            # which is how the glossary page is used in practice.
            text = correct.Corrector(self._store.glossary(),
                                     self._store.corrections()).fix(text)

            line = translate.Line(text=text, lang=used or forced, speaker=speaker.code)
            targets = [c for c in self._cfg.languages if c != line.lang]

            result = self._translate(line, targets)

            line_id = self._store.add_line(
                self._session, segment.start, speaker.code, line.lang, text, result.translations
            )
            self._emit(Emitted(line_id, segment.start, speaker.code, line.lang, text,
                               result.translations).event("line"))

            self._apply_refinement(result)

            self._previous = (line_id, segment.start, line)
            self._context = (self._context + [line])[-CONTEXT_LINES:]
        except Exception:
            self.errors += 1
            log.exception("segment at %.2fs failed", segment.start)

    def _translate(self, line: translate.Line, targets: list[str]) -> translate.Result:
        if not self._translator or not targets:
            return translate.Result({})
        prev_line = self._previous[2] if self._previous else None
        return self._translator.translate(
            line, targets, context=self._context, previous=prev_line, terms=self._store.glossary()
        )

    def _apply_refinement(self, result: translate.Result) -> None:
        """Rewrite the previous line if the model judged it wrong in hindsight.

        Each line gets exactly one chance: `_previous` advances every segment, so a corrected line
        is never revisited. Subtitles that keep shifting are harder to read than subtitles that are
        slightly off, which is why this is one-shot rather than iterative.
        """
        if not self._previous:
            return
        if not (result.previous_source or result.previous_translations):
            return

        prev_id, prev_start, prev_line = self._previous
        source = result.previous_source or prev_line.text
        self._store.update_line(prev_id, source, result.previous_translations)
        self._emit(Emitted(prev_id, prev_start, prev_line.speaker, prev_line.lang, source,
                           result.previous_translations, refined=True).event("update"))
