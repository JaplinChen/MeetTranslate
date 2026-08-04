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

# Utterances held back for a second attempt once their speaker's language is known. Each one keeps
# its raw float32 audio — 20 s of it is about 1.3 MB — so this is a memory budget as much as a
# policy one. Twenty-four is roughly thirty seconds of held speech spread across the room, past
# which a meeting is failing to decode so consistently that retrying is not the answer.
RETRY_BUFFER = 24
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
    status: str = "ok"

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
                "status": self.status,
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
        # GPU first: measured on this box it is both faster and markedly more accurate.
        self._hotwords = asr_gpu.hotwords_from(store.glossary())
        self._transcriber = (asr_gpu.maybe(cfg.languages, self._hotwords)
                             or asr.Transcriber(model_dir=cfg.whisper_dir(),
                                                languages=cfg.languages))
        self._diarizer = diarize.Diarizer(cfg=cfg, known=diarize.load_known(store))
        self._thread: threading.Thread | None = None

        self._context: list[translate.Line] = []
        # (line id, start seconds, line) of the utterance eligible for one refinement pass.
        self._previous: tuple[int, float, translate.Line] | None = None
        # Utterances that decoded to nothing, held for one retry once their speaker's language is
        # settled: (segment, speaker, language already tried).
        self._held: list[tuple[asr.Segment, diarize.Speaker, str]] = []
        self.recovered = 0
        self.dropped = 0
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
            self._drain_held()
        except Exception:  # a crashed pipeline must not take the recording with it
            log.exception("pipeline stopped")

    def _drain_held(self) -> None:
        """Last attempt at whatever is still held when the meeting ends.

        By now every speaker has said all they are going to, so a language that never settled never
        will. Anything still failing is left to the post-meeting pass, which re-derives the whole
        recording anyway — this is about not throwing away what one more try would recover.
        """
        for segment, speaker, tried in list(self._held):
            language = self._diarizer.language_for(speaker)
            if language and language != tried:
                self._recover(segment, speaker, language)
            else:
                self.dropped += 1
        self._held.clear()
        if self.recovered or self.dropped:
            log.info("held utterances: %d recovered, %d dropped", self.recovered, self.dropped)

    def _handle(self, segment: asr.Segment) -> None:
        try:
            speaker = self._diarizer.assign(segment.samples)
            # A voice the room already knows arrives named. The centroid is stored either way, so
            # naming an unknown speaker afterwards is enough to recognise them next time.
            self._store.save_voiceprint(self._session, speaker.code, speaker.centroid.tobytes())
            if name := self._diarizer.recognised.pop(speaker.code, ""):
                self._store.set_speaker_name(self._session, speaker.code, name)
            # The glossary is read per utterance so a term added mid-meeting takes effect at once,
            # which is how the glossary page is used in practice. Read before decoding, not after,
            # so the same read biases the recogniser as well as correcting what it returns —
            # a term used to reach only the corrector and bias nothing until the next meeting.
            terms = self._store.glossary()
            self._rebias(terms)

            forced = self._diarizer.language_for(speaker)
            text, used = self._transcriber.transcribe(segment.samples, forced)
            if not text:
                # Held rather than dropped. The post-meeting pass recovered 992 real lines this way
                # across seven interviews — a decode that fails under one language routinely
                # succeeds under the speaker's own, and the live path used to bin them in silence.
                self._hold(segment, speaker, forced)
                return
            self._diarizer.observe_language(speaker, used)
            text = correct.Corrector(terms, self._store.corrections()).fix(text)

            line = translate.Line(text=text, lang=used or forced, speaker=speaker.code)
            targets = [c for c in self._cfg.languages if c != line.lang]

            # A translation that fails must cost the translation, not the utterance. This used to
            # raise into the handler's catch-all, so an API hiccup dropped the whole line — the
            # room saw nothing at all where it should have seen the original text untranslated.
            status = "ok"
            try:
                result = self._translate(line, targets)
            except Exception:
                log.exception("translation failed at %.2fs", segment.start)
                result, status = translate.Result({}), "translate_failed"

            line_id = self._store.add_line(
                self._session, segment.start, speaker.code, line.lang, text, result.translations,
                status=status, end_time=segment.start + segment.duration,
            )
            self._emit(Emitted(line_id, segment.start, speaker.code, line.lang, text,
                               result.translations, status=status).event("line"))

            self._apply_refinement(result)

            self._previous = (line_id, segment.start, line)
            self._context = (self._context + [line])[-config.CONTEXT_LINES:]

            # Only now, with this speaker's language possibly just settled, is a retry worth
            # spending GPU on. Doing it here also keeps it off the path of a meeting that is
            # decoding fine, where `_held` is empty and this costs one list check.
            self._retry_held(speaker)
        except Exception:
            self.errors += 1
            log.exception("segment at %.2fs failed", segment.start)

    def _hold(self, segment: asr.Segment, speaker: diarize.Speaker, tried: str) -> None:
        """Keep a failed utterance for one more attempt, oldest evicted first."""
        if len(self._held) >= RETRY_BUFFER:
            evicted = self._held.pop(0)
            self.dropped += 1
            log.info("retry buffer full, giving up on the utterance at %.2fs", evicted[0].start)
        self._held.append((segment, speaker, tried))

    def _retry_held(self, speaker: diarize.Speaker) -> None:
        """Re-decode this speaker's held utterances now that their language may have settled.

        Only theirs, and only when the language to try differs from the one that already failed —
        re-running the same audio under the same language would produce the same nothing. Each
        utterance gets exactly one retry whatever the outcome, so a room full of noise cannot build
        a backlog of audio the pipeline keeps paying to decode.
        """
        language = self._diarizer.language_for(speaker)
        if not language:
            return
        ready = [held for held in self._held
                 if held[1].code == speaker.code and held[2] != language]
        for held in ready:
            self._held.remove(held)
            self._recover(held[0], speaker, language)

    def _recover(self, segment: asr.Segment, speaker: diarize.Speaker, language: str) -> None:
        # Never raises. The caller has already taken this utterance off the held list, so an
        # exception escaping here would lose it without even counting it — the silent drop this
        # whole retry path exists to remove. It also must not fail the live segment that triggered
        # the retry: recovering an old utterance is strictly a bonus on top of that one.
        try:
            self._recover_once(segment, speaker, language)
        except Exception:
            self.dropped += 1
            log.exception("retrying the utterance at %.2fs failed", segment.start)

    def _recover_once(self, segment: asr.Segment, speaker: diarize.Speaker, language: str) -> None:
        text, used = self._transcriber.transcribe(segment.samples, language)
        if not text:
            self.dropped += 1
            return

        # Deliberately not observe_language: this segment is being decoded a second time, and
        # letting it vote again would count one utterance twice toward what this speaker speaks.
        text = correct.Corrector(self._store.glossary(), self._store.corrections()).fix(text)
        line = translate.Line(text=text, lang=used or language, speaker=speaker.code)
        targets = [c for c in self._cfg.languages if c != line.lang]

        translations, status = {}, "ok"
        if self._translator and targets:
            try:
                # No context and no `previous`: this utterance is arriving out of order, so the
                # surrounding lines are not the ones that surrounded it, and offering them as
                # context would mislead the translator rather than help it.
                translations = self._translator.translate(
                    line, targets, terms=self._store.glossary()).translations
            except Exception:
                log.exception("late translation failed at %.2fs", segment.start)
                status = "translate_failed"

        line_id = self._store.add_line(self._session, segment.start, speaker.code, line.lang, text,
                                       translations, status=status,
                                       end_time=segment.start + segment.duration)
        # Not added to `_context` or `_previous`: those model what was just said, and this was said
        # earlier. Feeding it in would hand the next line the wrong neighbour and spend that line's
        # one refinement pass revising an utterance from further back.
        self._emit(Emitted(line_id, segment.start, speaker.code, line.lang, text, translations,
                           status=status).event("line"))
        self.recovered += 1
        log.info("recovered the utterance at %.2fs under %s", segment.start, line.lang)

    def _rebias(self, terms: list) -> None:
        """Push the glossary into the recogniser when it has changed since the last utterance.

        Compared as a string rather than tracked with a revision counter: the glossary is already
        being read for the corrector, so this is a comparison of two short strings on a path that
        was about to do a model inference. Nothing is stored that could go stale.
        """
        hotwords = asr_gpu.hotwords_from(terms)
        if hotwords != self._hotwords:
            self._hotwords = hotwords
            self._transcriber.set_hotwords(hotwords)
            log.info("glossary changed mid-meeting, re-biasing the recogniser")

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
