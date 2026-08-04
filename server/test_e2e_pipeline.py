"""The live path end to end: what reaches the page, and what a failure costs."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import asr, config, main, store as store_mod, translate
from .e2e_support import FixedTranscriber, OneSpeaker, StubTranslator, headless_pipeline
from .pipeline import Pipeline


def test_pipeline_emits_line_then_update(tmp: Path) -> None:
    """The core subtitle contract, exercised on a real wav through the real VAD and ASR."""
    wav = config.MODELS_DIR / "sherpa-onnx-whisper-tiny" / "test_wavs" / "1.wav"
    if not wav.is_file():
        print("  (skipped: whisper test wav not present)")
        return

    import soundfile as sf

    audio, sr = sf.read(str(wav), dtype="float32")
    assert sr == config.SAMPLE_RATE

    # This test is about the sherpa-onnx wiring and a known-language wav; the GPU model would
    # substitute a different recogniser and decode this English clip as Mandarin.
    os.environ["MEETTRANSLATE_NO_GPU"] = "1"

    st = store_mod.Store(tmp / "pipeline.db")
    session = st.start_session("now", str(wav))
    events: list[dict] = []
    cfg = config.Config(languages=["en", "zh"], whisper_model="tiny")

    # A second of silence between repeats, or VAD sees one unbroken 50 s utterance rather than
    # three — which is also how a real meeting separates turns.
    gap = np.zeros(config.SAMPLE_RATE, dtype="float32")
    feed = np.concatenate([audio, gap, audio, gap, audio, gap])

    pipe = Pipeline(cfg, st, session, StubTranslator(), events.append)
    pipe.start()
    try:
        for i in range(0, len(feed), config.BLOCK_SIZE):
            pipe.tap.put(feed[i : i + config.BLOCK_SIZE])
        pipe.tap.put(None)
        pipe.join()

        kinds = [e["type"] for e in events]
        assert kinds.count("line") >= 3, kinds
        assert "update" in kinds, f"no refinement emitted: {kinds}"
        assert pipe.errors == 0, f"{pipe.errors} segment errors"

        lines = {e["line"]["id"] for e in events if e["type"] == "line"}
        updates = {e["line"]["id"] for e in events if e["type"] == "update"}
        # An update must target a line already sent, or the page would have nothing to rewrite.
        assert updates <= lines, (lines, updates)

        stored = st.lines(session)
        assert len(stored) == len(lines), "every emitted line must be persisted"
        refined = [r for r in stored if r["refined"]]
        assert refined and refined[0]["source"] == "corrected source"
        assert refined[0]["translations"]["zh"] == "[zh] corrected"
    finally:
        st.close()


def test_a_rerun_always_answers_in_the_same_shape(tmp: Path) -> None:
    """The page reads the reply straight into state, so both outcomes must carry the same keys.

    They drifted apart once — one exit returned `line`, the other `lines` — which would have
    blanked the transcript the rerun was supposed to be repairing.
    """
    st = store_mod.Store(tmp / "shape.db")
    try:
        session_id = st.start_session("2026-01-01T09:00:00", str(tmp / "s.wav"))
        st.add_line(session_id, 0.0, "S1", "zh", "一行", {"en": "a line"})
        st.set_speaker_name(session_id, "S1", "陳經理")

        original, main.store = main.store, st
        try:
            for status in ("ok", "asr_failed", "translate_failed"):
                body = main._transcript(session_id, status)
                assert set(body) == {"lines", "speakers", "status"}, body
                assert body["status"] == status
                assert body["lines"] and body["speakers"] == {"S1": "陳經理"}, body
        finally:
            main.store = original
    finally:
        st.close()


def test_a_failed_translation_costs_the_translation_not_the_line(tmp: Path) -> None:
    """It used to raise into the handler's catch-all and drop the whole utterance.

    The room would then see nothing where it should have seen the original text untranslated —
    a translation outage reading as a speaker who never spoke.
    """
    class Exploding:
        def translate(self, *a, **k):
            raise RuntimeError("no key")

    st = store_mod.Store(tmp / "translate-fail.db")
    try:
        session_id = st.start_session("2026-01-01T09:00:00", str(tmp / "t.wav"))
        emitted: list[dict] = []
        pipe = headless_pipeline(config.Config(), st, session_id, Exploding(), emitted.append)
        pipe._transcriber = FixedTranscriber("這句話有說出來")
        pipe._diarizer = OneSpeaker()

        pipe._handle(asr.Segment(np.zeros(config.SAMPLE_RATE, dtype="float32"), 0.0))

        rows = st.lines(session_id)
        assert len(rows) == 1, rows
        assert rows[0]["source"] == "這句話有說出來", rows
        assert rows[0]["status"] == "translate_failed", rows
        assert rows[0]["translations"] == {}, rows
        assert pipe.errors == 0, "a translation outage is not a pipeline error"
        assert emitted and emitted[0]["line"]["status"] == "translate_failed", emitted
    finally:
        st.close()
