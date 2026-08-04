"""End-to-end checks over the HTTP API with a stubbed translator.

Run: python -m server.test_e2e

Real audio and a real Claude key are not available here, so the translator is replaced by a stub
and the pipeline is fed a recorded wav. What this proves is the wiring: capture control, the
store, the WebSocket fan-out, and that a `line` event followed by an `update` event rewrites the
subtitle in place rather than appending a duplicate.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from . import asr, asr_gpu, config, jobs, llm, main, postprocess as postprocess_mod
from . import pipeline as pipeline_mod, store as store_mod, translate
from .pipeline import Pipeline


def _isolate(tmp: Path) -> None:
    """Point every persistent path at a temp dir so a test run cannot touch real data."""
    config.CONFIG_PATH = tmp / "config.json"
    config.RECORDINGS_DIR = tmp / "recordings"
    llm.LLM_PATH = tmp / "llm.json"
    llm.KEYS_PATH = tmp / "llm_keys.json"
    main.store = store_mod.Store(tmp / "test.db")
    main.keys = llm.KeyStore(tmp / "llm_keys.json")
    main.state["cfg"] = config.Config()
    main.state["llm"] = llm.LlmConfig()
    # Stopping a recording now queues the post-meeting pass, and the real one loads a Whisper
    # model. Every lifecycle test would pull one in, so the pass records that it was asked instead.
    main.postprocess = _StubPostprocess()


class _StubPostprocess:
    """Stands in for the postprocess module: records calls, honours cancellation, writes nothing."""

    def __init__(self) -> None:
        self.calls: list[int] = []
        # Set means "return at once". A test that needs a pass to still be running clears it.
        self.block = threading.Event()
        self.block.set()

    def rewrite_session(self, store, session_id, wav, cfg, translator=None, should_stop=None):
        self.calls.append(session_id)
        while not self.block.is_set():
            if should_stop and should_stop():
                raise jobs.Cancelled()
            time.sleep(0.005)
        return []

    def to_markdown(self, store, session_id):
        return postprocess_mod.to_markdown(store, session_id)


class StubTranslator:
    """Echoes a deterministic translation, and revises the previous line on the third call."""

    def __init__(self) -> None:
        self.calls = 0

    def translate(self, line, targets, context=None, previous=None, terms=None):
        self.calls += 1
        out = {t: f"[{t}] {line.text}" for t in targets}
        if self.calls == 3 and previous is not None:
            return translate.Result(out, "corrected source", {t: f"[{t}] corrected" for t in targets})
        return translate.Result(out)


def test_health_and_config_roundtrip(client: TestClient) -> None:
    assert client.get("/api/health").json()["status"] == "ok"

    cfg = client.get("/api/config").json()
    assert cfg["languages"] == ["zh", "vi", "en"]
    assert cfg["display"]["show_source"] == "top"

    r = client.put("/api/config", json={"languages": ["zh", "vi"], "display": {"font_size": 64, "theme": "light"}})
    assert r.status_code == 200, r.text
    assert r.json()["languages"] == ["zh", "vi"]
    assert r.json()["display"]["font_size"] == 64
    # An unspecified field must survive a partial patch.
    assert r.json()["display"]["lines"] == 6


def test_config_rejects_bad_input(client: TestClient) -> None:
    assert client.put("/api/config", json={"languages": ["zh"]}).status_code == 400
    assert client.put("/api/config", json={"languages": ["zh", "zh"]}).status_code == 400
    assert client.put("/api/config", json={"display": {"lines": 99}}).status_code == 400
    assert client.put("/api/config", json={"display": {"show_source": "sideways"}}).status_code == 400


def test_glossary_crud(client: TestClient) -> None:
    r = client.post("/api/glossary", json={"source": "產能", "targets": {"vi": "công suất", "en": "capacity"}})
    assert r.status_code == 200, r.text
    assert r.json()[0]["targets"]["vi"] == "công suất"

    # `keep` is the code-switching case: the term must never be translated.
    client.post("/api/glossary", json={"source": "schedule", "mode": "keep"})
    modes = {t["source"]: t["mode"] for t in client.get("/api/glossary").json()}
    assert modes == {"產能": "translate", "schedule": "keep"}

    assert client.post("/api/glossary", json={"source": "x", "mode": "nonsense"}).status_code == 400
    assert client.post("/api/glossary", json={"source": "   "}).status_code == 400

    left = client.request("DELETE", "/api/glossary", params={"source": "schedule"}).json()
    assert [t["source"] for t in left] == ["產能"]


def test_glossary_reports_what_a_term_would_overwrite(client: TestClient) -> None:
    """Adding a term is not obviously destructive, which is the problem.

    料號 and 料耗 are both liaohao and 料耗 is a term of the trade; adding 料號 rewrote it
    forty-two times across seven interviews and nothing said so. The answer comes from the
    meetings already recorded — what these people say, not what Mandarin permits.
    """
    session = main.store.start_session("now", "x.wav")
    for text in ("這個料耗的部分", "料耗變動原則", "刀具的料耗很多"):
        main.store.add_line(session, 1.0, "S1", "zh", text, {})

    hits = client.get("/api/glossary/collisions", params={"source": "料號"}).json()
    assert hits["collisions"] == [{"text": "料耗", "count": 3}]

    # A word already in the glossary is not collateral: registering it is how two real
    # homophones are made to coexist.
    client.post("/api/glossary", json={"source": "料耗", "mode": "protect"})
    assert client.get("/api/glossary/collisions", params={"source": "料號"}).json()["collisions"] == []

    assert client.get("/api/glossary/collisions", params={"source": "交貨"}).json()["collisions"] == []
    assert client.get("/api/glossary/collisions", params={"source": " "}).status_code == 400


def test_llm_config_never_returns_the_key(client: TestClient) -> None:
    client.put("/api/translate/config", json={
        "llmProvider": "anthropic", "llmModel": "claude-opus-5", "llmApiKey": "sk-secret-value-1234",
        "llmEndpoint": "https://api.anthropic.com", "llmTemperature": 0,
        "llmFallbackModels": [], "llmProviderConfigs": {},
    })
    body = client.get("/api/translate/config").json()
    assert body["llmApiKey"] == ""
    assert body["apiKeySet"] is True
    assert "sk-secret-value-1234" not in json.dumps(body)

    # A blank key on a later save means "keep the stored one", not "clear it".
    client.put("/api/translate/config", json={"llmModel": "claude-sonnet-5", "llmApiKey": ""})
    body = client.get("/api/translate/config").json()
    assert body["apiKeySet"] is True
    assert body["llmModel"] == "claude-sonnet-5"


def test_keyproxy_masks_and_rotates(client: TestClient) -> None:
    client.post("/api/keyproxy/keys", json={"provider": "anthropic", "apiKey": "sk-aaaabbbbccccdddd", "account": "a"})
    client.post("/api/keyproxy/keys", json={"provider": "anthropic", "apiKey": "sk-eeeeffffgggghhhh", "account": "b"})
    listed = client.get("/api/keyproxy/keys").json()
    assert len(listed) == 2
    assert all("sk-aaaa" not in json.dumps(k) or k["masked"].startswith("sk-a") for k in listed)
    assert listed[0]["masked"] == "sk-a…dddd"

    got = {main.keys.next_key("anthropic") for _ in range(2)}
    assert got == {"sk-aaaabbbbccccdddd", "sk-eeeeffffgggghhhh"}, "rotation did not visit both keys"

    assert client.post("/api/keyproxy/keys", json={"provider": "anthropic", "apiKey": ""}).status_code == 400
    assert client.delete("/api/keyproxy/keys/anthropic/9").status_code == 404
    assert client.delete("/api/keyproxy/keys/anthropic/0").json() == client.get("/api/keyproxy/keys").json()


def test_naming_a_speaker_teaches_the_room_their_voice(client: TestClient) -> None:
    """The one piece of labelled data this system ever gets, kept instead of discarded."""
    session = main.store.start_session("now", "x.wav")
    main.store.save_voiceprint(session, "S1", np.array([1.0], dtype="float32").tobytes())

    client.put(f"/api/sessions/{session}/speakers", json={"S1": "Vincent"})
    assert [s["name"] for s in client.get("/api/speakers/known").json()] == ["Vincent"]

    # A speaker with no stored voiceprint is still nameable, it just teaches nothing.
    client.put(f"/api/sessions/{session}/speakers", json={"S9": "Nobody"})
    assert [s["name"] for s in client.get("/api/speakers/known").json()] == ["Vincent"]

    assert client.delete("/api/speakers/known/Vincent").json() == []


def test_editing_a_line_teaches_the_correction(client: TestClient) -> None:
    """An edit on the transcript page is the only ground truth this system gets: someone who was
    in the room saying what was actually said. Kept, the same mistake is fixed everywhere next
    time — live as well as after the fact."""
    session = main.store.start_session("now", "x.wav")
    line = main.store.add_line(session, 1.0, "S1", "zh", "那個申管會上系統", {})

    r = client.put(f"/api/sessions/{session}/lines/{line}", json={"source": "那個生管會上系統"})
    assert r.status_code == 200, r.text
    assert r.json()["lines"][0]["source"] == "那個生管會上系統"
    assert {c["wrong"]: c["right"] for c in client.get("/api/corrections").json()} == {"申管": "生管"}

    # What was learned is applied to text the recogniser has not seen yet.
    from . import correct as correct_mod
    fixed = correct_mod.Corrector([], main.store.corrections()).fix("剛剛申管講的")
    assert fixed == "剛剛生管講的"

    assert client.put(f"/api/sessions/{session}/lines/{line}", json={"source": " "}).status_code == 400
    assert client.put(f"/api/sessions/{session}/lines/9999", json={"source": "x"}).status_code == 404
    assert client.delete("/api/corrections/申管").json() == []


def test_every_script_flag_is_wired_to_something(tmp: Path) -> None:
    """A flag that nothing reads is a feature that silently stopped existing.

    scripts/learn_terms.py kept its --max-sound option for a while after a scripted edit removed
    the ranking that used it, so the tool went on accepting the flag and ignoring it. Nothing in
    the output said so.
    """
    import re

    scripts = sorted(Path("scripts").glob("*.py"))
    assert scripts, "no scripts found; is the working directory wrong?"
    for path in scripts:
        source = path.read_text(encoding="utf-8")
        for flag in re.findall(r'add_argument\("--([a-z-]+)"', source):
            assert f"args.{flag.replace('-', '_')}" in source, f"{path.name}: --{flag} is unused"


def test_recording_lifecycle(client: TestClient) -> None:
    assert client.post("/api/recording/stop").status_code == 409
    status = client.get("/api/recording/status").json()
    assert status["recording"] is False and status["sessionId"] is None


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


def test_websocket_receives_config_and_events(client: TestClient) -> None:
    with client.websocket_connect("/ws/live") as ws:
        first = ws.receive_json()
        assert first["type"] == "config"
        assert "display" in first and "languages" in first

        # Publishing crosses the thread boundary the pipeline uses.
        threading.Thread(target=lambda: main.hub.publish({"type": "line", "line": {"id": 1}})).start()
        assert ws.receive_json()["line"]["id"] == 1


def test_known_voice_can_be_heard_and_renamed(client: TestClient) -> None:
    """A learned voice is only inspectable if you can play it back and fix the name on it."""
    import soundfile as sf

    wav = config.RECORDINGS_DIR / "voice.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), np.zeros(config.SAMPLE_RATE * 10, dtype="float32"), config.SAMPLE_RATE)

    session = main.store.start_session("now", str(wav))
    main.store.add_line(session, 5.0, "S1", "en", "hello", {})
    main.store.save_voiceprint(session, "S1", b"\x00" * 8)
    assert client.put(f"/api/sessions/{session}/speakers", json={"S1": "Ana"}).status_code == 200

    known = client.get("/api/speakers/known").json()
    assert [s["name"] for s in known] == ["Ana"] and known[0]["sessions"] >= 1

    clip = client.get("/api/speakers/known/Ana/clip")
    assert clip.status_code == 200 and clip.headers["content-type"] == "audio/wav"
    heard, rate = sf.read(io.BytesIO(clip.content))
    assert len(heard) == main.CLIP_SECONDS * rate, len(heard)

    renamed = client.put("/api/speakers/known/Ana", json={"name": "Ana Lee"}).json()
    assert [s["name"] for s in renamed] == ["Ana Lee"]
    # The transcript must follow the rename, or it keeps showing a name that no longer exists.
    assert client.get(f"/api/sessions/{session}/lines").json()["speakers"]["S1"] == "Ana Lee"
    assert client.get("/api/speakers/known/Ana/clip").status_code == 404

    assert client.delete("/api/speakers/known/Ana%20Lee").json() == []


def test_importing_a_recording_makes_it_a_session(client: TestClient) -> None:
    """An uploaded file has to land as an ordinary session, or nothing can be learned from it."""
    if shutil.which("ffmpeg") is None:
        print("  (skipped: ffmpeg not installed)")
        return

    import soundfile as sf

    # Not named import-*: that prefix belongs to what the endpoint writes, and this test counts those.
    src = config.RECORDINGS_DIR / "fixture.m4a"
    src.parent.mkdir(parents=True, exist_ok=True)
    tone = np.sin(np.arange(config.SAMPLE_RATE * 3) * 0.05).astype("float32")
    raw = config.RECORDINGS_DIR / "fixture.wav"
    sf.write(str(raw), tone, config.SAMPLE_RATE)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), str(src)], check=True)

    before = len(client.get("/api/sessions").json())
    r = client.post("/api/sessions/import?filename=meeting.m4a", content=src.read_bytes())
    assert r.status_code == 200, r.text
    session_id = r.json()["id"]

    listed = client.get("/api/sessions").json()
    assert len(listed) == before + 1
    imported = next(s for s in listed if s["id"] == session_id)
    # Ended, or /reprocess and the clip endpoint would treat it as still recording.
    assert imported["ended"] and Path(imported["wav_path"]).is_file()
    # The upload itself is not kept: everything downstream reads the extracted wav.
    assert not list(config.RECORDINGS_DIR.glob("import-*-meeting.m4a"))

    assert client.post("/api/sessions/import?filename=empty.mp4", content=b"").status_code == 400
    assert client.post("/api/sessions/import?filename=x.mp4", content=b"not a video").status_code == 400
    # A rejected upload must leave nothing behind, or the next import inherits a stale wav.
    assert len(list(config.RECORDINGS_DIR.glob("import-*.wav"))) == 1

    # A second import in the same second must not overwrite the first one's audio.
    again = client.post("/api/sessions/import?filename=meeting.m4a", content=src.read_bytes())
    assert again.status_code == 200, again.text
    imports = [s for s in client.get("/api/sessions").json() if s["id"] in (session_id, again.json()["id"])]
    assert len({s["wav_path"] for s in imports}) == 2, imports


def test_unknown_api_path_is_json_404(client: TestClient) -> None:
    r = client.get("/api/definitely-not-a-route")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")

    # Whatever the method: a 405 here would report "wrong method" for a route that does not exist,
    # which is exactly how a server running older code presents itself.
    for send in (client.post, client.put, client.delete):
        r = send("/api/definitely-not-a-route")
        assert r.status_code == 404, (send.__name__, r.status_code)
        assert r.headers["content-type"].startswith("application/json")


def test_rerunning_a_line_refuses_what_it_should(client: TestClient) -> None:
    """The rerun endpoint has no authentication in front of it, so its guards are the only ones."""
    jobs.reset()
    session_id = _seed_session("rerun.wav")
    line_id = main.store.lines(session_id)[0]["id"]

    assert client.post(f"/api/sessions/{session_id}/lines/999999/rerun").status_code == 404
    assert client.post(f"/api/sessions/999999/lines/{line_id}/rerun").status_code == 404

    # A line id belonging to another session must not be reachable through this session's path.
    other = _seed_session("rerun-other.wav")
    other_line = main.store.lines(other)[0]["id"]
    assert client.post(f"/api/sessions/{session_id}/lines/{other_line}/rerun").status_code == 404

    # Not while that session is recording: the wav is still being written.
    main.state["session"] = session_id
    try:
        assert client.post(f"/api/sessions/{session_id}/lines/{line_id}/rerun").status_code == 409
    finally:
        main.state["session"] = None

    # A line with no duration is refused rather than decoded as a 60-second span.
    main.store.replace_line(line_id, "一行", "zh", {}, "ok")
    assert client.post(f"/api/sessions/{session_id}/lines/{line_id}/rerun").status_code == 400


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
        pipe = Pipeline(config.Config(), st, session_id, Exploding(), emitted.append)
        pipe._transcriber = _FixedTranscriber("這句話有說出來")
        pipe._diarizer = _OneSpeaker()

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


def test_a_failed_decode_is_retried_once_the_speaker_language_is_known(tmp: Path) -> None:
    """Live used to bin these in silence. The post-meeting pass recovers 992 such lines.

    The first utterance decodes to nothing under auto-detect. The second establishes that this
    speaker is speaking Chinese. The first must then be re-decoded under Chinese and appear.
    """
    st = store_mod.Store(tmp / "retry.db")
    try:
        session_id = st.start_session("2026-01-01T09:00:00", str(tmp / "r.wav"))
        emitted: list[dict] = []
        pipe = Pipeline(config.Config(languages=["zh", "en"]), st, session_id, None, emitted.append)
        pipe._transcriber = _ByLanguage({"": ("", ""), "zh": ("補回來的那一句", "zh")})
        pipe._diarizer = _OneSpeaker()

        # Fails under auto-detect and is held, not dropped.
        pipe._handle(asr.Segment(np.zeros(config.SAMPLE_RATE, dtype="float32"), 10.0))
        assert st.lines(session_id) == [], "a held utterance must not be stored yet"
        assert len(pipe._held) == 1, pipe._held

        # A later utterance settles the speaker's language, which triggers the retry.
        pipe._transcriber.table[""] = ("這句話正常", "zh")
        pipe._handle(asr.Segment(np.zeros(config.SAMPLE_RATE, dtype="float32"), 20.0))

        rows = st.lines(session_id)
        assert [r["source"] for r in rows] == ["補回來的那一句", "這句話正常"], rows
        assert pipe.recovered == 1 and pipe.dropped == 0
        assert pipe._held == []

        # It voted once, not twice: the retry must not count the same audio toward the speaker.
        assert pipe._diarizer.votes == ["zh"], pipe._diarizer.votes

        # And it is not offered to the next line as context, nor as the line to refine.
        assert pipe._previous[2].text == "這句話正常"
        assert [l.text for l in pipe._context] == ["這句話正常"]

        # The recovered line is emitted with its own start, so the page can place it correctly.
        late = [e for e in emitted if e["line"]["source"] == "補回來的那一句"]
        assert late and late[0]["line"]["start"] == 10.0, emitted
    finally:
        st.close()


def test_the_retry_buffer_cannot_grow_without_bound(tmp: Path) -> None:
    """Every held utterance keeps its raw audio, so a room that decodes nothing must not fill RAM."""
    st = store_mod.Store(tmp / "retry-cap.db")
    try:
        session_id = st.start_session("2026-01-01T09:00:00", str(tmp / "c.wav"))
        pipe = Pipeline(config.Config(languages=["zh"]), st, session_id, None, lambda e: None)
        pipe._transcriber = _ByLanguage({"": ("", ""), "zh": ("", "")})
        pipe._diarizer = _OneSpeaker()

        for i in range(pipeline_mod.RETRY_BUFFER + 8):
            pipe._handle(asr.Segment(np.zeros(1600, dtype="float32"), float(i)))

        assert len(pipe._held) == pipeline_mod.RETRY_BUFFER, len(pipe._held)
        assert pipe.dropped == 8, pipe.dropped
        # The oldest went first, so what is still held is the most recent audio.
        assert pipe._held[0][0].start == 8.0, pipe._held[0][0].start
    finally:
        st.close()


def test_a_retry_that_explodes_is_counted_not_lost(tmp: Path) -> None:
    """The held entry is already off the list by then, so an escape would lose it silently.

    It must also not fail the live utterance that triggered the retry: recovering something older
    is a bonus on top of that line, never a risk to it.
    """
    class Exploding(_ByLanguage):
        def transcribe(self, samples, language):
            if language == "zh" and len(samples) == 4242:
                raise RuntimeError("decoder blew up")
            return super().transcribe(samples, language)

    st = store_mod.Store(tmp / "retry-boom.db")
    try:
        session_id = st.start_session("2026-01-01T09:00:00", str(tmp / "b.wav"))
        pipe = Pipeline(config.Config(languages=["zh"]), st, session_id, None, lambda e: None)
        pipe._transcriber = Exploding({"": ("", ""), "zh": ("正常的一句", "zh")})
        pipe._diarizer = _OneSpeaker()

        pipe._handle(asr.Segment(np.zeros(4242, dtype="float32"), 5.0))  # decodes to nothing, held
        assert len(pipe._held) == 1, pipe._held

        # The next utterance decodes, which settles the speaker's language and fires the retry.
        pipe._transcriber.table[""] = ("正常的一句", "zh")
        pipe._handle(asr.Segment(np.zeros(1600, dtype="float32"), 12.0))

        assert pipe.dropped == 1, pipe.dropped
        assert pipe.recovered == 0
        assert pipe._held == []
        # The live line survived the failed retry.
        assert [r["source"] for r in st.lines(session_id)] == ["正常的一句"]
        assert pipe.errors == 0, "a failed retry is not a failure of the live segment"
    finally:
        st.close()


class _ByLanguage:
    """Transcriber returning a canned result per forced language."""

    def __init__(self, table: dict[str, tuple[str, str]]) -> None:
        self.table = table

    def set_hotwords(self, hotwords: str) -> None:
        pass

    def transcribe(self, samples, language):
        return self.table.get(language, ("", language))


class _FixedTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text

    def set_hotwords(self, hotwords: str) -> None:
        pass

    def transcribe(self, samples, language):
        return self.text, "zh"


class _OneSpeaker:
    """One speaker whose language is unknown until an utterance actually decodes."""

    class _S:
        code = "S1"
        centroid = np.zeros(4, dtype="float32")

    def __init__(self) -> None:
        self.recognised: dict = {}
        self.language = ""
        self.votes: list[str] = []
        self._speaker = self._S()

    def assign(self, samples):
        return self._speaker

    def language_for(self, speaker):
        return self.language

    def observe_language(self, speaker, used):
        self.votes.append(used)
        if used:
            self.language = used


def _wait_for(predicate, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _seed_session(name: str) -> int:
    """A finished session with one line and a real file on disk, without needing a microphone."""
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    wav = config.RECORDINGS_DIR / name
    wav.write_bytes(b"")
    session_id = main.store.start_session("2026-01-01T09:00:00", str(wav))
    main.store.end_session(session_id, "2026-01-01T10:00:00")
    main.store.add_line(session_id, 0.0, "S1", "zh", "精修前就在的一行", {})
    return session_id


def test_stopping_a_recording_refines_it_without_being_asked(client: TestClient) -> None:
    """The whole point: a transcript's quality must not depend on anyone clicking a button.

    An imported recording was always refined on the way in. A meeting the room captured was not,
    so the same audio came out better when uploaded through the dashboard than when recorded in
    the room this was built for.

    Driven through `_stop_capture` rather than the HTTP endpoints because starting a real capture
    needs a sound card, and a test that quietly skips on the build machine is not a test.
    """
    jobs.reset()
    stub = main.postprocess
    stub.calls.clear()
    session_id = _seed_session("stopped.wav")

    main.state.update(session=session_id, recorder=None, pipeline=None, gpu=False)
    main._stop_capture()

    assert _wait_for(lambda: session_id in stub.calls), f"never refined: {stub.calls}"
    assert _wait_for(
        lambda: client.get(f"/api/sessions/{session_id}/refine").json()["state"] == "refined"
    ), client.get(f"/api/sessions/{session_id}/refine").json()

    listed = next(s for s in client.get("/api/sessions").json() if s["id"] == session_id)
    assert listed["refine"]["state"] == "refined", listed
    assert stub.calls.count(session_id) == 1, stub.calls


def test_shutting_down_does_not_queue_a_pass_that_can_never_finish(client: TestClient) -> None:
    """The worker is a daemon thread: one queued on the way out is killed or holds the exit open."""
    jobs.reset()
    stub = main.postprocess
    stub.calls.clear()
    session_id = _seed_session("shutdown.wav")

    main.state.update(session=session_id, recorder=None, pipeline=None, gpu=False)
    main._stop_capture(refine=False)
    time.sleep(0.05)
    assert stub.calls == [], stub.calls


def test_a_meeting_takes_the_card_back_from_a_running_pass(client: TestClient) -> None:
    """One GPU. A pass in flight must yield to a meeting starting now, and yield without damage.

    Two Whisper models on one 16 GB card pushes the live realtime factor past 1, and once that
    happens the capture backlog fills and the room's subtitles start dropping. The meeting always
    wins, and winning must cost the transcript nothing.
    """
    jobs.reset()
    stub = main.postprocess
    stub.calls.clear()
    stub.block.clear()  # the pass spins until it is asked to stop
    session_id = _seed_session("held.wav")

    try:
        assert jobs.schedule(session_id, lambda cancel: main.postprocess.rewrite_session(
            main.store, session_id, None, None, None, should_stop=cancel.is_set))
        assert _wait_for(lambda: session_id in stub.calls), "pass never started"
        assert jobs.state(session_id)["state"] == "refining"

        # This is what /api/recording/start does before it builds a Pipeline.
        assert jobs.claim_gpu(timeout=5.0), "the meeting never got the card"
        try:
            assert _wait_for(lambda: jobs.state(session_id)["state"] == "cancelled"), \
                jobs.state(session_id)
            # Yielding cost nothing: the transcript it was part-way through rewriting is intact.
            assert [l["source"] for l in main.store.lines(session_id)] == ["精修前就在的一行"]
        finally:
            jobs.release_gpu()
    finally:
        stub.block.set()


def test_two_passes_over_one_session_do_not_overlap(client: TestClient) -> None:
    """Both would call replace_lines on the same session, and both would want the card."""
    jobs.reset()
    stub = main.postprocess
    stub.calls.clear()
    stub.block.clear()
    session_id = _seed_session("twice.wav")

    try:
        assert jobs.schedule(session_id, lambda cancel: main.postprocess.rewrite_session(
            main.store, session_id, None, None, None, should_stop=cancel.is_set))
        assert _wait_for(lambda: session_id in stub.calls)

        assert client.post(f"/api/sessions/{session_id}/reprocess").status_code == 409
        assert not jobs.schedule(session_id, lambda cancel: None)
    finally:
        stub.block.set()
        jobs.cancel_all(wait=1.0)


def test_a_database_made_before_the_columns_existed_gains_them(tmp: Path) -> None:
    """The meeting room's database predates `status` and `end_time`.

    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, so without a
    migration this only breaks where it matters: on the one machine holding real recordings, inside
    the capture thread, as a swallowed error count.
    """
    import sqlite3

    path = tmp / "legacy.db"
    old = sqlite3.connect(str(path))
    old.executescript(
        """
        CREATE TABLE session (id INTEGER PRIMARY KEY, started TEXT NOT NULL, ended TEXT,
                              wav_path TEXT NOT NULL);
        CREATE TABLE line (id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, start REAL NOT NULL,
                           speaker TEXT NOT NULL, lang TEXT NOT NULL, source TEXT NOT NULL,
                           refined INTEGER NOT NULL DEFAULT 0);
        INSERT INTO session (started, wav_path) VALUES ('2026-01-01T09:00:00', 'old.wav');
        INSERT INTO line (session_id, start, speaker, lang, source) VALUES (1, 1.5, 'S1', 'zh', '舊的一行');
        """
    )
    old.commit()
    old.close()

    st = store_mod.Store(path)
    try:
        columns = {r[1] for r in st._db.execute("PRAGMA table_info(line)")}
        assert "status" in columns, columns
        assert "end_time" in columns, columns

        kept = st.lines(1)
        assert len(kept) == 1, kept
        assert kept[0]["source"] == "舊的一行", kept
        assert kept[0]["status"] == "ok", kept
        assert kept[0]["end_time"] is None, kept

        # And the migrated table still accepts writes, which is the part that was breaking.
        st.replace_lines(1, [{"start": 0.0, "speaker": "S1", "lang": "zh", "source": "新的一行",
                              "translations": {"en": "a new line"}, "status": "ok"}])
        assert [l["source"] for l in st.lines(1)] == ["新的一行"]
    finally:
        st.close()


def test_a_failed_rewrite_leaves_the_old_transcript_alone(tmp: Path) -> None:
    """Replacing a transcript is all-or-nothing.

    The failure this guards is not a slow path, it is data loss: the delete lands, the inserts do
    not, and the meeting's transcript is gone while its recording sits on disk looking fine.
    """
    st = store_mod.Store(tmp / "atomic.db")
    try:
        session_id = st.start_session("2026-01-01T09:00:00", str(tmp / "a.wav"))
        st.add_line(session_id, 0.0, "S1", "zh", "原本就在的一行", {"en": "already here"})
        before = st.lines(session_id)
        assert len(before) == 1, before

        # The fifth row is missing "source". The delete and four inserts have already run.
        rows = [{"start": float(i), "speaker": "S1", "lang": "zh", "source": f"新 {i}",
                 "translations": {}} for i in range(4)]
        rows.append({"start": 4.0, "speaker": "S1", "lang": "zh", "translations": {}})

        failed = False
        try:
            st.replace_lines(session_id, rows)
        except KeyError:
            failed = True
        assert failed, "replace_lines should have raised on the malformed row"

        after = st.lines(session_id)
        assert len(after) == 1, f"transcript was clobbered by a failed rewrite: {after}"
        assert after[0]["source"] == "原本就在的一行", after

        # The rollback must not have poisoned the connection for everyone else.
        st.add_line(session_id, 9.0, "S2", "en", "still writable", {})
        assert len(st.lines(session_id)) == 2
    finally:
        st.close()


def main_() -> None:
    # ignore_cleanup_errors: SQLite on Windows keeps a handle briefly after close, and a failed
    # rmtree must not mask a passing run.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp = Path(tmpdir)
        _isolate(tmp)

        with TestClient(main.app) as client:
            for name, fn in sorted(globals().items()):
                if not name.startswith("test_"):
                    continue
                fn(tmp) if "tmp" in fn.__code__.co_varnames[: fn.__code__.co_argcount] else fn(client)
                print(f"ok  {name}")

        main.store.close()
    print("\nall e2e checks passed")


if __name__ == "__main__":
    main_()
