"""End-to-end checks over the HTTP API with a stubbed translator.

Run: python -m server.test_e2e

Real audio and a real Claude key are not available here, so the translator is replaced by a stub
and the pipeline is fed a recorded wav. What this proves is the wiring: capture control, the
store, the WebSocket fan-out, and that a `line` event followed by an `update` event rewrites the
subtitle in place rather than appending a duplicate.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from . import asr, asr_gpu, config, llm, main, store as store_mod, translate
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
    """Whisper drops everything past 30 s; the transcriber must chunk rather than lose speech."""
    wav = config.MODELS_DIR / "sherpa-onnx-whisper-tiny" / "test_wavs" / "1.wav"
    if not wav.is_file():
        print("  (skipped: whisper test wav not present)")
        return

    import soundfile as sf

    audio, _ = sf.read(str(wav), dtype="float32")
    tr = asr.Transcriber(model_dir=config.MODELS_DIR / "sherpa-onnx-whisper-tiny")

    short, _ = tr.transcribe(audio, "en")
    # ~50 s: over the limit, so it must be split and both halves must contribute.
    long_text, _ = tr.transcribe(np.concatenate([audio, audio, audio]), "en")

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


def test_unknown_api_path_is_json_404(client: TestClient) -> None:
    r = client.get("/api/definitely-not-a-route")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")


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
