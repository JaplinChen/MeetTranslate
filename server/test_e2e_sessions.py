"""Sessions over HTTP: recording control, the subtitle socket, import, and what 404s."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from . import config, jobs, main
from .e2e_support import seed_session


def test_recording_lifecycle(client: TestClient) -> None:
    assert client.post("/api/recording/stop").status_code == 409
    status = client.get("/api/recording/status").json()
    assert status["recording"] is False and status["sessionId"] is None


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
    session_id = seed_session("rerun.wav")
    line_id = main.store.lines(session_id)[0]["id"]

    assert client.post(f"/api/sessions/{session_id}/lines/999999/rerun").status_code == 404
    assert client.post(f"/api/sessions/999999/lines/{line_id}/rerun").status_code == 404

    # A line id belonging to another session must not be reachable through this session's path.
    other = seed_session("rerun-other.wav")
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
