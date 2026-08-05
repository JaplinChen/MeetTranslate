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


def test_unnamed_speaker_can_be_heard_before_naming(client: TestClient) -> None:
    """The naming screen shows S1..S35 and asks who they are; it has to let you hear them.

    /api/speakers/known/{name}/clip resolves a voice through the name attached to it, so it can only
    play back a speaker who has already been identified — no use to the screen doing the identifying.
    """
    import soundfile as sf

    wav = config.RECORDINGS_DIR / "unnamed.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), np.zeros(config.SAMPLE_RATE * 30, dtype="float32"), config.SAMPLE_RATE)

    session = main.store.start_session("now", str(wav))
    # "謝謝" first and longer speech later: picking the earliest line would play the useless one.
    main.store.add_line(session, 1.0, "S1", "zh", "謝謝", {}, end_time=1.4)
    main.store.add_line(session, 12.0, "S1", "zh", "這段話長得多，聽得出是誰", {}, end_time=20.0)
    main.store.add_line(session, 3.0, "S2", "zh", "另一個人", {}, end_time=5.0)

    # Nobody has been named — the endpoint keyed on names finds nothing to play.
    assert client.get("/api/speakers/known/S1/clip").status_code == 404

    clip = client.get(f"/api/sessions/{session}/speakers/S1/clip")
    assert clip.status_code == 200 and clip.headers["content-type"] == "audio/wav"
    heard, rate = sf.read(io.BytesIO(clip.content))
    assert len(heard) == main.CLIP_SECONDS * rate, len(heard)

    # Which line it picked cannot be heard — the fixture is silence — so assert it directly:
    # the 8-second utterance at 12.0s, not the 0.4-second "謝謝" that comes first.
    assert main.store.session_speaker_sample(session, "S1") == (str(wav), 12.0)

    assert client.get(f"/api/sessions/{session}/speakers/S2/clip").status_code == 200
    assert client.get(f"/api/sessions/{session}/speakers/S9/clip").status_code == 404
    assert client.get(f"/api/sessions/999999/speakers/S1/clip").status_code == 404


def test_a_learned_correction_can_be_fixed_in_place(client: TestClient) -> None:
    """Deleting and re-learning means reproducing the line it came from, which a typo rarely is."""
    session = seed_session("editable.wav")
    # This suite shares one store in one global order, so a check that seeds rows has to remove
    # them again — a later one asserts the exact contents of this table.
    before = len(client.get("/api/corrections").json())
    main.store.add_correction("缺消疫", "切削夜")
    main.store.add_correction("CNT", "吸菸")

    # The right-hand side was itself mistyped: fix it without touching what it matches.
    fixed = client.put("/api/corrections/缺消疫", json={"right": "切削液"}).json()
    assert {c["wrong"]: c["right"] for c in fixed}["缺消疫"] == "切削液"

    # The left-hand side is the key, so changing it is a rename: the old text stops matching.
    renamed = client.put("/api/corrections/缺消疫", json={"wrong": "缺خ疫", "right": "切削液"}).json()
    pairs = {c["wrong"]: c["right"] for c in renamed}
    assert "缺消疫" not in pairs and pairs["缺خ疫"] == "切削液"

    # Renaming onto an existing pair would drop whichever one the user was not looking at.
    assert client.put("/api/corrections/缺خ疫", json={"wrong": "CNT", "right": "切削液"}).status_code == 400
    assert client.put("/api/corrections/缺خ疫", json={"right": ""}).status_code == 400
    assert client.put("/api/corrections/缺خ疫", json={"wrong": "同", "right": "同"}).status_code == 400
    assert client.put("/api/corrections/nothing-here", json={"right": "x"}).status_code == 404

    # The rejected edits above left nothing behind: still the two seeded here and whatever existed.
    assert len(client.get("/api/corrections").json()) == before + 2
    assert client.get(f"/api/sessions/{session}/lines").status_code == 200

    client.delete("/api/corrections/缺خ疫")
    client.delete("/api/corrections/CNT")
    assert len(client.get("/api/corrections").json()) == before


def test_a_transcript_line_can_be_played_back(client: TestClient) -> None:
    """Correcting a line means judging text against audio; the page had only the text."""
    import soundfile as sf

    wav = config.RECORDINGS_DIR / "playable.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), np.zeros(config.SAMPLE_RATE * 120, dtype="float32"), config.SAMPLE_RATE)

    session = main.store.start_session("now", str(wav))
    short = main.store.add_line(session, 2.0, "S1", "zh", "短句", {}, end_time=5.0)
    # Longer than the 4s a voice sample uses: a sentence cut off there cannot be checked.
    long = main.store.add_line(session, 10.0, "S1", "zh", "很長的一句話", {}, end_time=25.0)
    unbounded = main.store.add_line(session, 40.0, "S1", "zh", "沒有結束時間", {})

    heard, rate = sf.read(io.BytesIO(client.get(f"/api/sessions/{session}/lines/{short}/clip").content))
    assert len(heard) == 3 * rate, len(heard)

    heard, rate = sf.read(io.BytesIO(client.get(f"/api/sessions/{session}/lines/{long}/clip").content))
    assert len(heard) == 15 * rate, len(heard)

    # No end_time: falls back to the sample length rather than reading to the end of the meeting.
    heard, rate = sf.read(io.BytesIO(client.get(f"/api/sessions/{session}/lines/{unbounded}/clip").content))
    assert len(heard) == main.CLIP_SECONDS * rate, len(heard)

    # A line id from another session must not be playable through this one's recording.
    other = seed_session("elsewhere.wav")
    assert client.get(f"/api/sessions/{other}/lines/{short}/clip").status_code == 404
    assert client.get(f"/api/sessions/{session}/lines/999999/clip").status_code == 404
