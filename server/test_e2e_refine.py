"""The post-meeting pass: when it is queued, who gets the card, and what a failure leaves behind."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from . import jobs, main, store as store_mod
from .e2e_support import seed_session, wait_for


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
    session_id = seed_session("stopped.wav")

    main.state.update(session=session_id, recorder=None, pipeline=None, gpu=False)
    main._stop_capture()

    assert wait_for(lambda: session_id in stub.calls), f"never refined: {stub.calls}"
    assert wait_for(
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
    session_id = seed_session("shutdown.wav")

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
    session_id = seed_session("held.wav")

    try:
        assert jobs.schedule(session_id, lambda cancel: main.postprocess.rewrite_session(
            main.store, session_id, None, None, None, should_stop=cancel.is_set))
        assert wait_for(lambda: session_id in stub.calls), "pass never started"
        assert jobs.state(session_id)["state"] == "refining"

        # This is what /api/recording/start does before it builds a Pipeline.
        assert jobs.claim_gpu(timeout=5.0), "the meeting never got the card"
        try:
            assert wait_for(lambda: jobs.state(session_id)["state"] == "cancelled"), \
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
    session_id = seed_session("twice.wav")

    try:
        assert jobs.schedule(session_id, lambda cancel: main.postprocess.rewrite_session(
            main.store, session_id, None, None, None, should_stop=cancel.is_set))
        assert wait_for(lambda: session_id in stub.calls)

        assert client.post(f"/api/sessions/{session_id}/reprocess").status_code == 409
        assert not jobs.schedule(session_id, lambda cancel: None)
    finally:
        stub.block.set()
        jobs.cancel_all(wait=1.0)


def test_llm_stages_do_not_hold_the_gpu_gate(client: TestClient) -> None:
    """The blocker this whole split exists for: a meeting must be able to start while a followup
    stage is still talking to a language model.

    Held inside the gate, a minutes-long Ollama call keeps `claim_gpu` waiting past its timeout
    and the room is told it cannot start recording — on account of work that was not using the
    GPU at all.
    """
    import threading

    jobs.reset()
    session_id = seed_session("llm-stage.wav")

    in_followup = threading.Event()
    release = threading.Event()

    def followup(cancel, set_stage):
        set_stage("summarize")
        in_followup.set()
        release.wait(10)

    try:
        assert jobs.schedule(session_id, lambda cancel: None, followup=followup)
        assert wait_for(in_followup.is_set), "followup never started"

        # The pass is mid-followup and must not be holding the card.
        assert jobs.state(session_id) == {"state": "refining", "stage": "summarize", "error": ""}
        assert jobs.claim_gpu(timeout=1.0), "the LLM stage is holding the GPU gate"
        jobs.release_gpu()
    finally:
        release.set()
        jobs.cancel_all(wait=1.0)

    assert wait_for(lambda: jobs.state(session_id)["state"] in ("refined", "cancelled"))


def test_a_failed_followup_keeps_what_the_rewrite_landed(client: TestClient) -> None:
    """Stages land independently: a summary that fails must not undo a refine that succeeded."""
    jobs.reset()
    session_id = seed_session("followup-fail.wav")

    def followup(cancel, set_stage):
        raise RuntimeError("summary model unreachable")

    assert jobs.schedule(session_id, lambda cancel: None, followup=followup)
    assert wait_for(lambda: jobs.state(session_id)["state"] == "failed")
    assert "summary model unreachable" in jobs.state(session_id)["error"]
    # The transcript the rewrite stage owns is untouched by the followup's failure.
    assert [l["source"] for l in main.store.lines(session_id)] == ["精修前就在的一行"]


def test_legacy_session_table_gains_lines_rev(tmp: Path) -> None:
    """Same trap as `status`/`end_time`: an old database's session table never gains the column."""
    import sqlite3

    path = tmp / "legacy-rev.db"
    old = sqlite3.connect(str(path))
    old.executescript(
        """
        CREATE TABLE session (id INTEGER PRIMARY KEY, started TEXT NOT NULL, ended TEXT,
                              wav_path TEXT NOT NULL);
        INSERT INTO session (started, wav_path) VALUES ('2026-01-01T09:00:00', 'old.wav');
        """
    )
    old.commit()
    old.close()

    st = store_mod.Store(path)
    try:
        row = st.session(1)
        assert row is not None and row["lines_rev"] == 0, row
        # And the migrated column actually moves when a line changes.
        st.add_line(1, 0.0, "S1", "zh", "一行", {})
        line_id = st.lines(1)[0]["id"]
        st.update_line(line_id, "改過的一行", {})
        assert st.session(1)["lines_rev"] == 1, st.session(1)
    finally:
        st.close()


def test_every_edit_path_moves_the_revision(tmp: Path) -> None:
    """update_line, replace_line and replace_lines each change what the transcript says.

    The revision is what lets the summary admit it describes an older transcript; an edit path
    that forgets to bump it makes stale look fresh, which is the exact lie this exists to stop.
    """
    st = store_mod.Store(tmp / "rev.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "r.wav")
        st.add_line(sid, 0.0, "S1", "zh", "原句", {})
        line_id = st.lines(sid)[0]["id"]
        assert st.session(sid)["lines_rev"] == 0

        st.update_line(line_id, "人工修正", {})
        assert st.session(sid)["lines_rev"] == 1

        st.replace_line(line_id, "重跑結果", "zh", {}, "ok")
        assert st.session(sid)["lines_rev"] == 2

        st.replace_lines(sid, [{"start": 0.0, "speaker": "S1", "lang": "zh",
                                "source": "精修結果", "translations": {}}])
        assert st.session(sid)["lines_rev"] == 3
    finally:
        st.close()


def test_summary_roundtrip_and_cascade_delete(tmp: Path) -> None:
    st = store_mod.Store(tmp / "summary.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "s.wav")
        assert st.summary(sid) is None

        st.set_summary(sid, '{"zh": {"title": "週會"}}', "ok", lines_rev=0,
                       created="2026-01-01T10:00:00")
        row = st.summary(sid)
        assert row["json"] == '{"zh": {"title": "週會"}}' and row["status"] == "ok", row

        # Latest wins: regeneration overwrites in place.
        st.set_summary(sid, '{"zh": {"title": "週會 v2"}}', "partial", lines_rev=3,
                       created="2026-01-01T11:00:00")
        row = st.summary(sid)
        assert row["lines_rev"] == 3 and row["status"] == "partial", row

        # The summary dies with its session.
        with st._lock:
            st._db.execute("DELETE FROM session WHERE id=?", (sid,))
            st._db.commit()
        assert st.summary(sid) is None
    finally:
        st.close()


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
