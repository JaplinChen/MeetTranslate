"""Who spoke: per-session voiceprints, and the names this room has learned.

Naming S1 as "Vincent" is knowledge the meeting room throws away every time. Kept here, the next
meeting recognises the voice instead of asking again. Three tables because the facts arrive at
different times: the centroid exists while the meeting runs, the name usually arrives afterwards
from the transcript page, and only then can it be promoted to something the room knows by voice.

A mixin over `Store` rather than an object of its own: every method here is one statement against
the same connection under the same lock, so giving it a connection of its own would mean a second
writer on a database whose whole locking story is "one connection, one lock".
"""

from __future__ import annotations

import sqlite3
import threading


def _sample(row: sqlite3.Row | None) -> tuple[str, float, float | None] | None:
    """A sample row as (recording, start, seconds), with seconds unknown on transcripts written
    before end_time was recorded."""
    if row is None:
        return None
    end = row["end_time"]
    span = float(end) - float(row["start"]) if end is not None else None
    return (row["wav"], row["start"], span if span and span > 0 else None)


class SpeakerStore:
    """Speaker tables. Mixed into `Store`, which opens the connection and owns the lock."""

    _db: sqlite3.Connection
    _lock: threading.Lock

    def set_speaker_name(self, session_id: int, code: str, name: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO speaker_name (session_id, code, name) VALUES (?,?,?) "
                "ON CONFLICT(session_id, code) DO UPDATE SET name=excluded.name",
                (session_id, code, name),
            )
            self._db.commit()

    def speaker_names(self, session_id: int) -> dict[str, str]:
        with self._lock:
            return {r["code"]: r["name"] for r in self._db.execute(
                "SELECT code, name FROM speaker_name WHERE session_id=?", (session_id,)
            )}

    def save_voiceprint(self, session_id: int, code: str, centroid: bytes) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO voiceprint (session_id, code, centroid) VALUES (?,?,?) "
                "ON CONFLICT(session_id, code) DO UPDATE SET centroid=excluded.centroid",
                (session_id, code, centroid),
            )
            self._db.commit()

    def voiceprint(self, session_id: int, code: str) -> bytes | None:
        with self._lock:
            row = self._db.execute(
                "SELECT centroid FROM voiceprint WHERE session_id=? AND code=?", (session_id, code)
            ).fetchone()
        return row["centroid"] if row else None

    def remember_speaker(self, name: str, centroid: bytes) -> None:
        """Promote one session's voiceprint to a name this room knows.

        The newest recording wins rather than being averaged in: a voice drifts with the room, the
        mic and the codec, and the most recent sample is the closest to the next meeting.
        """
        with self._lock:
            self._db.execute(
                "INSERT INTO known_speaker (name, centroid) VALUES (?,?) "
                "ON CONFLICT(name) DO UPDATE SET centroid=excluded.centroid, "
                "sessions=known_speaker.sessions + 1",
                (name, centroid),
            )
            self._db.commit()

    def known_speakers(self) -> list[tuple[str, bytes]]:
        with self._lock:
            return [(r["name"], r["centroid"]) for r in
                    self._db.execute("SELECT name, centroid FROM known_speaker ORDER BY sessions DESC")]

    def speaker_sessions(self) -> dict[str, int]:
        with self._lock:
            return {r["name"]: r["sessions"] for r in
                    self._db.execute("SELECT name, sessions FROM known_speaker")}

    def speaker_sample(self, name: str) -> tuple[str, float, float | None] | None:
        """Where to hear this voice, and how long that utterance lasts.

        Derived rather than stored — a name is only ever attached on the transcript page, so the
        transcript already knows which recording and which second to play.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT s.wav_path AS wav, l.start AS start, l.end_time AS end_time "
                "FROM speaker_name sn "
                "JOIN line l ON l.session_id=sn.session_id AND l.speaker=sn.code "
                "JOIN session s ON s.id=sn.session_id "
                "WHERE sn.name=? ORDER BY sn.session_id DESC, l.start LIMIT 1",
                (name,),
            ).fetchone()
        return _sample(row)

    def session_speaker_sample(self, session_id: int, code: str) -> tuple[str, float, float | None] | None:
        """Where to hear S3 in this meeting, before anyone has said who S3 is.

        speaker_sample() goes through speaker_name, so it can only find a voice that already has a
        name — which is exactly the voice nobody needs to hear. This one is keyed on the diariser's
        own code, so it works while the field beside it is still empty.

        The longest utterance, not the first: "謝謝" identifies nobody, and picking by length costs
        an ORDER BY. Falls back to text length where the recording has no end time.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT s.wav_path AS wav, l.start AS start, l.end_time AS end_time FROM line l "
                "JOIN session s ON s.id = l.session_id "
                "WHERE l.session_id=? AND l.speaker=? "
                "ORDER BY COALESCE(l.end_time - l.start, 0) DESC, LENGTH(l.source) DESC LIMIT 1",
                (session_id, code),
            ).fetchone()
        return _sample(row)

    def rename_speaker(self, old: str, new: str) -> None:
        """Rename a learned voice everywhere it is used, transcripts included.

        Leaving old transcripts on the wrong name would make the rename look like it half-worked.
        """
        with self._lock:
            self._db.execute("DELETE FROM known_speaker WHERE name=?", (new,))
            self._db.execute("UPDATE known_speaker SET name=? WHERE name=?", (new, old))
            self._db.execute("UPDATE speaker_name SET name=? WHERE name=?", (new, old))
            self._db.commit()

    def forget_speaker(self, name: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM known_speaker WHERE name=?", (name,))
            self._db.commit()
