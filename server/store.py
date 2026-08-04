"""SQLite persistence: glossary, sessions and transcript lines.

One connection with `check_same_thread=False` because the capture pipeline writes from a worker
thread while FastAPI reads from the event loop; a lock serialises them. WAL keeps a long-running
write from blocking the subtitle page's reads.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from . import config

DB_PATH = config.ROOT / "meettranslate.db"

# How a glossary term is applied. `keep` exists because code-switched English terms
# ("schedule", "delay") are shared vocabulary in cross-border teams — translating them into
# Vietnamese makes the subtitle harder to read, not easier.
#
# `protect` is the opposite of the others: it declares a word real so the corrector leaves it
# alone, and never rewrites anything into it. Needed because 才夠 and 採購 are homophones and both
# are ordinary speech — registering 才夠 to shield it made it a target instead, and 採購 was
# rewritten to 才夠 211 times across seven interviews.
TERM_MODES = ("translate", "keep", "hint", "protect")

SCHEMA = """
CREATE TABLE IF NOT EXISTS glossary (
    id        INTEGER PRIMARY KEY,
    source    TEXT NOT NULL,
    lang      TEXT NOT NULL DEFAULT '',
    mode      TEXT NOT NULL DEFAULT 'translate',
    category  TEXT NOT NULL DEFAULT '',
    UNIQUE(source, lang)
);
CREATE TABLE IF NOT EXISTS glossary_target (
    term_id   INTEGER NOT NULL REFERENCES glossary(id) ON DELETE CASCADE,
    lang      TEXT NOT NULL,
    text      TEXT NOT NULL,
    PRIMARY KEY (term_id, lang)
);
CREATE TABLE IF NOT EXISTS session (
    id        INTEGER PRIMARY KEY,
    started   TEXT NOT NULL,
    ended     TEXT,
    wav_path  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS line (
    id         INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    start      REAL NOT NULL,
    speaker    TEXT NOT NULL,
    lang       TEXT NOT NULL,
    source     TEXT NOT NULL,
    refined    INTEGER NOT NULL DEFAULT 0,
    status     TEXT NOT NULL DEFAULT 'ok',
    end_time   REAL
);
CREATE TABLE IF NOT EXISTS line_translation (
    line_id   INTEGER NOT NULL REFERENCES line(id) ON DELETE CASCADE,
    lang      TEXT NOT NULL,
    text      TEXT NOT NULL,
    PRIMARY KEY (line_id, lang)
);
CREATE TABLE IF NOT EXISTS correction (
    wrong  TEXT PRIMARY KEY,
    right  TEXT NOT NULL,
    lang   TEXT NOT NULL DEFAULT '',
    count  INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS voiceprint (
    session_id INTEGER REFERENCES session(id) ON DELETE CASCADE,
    code       TEXT NOT NULL,
    centroid   BLOB NOT NULL,
    PRIMARY KEY (session_id, code)
);
CREATE TABLE IF NOT EXISTS known_speaker (
    name     TEXT PRIMARY KEY,
    centroid BLOB NOT NULL,
    sessions INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS speaker_name (
    session_id INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    code       TEXT NOT NULL,
    name       TEXT NOT NULL,
    PRIMARY KEY (session_id, code)
);
CREATE INDEX IF NOT EXISTS line_session ON line(session_id, start);
"""

# `CREATE TABLE IF NOT EXISTS` is a no-op against a table that already exists, so a database
# created before a column was added never gets it. New machines and CI pass either way; the meeting
# room's database is the one that breaks, and it breaks inside the capture thread where
# Pipeline._handle swallows it as one more error count. Each entry is (column, DDL), applied only
# when the column is absent.
_LINE_COLUMNS = (
    ("status", "ALTER TABLE line ADD COLUMN status TEXT NOT NULL DEFAULT 'ok'"),
    # No NOT NULL: rows written before this column existed have no end to backfill, and guessing
    # one would be worse than admitting it is unknown.
    ("end_time", "ALTER TABLE line ADD COLUMN end_time REAL"),
)


@dataclass
class Term:
    id: int
    source: str
    lang: str
    mode: str
    category: str
    targets: dict[str, str]


class Store:
    def __init__(self, path: Path | None = None):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path or DB_PATH), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()
            self._migrate()

    def _migrate(self) -> None:
        """Add columns the schema gained after this database was created.

        Deliberately not caught: starting with a stale schema is worse than not starting. The
        alternative is a room that records a whole meeting into a table that rejects every insert.
        """
        have = {r["name"] for r in self._db.execute("PRAGMA table_info(line)")}
        added = [ddl for column, ddl in _LINE_COLUMNS if column not in have]
        for ddl in added:
            self._db.execute(ddl)
        if added:
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ── glossary ────────────────────────────────────────────────────────

    def add_term(self, source: str, targets: dict[str, str], lang: str = "", mode: str = "translate",
                 category: str = "") -> Term:
        if mode not in TERM_MODES:
            raise ValueError(f"mode must be one of {TERM_MODES}")
        source = source.strip()
        if not source:
            raise ValueError("source must not be empty")

        with self._lock:
            cur = self._db.execute(
                "INSERT INTO glossary (source, lang, mode, category) VALUES (?,?,?,?) "
                "ON CONFLICT(source, lang) DO UPDATE SET mode=excluded.mode, category=excluded.category "
                "RETURNING id",
                (source, lang, mode, category),
            )
            term_id = cur.fetchone()[0]
            self._db.execute("DELETE FROM glossary_target WHERE term_id=?", (term_id,))
            self._db.executemany(
                "INSERT INTO glossary_target (term_id, lang, text) VALUES (?,?,?)",
                [(term_id, k, v) for k, v in targets.items() if v.strip()],
            )
            self._db.commit()
        return Term(term_id, source, lang, mode, category, dict(targets))

    def remove_term(self, source: str, lang: str = "") -> None:
        with self._lock:
            self._db.execute("DELETE FROM glossary WHERE source=? AND lang=?", (source, lang))
            self._db.commit()

    def glossary(self) -> list[Term]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM glossary ORDER BY source").fetchall()
            targets: dict[int, dict[str, str]] = {}
            for t in self._db.execute("SELECT * FROM glossary_target"):
                targets.setdefault(t["term_id"], {})[t["lang"]] = t["text"]
        return [Term(r["id"], r["source"], r["lang"], r["mode"], r["category"], targets.get(r["id"], {})) for r in rows]

    # ── sessions and lines ──────────────────────────────────────────────

    def start_session(self, started: str, wav_path: str) -> int:
        with self._lock:
            cur = self._db.execute("INSERT INTO session (started, wav_path) VALUES (?,?)", (started, wav_path))
            self._db.commit()
            return int(cur.lastrowid)

    def end_session(self, session_id: int, ended: str) -> None:
        with self._lock:
            self._db.execute("UPDATE session SET ended=? WHERE id=?", (ended, session_id))
            self._db.commit()

    def add_line(self, session_id: int, start: float, speaker: str, lang: str, source: str,
                 translations: dict[str, str], status: str = "ok",
                 end_time: float | None = None) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO line (session_id, start, speaker, lang, source, status, end_time) "
                "VALUES (?,?,?,?,?,?,?)",
                (session_id, start, speaker, lang, source, status, end_time),
            )
            line_id = int(cur.lastrowid)
            self._db.executemany(
                "INSERT INTO line_translation (line_id, lang, text) VALUES (?,?,?)",
                [(line_id, k, v) for k, v in translations.items()],
            )
            self._db.commit()
        return line_id

    def update_line(self, line_id: int, source: str | None, translations: dict[str, str]) -> None:
        """Apply a refinement. Marks the line refined so it is never rewritten twice."""
        with self._lock:
            if source is not None:
                self._db.execute("UPDATE line SET source=? WHERE id=?", (source, line_id))
            self._db.executemany(
                "INSERT INTO line_translation (line_id, lang, text) VALUES (?,?,?) "
                "ON CONFLICT(line_id, lang) DO UPDATE SET text=excluded.text",
                [(line_id, k, v) for k, v in translations.items()],
            )
            self._db.execute("UPDATE line SET refined=1 WHERE id=?", (line_id,))
            self._db.commit()

    def line(self, line_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM line WHERE id=?", (line_id,)).fetchone()
        return dict(row) if row else None

    def replace_line(self, line_id: int, source: str, lang: str, translations: dict[str, str],
                     status: str) -> None:
        """Overwrite one line after re-running it. Leaves `refined` alone.

        `refined` records that the translator revised this line in hindsight, which is a different
        claim from "someone re-ran it", and conflating the two would let a rerun suppress the one
        refinement pass the line is still entitled to.
        """
        with self._lock:
            try:
                self._db.execute("UPDATE line SET source=?, lang=?, status=? WHERE id=?",
                                 (source, lang, status, line_id))
                self._db.execute("DELETE FROM line_translation WHERE line_id=?", (line_id,))
                self._db.executemany(
                    "INSERT INTO line_translation (line_id, lang, text) VALUES (?,?,?)",
                    [(line_id, k, v) for k, v in translations.items()],
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def replace_lines(self, session_id: int, rows: list[dict]) -> None:
        """Swap a session's whole transcript in one transaction.

        The obvious shape — delete the old lines, then insert each new one — commits after every
        line, so a run that dies in the middle leaves the transcript half-replaced, and a run that
        dies right after the delete leaves it empty. That is data loss, not a slow path: the
        recording is still on disk but the meeting's transcript is gone until someone notices.

        Every insert here shares one implicit transaction and one commit, so the transcript is
        either entirely the old one or entirely the new one. The caller must have finished
        translating before calling: holding a write lock across an LLM round trip would block the
        next meeting's first line on `database is locked`.
        """
        with self._lock:
            try:
                self._db.execute("DELETE FROM line WHERE session_id=?", (session_id,))
                for row in rows:
                    cur = self._db.execute(
                        "INSERT INTO line (session_id, start, speaker, lang, source, status, end_time) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (session_id, row["start"], row["speaker"], row["lang"], row["source"],
                         row.get("status", "ok"), row.get("end_time")),
                    )
                    self._db.executemany(
                        "INSERT INTO line_translation (line_id, lang, text) VALUES (?,?,?)",
                        [(int(cur.lastrowid), k, v) for k, v in row.get("translations", {}).items()],
                    )
                self._db.commit()
            except Exception:
                # Without this the half-finished transaction stays open on a connection every other
                # method shares, so the next commit anywhere in the Store would commit this delete.
                # The failure would surface later, somewhere else, as a transcript that vanished.
                self._db.rollback()
                raise

    def session(self, session_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM session WHERE id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def lines(self, session_id: int) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM line WHERE session_id=? ORDER BY start", (session_id,)
            ).fetchall()
            tr: dict[int, dict[str, str]] = {}
            for t in self._db.execute(
                "SELECT lt.* FROM line_translation lt JOIN line l ON l.id=lt.line_id WHERE l.session_id=?",
                (session_id,),
            ):
                tr.setdefault(t["line_id"], {})[t["lang"]] = t["text"]
        return [{**dict(r), "translations": tr.get(r["id"], {})} for r in rows]

    def sessions(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM line WHERE session_id=s.id) AS lines "
                "FROM session s ORDER BY s.id DESC"
            )]

    def set_speaker_name(self, session_id: int, code: str, name: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO speaker_name (session_id, code, name) VALUES (?,?,?) "
                "ON CONFLICT(session_id, code) DO UPDATE SET name=excluded.name",
                (session_id, code, name),
            )
            self._db.commit()

    # ── corrections ─────────────────────────────────────────────────────
    #
    # An edit made on the transcript page is ground truth: this is what the recogniser wrote and
    # this is what was actually said. Nothing else in the system is labelled by a human who was in
    # the room, so it outranks every heuristic that guesses from pinyin.

    def add_correction(self, wrong: str, right: str, lang: str = "") -> None:
        wrong, right = wrong.strip(), right.strip()
        if not wrong or not right or wrong == right:
            return
        with self._lock:
            self._db.execute(
                "INSERT INTO correction (wrong, right, lang) VALUES (?,?,?) "
                "ON CONFLICT(wrong) DO UPDATE SET right=excluded.right, count=correction.count + 1",
                (wrong, right, lang),
            )
            self._db.commit()

    def corrections(self) -> dict[str, str]:
        with self._lock:
            return {r["wrong"]: r["right"] for r in
                    self._db.execute("SELECT wrong, right FROM correction ORDER BY LENGTH(wrong) DESC")}

    def forget_correction(self, wrong: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM correction WHERE wrong=?", (wrong,))
            self._db.commit()

    # ── voiceprints ─────────────────────────────────────────────────────
    #
    # Naming S1 as "Vincent" is knowledge the meeting room throws away every time. Kept here, the
    # next meeting recognises the voice instead of asking again. Two tables because the two facts
    # arrive at different times: the centroid exists while the meeting runs, the name usually
    # arrives afterwards from the transcript page.

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

    def speaker_sample(self, name: str) -> tuple[str, float] | None:
        """Where to hear this voice: the newest line anyone attributed to that name.

        Derived rather than stored — a name is only ever attached on the transcript page, so the
        transcript already knows which recording and which second to play.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT s.wav_path AS wav, l.start AS start FROM speaker_name sn "
                "JOIN line l ON l.session_id=sn.session_id AND l.speaker=sn.code "
                "JOIN session s ON s.id=sn.session_id "
                "WHERE sn.name=? ORDER BY sn.session_id DESC, l.start LIMIT 1",
                (name,),
            ).fetchone()
        return (row["wav"], row["start"]) if row else None

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

    def transcript_text(self, limit: int = 20000) -> str:
        """Every line this room has recorded, as one string.

        The corpus for deciding whether a glossary term is safe is the meeting history itself —
        what these people actually say, rather than a dictionary of what Mandarin permits.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT source FROM line ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return chr(10).join(r["source"] for r in rows)

    def speaker_names(self, session_id: int) -> dict[str, str]:
        with self._lock:
            return {r["code"]: r["name"] for r in self._db.execute(
                "SELECT code, name FROM speaker_name WHERE session_id=?", (session_id,)
            )}
