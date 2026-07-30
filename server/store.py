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
TERM_MODES = ("translate", "keep", "hint")

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
    refined    INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS line_translation (
    line_id   INTEGER NOT NULL REFERENCES line(id) ON DELETE CASCADE,
    lang      TEXT NOT NULL,
    text      TEXT NOT NULL,
    PRIMARY KEY (line_id, lang)
);
CREATE TABLE IF NOT EXISTS speaker_name (
    session_id INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    code       TEXT NOT NULL,
    name       TEXT NOT NULL,
    PRIMARY KEY (session_id, code)
);
CREATE INDEX IF NOT EXISTS line_session ON line(session_id, start);
"""


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
                 translations: dict[str, str]) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO line (session_id, start, speaker, lang, source) VALUES (?,?,?,?,?)",
                (session_id, start, speaker, lang, source),
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

    def clear_lines(self, session_id: int) -> None:
        """Drop the live transcript before postprocess writes the re-derived one."""
        with self._lock:
            self._db.execute("DELETE FROM line WHERE session_id=?", (session_id,))
            self._db.commit()

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

    def speaker_names(self, session_id: int) -> dict[str, str]:
        with self._lock:
            return {r["code"]: r["name"] for r in self._db.execute(
                "SELECT code, name FROM speaker_name WHERE session_id=?", (session_id,)
            )}
