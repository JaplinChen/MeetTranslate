"""SQLite persistence: glossary, sessions, transcript lines and corrections.

One connection with `check_same_thread=False` because the capture pipeline writes from a worker
thread while FastAPI reads from the event loop; a lock serialises them. WAL keeps a long-running
write from blocking the subtitle page's reads.

The tables themselves live in `schema`, and the speaker half of the API in `speakers` — both share
this connection and this lock, so `Store` stays the only thing that talks to the database.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from . import config, schema
from .speakers import SpeakerStore

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


@dataclass
class Term:
    id: int
    source: str
    lang: str
    mode: str
    category: str
    targets: dict[str, str]


class Store(SpeakerStore):
    def __init__(self, path: Path | None = None):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path or DB_PATH), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            schema.apply(self._db)

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
            self._bump_rev_for_line(line_id)
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
                self._bump_rev_for_line(line_id)
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def _bump_rev_for_line(self, line_id: int) -> None:
        """Record that this line's session no longer says what it said. Caller holds the lock.

        The revision is what lets anything derived from the transcript — the summary — admit it
        is describing an older version. Bumped inside the same transaction as the edit, so a
        rollback takes the bump with it.
        """
        self._db.execute(
            "UPDATE session SET lines_rev = lines_rev + 1 "
            "WHERE id = (SELECT session_id FROM line WHERE id=?)", (line_id,))

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
                self._db.execute("UPDATE session SET lines_rev = lines_rev + 1 WHERE id=?",
                                 (session_id,))
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

    # ── summary ─────────────────────────────────────────────────────────

    def set_summary(self, session_id: int, json_text: str, status: str, lines_rev: int,
                    created: str) -> None:
        """One summary per session, latest wins.

        `lines_rev` is the session's revision at generation time. A later read that finds the
        session's current revision has moved on knows the summary describes a transcript that no
        longer exists — stale is a comparison, not a stored flag, so nothing has to remember to
        set it.
        """
        with self._lock:
            self._db.execute(
                "INSERT INTO summary (session_id, json, status, lines_rev, created) "
                "VALUES (?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET "
                "json=excluded.json, status=excluded.status, lines_rev=excluded.lines_rev, "
                "created=excluded.created",
                (session_id, json_text, status, lines_rev, created))
            self._db.commit()

    def summary(self, session_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM summary WHERE session_id=?",
                                   (session_id,)).fetchone()
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

    def edit_correction(self, old_wrong: str, wrong: str, right: str) -> None:
        """Fix a learned pair in place, either side of it.

        Until now the only repair was to delete and re-learn, which means finding the line it came
        from and correcting it again — and a pair learned from a typo is exactly the one you cannot
        reproduce on demand. `wrong` is the key, so changing it is a rename rather than an update.
        """
        wrong, right = wrong.strip(), right.strip()
        if not wrong or not right:
            raise ValueError("both sides of a correction must be filled in")
        if wrong == right:
            raise ValueError("a correction that rewrites text to itself would never stop matching")
        with self._lock:
            if wrong != old_wrong and self._db.execute(
                "SELECT 1 FROM correction WHERE wrong=?", (wrong,)
            ).fetchone():
                # Overwriting would silently discard whichever pair the user was not looking at.
                raise ValueError(f"there is already a correction for {wrong}")
            changed = self._db.execute(
                "UPDATE correction SET wrong=?, right=? WHERE wrong=?", (wrong, right, old_wrong)
            ).rowcount
            if not changed:
                raise KeyError(old_wrong)
            self._db.commit()

    def forget_correction(self, wrong: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM correction WHERE wrong=?", (wrong,))
            self._db.commit()
