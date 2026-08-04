"""Glossary terms and the collision check that runs before one is added."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from . import correct, main
from .store import TERM_MODES

router = APIRouter()


def _term_json(t) -> dict:
    return {"id": t.id, "source": t.source, "lang": t.lang, "mode": t.mode,
            "category": t.category, "targets": t.targets}


@router.get("/api/glossary")
def get_glossary() -> list[dict]:
    return [_term_json(t) for t in main.store.glossary()]


@router.post("/api/glossary")
def post_glossary(body: dict) -> list[dict]:
    mode = str(body.get("mode", "translate"))
    if mode not in TERM_MODES:
        raise HTTPException(400, f"mode must be one of {TERM_MODES}")
    try:
        main.store.add_term(
            source=str(body.get("source", "")),
            targets={str(k): str(v) for k, v in dict(body.get("targets", {})).items()},
            lang=str(body.get("lang", "")),
            mode=mode,
            category=str(body.get("category", "")),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_glossary()


@router.get("/api/glossary/collisions")
def get_collisions(source: str) -> dict:
    """What adding this term would overwrite in the meetings already recorded.

    Called before the term is added, because afterwards is too late to be useful: adding 料號
    rewrote 料耗 — a real term of the trade — forty-two times, silently.
    """
    source = source.strip()
    if not source:
        raise HTTPException(400, "source required")
    known = {t.source for t in main.store.glossary()} | {source}
    hits = correct.collisions(source, main.store.transcript_text(), known)
    return {"source": source,
            "collisions": [{"text": w, "count": n} for w, n in
                           sorted(hits.items(), key=lambda kv: -kv[1])]}


@router.delete("/api/glossary")
def delete_glossary(source: str, lang: str = "") -> list[dict]:
    main.store.remove_term(source, lang)
    return get_glossary()
