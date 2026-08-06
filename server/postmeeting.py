"""The stages of the post-meeting pass that talk to a language model instead of the GPU.

Kept out of `jobs` because they need the store and the LLM configuration, and out of `main`
because they are policy, not routing. The split that matters is the one `jobs.schedule` enforces:
everything here runs as a followup, after the GPU gate is released, because none of it touches
the card — and holding the gate through a minutes-long Ollama call meant the next meeting was
told it could not start recording.

Two stages, landing independently:

refine — the correction pass that until now only existed as a CLI script over exported
transcripts. The guards, the coverage accounting and the prompt were all written and validated
there; this wires the same `Refiner` over the stored lines, so every transcript gets the
context-aware fixes instead of only the ones someone exported and re-imported by hand.

summarize — one structured summary per configured language, generated from the refined lines so
it describes the transcript the reader will actually see.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Callable

from . import jobs, llm, refine, summarize
from .store import Store

log = logging.getLogger("meettranslate.postmeeting")


def chat_for(llm_cfg: llm.LlmConfig, api_key: str, max_tokens: int) -> Callable[[str], str] | None:
    """The chat callable the configured provider implies. None means no LLM stage can run.

    Follows the provider chosen on the LLM settings page rather than hard-coding Anthropic:
    `refine.ollama_chat` exists precisely because these transcripts may not be allowed to leave
    the machine, and a summary that quietly bypassed that choice would undo it.
    """
    if llm_cfg.provider == "ollama":
        return refine.ollama_chat(llm_cfg.model,
                                  llm_cfg.endpoint or llm.DEFAULT_ENDPOINTS["ollama"])
    if not api_key:
        return None
    return refine.anthropic_chat(api_key, llm_cfg.model, max_tokens=max_tokens)


def _cancellable(chat: Callable[[str], str], cancel: threading.Event) -> Callable[[str], str]:
    """Checked before every model call, so cancellation waits for one chunk, not one transcript."""
    def wrapped(prompt: str) -> str:
        if cancel.is_set():
            raise jobs.Cancelled()
        return chat(prompt)
    return wrapped


def _refine_stage(store: Store, session_id: int, chat: Callable[[str], str]) -> None:
    rows = store.lines(session_id)
    if not rows:
        return
    lines = [refine.Line(r["speaker"], r["lang"], r["source"]) for r in rows]
    coverage = refine.Coverage()
    corrected = refine.Refiner(chat).refine(lines, terms=store.glossary(), coverage=coverage)
    changed = 0
    for row, text in zip(rows, corrected):
        if text != row["source"]:
            # update_line, not replace_line: this is the one writer entitled to mark `refined`,
            # and it must not touch status — a line the decoder failed on stays visibly failed.
            store.update_line(row["id"], text, {})
            changed += 1
    log.info("refine stage: %d/%d lines corrected, %.0f%% of the transcript checked",
             changed, len(rows), (1 - coverage.fraction) * 100)


def _summarize_stage(store: Store, session_id: int, languages: list[str],
                     llm_cfg: llm.LlmConfig, api_key: str, cancel: threading.Event) -> None:
    # Read after the refine stage so the summary describes the transcript the reader will see,
    # and capture the revision the same read observed: stale is a comparison against this number.
    rows = store.lines(session_id)
    if not rows:
        return
    session = store.session(session_id)
    rev = int(session["lines_rev"]) if session else 0

    lines = [summarize.SummaryLine(r["speaker"], r["lang"], r["source"]) for r in rows]
    target = summarize.target_chars(sum(len(l.text) for l in lines))
    chat = chat_for(llm_cfg, api_key, max_tokens=summarize.max_tokens_for(target))
    if chat is None:
        return

    result, status = summarize.summarize(lines, languages, _cancellable(chat, cancel),
                                         should_stop=cancel.is_set)
    # Landed even when partial: two of three languages beats none, the card says so, and
    # regenerating later is one click. Nothing at all came back → failed is still worth storing,
    # because "tried and failed" and "never ran" are different answers to "where is my summary".
    store.set_summary(session_id, json.dumps(result, ensure_ascii=False), status, rev,
                      time.strftime("%Y-%m-%dT%H:%M:%S"))


def followup(store: Store, languages: list[str], llm_cfg: llm.LlmConfig, api_key: str,
             session_id: int) -> Callable[[threading.Event, Callable[[str], None]], None]:
    """The post-GPU stages, as one callable for `jobs.schedule`.

    Stages land independently: a summary that fails does not undo a refine that succeeded, which
    is why each writes to the store as it finishes rather than at the end.
    """
    def run(cancel: threading.Event, set_stage: Callable[[str], None]) -> None:
        chat = chat_for(llm_cfg, api_key, max_tokens=4000)
        if chat is None:
            log.warning("no LLM configured — transcript stays unrefined and unsummarized")
            return

        set_stage("refine")
        if cancel.is_set():
            raise jobs.Cancelled()
        _refine_stage(store, session_id, _cancellable(chat, cancel))

        set_stage("summarize")
        if cancel.is_set():
            raise jobs.Cancelled()
        _summarize_stage(store, session_id, languages, llm_cfg, api_key, cancel)

        if cancel.is_set():
            raise jobs.Cancelled()

    return run
