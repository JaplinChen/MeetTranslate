"""Self-checks for cross-meeting Q&A. Run: python -m server.test_ask

Model-free: keyword expansion, session picking, budget fitting and citation verification are
plain functions; the LLM is a fake chat callable that answers by looking at the prompt.
"""

from __future__ import annotations

import json

from . import ask as A
from .ask import AskLine, Budget


def _line(session_id: int, line_id: int, text: str = "hello", speaker: str = "S1",
          start: float = 1.0) -> AskLine:
    return AskLine(line_id, session_id, start, speaker, text)


def test_parse_keywords_drops_blank_and_short_and_survives_prose():
    raw = ('sure, here you go:\n{"keywords": ["交期", "", "  ", "x", "交期", "delivery date"],'
           ' "since": "2026-01-01", "until": null}\nhope that helps')
    words, since, until = A.parse_keywords(raw)
    assert words == ["交期", "delivery date"]
    assert since == "2026-01-01" and until is None


def test_parse_keywords_degrades_to_empty_on_garbage():
    assert A.parse_keywords("I cannot help with that") == ([], None, None)
    assert A.parse_keywords("{not json at all,,,}") == ([], None, None)


def test_budget_for_ollama_vs_anthropic():
    assert A.budget_for("ollama") == Budget(12_000, 2)
    assert A.budget_for("anthropic") == Budget(120_000, 6)
    assert A.budget_for("") == Budget(120_000, 6)


def test_pick_sessions_prefers_keyword_hits_and_respects_limit():
    index = [{"id": 9}, {"id": 8}, {"id": 7}, {"id": 6}]
    assert A.pick_sessions({7: 5, 6: 9}, index, 4) == [6, 7, 9, 8]
    assert A.pick_sessions({7: 5, 6: 9}, index, 2) == [6, 7]
    assert A.pick_sessions({}, index, 2) == [9, 8]
    # equal hit counts fall back to the index's own (most recent first) order
    assert A.pick_sessions({7: 3, 8: 3}, index, 2) == [8, 7]


def test_fit_leaves_small_transcript_alone():
    lines = [_line(1, i, "x" * 10) for i in range(5)]
    kept, truncated = A.fit(lines, Budget(10_000, 2))
    assert kept == lines and truncated == set()


def test_fit_samples_oversized_session_and_reports_it():
    small = [_line(1, i, "x" * 10) for i in range(3)]
    big = [_line(2, 100 + i, "y" * 100) for i in range(60)]
    kept, truncated = A.fit(small + big, Budget(2_000, 2))  # 1000 chars per session
    assert truncated == {2}
    assert [l for l in kept if l.session_id == 1] == small
    big_kept = [l for l in kept if l.session_id == 2]
    assert 5 <= len(big_kept) < 60
    assert [l.id for l in kept] == sorted(l.id for l in kept)


def test_answer_prompt_renders_markers_and_names_truncated_sessions():
    lines = [_line(3, 42, "改交期", speaker="S2")]
    prompt = A.answer_prompt("誰說要改交期?", [{"id": 3, "started": "2026-07-01"}], lines,
                             {"S2": "Amy"}, {3})
    assert "[3#42] Amy: 改交期" in prompt
    assert "2026-07-01" in prompt
    assert "only part of this meeting was read" in prompt
    clean = A.answer_prompt("q", [{"id": 3, "started": "d"}], lines, {"S2": "Amy"}, set())
    assert "only part of this meeting was read" not in clean


def _reply(answer="the answer", citations=None) -> str:
    return json.dumps({"answer": answer, "citations": citations or []}, ensure_ascii=False)


def test_parse_answer_drops_citation_with_unknown_line_id():
    lines = [_line(1, 10)]
    answer, cites, dropped = A.parse_answer(
        _reply(citations=[{"session_id": 1, "line_id": 10}, {"session_id": 1, "line_id": 99}]),
        lines)
    assert answer == "the answer"
    assert [(c.session_id, c.line_id) for c in cites] == [(1, 10)]
    assert dropped == 1


def test_parse_answer_overwrites_speaker_and_start_from_store_row():
    lines = [_line(1, 10, "real text", speaker="S3", start=12.5)]
    raw = json.dumps({"answer": "a", "citations": [
        {"session_id": 1, "line_id": 10, "speaker": "S9", "start": 999.0, "text": "invented"}]})
    _, cites, dropped = A.parse_answer(raw, lines)
    assert dropped == 0
    assert (cites[0].speaker, cites[0].start, cites[0].text) == ("S3", 12.5, "real text")


def test_parse_answer_raises_on_missing_answer():
    for raw in ("no json here", json.dumps({"citations": []}), _reply(answer="   ")):
        try:
            A.parse_answer(raw, [])
            raise AssertionError("should have raised")
        except ValueError:
            pass


def _fake_chat(replies: dict[str, list[str]], calls: list[str] | None = None):
    def chat(prompt: str) -> str:
        if "would appear verbatim" in prompt:
            kind = "keywords"
        elif "one line per meeting" in prompt:
            kind = "index"
        else:
            kind = "answer"
        if calls is not None:
            calls.append(kind)
        return replies[kind].pop(0)
    return chat


def _harness(answers: list[str], hits=None, keywords='{"keywords": ["交期"]}'):
    index = [{"id": 1, "started": "2026-07-01", "title": "T", "decisions": []},
             {"id": 2, "started": "2026-06-01", "title": "U", "decisions": ["d"]}]
    lines = {1: [_line(1, 10, "改交期")], 2: [_line(2, 20, "產能")]}
    return {
        "index_rows": index,
        "search": lambda kw, since, until: dict(hits if hits is not None else {1: 3, 2: 1}),
        "load_lines": lambda ids: [l for sid in ids for l in lines.get(sid, [])],
        "keywords": keywords,
    }


def test_ask_reports_unverified_when_every_citation_is_dropped():
    h = _harness([])
    calls: list[str] = []
    chat = _fake_chat({"keywords": [h["keywords"]],
                       "answer": [_reply(citations=[{"session_id": 1, "line_id": 777}])]}, calls)
    out = A.ask("誰說要改交期?", chat, h["search"], h["index_rows"], h["load_lines"])
    assert out["verified"] is False
    assert out["dropped_citations"] == 1 and out["citations"] == []
    assert out["sessions"] == [1, 2] and out["truncated"] == []
    assert calls == ["keywords", "answer"]  # two hits: the summary index is not consulted


def test_ask_rejects_oversized_question():
    h = _harness([])
    chat = _fake_chat({"keywords": [h["keywords"]], "answer": [_reply()]})
    for bad in ("", "   ", "x" * (A.MAX_QUESTION_CHARS + 1)):
        try:
            A.ask(bad, chat, h["search"], h["index_rows"], h["load_lines"])
            raise AssertionError("should have raised")
        except ValueError:
            pass


def test_ask_retries_once_then_succeeds():
    h = _harness([])
    calls: list[str] = []
    chat = _fake_chat({"keywords": [h["keywords"]],
                       "answer": ["sorry, no JSON here",
                                  _reply("he did", [{"session_id": 1, "line_id": 10}])]}, calls)
    out = A.ask("誰說要改交期?", chat, h["search"], h["index_rows"], h["load_lines"])
    assert out["answer"] == "he did"
    assert out["verified"] is True and out["dropped_citations"] == 0
    assert out["citations"][0]["speaker"] == "S1"
    assert calls == ["keywords", "answer", "answer"]


def test_ask_resolves_the_citation_speaker_to_a_name():
    """The page has no names map of its own, so the citation must arrive already resolved.

    Without this the answer says 陳經理 while the citation under it says S1, and they read as two
    different people. An unnamed voice keeps its code, which is what the transcript shows too.
    """
    h = _harness([])
    chat = _fake_chat({"keywords": [h["keywords"]],
                       "answer": [_reply(citations=[{"session_id": 1, "line_id": 10}])]})
    out = A.ask("誰改了交期?", chat, h["search"], h["index_rows"], h["load_lines"],
                names={"S1": "陳經理"})
    assert out["citations"][0]["speaker"] == "陳經理", out["citations"]

    # An unnamed speaker is left as its code, not blanked.
    out2 = A.ask("誰改了交期?",
                 _fake_chat({"keywords": [h["keywords"]],
                             "answer": [_reply(citations=[{"session_id": 1, "line_id": 10}])]}),
                 h["search"], h["index_rows"], h["load_lines"], names={"S9": "別人"})
    assert out2["citations"][0]["speaker"] == "S1", out2["citations"]


def test_ask_falls_back_to_the_summary_index_when_keyword_hits_are_thin():
    h = _harness([], hits={1: 2})
    calls: list[str] = []
    chat = _fake_chat({"keywords": [h["keywords"]],
                       "index": [json.dumps({"sessions": [2, 999]})],
                       "answer": [_reply()]}, calls)
    out = A.ask("q", chat, h["search"], h["index_rows"], h["load_lines"], provider="ollama")
    assert calls == ["keywords", "index", "answer"]
    assert out["sessions"] == [1, 2]  # 999 is not a known id and was dropped


def main() -> None:
    checks = sorted((n, f) for n, f in globals().items() if n.startswith("test_"))
    for name, fn in checks:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(checks)} passed")


if __name__ == "__main__":
    main()
