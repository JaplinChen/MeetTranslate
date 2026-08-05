"""Self-checks for the summary pass. Run: python -m server.test_summarize

Model-free: the prompt, the sampler, the schema check and the retry loop are plain functions;
the LLM is a fake chat callable.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Callable

from . import summarize as S
from .summarize import SummaryLine


def _lines(counts: dict[str, int], chars: int = 100) -> list[SummaryLine]:
    out, i = [], 0
    speakers = sorted(counts)
    remaining = dict(counts)
    while any(remaining.values()):
        for sp in speakers:
            if remaining[sp]:
                remaining[sp] -= 1
                out.append(SummaryLine(sp, "zh", f"{i:04d}" + "x" * (chars - 4)))
                i += 1
    return out


def test_target_chars_clamps_both_ends_and_scales():
    assert S.target_chars(0) == 200
    assert S.target_chars(2400) == 200
    assert S.target_chars(12_000) == 1000
    assert S.target_chars(1_000_000) == 2000


def test_load_rules_builtin_default_when_file_absent():
    saved = S.RULES_PATH
    try:
        S.RULES_PATH = Path(tempfile.gettempdir()) / "no_such_summary_rules.md"
        assert S.load_rules() == S.DEFAULT_RULES
        custom = Path(tempfile.mkdtemp()) / "summary_rules.md"
        custom.write_text("- custom rule", encoding="utf-8")
        S.RULES_PATH = custom
        assert S.load_rules() == "- custom rule"
    finally:
        S.RULES_PATH = saved


def test_build_prompt_asks_for_one_language_with_rules_and_excerpt_note():
    lines = [SummaryLine("S1", "zh", "hello")]
    prompt = S.build_prompt(lines, "vi", "- my rule", sampled=True)
    assert "Vietnamese" in prompt
    assert "Chinese" not in prompt and "English" not in prompt
    assert "- my rule" in prompt
    assert "excerpt" in prompt
    assert "S1(zh): hello" in prompt
    assert "excerpt" not in S.build_prompt(lines, "vi", "- my rule", sampled=False)


def test_build_prompt_targets_original_total_not_sampled():
    lines = [SummaryLine("S1", "zh", "x" * 100)]
    prompt = S.build_prompt(lines, "en", "-", sampled=True, total_chars=24_000)
    assert str(S.target_chars(24_000)) in prompt


def test_sample_identity_under_budget():
    lines = _lines({"S1": 5}, chars=10)
    out, sampled = S.sample(lines, budget_chars=1000)
    assert out == lines and sampled is False


def test_sample_proportional_and_ordered_over_budget():
    lines = _lines({"S1": 60, "S2": 30, "S3": 10}, chars=100)
    out, sampled = S.sample(lines, budget_chars=3000)  # keep ~30%
    assert sampled is True
    counts = {sp: sum(1 for l in out if l.speaker == sp) for sp in ("S1", "S2", "S3")}
    assert 15 <= counts["S1"] <= 21
    assert 7 <= counts["S2"] <= 11
    assert 2 <= counts["S3"] <= 4
    positions = [lines.index(l) for l in out]
    assert positions == sorted(positions)


def _valid(title="T", summary="S", decisions=None, actions=None) -> str:
    return json.dumps({"title": title, "summary": summary,
                       "decisions": decisions if decisions is not None else [],
                       "actions": actions if actions is not None else []})


def test_parse_response_accepts_valid_json_with_empty_arrays():
    got = S.parse_response("noise before " + _valid())
    assert got == {"title": "T", "summary": "S", "decisions": [], "actions": []}


def test_parse_response_rejects_missing_title():
    try:
        S.parse_response(json.dumps({"summary": "S", "decisions": [], "actions": []}))
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "title" in str(e)


def test_parse_response_rejects_wrong_type_decisions():
    try:
        S.parse_response(_valid(decisions=["ok", 5]))
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "decisions" in str(e)


def test_parse_response_rejects_action_without_text():
    try:
        S.parse_response(_valid(actions=[{"speaker": "S1"}]))
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "text" in str(e)


def test_retry_prompt_carries_error_and_truncates_bad_reply():
    prompt = S.retry_prompt("ORIGINAL", "y" * 900, "the error")
    assert "ORIGINAL" in prompt
    assert "the error" in prompt
    assert "y" * 500 in prompt
    assert "y" * 501 not in prompt


def test_max_tokens_for():
    assert S.max_tokens_for(1000) == 2500


def _chat_by_lang(replies: dict[str, list[str]]) -> Callable[[str], str]:
    def chat(prompt: str) -> str:
        for lang, name in (("zh", "Chinese"), ("en", "English"), ("vi", "Vietnamese")):
            if name in prompt.splitlines()[0]:
                return replies[lang].pop(0)
        raise AssertionError("unknown language in prompt")
    return chat


def test_summarize_happy_path():
    lines = [SummaryLine("S1", "zh", "hello")]
    out, status = S.summarize(lines, ["zh", "en"],
                              _chat_by_lang({"zh": [_valid("zt")], "en": [_valid("et")]}))
    assert status == "ok"
    assert out["zh"]["title"] == "zt" and out["en"]["title"] == "et"


def test_summarize_retries_once_then_partial():
    calls = {"zh": ["garbage", "still garbage"], "en": [_valid("et")]}
    out, status = S.summarize([SummaryLine("S1", "zh", "hi")], ["zh", "en"], _chat_by_lang(calls))
    assert status == "partial"
    assert "zh" not in out and out["en"]["title"] == "et"
    assert not calls["zh"]  # both attempts consumed: one retry, no more


def test_summarize_retry_succeeds():
    calls = {"zh": ["garbage", _valid("zt")]}
    out, status = S.summarize([SummaryLine("S1", "zh", "hi")], ["zh"], _chat_by_lang(calls))
    assert status == "ok" and out["zh"]["title"] == "zt"


def test_summarize_failed_when_all_garbage():
    calls = {"zh": ["g", "g"], "en": ["g", "g"]}
    out, status = S.summarize([SummaryLine("S1", "zh", "hi")], ["zh", "en"], _chat_by_lang(calls))
    assert status == "failed" and out == {}


def test_summarize_stops_early_when_asked():
    stopped = iter([False, True, True])
    out, status = S.summarize([SummaryLine("S1", "zh", "hi")], ["zh", "en", "vi"],
                              _chat_by_lang({"zh": [_valid("zt")]}),
                              should_stop=lambda: next(stopped))
    assert status == "partial"
    assert list(out) == ["zh"]


def main() -> None:
    checks = sorted((n, f) for n, f in globals().items() if n.startswith("test_"))
    for name, fn in checks:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(checks)} passed")


if __name__ == "__main__":
    main()
