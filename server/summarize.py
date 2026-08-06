"""Post-meeting summary: one LLM call per language, schema-checked, retried once.

The transcript is the input everyone already has; what a meeting produces is the part nobody wrote
down — what was decided and who left owning what. A model asked for that in free prose invents
structure on some days and skips it on others, so the reply is pinned to a JSON schema and a bad
reply is sent back once with the parser's complaint attached. Three languages in one call was
tried and truncates the JSON inside a 4000-token budget, hence one language per call.

Pure orchestration: the chat callable comes from the caller (refine.anthropic_chat or ollama_chat,
so the privacy choice made there carries over) and nothing here touches the store.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from . import config
from .translate import language_name

# Fits Ollama's num_ctx=8192 tokens for CJK-heavy text with prompt overhead.
INPUT_BUDGET = 12_000

RULES_PATH = config.ROOT / "summary_rules.md"

DEFAULT_RULES = "\n".join([
    "- 標題一句話",
    "- 摘要客觀轉述會議內容，不加評論",
    "- 決議＝已拍板的事",
    "- 行動項目＝誰要去做什麼",
    "- 沒有就留空陣列，不要硬湊",
])


@dataclass
class SummaryLine:
    speaker: str
    lang: str
    text: str


def target_chars(total_chars: int) -> int:
    """Summary length scales with the transcript, within reason either way."""
    return max(200, min(2000, total_chars // 12))


def load_rules() -> str:
    """User-editable format rules; the built-in default when the file does not exist."""
    if RULES_PATH.is_file():
        return RULES_PATH.read_text(encoding="utf-8")
    return DEFAULT_RULES


def sample(lines: list[SummaryLine], budget_chars: int = INPUT_BUDGET) -> tuple[list[SummaryLine], bool]:
    """Cut an over-budget transcript by evenly-spaced sampling, per speaker.

    Truncating from the front would summarize the first hour of a two-hour meeting. Sampling
    per speaker keeps each participant's share of the floor, so a quiet decision-maker is not
    sampled out by a talkative colleague.
    """
    total = sum(len(l.text) for l in lines)
    if total <= budget_chars:
        return lines, False

    ratio = budget_chars / total
    by_speaker: dict[str, list[int]] = {}
    for i, line in enumerate(lines):
        by_speaker.setdefault(line.speaker, []).append(i)

    kept: list[int] = []
    for indices in by_speaker.values():
        n = max(1, round(len(indices) * ratio))
        step = len(indices) / n
        kept += [indices[int(k * step)] for k in range(n)]

    return [lines[i] for i in sorted(kept)], True


def build_prompt(lines: list[SummaryLine], lang: str, rules: str,
                 speakers: dict[str, str] | None = None, sampled: bool = False,
                 total_chars: int | None = None) -> str:
    """One language per call — a multi-language reply truncates inside the token budget."""
    # Length target is set by the whole meeting, not by however much survived sampling.
    total = total_chars if total_chars is not None else sum(len(l.text) for l in lines)
    target = target_chars(total)

    parts = [
        f"Summarize this meeting transcript. Write everything in {language_name(lang)}.",
        "",
        "Rules:",
        rules,
        "",
        f"The summary text should be about {target} characters.",
    ]
    if sampled:
        parts += ["", "The transcript below is an evenly-sampled excerpt of a longer meeting."]

    parts += ["", "Transcript:"]
    parts += [f"{l.speaker}({l.lang}): {l.text}" for l in lines]

    parts += [
        "",
        "Reply with JSON only, no code fence and no commentary:",
        json.dumps({"title": "one sentence", "summary": "...", "decisions": ["..."],
                    "actions": [{"text": "...", "speaker": "speaker code"}]}, ensure_ascii=False),
        "Use the speaker codes exactly as they appear in the transcript; \"\" when unclear.",
        "decisions and actions may be empty arrays — do not invent items to fill them.",
    ]
    return "\n".join(parts)


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(raw: str, valid_speakers: frozenset[str] = frozenset()) -> dict:
    """Strict schema check: a wrong shape raises so the retry loop can quote the exact problem."""
    match = _JSON.search(raw)
    if not match:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    data = json.loads(match.group(0))

    for key in ("title", "summary"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string, got {value!r}")

    decisions = data.get("decisions")
    if not isinstance(decisions, list) or not all(isinstance(d, str) for d in decisions):
        raise ValueError(f"decisions must be a list of strings, got {decisions!r}")

    actions = data.get("actions")
    if not isinstance(actions, list):
        raise ValueError(f"actions must be a list, got {actions!r}")
    clean_actions = []
    for a in actions:
        if not isinstance(a, dict):
            raise ValueError(f"each action must be an object, got {a!r}")
        if not isinstance(a.get("text"), str) or not a["text"].strip():
            raise ValueError(f"action text must be a non-empty string, got {a.get('text')!r}")
        speaker = a.get("speaker")
        if not isinstance(speaker, str):
            raise ValueError(f"action speaker must be a string, got {speaker!r}")
        # A speaker code the transcript never contained is the model inventing an owner — the same
        # failure the citation path drops outright. Here the action text is still worth keeping, so
        # the false attribution is cleared to "" (the page shows "unassigned") rather than the whole
        # item. When no set is supplied — a test calling parse_response directly — nothing is dropped.
        if valid_speakers and speaker and speaker not in valid_speakers:
            speaker = ""
        clean_actions.append({"text": a["text"], "speaker": speaker})

    return {"title": data["title"], "summary": data["summary"],
            "decisions": decisions, "actions": clean_actions}


def retry_prompt(original: str, bad_reply: str, error: str) -> str:
    return "\n".join([
        original,
        "",
        "Your previous reply was rejected:",
        f"  error: {error}",
        f"  reply: {bad_reply[:500]}",
        "Answer again with ONLY valid JSON matching the schema above. No other text.",
    ])


def max_tokens_for(target: int) -> int:
    # Double the character target covers CJK tokenization plus title/decisions/actions overhead.
    return target * 2 + 500


def summarize(lines: list[SummaryLine], languages: list[str], chat: Callable[[str], str],
              speakers: dict[str, str] | None = None,
              should_stop: Callable[[], bool] | None = None) -> tuple[dict, str]:
    """One summary per language; a language whose reply fails schema twice is simply missing."""
    rules = load_rules()
    total = sum(len(l.text) for l in lines)
    sampled_lines, sampled = sample(lines)
    # The codes the transcript actually uses. The model is told to use these exactly; this is what
    # holds it to that when it does not.
    valid = frozenset(l.speaker for l in lines if l.speaker)

    out: dict[str, dict] = {}
    for lang in languages:
        if should_stop and should_stop():
            break
        prompt = build_prompt(sampled_lines, lang, rules, speakers, sampled, total_chars=total)
        raw = chat(prompt)
        try:
            out[lang] = parse_response(raw, valid)
        except ValueError as first:
            try:
                out[lang] = parse_response(chat(retry_prompt(prompt, raw, str(first))), valid)
            except ValueError:
                pass  # recorded as missing; status below says partial/failed

    if len(out) == len(languages):
        return out, "ok"
    return out, "partial" if out else "failed"
