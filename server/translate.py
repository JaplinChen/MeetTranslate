"""Translation via the Claude API — the only outbound call this app makes.

One request does two jobs: translate the new utterance into every other configured language, and
say whether the *previous* utterance now reads wrong given what was just said. Folding the
refinement into the next line's request is what keeps context-aware polishing from doubling the
API bill, and it is why the subtitle page has to support rewriting a line it already drew.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .store import Term

LANGUAGE_NAMES = {
    "zh": "Traditional Chinese as written in Taiwan (臺灣繁體中文)",
    "vi": "Vietnamese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "th": "Thai",
    "id": "Indonesian",
}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


@dataclass
class Line:
    text: str
    lang: str
    speaker: str = ""


@dataclass
class Result:
    translations: dict[str, str]
    # Set only when the model judged the previous line wrong in hindsight.
    previous_source: str | None = None
    previous_translations: dict[str, str] = field(default_factory=dict)


def relevant_terms(text: str, terms: list[Term]) -> list[Term]:
    """Glossary entries whose source string actually occurs in this utterance.

    Sending the whole glossary every time would be both slow and counterproductive — a long list
    of irrelevant terms measurably degrades adherence to the ones that matter.
    """
    lowered = text.casefold()
    return [t for t in terms if t.source.casefold() in lowered]


def _term_lines(terms: list[Term], targets: list[str]) -> list[str]:
    out = []
    for t in terms:
        if t.mode == "keep":
            out.append(f'- "{t.source}": keep exactly as-is in every language, do not translate it')
        elif t.mode == "translate":
            pairs = ", ".join(f"{language_name(k)} = \"{v}\"" for k, v in t.targets.items() if k in targets)
            if pairs:
                out.append(f'- "{t.source}": must be rendered as {pairs}')
    return out


def build_prompt(line: Line, targets: list[str], context: list[Line], previous: Line | None,
                 terms: list[Term]) -> str:
    """The user-turn content. Kept a pure function so the prompt is testable without the network."""
    parts = [
        "You translate live meeting speech. The text comes from speech recognition, so it may "
        "contain recognition errors, missing punctuation, and words from other languages mixed in.",
        "",
        f"Source language: {language_name(line.lang) if line.lang else 'unknown, infer it'}",
        f"Translate into: {', '.join(language_name(t) for t in targets)}",
    ]

    if glossary := _term_lines(relevant_terms(line.text, terms), targets):
        parts += ["", "Glossary (binding):", *glossary]

    if context:
        parts += ["", "Earlier lines, for context only — do not translate these:"]
        parts += [f"  [{c.speaker or '?'}] {c.text}" for c in context]

    parts += ["", f"Translate this line, spoken by {line.speaker or 'an unknown speaker'}:", line.text]

    if previous:
        parts += [
            "",
            "The line before it was translated without knowing what came next:",
            f"  original: {previous.text}",
            "If the new line shows that translation was wrong — a misheard word, a pronoun with no "
            "referent, a sentence split in the wrong place — provide a corrected version. If it "
            "reads correctly, omit the `previous` field entirely rather than restating it.",
        ]

    schema = {t: "translation" for t in targets}
    parts += [
        "",
        "Reply with JSON only, no code fence and no commentary:",
        json.dumps({"translations": schema, "previous": {"source": "...", "translations": schema}}, ensure_ascii=False),
        "Keep translations natural and spoken, not literal. Preserve the speaker's register.",
    ]
    if "zh" in targets:
        parts.append("Chinese output must be Traditional Chinese with Taiwanese vocabulary, never Simplified.")

    return "\n".join(parts)


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(raw: str, targets: list[str]) -> Result:
    """Tolerant parse: models wrap JSON in fences or prose often enough to matter."""
    match = _JSON.search(raw)
    if not match:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")

    data = json.loads(match.group(0))
    translations = {k: str(v) for k, v in (data.get("translations") or {}).items() if k in targets}

    prev = data.get("previous") or {}
    prev_source = prev.get("source")
    prev_translations = {k: str(v) for k, v in (prev.get("translations") or {}).items() if k in targets}

    # A `previous` block with nothing usable in it is the model echoing the schema; drop it so the
    # subtitle page is not told to rewrite a line with identical content.
    if not prev_source and not prev_translations:
        return Result(translations)

    return Result(translations, prev_source, prev_translations)


class Translator:
    """Thin wrapper over the Anthropic SDK. Import is deferred so the module loads without a key."""

    def __init__(self, api_key: str, model: str = "claude-opus-5", max_tokens: int = 1500):
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def translate(self, line: Line, targets: list[str], context: list[Line] | None = None,
                  previous: Line | None = None, terms: list[Term] | None = None) -> Result:
        prompt = build_prompt(line, targets, context or [], previous, terms or [])
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return parse_response("".join(b.text for b in message.content if b.type == "text"), targets)
