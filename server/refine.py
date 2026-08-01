"""LLM pass that repairs recognition errors using the surrounding conversation.

Whisper decodes one utterance at a time and cannot know it is sitting in an SAP ERP interview. A
reader who does knows that `一夕變更` is `工程變更` and that `生技` should be `生管`, because the
sentence around it is about change control and production planning. That is the only signal left
once the acoustics have been squeezed dry, and it is not available to the recogniser.

The danger is the same thing that makes it work: a model asked to fix a transcript will happily
rewrite it into something more fluent than what was said. Every guard here exists to keep it to
substitutions it can justify — same line count, same speakers, and a per-line ceiling on how much
may change before the result is treated as invention and thrown away.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass

from opencc import OpenCC

from .correct import edit_distance, pinyin_of
from .store import Term

log = logging.getLogger("meettranslate.refine")

# Lines per request. Large enough that a term corrected early informs the rest, small enough that
# the model still has every line in view when it answers.
CHUNK_LINES = 25
# Lines of already-refined text sent as read-only lead-in, so a chunk boundary does not sit in the
# middle of a sentence with no history.
CONTEXT_LINES = 4
# A correction that changes more than this fraction of a line is a rewrite. Measured against the
# original: `生技` for `生管` is 0.5 of a two-character word but a rounding error in a sentence,
# which is why this is applied per line rather than per term.
MAX_CHANGE = 0.30
# How far a correction may move the line's pronunciation. A recognition error is something the
# recogniser heard; a correction that sounds nothing like what it produced was invented instead.
MAX_SOUND_CHANGE = 0.20
# Glossary terms are exempt from most of that. The recogniser cannot know a term exists, so the
# text it produced may be acoustically far from the truth — and the glossary is the user saying
# this term belongs in this meeting.
# Measured on real corrections: 一夕變更 -> 工程變更 sits at 0.53 and is right, 浴室量 -> 收料
# at 0.60 and is not. There is not much room between them, and that is the honest width of this
# signal — a glossary term buys latitude, not immunity.
MAX_TERM_SOUND_CHANGE = 0.55
# And a hard ceiling in edits, because the ratio is measured over the whole line: two nonsense
# characters inside a sixty-character sentence are a rounding error to a ratio and still nonsense.
# `夢表` for `模具` and `監獄` for `零件` both slipped through on ratio alone.
MAX_SOUND_EDITS = 3

NUMBERED = re.compile(r"^\s*(\d+)\s*[:：.]\s*(.*)$")

# Character-level Simplified -> Traditional, deliberately not the s2twp used on ASR output. The
# model writes the odd Simplified character (保税 for 保稅) and that has to go, but s2twp also
# rewrites vocabulary — 對象 to 物件, 軟件 to 軟體 — which would edit words the speaker chose.
_to_traditional = OpenCC("s2t")


@dataclass
class Line:
    speaker: str
    lang: str
    text: str


def build_prompt(lines: list[Line], context: list[Line], terms: list[Term], topic: str) -> str:
    """Ask for substitutions, not improvements."""
    parts = [
        f"以下是一場「{topic}」的逐字稿片段，由語音辨識產生，含有辨識錯誤。",
        "",
        "你的工作是根據上下文修正**辨識錯誤**，不是改寫。規則：",
        "- 只改明顯錯字：同音字、專有名詞、術語。判斷依據是上下文語意",
        "- 不補字、不刪字、不潤飾語句、不改語氣、不合併或拆分句子",
        "- 口語的重複與贅字是說話者原本就有的，保留",
        "- 不確定的地方保持原樣。寧可留錯，不可改成沒說過的內容",
        "- 中文一律繁體。英文與越南語維持原文",
        "- 不要增刪空白或標點。加空白不算修正",
        "",
        "輸出格式：**只輸出需要修改的行**，每行 `編號: 修正後的完整句子`。",
        "沒有問題的行不要輸出。全部都沒問題就輸出 NONE。不要加任何說明。",
    ]

    if terms:
        parts += ["", "會議專有名詞（出現近似音時應修正為這些寫法）：",
                  "、".join(t.source for t in terms)]

    if context:
        parts += ["", "前文（僅供理解，不要輸出）："]
        parts += [f"{l.speaker}（{l.lang}）：{l.text}" for l in context]

    parts += ["", "待修正："]
    parts += [f"{i}: {l.text}" for i, l in enumerate(lines, 1)]
    return "\n".join(parts)


def _squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text)


def accept(original: str, candidate: str, terms: list[Term] | None = None) -> bool:
    """Whether a proposed line is a correction rather than a replacement.

    Two questions, and both must pass. Is it small enough to be a substitution rather than a
    rewrite? And could the recogniser plausibly have produced the original from the corrected
    audio — that is, do they sound alike?

    The second is what separates a fix from a guess. Measured on a local model's output: `早等`
    for `稍等` is 0.06 of the line's pinyin apart and right; `延伸` for `選項` is 0.23 apart and
    invented, because nothing that sounds like 選項 was ever spoken.

    A glossary term is allowed to travel much further. `一夕變更` and `工程變更` are 0.41 apart
    and the correction is still right — the recogniser had no idea the term existed, and the
    glossary is the user asserting that it does.
    """
    candidate = candidate.strip()
    if not candidate or candidate == original:
        return False
    # Re-spacing is not a correction. Asked not to think, a model reaches for the cheapest edit it
    # can justify, and inserting spaces between words is the cheapest of all.
    if _squeeze(candidate) == _squeeze(original):
        return False

    # A short line has no room for a ratio to mean anything; allow one or two characters.
    if edit_distance(original, candidate) > max(2, int(len(original) * MAX_CHANGE)):
        return False

    before, after = pinyin_of(original, tones=False), pinyin_of(candidate, tones=False)
    if not before or not after:
        return True  # nothing Chinese in it; the size check above is all there is to go on
    introduced = [t.source for t in terms or []
                  if t.source in candidate and t.source not in original]
    if not introduced:
        return edit_distance(before, after) <= min(MAX_SOUND_CHANGE * len(before), MAX_SOUND_EDITS)

    # A glossary term may travel further than an ordinary correction, but it is compared against
    # the text it replaced and nothing else. Measuring across the whole line let 土壤 become 交貨
    # and 祂 become 生管 — two unrelated syllables inside a sixty-character sentence look like a
    # rounding error. Measuring only the characters that differ is no better: 一夕 and 工程 share
    # nothing either, and yet 一夕變更 -> 工程變更 is right. The term is the unit that works,
    # because it carries the part both versions have in common.
    return all(_sounds_like(_replaced_by(original, candidate, term), term) for term in introduced)


@dataclass
class Coverage:
    """How much of the transcript the pass actually looked at.

    A chunk the guards throw out whole leaves its lines exactly as the recogniser wrote them, which
    is indistinguishable from a chunk that needed no corrections. Counted, because "checked and
    found clean" and "never checked" are not the same claim to make about a transcript.
    """
    lines: int = 0
    skipped: int = 0

    @property
    def fraction(self) -> float:
        return 0.0 if not self.lines else self.skipped / self.lines


@dataclass
class Rejected:
    """A correction the guards refused, kept because it is the most useful thing they produce.

    A model that repeatedly wants to write 工程變更 where the recogniser wrote 一夕變更 is telling
    you the term exists and that the glossary does not know it. That is the one signal in this
    system that names its own blind spots.
    """
    original: str
    candidate: str


def _replaced_by(original: str, candidate: str, term: str) -> str:
    """The text in `original` that sits where `term` now sits in `candidate`."""
    start = candidate.index(term)
    end = start + len(term)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, original, candidate).get_opcodes():
        if j1 < end and j2 > start:
            lo = i1 + max(0, start - j1) if tag == "equal" else i1
            hi = i1 + min(i2 - i1, end - j1) if tag == "equal" else i2
            out.append(original[lo:hi])
    return "".join(out)


def _sounds_like(was: str, now: str) -> bool:
    pa, pb = pinyin_of(was, tones=False), pinyin_of(now, tones=False)
    if not pa or not pb:
        return False
    moved = edit_distance(pa, pb)
    # Two edits is a misheard syllable at any length; past that it has to still sound similar.
    return moved <= 2 or moved / max(len(pa), len(pb)) <= MAX_TERM_SOUND_CHANGE


def parse_response(raw: str, lines: list[Line], terms: list[Term] | None = None,
                   rejected: list[Rejected] | None = None,
                   coverage: Coverage | None = None) -> list[str]:
    """Map the numbered reply back onto the input, keeping originals wherever it disagrees.

    Only changed lines come back. Asking for the whole chunk made the model copy out two dozen
    correct sentences to deliver two corrections — measured at 570 output tokens per 25 lines
    against 42 tokens a second, which was most of the runtime.

    An index outside the chunk means the model lost track of the numbering; that line is dropped
    rather than applied to whatever happens to sit at that position.
    """
    got: dict[int, str] = {}
    for row in raw.splitlines():
        if m := NUMBERED.match(row):
            index = int(m.group(1))
            if 1 <= index <= len(lines):
                got[index] = m.group(2)

    # "Most of the chunk" needs a chunk to be about: on the two or three lines left over at the
    # end of a transcript, one correction is already a majority.
    if len(lines) >= 4 and len(got) > len(lines) // 2:
        log.warning("refine rewrote %d of %d lines, keeping originals", len(got), len(lines))
        if coverage is not None:
            coverage.skipped += len(lines)
        return [l.text for l in lines]

    out = []
    for i, line in enumerate(lines, 1):
        candidate = got.get(i, "")
        if not accept(line.text, candidate, terms):
            if rejected is not None and candidate.strip() and candidate.strip() != line.text:
                rejected.append(Rejected(line.text, candidate.strip()))
            out.append(line.text)
            continue
        candidate = candidate.strip()
        out.append(_to_traditional.convert(candidate) if line.lang.startswith("zh") else candidate)
    return out


THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def anthropic_chat(api_key: str, model: str, max_tokens: int = 4000):
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    def chat(prompt: str) -> str:
        message = client.messages.create(model=model, max_tokens=max_tokens,
                                         messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in message.content if b.type == "text")

    return chat


def ollama_chat(model: str, endpoint: str = "http://127.0.0.1:11434", timeout: float = 900,
                think: bool = False):
    """Local models over Ollama's HTTP API — no key, and the transcript never leaves the machine.

    That second point is the reason to prefer it here: these are client interview recordings.
    """
    import json
    import urllib.request

    def chat(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            # Reasoning costs minutes per chunk and buys a different failure mode rather than a
            # better one — see --think in scripts/refine_transcript.py. Models with no thinking
            # mode ignore the field.
            "think": think,
            # Deterministic: the same transcript should not correct differently on a re-run.
            "options": {"temperature": 0, "num_ctx": 8192},
        }).encode()
        req = urllib.request.Request(f"{endpoint}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = json.load(r).get("response", "")
        # Qwen and other reasoning models narrate before answering; the answer is what follows.
        return THINK.sub("", text)

    return chat


class Refiner:
    """Correction driven by whatever `chat` is given — cloud or local, the guards are the same."""

    def __init__(self, chat, topic: str = "會議"):
        self._chat = chat
        self._topic = topic

    def refine(self, lines: list[Line], terms: list[Term] | None = None,
               rejected: list[Rejected] | None = None,
               coverage: Coverage | None = None) -> list[str]:
        """Correct a whole transcript, chunk by chunk. Returns one string per input line."""
        out: list[str] = []
        if coverage is not None:
            coverage.lines += len(lines)
        for start in range(0, len(lines), CHUNK_LINES):
            chunk = lines[start : start + CHUNK_LINES]
            context = [Line(l.speaker, l.lang, t)
                       for l, t in zip(lines[max(0, start - CONTEXT_LINES) : start],
                                       out[max(0, start - CONTEXT_LINES) : start])]
            prompt = build_prompt(chunk, context, terms or [], self._topic)
            try:
                out += parse_response(self._chat(prompt), chunk, terms, rejected, coverage)
            except Exception:
                log.exception("refine failed at line %d, keeping originals", start)
                if coverage is not None:
                    coverage.skipped += len(chunk)
                out += [l.text for l in chunk]
        return out
