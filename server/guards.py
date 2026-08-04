"""What separates a correction from an invention.

A model asked to fix a transcript will happily rewrite it into something more fluent than what was
said. Everything here exists to hold it to substitutions it can justify: a ceiling on how much of
a line may change, and a check that the recogniser could plausibly have heard the original in the
place of the correction.

The thresholds are calibrated against real output, not chosen for roundness — each one carries the
pair that set it. Kept apart from `refine` because they are pure functions over two strings and a
glossary: no prompt, no chunking, no model.
"""

from __future__ import annotations

import difflib
import re

from .correct import diff_terms, edit_distance, pinyin_of
from .store import Term

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
# A word the meeting room says often is a word, whatever a model thinks it heard. Below this many
# occurrences the corpus has not established anything either way.
ESTABLISHED = 3


def _squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text)


def displaces_a_word(original: str, candidate: str, corpus: str) -> bool:
    """Whether this correction replaces something the meeting room says with something rarer.

    Reported, not enforced. As a guard it was measured and backed out: comparing bare spans
    rejected 標準公司 -> 標準工時, which is right, because 公司 alone appears 83 times and nobody
    says 標準公司. Adding two characters of context fixed that and broke the rest — 之後才是
    appears twice in this corpus, too rare to establish anything, so the archaic 之後纔是 sailed
    through. Three thousand lines is not enough vocabulary for either version to be trusted with
    a veto.

    It still separates signal from noise well enough to be worth reading, which is what
    scripts/regress.py uses it for.
    """
    if not corpus:
        return False
    for was, now in diff_terms(original, candidate):
        seen = corpus.count(was)
        if seen >= ESTABLISHED and seen > corpus.count(now):
            return True
    return False


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
