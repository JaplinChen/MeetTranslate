"""Proper-noun correction over decoded text.

Whisper picks the wrong character far more often than it mishears the syllable. The decode-time
homophone replacer only fires on an exact tone-for-tone pinyin match, so it misses the ordinary
case where the tone is what went wrong. Comparing toneless pinyin within a small edit distance
catches those.

It does not catch everything, and is not meant to: `生管` decoded as `生氣` is a different syllable,
not a different character for the same one. That is an acoustic error and belongs to the model,
not to this pass.

Chinese terms are compared as pinyin, Latin ones as lowercase letters. Both use the same edit
distance and both refuse to fire above their threshold: a false insertion puts a word on the
meeting-room TV that nobody said, which is worse than leaving the original mistake alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .store import Term

# How far apart two spellings may be, as a fraction of the term's own length, before they stop
# counting as the same word. At 0.25 a two-syllable term tolerates one edit, which covers the
# common case — same sound, wrong character — without letting `料號` reach `了好`.
MAX_DISTANCE = 0.25
# Below this many characters a term is too small for a distance to mean anything: at three
# characters of pinyin, every second syllable is within the threshold of every other. Such terms
# are still corrected, but only on an exact toneless match — `直距` for `治具`, both `zhiju`.
MIN_KEY = 6
# Shorter than this and even an exact sound match is too likely to be a different word entirely.
MIN_TERM_KEY = 4

HAN = re.compile(r"[一-鿿]")
LATIN_TOKEN = re.compile(r"[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ'’-]*")


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def pinyin_of(text: str, tones: bool = True) -> str:
    """Pinyin with no separators. Neutral tone is written as 1, matching upstream.

    Correction drops the tones. Whisper picks the wrong character far more often than it mishears
    the syllable itself, and those wrong characters usually differ only in tone — `工單` for `公單`
    is one edit with tones and none without.
    """
    from pypinyin import Style, lazy_pinyin

    if not tones:
        return "".join(lazy_pinyin(text))
    return "".join(p.replace("5", "1") for p in
                   lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))


@dataclass
class _Rule:
    term: str
    key: str      # what we compare against: pinyin for Chinese, lowercase for Latin
    chinese: bool

    @property
    def limit(self) -> int:
        return int(len(self.key) * MAX_DISTANCE) if len(self.key) >= MIN_KEY else 0


def _rules(terms: list[Term]) -> list[_Rule]:
    out: list[_Rule] = []
    for t in terms:
        chinese = bool(HAN.search(t.source))
        key = pinyin_of(t.source, tones=False) if chinese else t.source.lower()
        if len(key) >= MIN_TERM_KEY:
            out.append(_Rule(t.source, key, chinese))
    # Longest first: a term that contains another must win, or the shorter one eats its prefix.
    return sorted(out, key=lambda r: -len(r.term))


class Corrector:
    """Rewrites near-misses of glossary terms to their canonical spelling."""

    def __init__(self, terms: list[Term]):
        self._rules = _rules(terms)

    def fix(self, text: str) -> str:
        if not text or not self._rules:
            return text
        for rule in self._rules:
            text = self._fix_chinese(text, rule) if rule.chinese else self._fix_latin(text, rule)
        return text

    def _fix_chinese(self, text: str, rule: _Rule) -> str:
        """Slide a window the width of the term over every run of Han characters.

        Only Han windows are considered, so a term never swallows the English half of a
        code-switched sentence, which is most of them in this meeting room.
        """
        width = len(rule.term)
        limit = rule.limit
        i = 0
        while i + width <= len(text):
            window = text[i : i + width]
            if len(HAN.findall(window)) != width or window == rule.term:
                i += 1
                continue
            if edit_distance(pinyin_of(window, tones=False), rule.key) <= limit:
                text = text[:i] + rule.term + text[i + width :]
                i += len(rule.term)
            else:
                i += 1
        return text

    def _fix_latin(self, text: str, rule: _Rule) -> str:
        limit = rule.limit

        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            if token == rule.term:
                return token
            return rule.term if edit_distance(token.lower(), rule.key) <= limit else token

        return LATIN_TOKEN.sub(replace, text)
