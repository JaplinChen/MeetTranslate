"""Turn the LLM pass's refused corrections into glossary candidates.

    python -m scripts.learn_terms transcripts/clean/*.txt --ollama qwen3:14b
    python -m scripts.learn_terms --apply          # write the accepted ones to the glossary

The refine pass throws away every correction its guards refuse. Those refusals are the most
informative thing it produces: a model that keeps wanting to write 工程變更 where the recogniser
wrote 一夕變更 is naming a term the glossary does not have. A term that shows up this way in
several places is worth adding — and once it is in the glossary, the same correction is allowed
through on the next run.

Refusals are stored as they are found, so mining and applying are separate steps and you get to
look at the list before it changes anything.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import correct, llm, refine  # noqa: E402
from server.store import Store  # noqa: E402

LINE = re.compile(r"^\[(\d+:\d+)\] (S\d+) \((\w+)\) (.*)$")
HAN = re.compile(r"[一-鿿]")
# A candidate has to look like a term rather than a phrase or a stray character.
MIN_LEN, MAX_LEN = 2, 8
# Widening runs into whatever sits next to the edit, and in speech that is usually a particle.
# Trimmed off both ends so the candidate is the term rather than the sentence around it.
PARTICLES = set("的那個這些他她我你們是在了就也都有會要跟和把被對從")


def _trim(before: str, after: str) -> tuple[str, str]:
    """Drop matching particles from both ends, keeping the two strings aligned."""
    while len(after) > MIN_LEN and before[:1] == after[:1] and after[0] in PARTICLES:
        before, after = before[1:], after[1:]
    while len(after) > MIN_LEN and before[-1:] == after[-1:] and after[-1] in PARTICLES:
        before, after = before[:-1], after[:-1]
    return before, after


def spans(original: str, candidate: str) -> list[tuple[str, str]]:
    """The pieces that differ, widened to something that looks like a term.

    The interesting correction is usually one character — 申管 for 生管, ELP for ERP — and one
    character is not a glossary entry. Widening into the Han characters on either side turns the
    edit back into the word it sits in, which is what a reader can recognise and what the
    corrector needs to match against.
    """
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, original, candidate).get_opcodes():
        if tag != "replace" or (j2 - j1) > MAX_LEN or (i2 - i1) > MAX_LEN:
            continue
        # Both strings share the text around the edit, so one offset widens both.
        left = right = 0
        while (left < 2 and j1 - left - 1 >= 0 and i1 - left - 1 >= 0
               and HAN.fullmatch(candidate[j1 - left - 1] or "")):
            left += 1
        while (right < 2 and j2 + right < len(candidate) and i2 + right < len(original)
               and HAN.fullmatch(candidate[j2 + right] or "")):
            right += 1
        before, after = _trim(original[i1 - left : i2 + right], candidate[j1 - left : j2 + right])
        if MIN_LEN <= len(after) <= MAX_LEN and before != after:
            out.append((before, after))
    return out


def read(path: Path) -> list[refine.Line]:
    lines = []
    for row in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := LINE.match(row):
            lines.append(refine.Line(m.group(2), m.group(3), m.group(4)))
    return lines


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("transcripts", nargs="*", type=Path)
    ap.add_argument("--ollama", nargs="?", const="qwen3:14b", default=None)
    ap.add_argument("--model", default="")
    ap.add_argument("--topic", default="SAP ERP 導入訪談")
    ap.add_argument("--min-count", type=int, default=2,
                    help="times a term must be proposed before it counts as evidence")
    ap.add_argument("--apply", action="store_true", help="add the surviving candidates to the glossary")
    args = ap.parse_args()

    if not args.transcripts:
        print("nothing to read", file=sys.stderr)
        return 1

    cfg = llm.load_llm()
    if args.ollama:
        chat = refine.ollama_chat(args.model or args.ollama, llm.DEFAULT_ENDPOINTS["ollama"], think=True)
    else:
        key = cfg.api_key
        if not key:
            print("no API key; pass --ollama to use a local model", file=sys.stderr)
            return 1
        chat = refine.anthropic_chat(key, args.model or cfg.model)

    store = Store()
    terms = store.glossary()
    known = {t.source for t in terms}
    proposed: Counter[tuple[str, str]] = Counter()

    for path in args.transcripts:
        lines = read(path)
        if not lines:
            continue
        rejected: list[refine.Rejected] = []
        refine.Refiner(chat, topic=args.topic).refine(lines, terms, rejected)
        for r in rejected:
            for before, after in spans(r.original, r.candidate):
                if HAN.search(after) and after not in known:
                    proposed[(before, after)] += 1
        print(f"{path.name}: {len(rejected)} refused corrections")

    survivors = [(before, after, n) for (before, after), n in proposed.most_common()
                 if n >= args.min_count]
    print(f"\n{len(survivors)} candidates seen at least {args.min_count} times:")
    for before, after, n in survivors:
        # A candidate the corrector could already reach needs no glossary entry.
        reachable = correct.pinyin_of(before, tones=False) == correct.pinyin_of(after, tones=False)
        print(f"  {n:3d}  {before} -> {after}{'   (already reachable)' if reachable else ''}")

    if args.apply:
        added = [after for _, after, _ in survivors if after not in known]
        for term in added:
            store.add_term(term, {}, mode="hint")
        print(f"\nadded {len(added)} terms to the glossary")
    elif survivors:
        print("\nrun again with --apply to add these, or add the ones you recognise by hand")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
