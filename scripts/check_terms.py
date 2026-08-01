"""What a glossary term would rewrite, before you commit to it.

    python -m scripts.check_terms 料號 採購 工序 --against transcripts/clean/*.txt

The post-decode corrector rewrites anything whose pinyin matches a term exactly, and Mandarin
supplies homophones for almost everything. `料號` and `料耗` are both liaohao, and 料耗 is real
vocabulary in a manufacturing interview — adding 料號 to the glossary silently destroyed it
twenty-one times. This finds that before it happens, by scanning transcripts you already have.

A collision is not automatically a veto. If both spellings are real words, add both: a term that
is in the glossary is never rewritten into another, so registering 供需 alongside 工序 keeps each
of them intact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.correct import collisions, pinyin_of  # noqa: E402
from server.store import Store  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("terms", nargs="+", help="terms you are considering adding")
    ap.add_argument("--against", nargs="*", type=Path, default=[],
                    help="transcripts to scan; defaults to transcripts/clean/*.txt")
    ap.add_argument("--apply", action="store_true", help="add the terms that collide with nothing")
    args = ap.parse_args()

    paths = args.against or sorted(Path("transcripts/clean").glob("*.txt"))
    if not paths:
        print("nothing to scan; pass --against with some transcripts", file=sys.stderr)
        return 1

    text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in paths)
    store = Store()
    known = {t.source for t in store.glossary()} | set(args.terms)

    clean = []
    for term in args.terms:
        hits = collisions(term, text, known)
        if not hits:
            clean.append(term)
            print(f"{term}: no collisions in {len(paths)} transcripts")
            continue
        total = sum(hits.values())
        print(f"{term}: would rewrite {total} occurrences of {len(hits)} other spellings")
        for other, n in sorted(hits.items(), key=lambda kv: -kv[1]):
            # Tones are shown, not enforced. Measured across every collision in seven interviews,
            # requiring them to match would have protected three real words (才夠, 升官, 供需) at
            # the cost of seven genuine fixes (財購, 盛管, 省管...), and would not have caught the
            # one that did damage: 料耗 and 料號 are both liao4hao4. Whisper mishears tone as
            # readily as it mishears the syllable, so a tone difference is as often the error as
            # the evidence. It still tells a reader something, so it is on the line.
            same = pinyin_of(other) == pinyin_of(term)
            print(f"    {n:4d}  {other} -> {term}"
                  f"   {'identical tones' if same else 'different tones'}")
        print("    keep any of these that are real words by adding them to the glossary too")

    if args.apply:
        added = [t for t in clean if t not in {x.source for x in store.glossary()}]
        for term in added:
            store.add_term(term, {}, mode="hint")
        print(f"\nadded {len(added)} collision-free terms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
