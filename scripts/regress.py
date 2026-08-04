"""Replay every correction layer over real transcripts and say what changed.

    python -m scripts.regress                 # measure against the stored baseline
    python -m scripts.regress --save          # accept the current numbers as the baseline

Three thresholds in this project decide whether a transcript is repaired or corrupted, and every
one of them looked correct against hand-written examples and was wrong on real speech. The pinyin
distance in the corrector rewrote 知道 to 製造 156 times. The sound check in the refine pass turned
夢表 into 模具. Its glossary branch turned 土壤 into 交貨. None of those cases would occur to
someone inventing test data, and all of them are sitting in the transcripts already on disk.

So this is not a unit test. It runs the real thresholds over the real corpus and counts what they
do, split into corrections and suspected corruptions, with a stored baseline to compare against.
Change a threshold, run this, and the cost of the change is a number rather than an opinion.

A suspected corruption is a rewrite whose *original* is established vocabulary — a string the
corpus uses often enough that it is a word, not a mishearing. That is exactly how 料號 destroyed
料耗: 料耗 appears 42 times and is a term of the trade.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import correct, guards  # noqa: E402
from server.store import Store  # noqa: E402

LINE = re.compile(r"^\[(\d+:\d+)\] (S\d+) \((\w+)\) (.*)$")
PROPOSAL = re.compile(r"^  \[\d+:\d+\] (.*)$")
BASELINE = Path("transcripts/regress-baseline.json")
# How often a string must appear before it counts as established vocabulary rather than a
# one-off misrecognition. Low, because a real term said twice is still a real term.
ESTABLISHED = 3


def corpus(paths: list[Path]) -> list[tuple[str, str]]:
    """(language, text) for every transcript line found."""
    out = []
    for path in paths:
        for row in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if m := LINE.match(row):
                out.append((m.group(3), m.group(4)))
    return out


def proposals(paths: list[Path]) -> list[tuple[str, str]]:
    """(before, after) pairs the LLM actually proposed, recovered from the refine logs."""
    out = []
    for path in paths:
        rows = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, row in enumerate(rows):
            m = PROPOSAL.match(row)
            if m and i + 1 < len(rows) and rows[i + 1].strip().startswith("->"):
                out.append((m.group(1).strip(), rows[i + 1].split("->", 1)[1].strip()))
    return out


def vocabulary(lines: list[tuple[str, str]]) -> Counter[str]:
    """Every 2-to-4 character Han string in the corpus, counted."""
    seen: Counter[str] = Counter()
    for _, text in lines:
        for run in re.findall(r"[一-鿿]+", text):
            for n in (2, 3, 4):
                for i in range(len(run) - n + 1):
                    seen[run[i : i + n]] += 1
    return seen


def score_corrector(lines: list[tuple[str, str]], vocab: Counter[str]) -> dict:
    """What the glossary and the learned corrections would do to this corpus."""
    store = Store()
    corrector = correct.Corrector(store.glossary(), store.corrections())

    fixes: Counter[tuple[str, str]] = Counter()
    suspect: Counter[tuple[str, str]] = Counter()
    for lang, text in lines:
        fixed = corrector.fix(text)
        if fixed == text:
            continue
        for was, now in correct.diff_terms(text, fixed):
            # Replacing something the corpus treats as a word is how 料耗 became 料號 42 times.
            (suspect if vocab[was] >= ESTABLISHED else fixes)[(was, now)] += 1

    return {
        "lines": len(lines),
        "fixes": sum(fixes.values()),
        "suspect": sum(suspect.values()),
        "top_fixes": [[w, n, c] for (w, n), c in fixes.most_common(10)],
        "top_suspect": [[w, n, c] for (w, n), c in suspect.most_common(10)],
    }


def score_guards(pairs: list[tuple[str, str]], vocab: Counter[str], corpus: str) -> dict:
    """Which of the LLM's real proposals today's guards would let through.

    Judged the same way as the corrector, because it is the same failure: an accepted correction
    that overwrites established vocabulary is how 土壤 became 交貨 three times. Symmetric checks
    on both layers, so a change to either is measured against the same question.
    """
    terms = Store().glossary()
    accepted = [(a, b) for a, b in pairs if guards.accept(a, b, terms)]
    # Reported rather than enforced — see guards.displaces_a_word.
    displaced = sum(guards.displaces_a_word(a, b, corpus) for a, b in accepted)
    suspect: Counter[tuple[str, str]] = Counter()
    for before, after in accepted:
        for was, now in correct.diff_terms(before, after):
            if vocab[was] >= ESTABLISHED:
                suspect[(was, now)] += 1
    return {
        "proposals": len(pairs),
        "accepted": len(accepted),
        "rejected": len(pairs) - len(accepted),
        "suspect": sum(suspect.values()),
        "displaces": displaced,
        "top_suspect": [[w, n, c] for (w, n), c in suspect.most_common(10)],
    }


def compare(name: str, now: dict, before: dict | None) -> None:
    print(f"\n{name}")
    for key, value in now.items():
        if key.startswith("top_"):
            continue
        was = (before or {}).get(key)
        delta = "" if was is None or was == value else f"   ({was:+d} -> {value:+d})".replace("+", "")
        print(f"  {key:12} {value}{delta}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", type=Path, default=Path("transcripts/final"))
    ap.add_argument("--logs", type=Path, default=Path("transcripts/raw"))
    ap.add_argument("--save", action="store_true", help="accept these numbers as the baseline")
    args = ap.parse_args()

    lines = corpus(sorted(args.transcripts.glob("*.txt")))
    if not lines:
        print(f"no transcripts under {args.transcripts}", file=sys.stderr)
        return 1
    vocab = vocabulary(lines)

    result = {"corrector": score_corrector(lines, vocab)}
    pairs = proposals(sorted(args.logs.glob("*.refine.log")))
    if pairs:
        result["guards"] = score_guards(pairs, vocab, chr(10).join(t for _, t in lines))

    before = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else None
    for name, block in result.items():
        compare(name, block, (before or {}).get(name))

    for layer in ("corrector", "guards"):
        rows = result.get(layer, {}).get("top_suspect") or []
        if rows:
            print(f"\n{layer}: suspected corruptions — the corpus uses these as words:")
            for was, now, n in rows:
                print(f"  {n:4d}  {was} -> {now}   ({vocab[was]} occurrences of {was})")

    if args.save:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nbaseline written to {BASELINE}")
    elif before is None:
        print("\nno baseline yet; run with --save once these numbers look right")

    # Judged against the baseline rather than against zero. The heuristic cannot tell a real word
    # from a misrecognition that happens to recur — 生館 appears three times and is neither — so
    # some suspicion is permanent. What matters is whether a change added any.
    for layer in ("corrector", "guards"):
        was = (before or {}).get(layer, {}).get("suspect")
        now = result.get(layer, {}).get("suspect")
        if was is not None and now is not None and now > was:
            print(f"\nREGRESSION: {layer} suspected corruptions rose from {was} to {now}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
