"""Fixed evaluation harness: score a hypothesis transcript against a labelled reference.

    python -m scripts.eval_harness --ref transcripts/eval/meet01.ref.txt \
                                   --hyp transcripts/eval/meet01.hyp.txt
    python -m scripts.eval_harness --ref ... --hyp ... --save   # accept as baseline
    python -m scripts.eval_harness --selfcheck                  # run the built-in asserts

Both files are standard transcripts, one line each: `[M:SS] S1 (lang) text`. The reference is a
hand-corrected transcript with the true speaker on every line; the hypothesis is whatever the
pipeline produced (bench_wav output, or an exported session).

Three numbers decide whether a change to diarisation, language selection, or the hallucination
filter helped or hurt, and none of them can be read off a single transcript — you need the
reference beside it. This is the multiplier the roadmap put first: every later improvement is
measured here, against a stored baseline, so the cost of a change is a number and not an opinion.

- Accuracy: per-language CER (Han) / WER (Latin). Text is bucketed by the language each line was
  decoded as, so mis-assigned language shows up as accuracy loss in the wrong bucket — which is
  honest, that is a real error.
- Speaker attribution: the room's own pain. Reference and hypothesis speaker labels are matched
  optimally (S1-here is not S1-there), then the share of reference speech landing on the wrong
  speaker is the error. `hyp_speakers` vs `ref_speakers` exposes the collapse directly: 4 real
  speakers flattened to 1 is the 97-99%-one-speaker failure the README documents.
- Hallucination: the share of hypothesis speech the phrase filter flags as YouTube boilerplate.
  On a clean run this is ~0 because the live path already filters it; a rise means the filter
  regressed or leaked, which is exactly what a threshold change to chase Vietnamese hallucination
  might do by accident.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.bench_wav import edit_distance, normalize  # noqa: E402
from server import asr  # noqa: E402

LINE = re.compile(r"^\[(\d+):(\d+)\] (S\d+) \((\w+)\) (.*)$")
HAN = re.compile(r"[一-鿿]")
LATIN_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
BASELINE = Path("transcripts/eval/baseline.json")


class Line:
    __slots__ = ("start", "speaker", "lang", "text")

    def __init__(self, start: int, speaker: str, lang: str, text: str):
        self.start, self.speaker, self.lang, self.text = start, speaker, lang, text


def parse(path: Path) -> list[Line]:
    out = []
    for row in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := LINE.match(row):
            mm, ss, spk, lang, text = m.groups()
            out.append(Line(int(mm) * 60 + int(ss), spk, lang, text))
    return out


def is_han_lang(lines: list[Line]) -> bool:
    """A language bucket is scored by characters if its reference text is mostly Han."""
    joined = "".join(l.text for l in lines)
    return bool(joined) and len(HAN.findall(joined)) >= len(normalize(joined)) / 2


def wer(ref: str, hyp: str) -> tuple[int, int]:
    """Word-level edit distance for Latin-script languages; returns (errors, ref_words)."""
    r = LATIN_PUNCT.sub(" ", ref.lower()).split()
    h = LATIN_PUNCT.sub(" ", hyp.lower()).split()
    if not r:
        return 0, 0
    # edit_distance works on any sequence with == and len; a list of words qualifies.
    return edit_distance(r, h), len(r)


def accuracy(ref: list[Line], hyp: list[Line]) -> dict:
    """Per-language CER/WER, text bucketed by decoded language."""
    ref_by: dict[str, list[Line]] = defaultdict(list)
    hyp_by: dict[str, list[Line]] = defaultdict(list)
    for l in ref:
        ref_by[l.lang].append(l)
    for l in hyp:
        hyp_by[l.lang].append(l)

    out = {}
    for lang in sorted(ref_by):
        rlines = ref_by[lang]
        hlines = hyp_by.get(lang, [])
        if is_han_lang(rlines):
            r = normalize("".join(l.text for l in rlines))
            h = normalize("".join(l.text for l in hlines))
            errors, total = (edit_distance(r, h), len(r)) if r else (0, 0)
            metric = "cer"
        else:
            errors, total = wer("".join(l.text + " " for l in rlines),
                                 "".join(l.text + " " for l in hlines))
            metric = "wer"
        out[lang] = {"metric": metric, "rate": round(errors / total, 4) if total else 0.0,
                     "errors": errors, "total": total}
    return out


def _weight(text: str) -> int:
    """Speech-amount proxy: reference character count. DER is duration-weighted; without per-word
    timings, characters are the closest honest stand-in and never over-credit a one-word line."""
    return len(normalize(text)) or 1


def speaker_error(ref: list[Line], hyp: list[Line]) -> dict:
    """Optimal-mapped speaker attribution error, weighted by reference speech amount."""
    hyp_sorted = sorted(hyp, key=lambda l: l.start)

    def nearest(t: int) -> Line | None:
        best, gap = None, 6  # a match beyond 5s apart is not the same utterance
        for h in hyp_sorted:
            if abs(h.start - t) < gap:
                best, gap = h, abs(h.start - t)
        return best

    pairs: list[tuple[str, str, int]] = []  # (ref_speaker, hyp_speaker, weight)
    for l in ref:
        h = nearest(l.start)
        pairs.append((l.speaker, h.speaker if h else "∅", _weight(l.text)))

    ref_spk = sorted({r for r, _, _ in pairs})
    hyp_spk = sorted({h for _, h, _ in pairs})
    # Brute-force the best hyp->ref label map. Room transcripts have a handful of speakers, so the
    # permutation space is tiny; guard anyway so a pathological file can't hang the harness.
    total = sum(w for _, _, w in pairs)
    if len(hyp_spk) > 7:
        return {"error": None, "note": f"{len(hyp_spk)} hyp speakers — too many to map",
                "ref_speakers": len(ref_spk), "hyp_speakers": len(hyp_spk)}

    best_correct = 0
    targets = ref_spk + [s for s in hyp_spk if s not in ref_spk]
    for perm in permutations(targets, len(hyp_spk)):
        mapping = dict(zip(hyp_spk, perm))
        correct = sum(w for r, h, w in pairs if mapping[h] == r)
        best_correct = max(best_correct, correct)

    return {"error": round(1 - best_correct / total, 4) if total else 0.0,
            "ref_speakers": len(ref_spk), "hyp_speakers": len(hyp_spk)}


def hallucination_rate(hyp: list[Line]) -> dict:
    """Share of hypothesis speech the phrase filter flags, overall and per language."""
    total: Counter[str] = Counter()
    flagged: Counter[str] = Counter()
    for l in hyp:
        w = _weight(l.text)
        total[l.lang] += w
        total["_all"] += w
        if asr.is_hallucination(l.text):
            flagged[l.lang] += w
            flagged["_all"] += w
    per_lang = {lang: round(flagged[lang] / total[lang], 4)
                for lang in sorted(total) if lang != "_all" and total[lang]}
    overall = round(flagged["_all"] / total["_all"], 4) if total["_all"] else 0.0
    return {"rate": overall, "per_language": per_lang}


def score(ref: list[Line], hyp: list[Line]) -> dict:
    return {"accuracy": accuracy(ref, hyp),
            "speaker": speaker_error(ref, hyp),
            "hallucination": hallucination_rate(hyp)}


def report(result: dict, before: dict | None) -> bool:
    """Print the scorecard; return True if any tracked number regressed."""
    regressed = False
    print("--- accuracy (per language) ---")
    for lang, a in result["accuracy"].items():
        was = (before or {}).get("accuracy", {}).get(lang, {}).get("rate")
        delta = f"   (was {was:.1%})" if was is not None and was != a["rate"] else ""
        print(f"  {lang:4} {a['metric'].upper()} {a['rate']:.1%}  ({a['errors']}/{a['total']}){delta}")
        if was is not None and a["rate"] > was + 1e-9:
            regressed = True

    s = result["speaker"]
    print("\n--- speaker attribution ---")
    if s.get("error") is None:
        print(f"  {s.get('note')}")
    else:
        was = (before or {}).get("speaker", {}).get("error")
        delta = f"   (was {was:.1%})" if was is not None and was != s["error"] else ""
        print(f"  error {s['error']:.1%}  ref_speakers={s['ref_speakers']} "
              f"hyp_speakers={s['hyp_speakers']}{delta}")
        if s["hyp_speakers"] < s["ref_speakers"]:
            print(f"  COLLAPSE: {s['ref_speakers']} real speakers flattened to {s['hyp_speakers']}")
        if was is not None and s["error"] > was + 1e-9:
            regressed = True

    h = result["hallucination"]
    print("\n--- hallucination ---")
    was = (before or {}).get("hallucination", {}).get("rate")
    delta = f"   (was {was:.1%})" if was is not None and was != h["rate"] else ""
    print(f"  rate {h['rate']:.1%}{delta}"
          + ("  " + ", ".join(f"{k}={v:.1%}" for k, v in h["per_language"].items())
             if h["per_language"] else ""))
    if was is not None and h["rate"] > was + 1e-9:
        regressed = True
    return regressed


def demo() -> None:
    """Self-check on tiny in-memory transcripts — asserts the three metrics compute as expected."""
    ref = [Line(0, "S1", "zh", "工單已經完成"), Line(10, "S2", "vi", "toi dong y"),
           Line(20, "S1", "zh", "料號要確認")]
    # hyp: one zh char wrong, vietnamese one word wrong, and S2 collapsed into S1.
    hyp = [Line(0, "S1", "zh", "工單已經完成"), Line(10, "S1", "vi", "toi dong yy"),
           Line(20, "S1", "zh", "料號要確任")]
    r = score(ref, hyp)
    assert r["accuracy"]["zh"]["metric"] == "cer" and r["accuracy"]["zh"]["errors"] == 1, r["accuracy"]
    assert r["accuracy"]["vi"]["metric"] == "wer" and r["accuracy"]["vi"]["errors"] == 1, r["accuracy"]
    # 2 real speakers, hyp found 1 → collapse, and S2's speech is all misattributed.
    assert r["speaker"]["ref_speakers"] == 2 and r["speaker"]["hyp_speakers"] == 1, r["speaker"]
    assert r["speaker"]["error"] > 0, r["speaker"]
    # No boilerplate here.
    assert r["hallucination"]["rate"] == 0.0, r["hallucination"]
    # A real sign-off must be caught.
    assert asr.is_hallucination("请订阅我们的频道") or asr.is_hallucination("訂閱我們的頻道")
    hallu = hallucination_rate([Line(0, "vi", "vi", "đăng ký kênh")])
    assert hallu["rate"] == 1.0, hallu
    print("selfcheck ok")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, help="reference transcript (true speakers)")
    ap.add_argument("--hyp", type=Path, help="hypothesis transcript to score")
    ap.add_argument("--save", action="store_true", help="accept these numbers as the baseline")
    ap.add_argument("--selfcheck", action="store_true", help="run built-in asserts and exit")
    args = ap.parse_args()

    if args.selfcheck:
        demo()
        return 0
    if not args.ref or not args.hyp:
        ap.error("--ref and --hyp are required (or use --selfcheck)")

    ref, hyp = parse(args.ref), parse(args.hyp)
    if not ref:
        print(f"no parseable lines in {args.ref}", file=sys.stderr)
        return 1

    result = score(ref, hyp)
    before = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else None
    regressed = report(result, before)

    if args.save:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nbaseline written to {BASELINE}")
    elif before is None:
        print("\nno baseline yet; run with --save once these numbers look right")
    elif regressed:
        print("\nREGRESSION: a tracked metric got worse against the baseline")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
