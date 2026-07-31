"""Offline recognition benchmark: run one wav through the post-meeting pipeline.

Usage:
    python -m scripts.bench_wav recordings/test01.wav [--model medium] [--ref ref.txt]

The wav must be 16 kHz mono (ffmpeg -i in.mp4 -ac 1 -ar 16000 -c:a pcm_s16le out.wav).
With --ref, prints CER against a reference transcript; without it, just the transcript.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import asr, asr_gpu, config, diarize, postprocess  # noqa: E402
from server.store import Store  # noqa: E402

PUNCT = re.compile(r"[\s，。、！？：；「」『』（）,.!?:;\"'()\-—…]+")


def normalize(text: str) -> str:
    return PUNCT.sub("", text)


def edit_distance(a: str, b: str) -> int:
    # ponytail: O(n*m) in pure Python — fine for the 10–15 minute sample this is meant for,
    # swap in rapidfuzz if you ever score a whole meeting.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def report_glossary(hypothesis: str) -> None:
    """Which glossary terms survived recognition.

    A Chinese term that misses is a candidate for the homophone replacer; an English or Vietnamese
    one has no such fallback, so it either needs a better model tier or has to be lived with.
    """
    terms = Store().glossary()
    if not terms:
        return
    hay = normalize(hypothesis).lower()
    hit = [t.source for t in terms if normalize(t.source).lower() in hay]
    miss = [t.source for t in terms if normalize(t.source).lower() not in hay]
    print(f"glossary: {len(hit)}/{len(terms)} recognized")
    if miss:
        print("  missing: " + ", ".join(miss))


def main() -> int:
    # The Windows console defaults to cp950 here, which cannot encode a Vietnamese transcript and
    # would abort the run partway through printing it.
    #
    # line_buffering matters as much as the encoding: redirected to a file this is block-buffered,
    # so a ninety-minute run that is interrupted writes nothing at all. Forty-two minutes of a real
    # transcript were lost that way.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path)
    ap.add_argument("--model", choices=list(config.WHISPER_DIRS), help="default: largest on disk")
    ap.add_argument("--ref", type=Path, help="reference transcript for CER")
    ap.add_argument("--hr", action="store_true", help="apply homophone replacer (models/hr, Chinese only)")
    ap.add_argument("--threads", type=int, default=max(2, (os.cpu_count() or 4) // 2),
                    help="decode threads; the default leaves half the machine usable")
    ap.add_argument("--gpu", action="store_true", help="decode on the GPU via CTranslate2")
    args = ap.parse_args()

    if args.hr and config.hr_files() is None:
        print(f"homophone replacer not set up under {config.HR_DIR} — run scripts/build_hr.py", file=sys.stderr)
        return 1

    model = config.WHISPER_DIRS[args.model] if args.model else postprocess.best_model()
    if not args.gpu and not model.is_dir():
        print(f"model not found: {model}", file=sys.stderr)
        return 1

    cfg = config.load()
    if args.gpu:
        if not asr_gpu.available():
            print("no CUDA device or CTranslate2 runtime found", file=sys.stderr)
            return 1
        terms = Store().glossary()
        transcriber = asr_gpu.Transcriber(languages=cfg.languages,
                                          hotwords=asr_gpu.hotwords_from(terms))
        label = config.gpu_model(cfg.languages)
    else:
        transcriber = asr.Transcriber(model_dir=model, quantized=False, num_threads=args.threads,
                                      homophones=args.hr, languages=cfg.languages)
        label = model.name
    diarizer = diarize.Diarizer(cfg=cfg)

    started = time.monotonic()
    utterances = postprocess.segment(args.wav)
    if not utterances:
        print("no speech detected", file=sys.stderr)
        return 1
    postprocess.assign_speakers(utterances, diarizer)

    def line(u: postprocess.Utterance) -> str:
        return f"[{int(u.start) // 60}:{int(u.start) % 60:02d}] {u.speaker} ({u.lang}) {u.text}"

    def show(u: postprocess.Utterance, done: int, total: int) -> None:
        if u.text:
            print(line(u))
        print(f"  {done}/{total} {u.start / 60:.0f}min elapsed={time.monotonic() - started:.0f}s",
              file=sys.stderr, flush=True)

    # Printed as they decode rather than at the end: on a ninety-minute recording the first pass
    # runs for most of an hour, and an interrupted run must not lose all of it.
    postprocess.transcribe_all(utterances, transcriber, progress=show)
    elapsed = time.monotonic() - started

    audio_seconds = sum(len(u.samples) for u in utterances) / config.SAMPLE_RATE
    corrected = [u for u in utterances if u.text]
    print("\n--- after language reconciliation ---")
    for u in corrected:
        print(line(u))

    print(f"\nmodel={label} hr={'on' if args.hr else 'off'} threads={args.threads} "
          f"utterances={len(utterances)} speech={audio_seconds:.1f}s wall={elapsed:.1f}s "
          f"rtf={elapsed / audio_seconds:.2f}")
    print("languages: " + ", ".join(f"{s}={l}" for s, l in postprocess.dominant_languages(utterances).items()))

    report_glossary("".join(u.text for u in utterances))

    if args.ref:
        hyp = normalize("".join(u.text for u in utterances))
        ref = normalize(args.ref.read_text(encoding="utf-8"))
        if not ref:
            print("reference is empty", file=sys.stderr)
            return 1
        print(f"CER: {edit_distance(ref, hyp) / len(ref):.1%}  (ref {len(ref)} chars, hyp {len(hyp)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
