"""Do the same voices turn up in different meetings, and at what similarity?

    python -m scripts.match_speakers "C:/videos/*.mp4"

Voiceprint recognition is the one mechanism here with no evidence behind it: the unit tests use
made-up vectors, and a threshold picked against made-up vectors means nothing. What it needs is
the same person recorded twice on different days, through the same room and the same codec.

This measures that. Each recording is clustered on its own, every speaker with enough speech
gets a centroid, and every centroid is compared with every centroid from a *different* recording.
If two meetings share a participant the pair shows up near the top of the list; the question the
output answers is whether there is a gap between those pairs and everyone else, and whether
KNOWN_SPEAKER_THRESHOLD sits in it.

No labels are needed to see the gap, and none are available — but the timestamps let you check a
pair against the transcripts and say whether it really is the same person.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import config, diarize, postprocess  # noqa: E402

# A centroid built from less speech than this is noise wearing a speaker's name.
MIN_SPEECH_SECONDS = 20.0


def centroids(video: Path, keep_wav: Path | None = None) -> dict[str, tuple[np.ndarray, float]]:
    """One centroid per speaker in this recording, with how many seconds it was built from."""
    wav = keep_wav or Path(tempfile.gettempdir()) / f"{video.stem}.wav"
    if not wav.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-ac", "1",
                        "-ar", str(config.SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav)], check=True)

    utterances = postprocess.segment(wav)
    diarizer = diarize.Diarizer()
    postprocess.assign_speakers(utterances, diarizer)

    grouped: dict[str, list[np.ndarray]] = {}
    seconds: dict[str, float] = {}
    for u in utterances:
        length = len(u.samples) / config.SAMPLE_RATE
        if length < config.MIN_EMBED_SECONDS:
            continue
        grouped.setdefault(u.speaker, []).append(diarizer.embed(u.samples))
        seconds[u.speaker] = seconds.get(u.speaker, 0.0) + length

    if keep_wav is None:
        wav.unlink(missing_ok=True)

    return {code: (np.mean(vectors, axis=0), seconds[code])
            for code, vectors in grouped.items() if seconds[code] >= MIN_SPEECH_SECONDS}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--top", type=int, default=25, help="how many pairs to list")
    args = ap.parse_args()

    people: list[tuple[str, str, np.ndarray, float]] = []
    for video in args.videos:
        found = centroids(video)
        print(f"{video.stem}: {len(found)} speakers with more than "
              f"{MIN_SPEECH_SECONDS:.0f}s of speech")
        for code, (vector, seconds) in found.items():
            people.append((video.stem, code, vector, seconds))

    pairs = []
    for i, (film_a, code_a, vec_a, sec_a) in enumerate(people):
        for film_b, code_b, vec_b, sec_b in people[i + 1:]:
            if film_a == film_b:  # same meeting: clustering already decided these are different
                continue
            pairs.append((diarize.cosine(vec_a, vec_b),
                          f"{film_a}/{code_a} ({sec_a:.0f}s)", f"{film_b}/{code_b} ({sec_b:.0f}s)"))
    pairs.sort(reverse=True)

    print(f"\n{len(pairs)} cross-meeting pairs, most similar first "
          f"(threshold {config.KNOWN_SPEAKER_THRESHOLD}):")
    for score, a, b in pairs[: args.top]:
        mark = "  <- would be named" if score >= config.KNOWN_SPEAKER_THRESHOLD else ""
        print(f"  {score:.3f}  {a:34} {b:34}{mark}")

    above = sum(1 for s, _, _ in pairs if s >= config.KNOWN_SPEAKER_THRESHOLD)
    print(f"\n{above} of {len(pairs)} pairs would be treated as the same person")
    if pairs:
        scores = [s for s, _, _ in pairs]
        print(f"similarity runs from {min(scores):.3f} to {max(scores):.3f}, "
              f"median {sorted(scores)[len(scores) // 2]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
