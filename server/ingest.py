"""Turning a file someone recorded elsewhere into something this room can learn from.

A meeting that was captured by Teams, a phone or a camera is the same evidence as a live capture —
voices to attach names to, sentences to correct — but it arrives as an mp4 rather than the 16 kHz
mono wav every stage downstream assumes. Converting on the way in means nothing after this point
has to know an upload happened.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import config


def extract_audio(src: Path, dest: Path) -> None:
    """Write `src`'s audio to `dest` as the wav the pipeline expects, whatever container it was in.

    ffmpeg rather than an in-process decoder: the formats a meeting arrives in (mp4, mkv, m4a, the
    occasional webm) are exactly what it already handles, and it is the same tool the offline
    scripts have always used.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed — it is needed to read audio out of a video")

    done = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1",
         "-ar", str(config.SAMPLE_RATE), "-c:a", "pcm_s16le", str(dest)],
        capture_output=True, text=True,
    )
    if done.returncode != 0:
        # ffmpeg's last line names the actual problem; the rest is banner noise no one can act on.
        detail = (done.stderr or "").strip().splitlines()
        raise ValueError(detail[-1] if detail else "ffmpeg could not read that file")
