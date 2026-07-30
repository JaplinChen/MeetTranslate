"""Audio capture. PortAudio (sounddevice) so Windows and macOS share one code path.

The meeting audio must reach us through a virtual audio device (VB-Cable on Windows,
BlackHole on macOS) with Teams output pointed at it. Muting the system playback device
instead would make the capture silent — see plan.md.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from . import config


class DeviceNotFound(Exception):
    pass


def list_input_devices() -> list[dict]:
    """Every device that can be recorded from, in PortAudio index order."""
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"], "hostapi": sd.query_hostapis(d["hostapi"])["name"]}
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


# Windows exposes the same device under MME, DirectSound, WASAPI and WDM-KS; macOS uses Core Audio.
# Matching by name alone would pick whichever comes first (MME), the highest-latency of the four.
_PREFERRED_HOSTAPIS = ("Windows WASAPI", "Core Audio")


def resolve_device(name_fragment: str) -> int | None:
    """Device index whose name contains `name_fragment` (case-insensitive). None = system default.

    Raises DeviceNotFound with the available names rather than falling back silently: a silent
    fallback to the wrong device is the failure mode that looks like "it recorded nothing".
    """
    if not name_fragment:
        return None

    needle = name_fragment.casefold()
    matches = [d for d in list_input_devices() if needle in d["name"].casefold()]

    if not matches:
        available = ", ".join(d["name"] for d in list_input_devices()) or "(none)"
        raise DeviceNotFound(f"No input device matching {name_fragment!r}. Available: {available}")

    for dev in matches:
        if dev["hostapi"] in _PREFERRED_HOSTAPIS:
            return dev["index"]
    return matches[0]["index"]


@dataclass
class RecorderStatus:
    recording: bool
    path: str | None
    seconds: float
    peak: float  # 0.0-1.0 of the most recent block; 0.0 for a long stretch means no audio is arriving
    dropped_blocks: int


class Recorder:
    """Captures to a wav file on a writer thread.

    File IO is kept off the PortAudio callback so a slow disk cannot cause an overrun, and the
    recording is deliberately independent of everything downstream: without Graph API this wav is
    the only source for the post-meeting transcript, so it must survive a pipeline crash.
    """

    def __init__(self, device: int | None = None):
        self._device = device
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=256)
        self._stream: sd.InputStream | None = None
        self._writer: threading.Thread | None = None
        self._path: Path | None = None
        self._frames = 0
        self._peak = 0.0
        self._dropped = 0

    def _callback(self, indata, _frames, _time, status) -> None:
        if status:
            self._dropped += 1
        block = indata.copy()
        self._peak = float(np.abs(block).max())
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            # Writer is wedged. Losing a block beats blocking the audio thread and cascading.
            self._dropped += 1

    def _write_loop(self, path: Path) -> None:
        with sf.SoundFile(path, mode="w", samplerate=config.SAMPLE_RATE, channels=config.CHANNELS, subtype="PCM_16") as f:
            while (block := self._queue.get()) is not None:
                f.write(block)
                self._frames += len(block)

    def start(self, path: Path) -> None:
        if self._stream is not None:
            raise RuntimeError("already recording")

        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._frames = 0
        self._peak = 0.0
        self._dropped = 0

        self._writer = threading.Thread(target=self._write_loop, args=(path,), daemon=True)
        self._writer.start()

        self._stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            blocksize=config.BLOCK_SIZE,
            device=self._device,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> Path | None:
        if self._stream is None:
            return None

        self._stream.stop()
        self._stream.close()
        self._stream = None

        self._queue.put(None)
        if self._writer:
            self._writer.join(timeout=10)
            self._writer = None

        return self._path

    def status(self) -> RecorderStatus:
        return RecorderStatus(
            recording=self._stream is not None,
            path=str(self._path) if self._path else None,
            seconds=self._frames / config.SAMPLE_RATE,
            peak=self._peak,
            dropped_blocks=self._dropped,
        )


def new_session_path() -> Path:
    return config.RECORDINGS_DIR / f"session_{time.strftime('%Y%m%d_%H%M%S')}.wav"
