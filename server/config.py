"""Settings for MeetTranslate. Loaded from config.json next to the repo root, env vars override."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
RECORDINGS_DIR = ROOT / "recordings"

# Whisper wants 16 kHz mono; capturing at that rate avoids a resample step later.
SAMPLE_RATE = 16_000
CHANNELS = 1
# Frames per callback. 1600 @ 16 kHz = 100 ms — small enough that stopping feels instant,
# large enough that the callback isn't called so often it starves.
BLOCK_SIZE = 1600


@dataclass
class Config:
    # Language codes present in the meeting. First entry is the display-order primary.
    # zh = Traditional Chinese (Taiwan) — ASR output is converted, see plan.md.
    languages: list[str] = field(default_factory=lambda: ["zh", "vi", "en"])
    # Substring matched against input device names, case-insensitive. Empty = system default.
    # Set this to the virtual audio device carrying the meeting audio (VB-Cable / BlackHole).
    input_device: str = ""

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")


def load() -> Config:
    data = {}
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    cfg = Config(**{k: v for k, v in data.items() if k in Config.__dataclass_fields__})

    if env := os.environ.get("MEETTRANSLATE_INPUT_DEVICE"):
        cfg.input_device = env
    if env := os.environ.get("MEETTRANSLATE_LANGUAGES"):
        cfg.languages = [s.strip() for s in env.split(",") if s.strip()]

    return cfg
