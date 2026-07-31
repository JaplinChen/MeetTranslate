"""Settings for MeetTranslate. Loaded from config.json next to the repo root, env vars override."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
RECORDINGS_DIR = ROOT / "recordings"
MODELS_DIR = ROOT / "models"

# Whisper wants 16 kHz mono; capturing at that rate avoids a resample step later.
SAMPLE_RATE = 16_000
CHANNELS = 1
# Frames per callback. 1600 @ 16 kHz = 100 ms — small enough that stopping feels instant,
# large enough that the callback isn't called so often it starves.
BLOCK_SIZE = 1600

VAD_MODEL = MODELS_DIR / "silero_vad.onnx"
SPEAKER_MODEL = MODELS_DIR / "speaker_embedding.onnx"
# Homophone replacer: dict/, lexicon.txt from the sherpa-onnx hr-files release, replace.fst built
# from the glossary. Chinese only — pinyin is the matching key.
HR_DIR = MODELS_DIR / "hr"

# Whisper model directories, smallest first. The realtime tier is picked from `whisper_model`;
# postprocess always uses the largest available.
WHISPER_DIRS = {
    "tiny": MODELS_DIR / "sherpa-onnx-whisper-tiny",
    "base": MODELS_DIR / "sherpa-onnx-whisper-base",
    "small": MODELS_DIR / "sherpa-onnx-whisper-small",
    "medium": MODELS_DIR / "sherpa-onnx-whisper-medium",
    "large-v3": MODELS_DIR / "sherpa-onnx-whisper-large-v3",
}

# Below this cosine similarity to every known centroid, a segment starts a new speaker.
#
# Swept over two real interview recordings, counting clusters per threshold. At 0.55 a three-person
# interview split into fourteen speakers, twelve of them holding a single utterance; the room mic
# and Teams' noise suppression leave the same voice further from itself than a studio recording
# would. 0.45 is the knee on both recordings — three balanced clusters on the multi-speaker one,
# one dominant cluster on the single-presenter one, and no singleton tail on either.
SPEAKER_THRESHOLD = 0.45
# Segments shorter than this give unstable embeddings; they inherit the previous speaker.
MIN_EMBED_SECONDS = 1.0


@dataclass
class Display:
    """Subtitle presentation. Tuned for a TV at meeting-room viewing distance, not a desk monitor."""

    font_size: int = 40           # px for the source line; translations scale from this
    lines: int = 6                # utterances kept on screen before older ones scroll away
    show_source: str = "top"      # top | bottom | hidden
    show_speaker: bool = True
    colour_speakers: bool = True
    theme: str = "dark"           # dark | light


@dataclass
class Config:
    # Language codes present in the meeting. First entry is the display-order primary.
    # zh = Traditional Chinese (Taiwan) — ASR output is converted, see plan.md.
    languages: list[str] = field(default_factory=lambda: ["zh", "vi", "en"])
    # Substring matched against input device names, case-insensitive. Empty = system default.
    # Set this to the virtual audio device carrying the meeting audio (VB-Cable / BlackHole).
    input_device: str = ""
    whisper_model: str = "small"
    # Consecutive detections disagreeing with a speaker's established language before switching.
    # Higher for zh<->en because Taiwanese Mandarin routinely embeds English words and would
    # otherwise flip the speaker's language mid-meeting. See plan.md decision 5.
    language_switch_after: int = 3
    language_switch_after_zh_en: int = 6
    # Speaker code -> language code. Pins a speaker so detection never overrides it.
    pinned_languages: dict[str, str] = field(default_factory=dict)
    display: Display = field(default_factory=Display)

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    def whisper_dir(self) -> Path:
        return WHISPER_DIRS.get(self.whisper_model, WHISPER_DIRS["small"])


def load() -> Config:
    data = {}
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    known = {k: v for k, v in data.items() if k in Config.__dataclass_fields__}
    # Nested dataclass: json gives a plain dict, and an older file may lack newer display keys.
    if isinstance(known.get("display"), dict):
        known["display"] = Display(**{k: v for k, v in known["display"].items()
                                      if k in Display.__dataclass_fields__})
    cfg = Config(**known)

    if env := os.environ.get("MEETTRANSLATE_INPUT_DEVICE"):
        cfg.input_device = env
    if env := os.environ.get("MEETTRANSLATE_LANGUAGES"):
        cfg.languages = [s.strip() for s in env.split(",") if s.strip()]
    if env := os.environ.get("MEETTRANSLATE_WHISPER_MODEL"):
        cfg.whisper_model = env

    return cfg


def gpu_model(languages: list[str] | None = None) -> str:
    """CTranslate2 model for the GPU path.

    Breeze ASR 25, a large-v2 fine-tune for Taiwanese Mandarin and Mandarin-English code-switching,
    was tried here and dropped. On a real interview it and large-v3 differed on five lines out of
    a hundred and thirty-seven, all of them pre-meeting chatter where neither was clearly right,
    at the same realtime factor. What actually improved the transcript was leaving Whisper small
    behind; the fine-tune added nothing on top of that, and it does not know Vietnamese, which
    every meeting in this room contains.

    `languages` is accepted because the choice is language-dependent in principle — it just has
    one answer today.
    """
    return os.environ.get("MEETTRANSLATE_GPU_MODEL", "large-v3")


def hr_files() -> dict[str, str] | None:
    """Homophone replacer paths for sherpa-onnx, or None if not set up.

    All three must be present: the dict and lexicon convert decoded Chinese to pinyin, replace.fst
    holds the pinyin→term rules built from the glossary by scripts/build_hr.py.
    """
    paths = {"hr_dict_dir": HR_DIR / "dict", "hr_lexicon": HR_DIR / "lexicon.txt",
             "hr_rule_fsts": HR_DIR / "replace.fst"}
    if not all(p.exists() for p in paths.values()):
        return None
    return {k: str(v) for k, v in paths.items()}


def available_whisper_models() -> list[str]:
    """Model tiers actually present on disk, smallest first."""
    return [name for name, path in WHISPER_DIRS.items() if path.is_dir()]
