"""Self-checks for device resolution and config loading. Run: python -m server.test_audio

No hardware needed — device lookup is exercised against whatever this machine reports, and the
failure paths are the point: silently recording from the wrong device is the bug that looks like
"it captured nothing", so resolve_device must raise rather than fall back.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from . import audio, config


def test_empty_fragment_means_default() -> None:
    assert audio.resolve_device("") is None


def test_unknown_device_raises_with_available_names() -> None:
    try:
        audio.resolve_device("no-such-device-xyzzy")
    except audio.DeviceNotFound as exc:
        assert "no-such-device-xyzzy" in str(exc)
        assert "Available:" in str(exc)
    else:
        raise AssertionError("expected DeviceNotFound")


def test_match_is_case_insensitive_and_partial() -> None:
    devices = audio.list_input_devices()
    if not devices:
        return  # CI box with no audio hardware; nothing to assert against

    name = devices[0]["name"]
    fragment = name[: max(4, len(name) // 2)]
    assert audio.resolve_device(fragment.upper()) == devices[0]["index"]
    assert audio.resolve_device(fragment.lower()) == devices[0]["index"]


def test_duplicate_names_are_ranked_by_hostapi() -> None:
    """Windows lists one device under four host APIs; MME comes first but has the worst latency.

    Every match is returned, best first, because a device PortAudio lists is not necessarily one
    it can open — start() falls through to the next candidate.
    """
    fake = [
        {"index": 1, "name": "Stereo Mix", "channels": 2, "hostapi": "MME"},
        {"index": 7, "name": "Stereo Mix", "channels": 2, "hostapi": "Windows DirectSound"},
        {"index": 15, "name": "Stereo Mix", "channels": 2, "hostapi": "Windows WASAPI"},
        {"index": 22, "name": "Stereo Mix", "channels": 2, "hostapi": "Windows WDM-KS"},
        {"index": 30, "name": "Headset", "channels": 1, "hostapi": "Windows WASAPI"},
    ]
    with patch.object(audio, "list_input_devices", return_value=fake):
        assert audio.candidate_devices("stereo mix") == [15, 22, 7, 1]
        assert audio.resolve_device("stereo mix") == 15
        # Non-matching devices must never appear as a fallback.
        assert 30 not in audio.candidate_devices("stereo mix")

    with patch.object(audio, "list_input_devices", return_value=fake[:2]):
        assert audio.candidate_devices("stereo mix") == [7, 1]


def test_empty_fragment_yields_the_default_only() -> None:
    assert audio.candidate_devices("") == [None]


def test_config_defaults_and_env_override() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "absent.json"
        with patch.object(config, "CONFIG_PATH", missing), patch.dict(os.environ, {}, clear=True):
            cfg = config.load()
            assert cfg.languages == ["zh", "vi", "en"]
            assert cfg.input_device == ""

        with patch.object(config, "CONFIG_PATH", missing), \
             patch.dict(os.environ, {"MEETTRANSLATE_LANGUAGES": "zh, vi", "MEETTRANSLATE_INPUT_DEVICE": "Cable"}):
            cfg = config.load()
            assert cfg.languages == ["zh", "vi"]
            assert cfg.input_device == "Cable"


def test_unknown_config_keys_are_ignored() -> None:
    """A config.json written by a newer version must not crash an older one."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text('{"languages": ["zh", "en"], "someFutureKey": 1}', encoding="utf-8")
        with patch.object(config, "CONFIG_PATH", path), patch.dict(os.environ, {}, clear=True):
            assert config.load().languages == ["zh", "en"]


def test_config_save_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        with patch.object(config, "CONFIG_PATH", path), patch.dict(os.environ, {}, clear=True):
            config.Config(languages=["zh", "vi"], input_device="立體聲混音").save()
            reloaded = config.load()
            assert reloaded.languages == ["zh", "vi"]
            assert reloaded.input_device == "立體聲混音"  # non-ASCII must survive the round trip


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    main()
