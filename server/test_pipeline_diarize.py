"""Who is speaking: when a speaker's language may change, and how voices are grouped.

The language rules are hysteresis -- one disagreement is noise, several in a row is a speaker who
was misidentified -- and the clustering is judged by the speech it groups, not by cluster count.
"""

from __future__ import annotations

import numpy as np

from . import config, diarize, postprocess


def test_a_speaker_needs_evidence_before_setting_their_own_language() -> None:
    """Separating real participants also produces a tail of speakers holding two or three
    utterances, and a majority over two samples is a coin flip. Letting those establish their own
    language put 433 Chinese lines under an English label across seven interviews.
    """
    import numpy as np

    def said(speaker: str, lang: str) -> postprocess.Utterance:
        return postprocess.Utterance(0.0, np.zeros(1, dtype="float32"), speaker, lang, "x")

    meeting = [said("S1", "zh")] * 20 + [said("S2", "en")] * 6 + [said("S3", "en")] * 2
    dominant = postprocess.dominant_languages(meeting)
    assert dominant["S1"] == "zh"
    # Enough of its own to disagree with the room.
    assert dominant["S2"] == "en"
    # Not enough; inherits the meeting rather than guessing.
    assert dominant["S3"] == "zh"
    assert postprocess.dominant_languages([]) == {}


def test_clustering_is_judged_by_speech_not_cluster_count() -> None:
    """Two speakers who each hold a real share of the meeting must not be merged.

    The previous threshold was picked by counting clusters, which rewarded merging everyone into
    one: on a 37-minute interview it produced a single speaker holding 100% of the speech, and on
    a 67-minute one it produced 49 minutes against 14 where 0.65 finds 21 / 14 / 12 / 8.
    """
    import numpy as np

    # Two voices that a room microphone would leave closer together than a studio would.
    a = np.array([1.0, 0.35, 0.0], dtype=np.float32)
    b = np.array([0.35, 1.0, 0.0], dtype=np.float32)
    assert diarize.cosine(a, b) < config.SPEAKER_THRESHOLD, "the fixture must be separable"

    labels = diarize.cluster_offline([a, a * 0.9, b, b * 1.1])
    assert len(set(labels)) == 2, labels
    assert labels[0] == labels[1] and labels[2] == labels[3]


def test_known_voice_is_named_on_sight() -> None:
    """A voice the room has met before arrives named instead of as another anonymous Sn.

    Held to a stricter bar than in-meeting clustering: a wrong merge shows up as a split
    transcript, a wrong name is attributed to a real person and nobody thinks to check it.
    """
    import numpy as np

    vincent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    stranger = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    d = diarize.Diarizer.__new__(diarize.Diarizer)
    d._known = [("Vincent", vincent)]

    assert d._recognise(vincent) == "Vincent"
    # Close, but not close enough to put someone's name on it.
    nearly = np.array([1.0, 1.3, 0.0], dtype=np.float32)
    assert diarize.cosine(nearly, vincent) < config.KNOWN_SPEAKER_THRESHOLD
    assert d._recognise(nearly) == ""
    assert d._recognise(stranger) == ""
    # An empty roster never guesses.
    d._known = []
    assert d._recognise(vincent) == ""


def test_cosine() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert abs(diarize.cosine(a, a) - 1.0) < 1e-6
    assert abs(diarize.cosine(a, np.array([0.0, 1.0, 0.0], dtype=np.float32))) < 1e-6
    assert diarize.cosine(a, np.zeros(3, dtype=np.float32)) == 0.0


def _speaker(lang: str = "zh") -> diarize.Speaker:
    return diarize.Speaker(code="S1", centroid=np.zeros(4, dtype=np.float32), language=lang)


def _diarizer(cfg: config.Config) -> diarize.Diarizer:
    """A Diarizer without the ONNX extractor — language bookkeeping needs no model."""
    d = diarize.Diarizer.__new__(diarize.Diarizer)
    d._cfg = cfg
    d.speakers = []
    d._last_code = None
    return d


def test_first_language_is_adopted_immediately() -> None:
    d, spk = _diarizer(config.Config()), diarize.Speaker(code="S1", centroid=np.zeros(4, dtype=np.float32))
    d.observe_language(spk, "vi")
    assert spk.language == "vi"


def test_single_disagreement_does_not_switch() -> None:
    d, spk = _diarizer(config.Config()), _speaker("vi")
    d.observe_language(spk, "en")
    assert spk.language == "vi"


def test_switch_after_enough_consecutive_disagreements() -> None:
    cfg = config.Config(language_switch_after=3)
    d, spk = _diarizer(cfg), _speaker("vi")
    for _ in range(3):
        d.observe_language(spk, "en")
    assert spk.language == "en"


def test_agreement_resets_the_pending_switch() -> None:
    """Alternating detections must not accumulate into a switch."""
    cfg = config.Config(language_switch_after=3)
    d, spk = _diarizer(cfg), _speaker("vi")
    for _ in range(2):
        d.observe_language(spk, "en")
        d.observe_language(spk, "vi")
    assert spk.language == "vi"


def test_zh_en_needs_a_higher_bar() -> None:
    """Taiwanese Mandarin embeds English constantly; the zh<->en pair must resist flipping."""
    cfg = config.Config(language_switch_after=3, language_switch_after_zh_en=6)
    d, spk = _diarizer(cfg), _speaker("zh")
    for _ in range(5):
        d.observe_language(spk, "en")
    assert spk.language == "zh", "flipped too early on code-switched speech"
    d.observe_language(spk, "en")
    assert spk.language == "en"


def test_pinned_language_never_changes() -> None:
    cfg = config.Config(pinned_languages={"S1": "zh"}, language_switch_after=1)
    d, spk = _diarizer(cfg), _speaker("zh")
    for _ in range(10):
        d.observe_language(spk, "en")
    assert spk.language == "zh"
    assert d.language_for(spk) == "zh"


def test_offline_clustering_groups_similar_embeddings() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=64).astype(np.float32)
    b = rng.normal(size=64).astype(np.float32)
    # Three noisy takes of speaker A, two of speaker B.
    embeddings = [a + rng.normal(scale=0.05, size=64).astype(np.float32) for _ in range(3)]
    embeddings += [b + rng.normal(scale=0.05, size=64).astype(np.float32) for _ in range(2)]

    labels = diarize.cluster_offline(embeddings, threshold=0.5)
    assert len(set(labels)) == 2, labels
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]


def test_offline_clustering_edge_cases() -> None:
    assert diarize.cluster_offline([]) == []
    single = [np.array([1.0, 0.0], dtype=np.float32)]
    assert diarize.cluster_offline(single) == [0]
    # Zero vectors score zero against everything rather than dividing by nothing, so they never
    # merge — the same rule `cosine` applies.
    zeros = [np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32)]
    assert len(set(diarize.cluster_offline(zeros))) == 2


def test_offline_clustering_scales_to_a_long_meeting() -> None:
    """A two-hour recording segments into the thousands, and clustering must not be the wall.

    Comparing every pair in Python each round is O(n^3): at this size it ran for over an hour with
    the GPU idle, so an import never reached transcription at all. The bound is deliberately loose
    — it is here to catch a return to cubic, not to police milliseconds.
    """
    import time

    rng = np.random.default_rng(7)
    bases = rng.normal(size=(5, 192)).astype(np.float32)
    embeddings = [(bases[k % 5] + rng.normal(scale=0.3, size=192)).astype(np.float32)
                  for k in range(1500)]

    start = time.perf_counter()
    labels = diarize.cluster_offline(embeddings)
    elapsed = time.perf_counter() - start

    assert len(set(labels)) == 5, len(set(labels))
    assert elapsed < 20, f"clustering 1500 segments took {elapsed:.1f}s"
