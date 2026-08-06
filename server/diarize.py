"""Speaker separation and per-speaker language tracking.

Everything arrives on one audio stream — the machine is a silent listener in the meeting — so
speaker identity is the only way to tell participants apart, and it also decides which language
each utterance is transcribed in. A clustering mistake therefore costs twice: wrong name and
wrong language. Hence the hysteresis before ever changing a speaker's language.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

from . import config

log = logging.getLogger("meettranslate.diarize")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _similarities(rows: np.ndarray) -> np.ndarray:
    """Cosine of every row against every other — same rule as `cosine`, including the zero vector
    scoring zero rather than dividing by nothing."""
    norms = np.linalg.norm(rows, axis=1)
    safe = np.where(norms == 0, 1.0, norms)
    unit = rows / safe[:, None]
    unit[norms == 0] = 0.0
    return (unit @ unit.T).astype(np.float32)


@dataclass
class Speaker:
    code: str
    centroid: np.ndarray
    segments: int = 0
    language: str = ""  # established language; '' until the first transcription lands
    counts: dict[str, int] = field(default_factory=dict)
    _pending: tuple[str, int] = ("", 0)


@dataclass(frozen=True)
class Turn:
    """One stretch of one person talking, as the segmentation model heard it."""

    start: float
    end: float
    speaker: int


# Speech shorter than this is not a turn, and a gap shorter than this does not end one. Both are
# sherpa's own defaults for meeting audio; loosening either turns a breath into a speaker change.
MIN_TURN_ON = 0.3
MIN_TURN_OFF = 0.5


def turns(wav: Path, threshold: float | None = None) -> list[Turn] | None:
    """Where the speaker changes across a whole recording, or None if the model is not installed.

    The VAD cannot answer this. It hears speech against silence, so two people talking without a
    pause between them arrive as one utterance and get one embedding — an average of both, which
    then clusters as if it were a third person. Measured on a 2h19m meeting: 72 of 859 utterances
    (8.4%, ten minutes of speech) held more than one voice, one of them four.

    Optional on purpose: the model is a separate 6 MB download, and a machine without it should
    still be able to process a recording.
    """
    model = config.SPEAKER_SEGMENTATION_MODEL
    if not model.is_file():
        return None

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(model)),
            num_threads=max((os.cpu_count() or 4) // 2, 1),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(config.SPEAKER_MODEL),
            num_threads=max((os.cpu_count() or 4) // 2, 1),
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            threshold=config.SPEAKER_THRESHOLD if threshold is None else threshold),
        min_duration_on=MIN_TURN_ON,
        min_duration_off=MIN_TURN_OFF,
    )
    if not cfg.validate():
        log.warning("speaker segmentation config rejected; falling back to clustering utterances")
        return None

    audio, rate = sf.read(str(wav), dtype="float32")
    if rate != config.SAMPLE_RATE:
        raise ValueError(f"{wav} is {rate} Hz, expected {config.SAMPLE_RATE}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    result = sherpa_onnx.OfflineSpeakerDiarization(cfg).process(audio).sort_by_start_time()
    return [Turn(float(s.start), float(s.end), int(s.speaker)) for s in result]


class Diarizer:
    """Online clustering plus language bookkeeping.

    Online rather than offline because subtitles cannot wait for the meeting to end. The offline
    pass in postprocess sees every segment at once and corrects what this got wrong.
    """

    def __init__(self, model: str | None = None, threshold: float | None = None,
                 cfg: config.Config | None = None, known: list[tuple[str, np.ndarray]] | None = None):
        ec = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model or config.SPEAKER_MODEL), num_threads=1
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(ec)
        self._threshold = config.SPEAKER_THRESHOLD if threshold is None else threshold
        self._cfg = cfg or config.Config()
        self.speakers: list[Speaker] = []
        self._last_code: str | None = None
        # Voices this room has met before, as (name, centroid). A new speaker whose embedding
        # matches one is named on the spot instead of arriving as another anonymous Sn.
        self._known = list(known or [])
        self.recognised: dict[str, str] = {}

    def embed(self, samples: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(config.SAMPLE_RATE, samples)
        stream.input_finished()
        return np.array(self._extractor.compute(stream), dtype=np.float32)

    def assign(self, samples: np.ndarray) -> Speaker:
        """Identify the speaker of one utterance, creating a new one if nothing matches."""
        duration = len(samples) / config.SAMPLE_RATE

        # Short clips give unstable embeddings — a hummed "OK" would otherwise mint a new speaker
        # every time. Inheriting the previous speaker is right far more often than guessing.
        if duration < config.MIN_EMBED_SECONDS and self._last_code:
            return self._by_code(self._last_code)

        emb = self.embed(samples)

        best, best_score = None, -1.0
        for spk in self.speakers:
            score = cosine(emb, spk.centroid)
            if score > best_score:
                best, best_score = spk, score

        if best is None or best_score < self._threshold:
            best = Speaker(code=f"S{len(self.speakers) + 1}", centroid=emb)
            if name := self._recognise(emb):
                self.recognised[best.code] = name
            self.speakers.append(best)
        else:
            # Running mean: later segments refine the centroid without a stored history.
            n = best.segments
            best.centroid = (best.centroid * n + emb) / (n + 1)

        best.segments += 1
        self._last_code = best.code
        return best

    def _recognise(self, emb: np.ndarray) -> str:
        """Name a freshly minted speaker if a known voiceprint is close enough.

        Held to a higher bar than in-meeting clustering. Merging two segments of one meeting wrongly
        costs a split transcript; putting last week's name on this week's stranger is a mistake
        nobody reading the transcript would think to check.
        """
        best, score = "", -1.0
        for name, centroid in self._known:
            if (s := cosine(emb, centroid)) > score:
                best, score = name, s
        return best if score >= config.KNOWN_SPEAKER_THRESHOLD else ""

    def language_for(self, speaker: Speaker) -> str:
        """Language to force on this speaker's next utterance. '' means let Whisper auto-detect."""
        if pinned := self._cfg.pinned_languages.get(speaker.code):
            return pinned
        return speaker.language

    def observe_language(self, speaker: Speaker, detected: str) -> None:
        """Record which language an utterance actually turned out to be.

        Switching needs several consecutive disagreements, and many more between Chinese and
        English: Taiwanese Mandarin routinely embeds English words, so a single English-heavy
        sentence must not flip the speaker and wreck every following transcription.
        """
        if not detected or speaker.code in self._cfg.pinned_languages:
            return

        speaker.counts[detected] = speaker.counts.get(detected, 0) + 1

        if not speaker.language:
            speaker.language = detected
            return

        if detected == speaker.language:
            speaker._pending = ("", 0)
            return

        lang, count = speaker._pending
        count = count + 1 if lang == detected else 1
        needed = self._switch_threshold(speaker.language, detected)

        if count >= needed:
            speaker.language = detected
            speaker._pending = ("", 0)
        else:
            speaker._pending = (detected, count)

    def _switch_threshold(self, current: str, candidate: str) -> int:
        if {current, candidate} == {"zh", "en"}:
            return self._cfg.language_switch_after_zh_en
        return self._cfg.language_switch_after

    def _by_code(self, code: str) -> Speaker:
        return next(s for s in self.speakers if s.code == code)


def load_known(store) -> list[tuple[str, np.ndarray]]:
    """Known voiceprints as arrays. Stored as raw float32 bytes — the embedder's own layout."""
    return [(name, np.frombuffer(blob, dtype=np.float32)) for name, blob in store.known_speakers()]


def cluster_offline(embeddings: list[np.ndarray], threshold: float | None = None) -> list[int]:
    """Agglomerative clustering over every segment of a finished meeting, on complete linkage.

    Seeing all segments at once fixes the online pass's mistakes: two clusters that online kept
    apart because the speaker's first few seconds were atypical get merged here.

    Two clusters join only when their *worst* cross-pair still clears the threshold. Averaging
    them into a centroid instead — which this did — produces a mean voice that resembles everyone
    a little, so the biggest cluster keeps absorbing and one speaker ends up holding the meeting.
    On a 2h19m morning meeting, 859 segments:

        linkage    thr    clusters   biggest   different speakers kept apart
        centroid   0.65        35     93.2%     0 of 33
        centroid   0.80       176     31.1%    33 of 33
        complete   0.65       134     14.1%    33 of 33

    Ten utterances there were attributable by content — the chair, a sales report, a production
    report — and at 0.65 every one of the 33 pairs from different people had been merged. That is
    the transcript naming almost every line after whoever S1 turned out to be.

    The threshold stays where two earlier interviews put it. Complete linkage splits one speaker
    across more codes than centroid merging did, which is the trade taken deliberately: naming the
    same person into several boxes is tedious but possible, while a cluster holding four people
    cannot be separated by anything the person reading the transcript can do.
    """
    if not embeddings:
        return []

    thr = config.SPEAKER_THRESHOLD if threshold is None else threshold
    n = len(embeddings)
    members: list[list[int]] = [[i] for i in range(n)]

    # Every pair's similarity, computed once and then repaired in place. Rescanning all pairs in
    # Python each round is O(n^3), and a two-hour meeting segments into the thousands: that spent
    # over an hour here with the GPU sitting idle, never reaching transcription at all.
    #
    # A merged cluster is masked rather than deleted — deleting means copying the whole matrix
    # every round, which is the same cubic cost again with a smaller constant.
    sims = _similarities(np.asarray(embeddings, dtype=np.float32))
    np.fill_diagonal(sims, -np.inf)
    alive = np.ones(n, dtype=bool)

    # Each row's best partner, so choosing the pair to merge is a scan of n values, not n^2.
    best_at = sims.argmax(axis=1) if n > 1 else np.zeros(n, dtype=int)
    best = sims[np.arange(n), best_at] if n > 1 else np.full(n, -np.inf)

    for _ in range(n - 1):
        i = int(np.argmax(np.where(alive, best, -np.inf)))
        j = int(best_at[i])
        if best[i] < thr:
            break
        if i > j:
            i, j = j, i

        # Complete linkage: the merged cluster sits as far from every other as its worse half did.
        row = np.minimum(sims[i], sims[j])
        members[i] += members[j]
        members[j] = []
        alive[j] = False

        row[j] = -np.inf
        row[i] = -np.inf
        sims[i, :] = row
        sims[:, i] = row
        sims[j, :] = -np.inf
        sims[:, j] = -np.inf
        best[j] = -np.inf
        best_at[i] = int(np.argmax(row))
        best[i] = row[best_at[i]]

        # Similarities only ever fall here, so a row's cached best can only be stale downwards —
        # and only if it pointed at one of the two clusters that just became one.
        stale = alive & ((best_at == i) | (best_at == j))
        stale[i] = False
        for r in np.flatnonzero(stale):
            best_at[r] = int(np.argmax(sims[r]))
            best[r] = sims[r, best_at[r]]

    labels = [0] * n
    for label, group in enumerate([m for m in members if m]):
        for idx in group:
            labels[idx] = label
    return labels
