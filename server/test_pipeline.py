"""Self-checks for the ASR and diarization logic. Run: python -m server.test_pipeline

Model-free: the parts worth testing are the decision rules — when a collapsed decode is detected,
when a speaker's language is allowed to change, how offline clustering merges — and those are
plain functions over data.
"""

from __future__ import annotations

import numpy as np

from . import asr, config, correct, diarize, store


def test_degenerate_detects_collapsed_decode() -> None:
    # The real output of forcing zh on English audio.
    assert asr.is_degenerate("前來,前來,前來,前來,前來,前來,前來,前來,前來,前來,前來")
    assert asr.is_degenerate("the the the the the the the the the the the")


def test_noise_annotations_are_dropped() -> None:
    """Every one of these came out of ten minutes of room noise before a real meeting started."""
    for text in ("[MUSIC PLAYING]", "(static)", "[BLANK_AUDIO]", "(upbeat music)", "(indistinct)", "[static"):
        assert asr.is_noise(text), text


def test_youtube_boilerplate_is_dropped() -> None:
    """Whisper answers unreadable audio with subtitle sign-offs. On seven real interviews these
    were 15% of every Vietnamese line, and none of it was spoken."""
    for text in ("Cảm ơn các bạn đã theo dõi và đăng ký kênh của mình.",
                 "Hãy subscribe cho kênh La La School",
                 "Cảm ơn các bạn đã theo dõi và hẹn gặp lại.",
                 "您可以訂閱我們的頻道,並且請點選訂閱",
                 "明鏡及點點欄目",
                 "I'll see you in a minute. Thanks for watching."):
        assert asr.is_hallucination(text), text


def test_hallucination_filter_spares_real_speech() -> None:
    """Matched as phrases: a meeting may say 訂閱 or subscribe without meaning a channel."""
    for text in ("Bây giờ mình hiện tại đang làm thủ công bằng Excel.",
                 "我們要訂閱這個服務嗎",
                 "我們的料號其實變動很大",
                 "這個 schedule 要 delay 一週"):
        assert not asr.is_hallucination(text), text


def test_noise_keeps_speech_containing_brackets() -> None:
    assert not asr.is_noise("這個 (ERP) 系統要換掉")
    assert not asr.is_noise("- All right")
    assert not asr.is_noise("")


def test_language_whitelist_rejects_what_was_never_configured() -> None:
    """A zh/vi/en meeting produced pt, bo, ja, ko and it on a real recording — all from noise."""
    tr = asr.Transcriber(languages=["zh", "vi", "en"])
    assert tr._allowed("zh") and tr._allowed("vi") and tr._allowed("en")
    assert not tr._allowed("pt") and not tr._allowed("bo") and not tr._allowed("ja")
    # Whisper reports bare codes, settings may carry a region; both must match.
    assert asr.Transcriber(languages=["zh-TW", "en"])._allowed("zh")
    # Auto-detect that reported nothing, and an unconfigured meeting, stay permissive.
    assert tr._allowed("") and asr.Transcriber()._allowed("pt")


def test_corrector_fixes_near_misses_only() -> None:
    def term(source: str) -> store.Term:
        return store.Term(id=0, source=source, lang="", mode="hint", category="", targets={})

    c = correct.Corrector([term("工單"), term("威剛科技"), term("Vincent"), term("治具")])
    # Wrong character, same sound — what the decode-time replacer misses once the tone is wrong.
    assert c.fix("公單的管理") == "工單的管理"
    assert c.fix("微剛科技的部分") == "威剛科技的部分"
    assert c.fix("直距的管理") == "治具的管理"
    assert c.fix("線上還有問incent") == "線上還有問Vincent"
    # Different words that merely rhyme must survive untouched.
    assert c.fix("我們公司的工作單位") == "我們公司的工作單位"
    assert c.fix("這個 schedule 要 delay 一週") == "這個 schedule 要 delay 一週"
    assert correct.Corrector([]).fix("原文不動") == "原文不動"


def test_corrector_never_rewrites_a_near_rhyme() -> None:
    """A single edit of Mandarin pinyin is a different word, not a misspelling of the same one.

    Allowing one, over seven real transcripts and a thirty-three term glossary, rewrote 知道 to
    製造 156 times and 生產 to 生管 146 times — 1578 corruptions. Chinese must match exactly.
    """
    def term(source: str) -> store.Term:
        return store.Term(id=0, source=source, lang="", mode="hint", category="", targets={})

    c = correct.Corrector([term("製造"), term("生管"), term("呆料"), term("委外")])
    assert c.fix("我不知道這件事") == "我不知道這件事"
    assert c.fix("生產線的狀況") == "生產線的狀況"
    assert c.fix("這批材料還在") == "這批材料還在"
    assert c.fix("未來五年的規劃") == "未來五年的規劃"
    # What it must still catch: the same sound, a different character.
    assert c.fix("生館的排程") == "生管的排程"


def test_degenerate_accepts_normal_speech() -> None:
    assert not asr.is_degenerate("這個 schedule 要 delay 一週，我們下週再確認一次時程")
    assert not asr.is_degenerate("After early nightfall the yellow lamps would light up the squalid quarter")
    assert not asr.is_degenerate("Chúng ta cần xác nhận lại lịch trình vào tuần sau nhé")


def test_degenerate_ignores_short_text() -> None:
    """A terse reply must never be mistaken for a collapse."""
    assert not asr.is_degenerate("好的")
    assert not asr.is_degenerate("OK OK")


def test_chinese_output_is_converted_to_taiwan_traditional() -> None:
    """Whisper emits Simplified for zh regardless of the speaker; Simplified on the meeting-room
    TV is an immediately visible failure, so conversion is not optional."""
    assert asr._post("这个软件的质量", "zh") == "這個軟體的質量"
    # Taiwan vocabulary, not just character shapes: 軟件 -> 軟體, 下周 -> 下週.
    converted = asr._post("我们下周确认软件进度", "zh")
    assert "軟體" in converted and "下週" in converted, converted
    # Other languages must pass through untouched, diacritics included.
    assert asr._post("Chúng ta cần xác nhận", "vi") == "Chúng ta cần xác nhận"
    assert asr._post("schedule and delay", "en") == "schedule and delay"


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


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    main()
