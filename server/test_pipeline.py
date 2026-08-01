"""Self-checks for the ASR and diarization logic. Run: python -m server.test_pipeline

Model-free: the parts worth testing are the decision rules — when a collapsed decode is detected,
when a speaker's language is allowed to change, how offline clustering merges — and those are
plain functions over data.
"""

from __future__ import annotations

import numpy as np

from . import asr, config, correct, diarize, refine, store


def test_refine_keeps_the_original_when_the_model_rewrites() -> None:
    """The LLM pass may substitute, never restructure.

    Asked to fix a transcript a model will happily improve it, and an improved sentence is one
    nobody said. Anything that changes too much of a line, or that changes the number of lines,
    is discarded in favour of the recognised text.
    """
    lines = [refine.Line("S1", "zh", "我們的料耗其實變動很大"),
             refine.Line("S1", "zh", "生技這邊先開始")]
    fixed = ["我們的料號其實變動很大", "生管這邊先開始"]
    rewrite = "我們的料號變動幅度相當大，這點需要注意"
    terms = [store.Term(id=0, source="生管", lang="", mode="hint", category="", targets={})]

    # Only changed lines come back, and an unmentioned line keeps its original text.
    assert refine.parse_response("1: " + fixed[0], lines, terms) == [fixed[0], lines[1].text]

    # A fluent rewrite of the same meaning must be refused.
    assert refine.parse_response("1: " + rewrite, lines, terms)[0] == lines[0].text

    # An index outside the chunk is the model losing count; it must not land on another line.
    assert refine.parse_response("7: " + fixed[0], lines, terms) == [l.text for l in lines]
    assert refine.parse_response("nothing numbered here", lines, terms) == [l.text for l in lines]

    # Rewriting most of a chunk is restructuring, not correcting; keep all of it.
    many = [refine.Line("S1", "zh", f"第{i}句話沒有問題") for i in range(6)]
    reply_all = chr(10).join(f"{i}: 第{i}句話有問題" for i in range(1, 6))
    assert refine.parse_response(reply_all, many, terms) == [l.text for l in many]


def test_refine_rejects_corrections_that_do_not_sound_alike() -> None:
    """A recognition error is something the recogniser heard. A correction that sounds nothing
    like the text it replaces was invented from context, not recovered from audio.

    All five cases came out of a local model correcting a real interview transcript.
    """
    term = lambda source: store.Term(id=0, source=source, lang="", mode="hint",
                                     category="", targets={})
    terms = [term("工程變更"), term("生管")]

    # Heard: 稍 as 早, 料號 as 料耗, 生管 as 生技.
    assert refine.accept("有聽到聲音嗎早等我一下", "有聽到聲音嗎稍等我一下", terms)
    assert refine.accept("我們的料耗其實變動很大", "我們的料號其實變動很大", terms)
    assert refine.accept("生技這邊先開始", "生管這邊先開始", terms)

    # Guessed: nothing that sounds like 選項 was spoken.
    assert not refine.accept("用延伸的吧他還沒投出來", "用選項的吧他還沒跳出來", terms)

    # Two nonsense characters inside a long sentence are a rounding error to a ratio and still
    # nonsense, so the sound test has an absolute ceiling as well. Both of these were proposed by
    # a local model on a real transcript.
    long_before = "因為你所有的夢表那些什麼包含你的一些標準工時那些全部都要工單的管理"
    assert not refine.accept(long_before, long_before.replace("夢表", "模具"), terms)
    # The same span may still be corrected when the glossary names the destination.
    assert refine.accept(long_before, long_before.replace("夢表", "報表"),
                         terms + [term("報表")])

    # Re-spacing is not a correction.
    assert not refine.accept("呃right nowswitch", "呃 right now switch", terms)

    # A glossary term may travel further, because the recogniser never knew it existed.
    assert refine.accept("一夕變更的流程", "工程變更的流程", terms)
    assert not refine.accept("一夕變更的流程", "工程變更的流程", [])


def test_refine_converts_what_the_model_writes_in_simplified() -> None:
    """The recogniser's output is already Traditional; a Simplified character in a correction can
    only have come from the model. Converted character by character, not with the phrase table
    used on ASR output — that one rewrites 對象 to 物件, which is the speaker's word, not an error.
    """
    lines = [refine.Line("S1", "zh", "申報的保税料件"),
             refine.Line("S1", "zh", "這個對象要處理"),
             refine.Line("S1", "en", "the tax iten")]
    assert refine.parse_response("1: 申報的保税料號", lines)[0] == "申報的保稅料號"
    assert refine.parse_response("2: 這個對象要處裡", lines)[1] == "這個對象要處裡"
    assert refine.parse_response("3: the tax item", lines)[2] == "the tax item"


def test_refine_prompt_states_the_domain_and_the_terms() -> None:
    said, earlier, term = "一夕變更的流程", "前面說過的話", "工程變更"
    prompt = refine.build_prompt(
        [refine.Line("S1", "zh", said)],
        [refine.Line("S1", "zh", earlier)],
        [store.Term(id=0, source=term, lang="", mode="hint", category="", targets={})],
        "SAP ERP interview",
    )
    assert "SAP ERP interview" in prompt
    assert term in prompt and earlier in prompt
    assert f"1: {said}" in prompt


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
