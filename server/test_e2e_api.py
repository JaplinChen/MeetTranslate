"""Config, glossary, LLM settings and the two things an edit teaches the room."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from . import config, llm_probe, main, translate


def test_health_and_config_roundtrip(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    # From the VERSION file, not a literal: the dashboard overwrites its own build-time version with
    # this answer, so a literal here decided what the sidebar said no matter what was released.
    assert health["version"] == config.VERSION
    assert health["version"] == (config.ROOT / "VERSION").read_text(encoding="utf-8").strip()

    cfg = client.get("/api/config").json()
    assert cfg["languages"] == ["zh", "vi", "en"]
    assert cfg["display"]["show_source"] == "top"

    r = client.put("/api/config", json={"languages": ["zh", "vi"], "display": {"font_size": 64, "theme": "light"}})
    assert r.status_code == 200, r.text
    assert r.json()["languages"] == ["zh", "vi"]
    assert r.json()["display"]["font_size"] == 64
    # An unspecified field must survive a partial patch.
    assert r.json()["display"]["lines"] == 6


def test_config_rejects_bad_input(client: TestClient) -> None:
    assert client.put("/api/config", json={"languages": ["zh"]}).status_code == 400
    assert client.put("/api/config", json={"languages": ["zh", "zh"]}).status_code == 400
    assert client.put("/api/config", json={"display": {"lines": 99}}).status_code == 400
    assert client.put("/api/config", json={"display": {"show_source": "sideways"}}).status_code == 400


def test_glossary_crud(client: TestClient) -> None:
    r = client.post("/api/glossary", json={"source": "產能", "targets": {"vi": "công suất", "en": "capacity"}})
    assert r.status_code == 200, r.text
    assert r.json()[0]["targets"]["vi"] == "công suất"

    # `keep` is the code-switching case: the term must never be translated.
    client.post("/api/glossary", json={"source": "schedule", "mode": "keep"})
    modes = {t["source"]: t["mode"] for t in client.get("/api/glossary").json()}
    assert modes == {"產能": "translate", "schedule": "keep"}

    assert client.post("/api/glossary", json={"source": "x", "mode": "nonsense"}).status_code == 400
    assert client.post("/api/glossary", json={"source": "   "}).status_code == 400

    left = client.request("DELETE", "/api/glossary", params={"source": "schedule"}).json()
    assert [t["source"] for t in left] == ["產能"]


def test_glossary_reports_what_a_term_would_overwrite(client: TestClient) -> None:
    """Adding a term is not obviously destructive, which is the problem.

    料號 and 料耗 are both liaohao and 料耗 is a term of the trade; adding 料號 rewrote it
    forty-two times across seven interviews and nothing said so. The answer comes from the
    meetings already recorded — what these people say, not what Mandarin permits.
    """
    session = main.store.start_session("now", "x.wav")
    for text in ("這個料耗的部分", "料耗變動原則", "刀具的料耗很多"):
        main.store.add_line(session, 1.0, "S1", "zh", text, {})

    hits = client.get("/api/glossary/collisions", params={"source": "料號"}).json()
    assert hits["collisions"] == [{"text": "料耗", "count": 3}]

    # A word already in the glossary is not collateral: registering it is how two real
    # homophones are made to coexist.
    client.post("/api/glossary", json={"source": "料耗", "mode": "protect"})
    assert client.get("/api/glossary/collisions", params={"source": "料號"}).json()["collisions"] == []

    assert client.get("/api/glossary/collisions", params={"source": "交貨"}).json()["collisions"] == []
    assert client.get("/api/glossary/collisions", params={"source": " "}).status_code == 400


def test_llm_config_never_returns_the_key(client: TestClient) -> None:
    client.put("/api/translate/config", json={
        "llmProvider": "anthropic", "llmModel": "claude-opus-5", "llmApiKey": "sk-secret-value-1234",
        "llmEndpoint": "https://api.anthropic.com", "llmTemperature": 0,
        "llmFallbackModels": [], "llmProviderConfigs": {},
    })
    body = client.get("/api/translate/config").json()
    assert body["llmApiKey"] == ""
    assert body["apiKeySet"] is True
    assert "sk-secret-value-1234" not in json.dumps(body)

    # A blank key on a later save means "keep the stored one", not "clear it".
    client.put("/api/translate/config", json={"llmModel": "claude-sonnet-5", "llmApiKey": ""})
    body = client.get("/api/translate/config").json()
    assert body["apiKeySet"] is True
    assert body["llmModel"] == "claude-sonnet-5"


def test_keyproxy_masks_and_rotates(client: TestClient) -> None:
    client.post("/api/keyproxy/keys", json={"provider": "anthropic", "apiKey": "sk-aaaabbbbccccdddd", "account": "a"})
    client.post("/api/keyproxy/keys", json={"provider": "anthropic", "apiKey": "sk-eeeeffffgggghhhh", "account": "b"})
    listed = client.get("/api/keyproxy/keys").json()
    assert len(listed) == 2
    assert all("sk-aaaa" not in json.dumps(k) or k["masked"].startswith("sk-a") for k in listed)
    assert listed[0]["masked"] == "sk-a…dddd"

    got = {main.keys.next_key("anthropic") for _ in range(2)}
    assert got == {"sk-aaaabbbbccccdddd", "sk-eeeeffffgggghhhh"}, "rotation did not visit both keys"

    assert client.post("/api/keyproxy/keys", json={"provider": "anthropic", "apiKey": ""}).status_code == 400
    assert client.delete("/api/keyproxy/keys/anthropic/9").status_code == 404
    assert client.delete("/api/keyproxy/keys/anthropic/0").json() == client.get("/api/keyproxy/keys").json()


def test_naming_a_speaker_teaches_the_room_their_voice(client: TestClient) -> None:
    """The one piece of labelled data this system ever gets, kept instead of discarded."""
    session = main.store.start_session("now", "x.wav")
    main.store.save_voiceprint(session, "S1", np.array([1.0], dtype="float32").tobytes())

    client.put(f"/api/sessions/{session}/speakers", json={"S1": "Vincent"})
    assert [s["name"] for s in client.get("/api/speakers/known").json()] == ["Vincent"]

    # A speaker with no stored voiceprint is still nameable, it just teaches nothing.
    client.put(f"/api/sessions/{session}/speakers", json={"S9": "Nobody"})
    assert [s["name"] for s in client.get("/api/speakers/known").json()] == ["Vincent"]

    assert client.delete("/api/speakers/known/Vincent").json() == []


def test_speaker_session_count_is_distinct_meetings_not_saves(client: TestClient) -> None:
    """The "N meetings" figure counts meetings the voice was named in, not times it was saved.

    Naming a speaker then fixing a typo in the name both save, and remember_speaker used to +1 on
    each — so a within-meeting rename read as an extra meeting. Counted from speaker_name, the
    figure is the number of distinct sessions regardless of how many saves produced it.
    """
    a = main.store.start_session("2026-01-01T09:00:00", "a.wav")
    main.store.save_voiceprint(a, "S1", np.array([1.0], dtype="float32").tobytes())

    # Names unique to this check, so a shared-store predecessor cannot seed them.
    OLD, NEW = "審查臨時甲", "審查臨時乙"
    # Name, then correct the name — same meeting, two saves.
    client.put(f"/api/sessions/{a}/speakers", json={"S1": OLD})
    client.put(f"/api/sessions/{a}/speakers", json={"S1": NEW})
    known = {s["name"]: s["sessions"] for s in client.get("/api/speakers/known").json()}
    assert OLD not in known, known  # the old name is gone — no orphan on the Learned page
    assert known.get(NEW) == 1, known  # one meeting, not two saves

    # The same person named in a second meeting is two distinct meetings.
    b = main.store.start_session("2026-01-02T09:00:00", "b.wav")
    main.store.save_voiceprint(b, "S1", np.array([1.0], dtype="float32").tobytes())
    client.put(f"/api/sessions/{b}/speakers", json={"S1": NEW})
    known = {s["name"]: s["sessions"] for s in client.get("/api/speakers/known").json()}
    assert known.get(NEW) == 2, known

    # Shared store: leave the table as it was found.
    from urllib.parse import quote
    client.delete(f"/api/speakers/known/{quote(NEW)}")


def test_forgetting_a_speaker_leaves_no_count_behind(client: TestClient) -> None:
    """forget_speaker drops the voice from known_speaker but keeps its historical transcript names.

    speaker_sessions is joined to known_speaker so it counts only voices the room still knows —
    without that join a forgotten voice kept a phantom count, invisible only because get_known_speakers
    happens to ask about it through known_speakers(). The two now agree at the source.
    """
    from urllib.parse import quote
    name = "審查忘記測試"
    sid = main.store.start_session("2026-03-01T09:00:00", "f.wav")
    main.store.save_voiceprint(sid, "S1", np.array([1.0], dtype="float32").tobytes())
    client.put(f"/api/sessions/{sid}/speakers", json={"S1": name})
    assert {s["name"]: s["sessions"] for s in client.get("/api/speakers/known").json()}.get(name) == 1

    client.delete(f"/api/speakers/known/{quote(name)}")
    # Gone from the list, and gone from the count that backs it — no phantom left in either source.
    assert name not in {s["name"] for s in client.get("/api/speakers/known").json()}
    assert name not in main.store.speaker_sessions()
    # The past meeting keeps the name it was given — forget is not a transcript edit.
    assert main.store.speaker_names(sid).get("S1") == name


def test_editing_a_line_teaches_the_correction(client: TestClient) -> None:
    """An edit on the transcript page is the only ground truth this system gets: someone who was
    in the room saying what was actually said. Kept, the same mistake is fixed everywhere next
    time — live as well as after the fact."""
    session = main.store.start_session("now", "x.wav")
    line = main.store.add_line(session, 1.0, "S1", "zh", "那個申管會上系統", {})

    r = client.put(f"/api/sessions/{session}/lines/{line}", json={"source": "那個生管會上系統"})
    assert r.status_code == 200, r.text
    assert r.json()["lines"][0]["source"] == "那個生管會上系統"
    assert {c["wrong"]: c["right"] for c in client.get("/api/corrections").json()} == {"申管": "生管"}

    # What was learned is applied to text the recogniser has not seen yet.
    from . import correct as correct_mod
    fixed = correct_mod.Corrector([], main.store.corrections()).fix("剛剛申管講的")
    assert fixed == "剛剛生管講的"

    assert client.put(f"/api/sessions/{session}/lines/{line}", json={"source": " "}).status_code == 400
    assert client.put(f"/api/sessions/{session}/lines/9999", json={"source": "x"}).status_code == 404
    assert client.delete("/api/corrections/申管").json() == []


def test_every_script_flag_is_wired_to_something(tmp: Path) -> None:
    """A flag that nothing reads is a feature that silently stopped existing.

    scripts/learn_terms.py kept its --max-sound option for a while after a scripted edit removed
    the ranking that used it, so the tool went on accepting the flag and ignoring it. Nothing in
    the output said so.
    """
    import re

    scripts = sorted(Path("scripts").glob("*.py"))
    assert scripts, "no scripts found; is the working directory wrong?"
    for path in scripts:
        source = path.read_text(encoding="utf-8")
        for flag in re.findall(r'add_argument\("--([a-z-]+)"', source):
            assert f"args.{flag.replace('-', '_')}" in source, f"{path.name}: --{flag} is unused"


def test_the_llm_probe_endpoints_answer_in_the_shape_the_page_reads(client: TestClient) -> None:
    """Both buttons on the provider form. The provider itself is stubbed; the wiring is the point.

    These two routes did not exist for a while, and because an unknown /api/* path returns a JSON
    404 rather than the HTML shell, the page reported it as a provider that refused the key —
    a backend gap wearing the costume of a wrong API key.
    """
    seen: list[tuple] = []
    real_check, real_list = llm_probe.check, llm_probe.list_models
    try:
        llm_probe.check = lambda *a: (seen.append(a), (True, "ok"))[1]
        llm_probe.list_models = lambda *a: (seen.append(a), ["m-2", "m-1"])[1]

        body = {"provider": "openai", "endpoint": "https://api.openai.com/v1/chat/completions",
                "model": "gpt-4o", "apiKey": "sk-typed"}
        r = client.post("/api/translate/llm/test", json=body)
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True, "message": "ok"}
        assert seen[-1] == ("openai", "https://api.openai.com/v1/chat/completions", "gpt-4o", "sk-typed")

        assert client.post("/api/translate/llm/models", json=body).json() == {"models": ["m-2", "m-1"]}

        # No endpoint: the provider's default is used rather than 400ing on a field the page
        # deliberately hides for providers whose endpoint is fixed.
        client.post("/api/translate/llm/test", json={"provider": "groq", "model": "x", "apiKey": "k"})
        assert seen[-1][1] == "https://api.groq.com/openai/v1"

        # A provider the page could not know the endpoint of, and no endpoint typed.
        assert client.post("/api/translate/llm/test", json={"provider": "azure"}).status_code == 400
        assert client.post("/api/translate/llm/test", json={"model": "x"}).status_code == 400

        # A provider that refuses is a 502 on the models route: the page shows the reason, and a
        # 200 with an empty list would read as "this provider has no models".
        def refuse(*_a):
            raise llm_probe.ProbeError("401 Unauthorized")

        llm_probe.list_models = refuse
        assert client.post("/api/translate/llm/models", json=body).status_code == 502
    finally:
        llm_probe.check, llm_probe.list_models = real_check, real_list


def test_the_llm_probe_falls_back_to_the_key_already_stored(client: TestClient) -> None:
    """The page cannot send a key it was never given: `llmApiKey` comes back empty by design.

    Without this, Verify worked once — right after typing the key — and failed on every later
    visit, which reads as a key that expired rather than a page that never had one.
    """
    client.put("/api/translate/config", json={"llmProvider": "anthropic", "llmApiKey": "sk-stored"})
    seen: list[tuple] = []
    real_check = llm_probe.check
    try:
        llm_probe.check = lambda *a: (seen.append(a), (True, "ok"))[1]
        client.post("/api/translate/llm/test", json={"provider": "anthropic", "model": "claude-opus-5"})
        assert seen[-1][3] == "sk-stored", seen
    finally:
        llm_probe.check = real_check


def test_an_endpoint_that_is_not_http_is_refused_by_the_route(client: TestClient) -> None:
    """This process fetches whatever the endpoint box holds, so the scheme is checked at the edge.

    400 rather than 502: nothing upstream failed — there is no upstream for file://.
    """
    for endpoint in ("file:///C:/Windows/win.ini", "ftp://example.com/x"):
        body = {"provider": "ollama", "endpoint": endpoint, "model": "x"}
        assert client.post("/api/translate/llm/models", json=body).status_code == 400, endpoint
        assert client.post("/api/translate/llm/test", json=body).status_code == 400, endpoint


def test_an_unknown_api_path_says_it_is_the_build_not_the_data(client: TestClient) -> None:
    """A stale server answers 404 for every endpoint it has not been restarted into.

    That is the same status a live endpoint uses to say "no such line", so the detail has to
    distinguish them — otherwise the dashboard reports missing data that is sitting right there.
    """
    gone = client.get("/api/sessions/1/lines/1/clip-that-does-not-exist")
    assert gone.status_code == 404
    assert gone.json()["detail"] == main.NO_SUCH_ENDPOINT

    # A real endpoint's 404 must not look like it.
    real = client.get("/api/sessions/999999/lines/1/clip")
    assert real.status_code == 404 and real.json()["detail"] != main.NO_SUCH_ENDPOINT
