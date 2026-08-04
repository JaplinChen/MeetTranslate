"""Config, glossary, LLM settings and the two things an edit teaches the room."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from . import config, main, translate


def test_health_and_config_roundtrip(client: TestClient) -> None:
    assert client.get("/api/health").json()["status"] == "ok"

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
