"""The cross-meeting question endpoint: it answers with verified citations, and it refuses abuse.

The model is stubbed through `main.postmeeting.chat_for` — a real ANTHROPIC_API_KEY would otherwise
send these questions to Anthropic from inside the suite, which is how the summary checks first
failed. The retrieval and citation logic is covered as pure functions in test_ask; here we prove
the endpoint wires the store, the gate and the model together.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from . import jobs, main, routes_ask
from .e2e_support import seed_session


def _reset_ask_gate() -> None:
    """The endpoint enforces a global cooldown between questions; the suite asks several back to
    back, so each check clears it rather than sleeping three seconds between them."""
    routes_ask._last_answered = 0.0


def _seed_two_meetings() -> tuple[int, int, int]:
    a = seed_session("ask-a.wav")
    b = seed_session("ask-b.wav")
    main.store.add_line(a, 12.5, "S1", "zh", "交貨日期我建議延後到下週五", {})
    main.store.add_line(a, 20.0, "S2", "zh", "採購單那邊還沒回覆", {})
    main.store.add_line(b, 3.0, "S3", "zh", "這場談的是產能規劃", {})
    main.store.set_speaker_name(a, "S1", "陳經理")
    line_id = main.store.lines(a)[-2]["id"]  # the 交貨 line (seed_session adds one line first)
    return a, b, line_id


def test_ask_answers_with_a_citation_from_the_store(client: TestClient) -> None:
    jobs.reset()
    _reset_ask_gate()
    a, _b, _ = _seed_two_meetings()
    # The 交貨 line is whichever one actually contains it, found rather than assumed.
    line = next(l for l in main.store.lines(a) if "交貨" in l["source"])

    main.postmeeting.replies = [
        json.dumps({"keywords": ["交期", "交貨"], "since": None, "until": None}),
        json.dumps({"sessions": [a]}),  # thin keyword hits → summary-index fallback runs
        json.dumps({"answer": "陳經理提議把交貨日期延後到下週五。",
                    "citations": [{"session_id": a, "line_id": line["id"]},
                                  {"session_id": a, "line_id": 999999}]}),
    ]
    try:
        r = client.post("/api/ask", json={"question": "上週誰說要改交期？"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "延後" in body["answer"], body
        assert len(body["citations"]) == 1, body  # the invented line_id 999999 was dropped
        assert body["dropped_citations"] == 1
        assert body["verified"] is True
        cite = body["citations"][0]
        # start comes from the store, not from whatever the model wrote; the speaker arrives
        # resolved to the name (S1 → 陳經理), because the page has no names map to resolve it with.
        assert cite["speaker"] == "陳經理" and cite["start"] == 12.5, cite
    finally:
        main.postmeeting.replies = None


def test_ask_reports_unverified_when_every_citation_is_invented(client: TestClient) -> None:
    jobs.reset()
    _reset_ask_gate()
    a, _b, _ = _seed_two_meetings()
    main.postmeeting.replies = [
        json.dumps({"keywords": ["交期"], "since": None, "until": None}),
        json.dumps({"sessions": [a]}),
        json.dumps({"answer": "有人提過。", "citations": [{"session_id": a, "line_id": 888888}]}),
    ]
    try:
        body = client.post("/api/ask", json={"question": "誰提過交期？"}).json()
        assert body["citations"] == [] and body["dropped_citations"] == 1
        assert body["verified"] is False, body
    finally:
        main.postmeeting.replies = None


def test_ask_without_a_model_is_503(client: TestClient) -> None:
    jobs.reset()
    _reset_ask_gate()
    _seed_two_meetings()
    main.postmeeting.replies = None  # chat_for returns None → no model configured
    body = client.post("/api/ask", json={"question": "隨便問"})
    assert body.status_code == 503, body.text
    assert "settings" in body.json()["detail"].lower()


def test_ask_rejects_an_empty_or_oversized_question(client: TestClient) -> None:
    jobs.reset()
    _reset_ask_gate()
    main.postmeeting.replies = ["{}"]
    try:
        assert client.post("/api/ask", json={"question": "  "}).status_code == 400
        assert client.post("/api/ask", json={"question": "字" * 501}).status_code == 400
    finally:
        main.postmeeting.replies = None


def test_a_failed_answer_still_starts_the_cooldown(client: TestClient) -> None:
    """A question the model cannot answer spent two calls; retrying it at once must be refused.

    The cooldown used to advance only after a success, so a persistently-failing question was an
    unlimited way to spend model calls — exactly what the gate exists to stop.
    """
    jobs.reset()
    _reset_ask_gate()
    _seed_two_meetings()
    # Keyword reply is fine; the answer reply is unparseable twice, so ask() raises ValueError.
    main.postmeeting.replies = [
        json.dumps({"keywords": ["交期"], "since": None, "until": None}),
        json.dumps({"sessions": []}),
        "not json", "still not json",
    ]
    try:
        first = client.post("/api/ask", json={"question": "上週誰說要改交期？"})
        assert first.status_code == 502, first.text

        # Immediately again: blocked by the cooldown the failure started, not let through.
        main.postmeeting.replies = [
            json.dumps({"keywords": ["交期"], "since": None, "until": None}),
            json.dumps({"sessions": []}), "not json", "still not json",
        ]
        second = client.post("/api/ask", json={"question": "再問一次"})
        assert second.status_code == 429, second.text
        assert "try again" in second.json()["detail"]
    finally:
        main.postmeeting.replies = None


def test_a_cooldown_bounce_does_not_extend_the_cooldown(client: TestClient) -> None:
    """A 429 costs nothing, so being bounced must not push the window further out."""
    import time as _time

    from . import routes_ask
    jobs.reset()
    _reset_ask_gate()
    a, _b, _ = _seed_two_meetings()
    # Land one successful answer to arm the cooldown, then read the timestamp it set.
    line = next(l for l in main.store.lines(a) if "交貨" in l["source"])
    main.postmeeting.replies = [
        json.dumps({"keywords": ["交期"], "since": None, "until": None}),
        json.dumps({"sessions": [a]}),
        json.dumps({"answer": "答案", "citations": [{"session_id": a, "line_id": line["id"]}]}),
    ]
    try:
        assert client.post("/api/ask", json={"question": "誰改交期？"}).status_code == 200
        armed_at = routes_ask._last_answered
        assert armed_at > 0

        # A bounced request during the cooldown must leave the timestamp untouched.
        bounced = client.post("/api/ask", json={"question": "馬上再問"})
        assert bounced.status_code == 429
        assert routes_ask._last_answered == armed_at, "a bounce moved the cooldown window"
    finally:
        main.postmeeting.replies = None
