"""Self-checks for multi-provider chat wiring. Run: python -m server.test_provider_chat

Model-free and offline: urllib.request.urlopen is faked, so these assert the request SHAPE each
provider builds and that the dispatcher routes to it — the bug was that every provider but Anthropic
(and Ollama, post-meeting only) fell through to the Anthropic client, so a verified OpenAI/Gemini/
Groq key silently translated nothing.
"""

from __future__ import annotations

import io
import json
import urllib.request

from . import llm, postmeeting, refine, translate


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _capture(reply: dict):
    """Patch urlopen to record the request and return `reply` as JSON. Returns (calls, restore)."""
    calls: list[urllib.request.Request] = []
    real = urllib.request.urlopen

    def fake(req, timeout=None):
        calls.append(req)
        return _FakeResponse(json.dumps(reply).encode())

    urllib.request.urlopen = fake
    return calls, (lambda: setattr(urllib.request, "urlopen", real))


def test_openai_chat_posts_chat_completions_with_bearer():
    reply = {"choices": [{"message": {"content": "hola"}}]}
    calls, restore = _capture(reply)
    try:
        chat = refine.openai_chat("sk-key", "gpt-4o", "https://api.openai.com/v1", max_tokens=50)
        out = chat("hi")
    finally:
        restore()
    assert out == "hola"
    req = calls[0]
    assert req.full_url == "https://api.openai.com/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer sk-key"
    assert json.loads(req.data)["model"] == "gpt-4o"


def test_gemini_chat_posts_generatecontent_with_key_in_query():
    reply = {"candidates": [{"content": {"parts": [{"text": "bonjour"}]}}]}
    calls, restore = _capture(reply)
    try:
        chat = refine.gemini_chat("gk", "gemini-2.0-flash",
                                  "https://generativelanguage.googleapis.com", max_tokens=50)
        out = chat("hi")
    finally:
        restore()
    assert out == "bonjour"
    assert calls[0].full_url.endswith("/models/gemini-2.0-flash:generateContent?key=gk")


def test_chat_for_routes_openai_provider_to_openai_shape():
    """The regression guard: provider=openai must NOT reach the Anthropic client."""
    cfg = llm.LlmConfig(provider="openai", model="gpt-4o",
                        endpoint="https://api.openai.com/v1")
    reply = {"choices": [{"message": {"content": "ok"}}]}
    calls, restore = _capture(reply)
    try:
        chat = postmeeting.chat_for(cfg, "sk-key", max_tokens=50)
        assert chat is not None
        chat("hi")
    finally:
        restore()
    assert calls[0].full_url == "https://api.openai.com/v1/chat/completions"


def test_chat_for_returns_none_without_key_for_cloud_provider():
    cfg = llm.LlmConfig(provider="groq", model="llama", endpoint="")
    assert postmeeting.chat_for(cfg, "", max_tokens=50) is None


def test_translator_uses_injected_chat_callable():
    seen: list[str] = []

    def chat(prompt: str) -> str:
        seen.append(prompt)
        return json.dumps({"translations": {"en": "hello"}})

    tr = translate.Translator(chat)
    result = tr.translate(translate.Line("你好", "zh"), ["en"])
    assert result.translations == {"en": "hello"}
    assert len(seen) == 1  # the prompt reached the injected callable, not an SDK


def test_previous_refinement_targets_the_previous_lines_language():
    """In a mixed-language meeting the previous line's language differs from the current one, so its
    hindsight correction must be solicited and kept in ITS languages, not the current line's targets.
    cfg=[zh,en], previous line zh (needs an en correction), current line en (targets=[zh])."""
    seen: list[str] = []

    def chat(prompt: str) -> str:
        seen.append(prompt)
        return json.dumps({"translations": {"zh": "你好"},
                           "previous": {"source": "前一句修正", "translations": {"en": "corrected"}}})

    tr = translate.Translator(chat)
    prev = translate.Line("前一句中文", "zh")
    result = tr.translate(translate.Line("hello", "en"), ["zh"],
                          previous=prev, prev_targets=["en"])

    assert result.previous_translations == {"en": "corrected"}, result.previous_translations
    # The previous block must ask for the language the previous line actually needs (en), not zh.
    prev_block = seen[0].split('"previous"')[1]
    assert '"en"' in prev_block and '"zh"' not in prev_block, prev_block


def test_translator_reports_rejection_then_reraises():
    rejected: list[Exception] = []

    def chat(prompt: str) -> str:
        raise RuntimeError("boom")

    tr = translate.Translator(chat, on_reject=rejected.append)
    try:
        tr.translate(translate.Line("你好", "zh"), ["en"])
        raise AssertionError("expected the error to re-raise")
    except RuntimeError:
        pass
    assert len(rejected) == 1


def main() -> None:
    checks = sorted((n, f) for n, f in globals().items() if n.startswith("test_"))
    for name, fn in checks:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(checks)} passed")


if __name__ == "__main__":
    main()
