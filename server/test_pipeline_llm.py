"""Where each provider's probe request goes, and what must never come back out of it.

No network: the two things worth pinning down are the URL and headers built for each provider —
they differ in every field and are easy to get subtly wrong — and that an API key cannot escape
through an error message.
"""

from __future__ import annotations

from . import llm_probe


def test_the_endpoint_box_is_normalised_to_a_root() -> None:
    """The settings page stores a chat URL for some providers and a bare root for others.

    That is not sloppiness, it is what each provider's own documentation shows, so every call site
    would otherwise have to know which kind it was holding.
    """
    assert llm_probe.base_url("https://api.openai.com/v1/chat/completions") == "https://api.openai.com/v1"
    assert llm_probe.base_url("http://127.0.0.1:11434/api/chat") == "http://127.0.0.1:11434"
    assert llm_probe.base_url("https://api.anthropic.com/") == "https://api.anthropic.com"
    # Already a root: left alone rather than guessed at.
    assert llm_probe.base_url("https://generativelanguage.googleapis.com/v1beta") == \
        "https://generativelanguage.googleapis.com/v1beta"


def test_each_provider_is_asked_for_models_in_its_own_dialect() -> None:
    url, headers = llm_probe.models_call("anthropic", "https://api.anthropic.com", "sk-a")
    assert url == "https://api.anthropic.com/v1/models"
    assert headers["x-api-key"] == "sk-a" and "anthropic-version" in headers

    url, headers = llm_probe.models_call("groq", "https://api.groq.com/openai/v1/chat/completions", "gsk")
    assert url == "https://api.groq.com/openai/v1/models"
    assert headers["Authorization"] == "Bearer gsk"

    url, headers = llm_probe.models_call("ollama", "http://127.0.0.1:11434/api/chat", "")
    assert url == "http://127.0.0.1:11434/api/tags"
    # No key, and no empty Authorization header pretending there is one.
    assert headers == {}

    url, _ = llm_probe.models_call("gemini", "https://generativelanguage.googleapis.com/v1beta", "AIza x")
    assert url.startswith("https://generativelanguage.googleapis.com/v1beta/models?key=")
    assert " " not in url, "the key is in a query string and has to be percent-encoded"


def test_azure_says_why_rather_than_guessing_a_url() -> None:
    """Its endpoint names one deployment, and a deployment already is the model."""
    try:
        llm_probe.models_call("azure", "https://x.openai.azure.com/openai/deployments/d/chat/completions", "k")
    except llm_probe.ProbeError as exc:
        assert "deployment" in str(exc)
    else:
        raise AssertionError("azure should not have produced a model-list URL")


def test_model_ids_are_read_out_of_whichever_envelope_arrived() -> None:
    assert llm_probe.parse_models("openai", {"data": [{"id": "gpt-4o"}, {"id": "gpt-4"}]}) == ["gpt-4", "gpt-4o"]
    assert llm_probe.parse_models("ollama", {"models": [{"name": "qwen3:8b"}]}) == ["qwen3:8b"]
    # Gemini's names carry the collection they live in; the page wants the model.
    assert llm_probe.parse_models("gemini", {"models": [{"name": "models/gemini-2.0-flash"}]}) == ["gemini-2.0-flash"]
    # An envelope with nothing in it is empty, not an exception.
    assert llm_probe.parse_models("openai", {}) == []


def test_the_test_request_is_the_smallest_one_that_proves_anything() -> None:
    url, headers, body = llm_probe.chat_call("anthropic", "https://api.anthropic.com", "claude-opus-5", "sk-a")
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "sk-a"
    assert body["max_tokens"] == 1, "a test button must not be a bill"

    url, headers, body = llm_probe.chat_call("openai", "https://api.openai.com/v1/chat/completions", "gpt-4o", "sk-o")
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-o" and body["model"] == "gpt-4o"

    # Azure is deployment-scoped: the model is in the URL the user pasted, so it is posted as given.
    given = "https://x.openai.azure.com/openai/deployments/d/chat/completions?api-version=2024-02-15-preview"
    url, headers, body = llm_probe.chat_call("azure", given, "", "azk")
    assert url == given and headers["api-key"] == "azk" and "model" not in body


def test_a_key_cannot_escape_through_an_error() -> None:
    """Gemini carries it in the query string, so provider error text is not safe to pass through."""
    leaked = "provider said: GET https://x/v1beta/models?key=AIzaSECRET failed for sk-live-SECRET"
    safe = llm_probe._scrub(leaked, "sk-live-SECRET")
    assert "sk-live-SECRET" not in safe
    assert "AIzaSECRET" not in safe
    assert "key=…" in safe


def test_a_non_http_endpoint_is_refused_before_it_is_fetched() -> None:
    """The endpoint is user-supplied and this process fetches it; file:// is not a provider."""
    for url in ("file:///etc/passwd", "ftp://example.com/x", "gopher://x"):
        try:
            llm_probe._require_http(url)
        except llm_probe.ProbeError:
            continue
        raise AssertionError(f"{url} should have been refused")
    assert llm_probe._require_http("http://127.0.0.1:11434") == "http://127.0.0.1:11434"


def test_a_missing_model_is_answered_without_a_round_trip() -> None:
    ok, message = llm_probe.check("openai", "https://api.openai.com/v1/chat/completions", "", "sk")
    assert ok is False and "model" in message


def test_a_slow_provider_is_an_answer_not_a_crash() -> None:
    """A read that times out comes straight out of the socket, not as a URLError.

    Uncaught it reached the models route as a 500 — a provider still loading a model reported as
    a broken server.
    """
    import urllib.request

    real = urllib.request.urlopen
    try:
        def slow(*_a, **_k):
            raise TimeoutError("timed out")

        urllib.request.urlopen = slow
        try:
            llm_probe.list_models("ollama", "http://127.0.0.1:11434/api/chat", "")
        except llm_probe.ProbeError as exc:
            assert "loading" in str(exc), exc
        else:
            raise AssertionError("a timeout should have become a ProbeError")
    finally:
        urllib.request.urlopen = real
