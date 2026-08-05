"""LLM settings and the API key rotation pool."""

from __future__ import annotations

from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException

from . import llm, llm_probe, main

router = APIRouter()


def _probe(body: dict) -> tuple[str, str, str, str]:
    """The four fields both probe endpoints take, with the saved key as the fallback.

    The page sends the key only when someone has just typed one; on a revisit the field is empty
    because the key never comes back down. Falling through to the stored key is what makes Verify
    work on a provider that was configured last week.
    """
    provider = str(body.get("provider", "")).strip()
    if not provider:
        raise HTTPException(400, "provider required")
    endpoint = str(body.get("endpoint", "")).strip() or llm.DEFAULT_ENDPOINTS.get(provider, "")
    if not endpoint:
        raise HTTPException(400, f"no endpoint for provider {provider}")
    # Checked here rather than left to the probe: this process is about to fetch a URL somebody
    # typed, and "you typed file://" is a bad request, not an upstream that failed.
    if urlparse(endpoint).scheme not in ("http", "https"):
        raise HTTPException(400, "endpoint must be an http or https URL")
    key = str(body.get("apiKey", ""))
    if not key:
        cfg: llm.LlmConfig = main.state["llm"]
        saved = cfg.providers.get(provider, {})
        key = str(saved.get("api_key", "")) or (cfg.api_key if cfg.provider == provider else "")
    return provider, endpoint, str(body.get("model", "")).strip(), key


@router.get("/api/translate/config")
def get_llm_config() -> dict:
    return main.state["llm"].to_json()


@router.put("/api/translate/config")
def put_llm_config(body: dict) -> dict:
    cfg: llm.LlmConfig = main.state["llm"]
    cfg.apply(body)
    cfg.save()
    return cfg.to_json()


@router.post("/api/translate/llm/test")
def test_llm(body: dict) -> dict:
    """Does this provider, endpoint, key and model answer?

    Always 200: a provider refusing the key is the answer the page asked for, not a failure of
    this endpoint, and the form renders `message` beside the button either way.
    """
    provider, endpoint, model, key = _probe(body)
    ok, message = llm_probe.check(provider, endpoint, model, key)
    return {"ok": ok, "message": message}


@router.post("/api/translate/llm/models")
def list_llm_models(body: dict) -> dict:
    """What this provider will let the user pick from, asked live rather than hard-coded.

    A model list that ships in the frontend is wrong the week after it ships, and wrong in the
    direction that hides the model someone is paying for.
    """
    provider, endpoint, _model, key = _probe(body)
    try:
        return {"models": llm_probe.list_models(provider, endpoint, key)}
    except llm_probe.ProbeError as exc:
        raise HTTPException(502, str(exc)) from exc


@router.get("/api/keyproxy/keys")
def get_keys() -> list[dict]:
    return main.keys.list()


@router.post("/api/keyproxy/keys")
def post_key(body: dict) -> list[dict]:
    try:
        return main.keys.add(str(body.get("provider", "")), str(body.get("apiKey", "")),
                             str(body.get("account", "")))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/api/keyproxy/keys/{provider}/{index}")
def delete_key(provider: str, index: int) -> list[dict]:
    try:
        return main.keys.remove(provider, index)
    except IndexError as exc:
        raise HTTPException(404, str(exc)) from exc
