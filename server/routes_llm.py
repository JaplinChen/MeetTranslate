"""LLM settings and the API key rotation pool."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from . import llm, main

router = APIRouter()


@router.get("/api/translate/config")
def get_llm_config() -> dict:
    return main.state["llm"].to_json()


@router.put("/api/translate/config")
def put_llm_config(body: dict) -> dict:
    cfg: llm.LlmConfig = main.state["llm"]
    cfg.apply(body)
    cfg.save()
    return cfg.to_json()


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
