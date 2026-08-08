"""LLM provider settings and API-key rotation.

Keys live in their own JSON file rather than config.json so that config.json stays safe to read,
diff and share while the secrets sit in one file that is easy to lock down. Only masked forms
ever leave this module.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import config

LLM_PATH = config.ROOT / "llm.json"
KEYS_PATH = config.ROOT / "llm_keys.json"

# How long a rate-limited key is benched before it is tried again. A 429 is transient — the window
# resets — so the key comes back on its own rather than needing a human to clear its status.
LIMITED_COOLDOWN = 60.0


def rejection(exc: Exception) -> str | None:
    """Classify a provider error: 'limited' for a rate limit, 'failed' for a rejected key, else None.

    Read off the HTTP status the SDK carries rather than the exception type, so one rule covers every
    provider's client with none of them imported here. 429 is a rate limit — transient, worth
    retrying after a cooldown. 401/403 is the key itself — wrong, disabled or out of quota — and no
    waiting fixes it. Anything else (a 500, a dropped connection) is not the key's fault, so the key
    keeps its place in the rotation.
    """
    code = getattr(exc, "status_code", None)
    if code == 429:
        return "limited"
    if code in (401, 403):
        return "failed"
    return None

DEFAULT_ENDPOINTS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434",
    "mistral": "https://api.mistral.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "nvidia_nim": "https://integrate.api.nvidia.com/v1",
}


def mask(key: str) -> str:
    """Enough to recognise a key, never enough to use it."""
    return f"{key[:4]}…{key[-4:]}" if len(key) > 12 else "…" * 3


@dataclass
class ProviderConfig:
    endpoint: str = ""
    model: str = ""
    api_key: str = ""
    temperature: float = 0.0


@dataclass
class LlmConfig:
    provider: str = "anthropic"
    endpoint: str = DEFAULT_ENDPOINTS["anthropic"]
    model: str = "claude-opus-5"
    api_key: str = ""
    temperature: float = 0.0
    # Flat "provider:model" chain tried in order when the primary fails.
    fallback_models: list[str] = field(default_factory=list)
    providers: dict[str, dict] = field(default_factory=dict)
    # Per-function model overrides; empty falls back to `model`. The morning meeting is Taiwanese
    # Mandarin most days (Vietnamese only when the manager is out), so translation wants a Traditional
    # Chinese + Vietnamese model while the summary wants a long-context reasoner — different jobs, so
    # a room can point each at the model that fits instead of compromising on one for both.
    translate_model: str = ""
    summary_model: str = ""

    def save(self) -> None:
        LLM_PATH.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    def to_json(self) -> dict:
        """The shape the dashboard's LLM settings page expects. Keys are masked, never returned."""
        providers = {
            name: {
                "endpoint": p.get("endpoint", ""),
                "model": p.get("model", ""),
                "temperature": p.get("temperature", 0.0),
                "apiKeySet": bool(p.get("api_key")),
            }
            for name, p in self.providers.items()
        }
        return {
            "llmProvider": self.provider,
            "llmEndpoint": self.endpoint,
            "llmModel": self.model,
            "llmTranslateModel": self.translate_model,
            "llmSummaryModel": self.summary_model,
            "llmApiKey": "",
            "llmTemperature": self.temperature,
            "llmFallbackModels": self.fallback_models,
            "llmProviderConfigs": providers,
            "apiKeySet": bool(self.api_key),
        }

    def apply(self, body: dict) -> None:
        self.provider = str(body.get("llmProvider", self.provider))
        self.endpoint = str(body.get("llmEndpoint", self.endpoint)) or DEFAULT_ENDPOINTS.get(self.provider, "")
        self.model = str(body.get("llmModel", self.model))
        self.translate_model = str(body.get("llmTranslateModel", self.translate_model))
        self.summary_model = str(body.get("llmSummaryModel", self.summary_model))
        self.temperature = float(body.get("llmTemperature", self.temperature))
        self.fallback_models = [str(s) for s in body.get("llmFallbackModels", self.fallback_models)]

        # An empty key means "keep the stored one" — the page never receives the real value back,
        # so it cannot echo it, and blanking it out on every save would be a nasty surprise.
        if incoming := str(body.get("llmApiKey", "")):
            self.api_key = incoming

        for name, saved in dict(body.get("llmProviderConfigs", {})).items():
            existing = self.providers.get(name, {})
            merged = {
                "endpoint": saved.get("endpoint", existing.get("endpoint", "")),
                "model": saved.get("model", existing.get("model", "")),
                "temperature": saved.get("temperature", existing.get("temperature", 0.0)),
                "api_key": saved.get("apiKey") or existing.get("api_key", ""),
            }
            self.providers[name] = merged

        if self.provider in self.providers and (key := self.providers[self.provider].get("api_key")):
            self.api_key = key


def load_llm() -> LlmConfig:
    if not LLM_PATH.exists():
        return LlmConfig()
    data = json.loads(LLM_PATH.read_text(encoding="utf-8"))
    return LlmConfig(**{k: v for k, v in data.items() if k in LlmConfig.__dataclass_fields__})


@dataclass
class ApiKey:
    provider: str
    key: str
    account: str = ""
    requests: int = 0
    voice_requests: int = 0
    failures: int = 0
    status: str = "ready"  # ready | limited | failed
    # Epoch seconds a `limited` key is benched until; 0 for ready/failed. Read by next_key to heal a
    # rate-limited key back to ready once its window has passed.
    limited_until: float = 0.0

    def to_json(self, index: int) -> dict:
        return {
            "provider": self.provider,
            "index": index,
            "account": self.account,
            "masked": mask(self.key),
            "status": self.status,
            "requestCount": self.requests,
            "failureCount": self.failures,
            "voiceRequestCount": self.voice_requests,
            # Providers report what is left, not what was spent, and only after a call. Until one
            # has been made there is nothing honest to show, so this stays null.
            "quota": None,
        }


class KeyStore:
    """Round-robin over keys, skipping ones the provider has rate-limited."""

    def __init__(self, path: Path | None = None):
        self._path = path or KEYS_PATH
        self._keys: list[ApiKey] = []
        self._cursor: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._keys = [ApiKey(**{k: v for k, v in d.items() if k in ApiKey.__dataclass_fields__}) for d in data]

    def _save(self) -> None:
        self._path.write_text(
            json.dumps([asdict(k) for k in self._keys], indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def list(self) -> list[dict]:
        return [k.to_json(i) for i, k in enumerate(self._keys)]

    def add(self, provider: str, key: str, account: str = "") -> list[dict]:
        key = key.strip()
        if not key:
            raise ValueError("apiKey must not be empty")
        self._keys.append(ApiKey(provider=provider, key=key, account=account.strip()))
        self._save()
        return self.list()

    def remove(self, provider: str, index: int) -> list[dict]:
        if not 0 <= index < len(self._keys) or self._keys[index].provider != provider:
            raise IndexError("no such key")
        del self._keys[index]
        self._save()
        return self.list()

    def next_key(self, provider: str) -> str | None:
        """Next ready key for a provider, or None when every one of them is benched.

        A rate-limited key heals back to ready once its cooldown has passed, so a passing 429 costs
        it a minute, not the rest of the session. A key rejected outright (`failed`) is never chosen
        again until someone removes and re-adds it — waiting does not fix a wrong key.
        """
        now = time.time()
        healed = False
        for k in self._keys:
            if k.status == "limited" and now >= k.limited_until:
                k.status, k.limited_until, healed = "ready", 0.0, True
        candidates = [k for k in self._keys if k.provider == provider and k.status == "ready"]
        if not candidates:
            if healed:
                self._save()
            return None
        start = self._cursor.get(provider, 0) % len(candidates)
        chosen = candidates[start]
        self._cursor[provider] = start + 1
        chosen.requests += 1
        self._save()
        return chosen.key

    def mark_failure(self, key: str, limited: bool = False) -> None:
        """Record that the provider rejected this key. A rate limit benches it for the cooldown; a
        rejection benches it until it is removed."""
        for k in self._keys:
            if k.key == key:
                k.failures += 1
                if limited:
                    k.status, k.limited_until = "limited", time.time() + LIMITED_COOLDOWN
                else:
                    k.status, k.limited_until = "failed", 0.0
                self._save()
                return
