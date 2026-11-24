"""Клиент модели. Один протокол — OpenAI-совместимый /chat/completions.

По нему работают локальная Ollama (http://127.0.0.1:11434/v1, ключ не нужен), LM Studio, vLLM
и облачные провайдеры с бесплатным тарифом (OpenRouter и подобные) — меняются только адрес, модель и ключ.
"""
from __future__ import annotations

from urllib.parse import urlparse

import httpx

OLLAMA_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen2.5:3b"


class ChatLLM:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 120.0,
                 client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._client = client or httpx.Client()

    def describe(self) -> dict[str, str]:
        host = urlparse(self.base_url)
        local = host.hostname in ("127.0.0.1", "localhost", "::1")
        provider = "ollama" if host.port == 11434 else ("local" if local else (host.hostname or "unknown"))
        return {"provider": provider, "model": self.model}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0, "seed": 0,
            "response_format": {"type": "json_object"},
        }
        r = self._client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers(),
                              timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise ValueError("модель вернула ответ в неожиданном формате")

    def models(self) -> list[str]:
        """Список доступных моделей; пустой, если сервер его не отдаёт."""
        r = self._client.get(f"{self.base_url}/models", headers=self._headers(), timeout=5.0)
        r.raise_for_status()
        data = r.json().get("data") or []
        return [m.get("id", "") for m in data if isinstance(m, dict)]


def from_config(cfg, client: httpx.Client | None = None) -> ChatLLM | None:
    """Клиент по настройкам. None — модель не настроена или недоступна: тогда работают одни правила.

    Без настроек пробуем локальную Ollama: это способ запуска без ключей и аккаунта.
    """
    explicit = bool(cfg.llm_base_url)
    base = cfg.llm_base_url or OLLAMA_URL
    llm = ChatLLM(base, cfg.llm_model or DEFAULT_MODEL, cfg.llm_api_key, client=client)
    try:
        available = llm.models()
    except (httpx.HTTPError, ValueError):
        if explicit and cfg.llm_api_key:
            return llm          # облачный провайдер может не отдавать список моделей — это не признак недоступности
        return None
    if available and not cfg.llm_model and llm.model not in available:
        # модель не задана, а модели по умолчанию нет — берём ту, что установлена
        family = [m for m in available if m.split(":")[0] == DEFAULT_MODEL.split(":")[0]]
        llm.model = (family or available)[0]
    return llm
