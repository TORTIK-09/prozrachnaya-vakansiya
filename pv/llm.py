"""Клиент модели. Один протокол — OpenAI-совместимый /chat/completions.

По нему работают локальная Ollama (http://127.0.0.1:11434/v1, ключ не нужен), LM Studio, vLLM
и облачные провайдеры с бесплатным или собственным тарифом (OpenRouter, OpenAI, Gemini, DeepSeek, Groq и др.) —
меняются только адрес, модель и ключ.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

OLLAMA_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen2.5:3b"

PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "ollama": {
        "label": "Ollama (Локальная LLM)",
        "mode_type": "local",
        "base_url": "http://127.0.0.1:11434/v1",
        "default_model": "qwen2.5:3b",
    },
    "lmstudio": {
        "label": "LM Studio / Local Server (Локальный API)",
        "mode_type": "local",
        "base_url": "http://127.0.0.1:1234/v1",
        "default_model": "qwen2.5-3b-instruct",
    },
    "openrouter": {
        "label": "OpenRouter API (любые облачные и :free модели)",
        "mode_type": "custom_api",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "google/gemma-3-27b-it:free",
    },
    "openai": {
        "label": "OpenAI API",
        "mode_type": "custom_api",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "gemini": {
        "label": "Google Gemini API (OpenAI-совместимый)",
        "mode_type": "custom_api",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.5-flash",
    },
    "deepseek": {
        "label": "DeepSeek API",
        "mode_type": "custom_api",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    "groq": {
        "label": "Groq Cloud API",
        "mode_type": "custom_api",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
    "custom": {
        "label": "Свой OpenAI-совместимый API",
        "mode_type": "custom_api",
        "base_url": "",
        "default_model": "",
    },
}


class ChatLLM:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 120.0,
                 client: httpx.Client | None = None, mode_type: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        host = urlparse(self.base_url)
        self.is_local = host.hostname in ("127.0.0.1", "localhost", "::1")
        self.mode_type = mode_type or ("local" if self.is_local else "custom_api")
        limits = httpx.Limits(max_connections=20, max_keepalive_connections=12)
        self._client = client or httpx.Client(trust_env=not self.is_local, limits=limits)

    def describe(self) -> dict[str, str]:
        host = urlparse(self.base_url)
        provider = "ollama" if host.port == 11434 else ("local" if self.is_local else (host.hostname or "custom_api"))
        return {
            "provider": provider,
            "model": self.model,
            "mode_type": self.mode_type,
            "base_url": self.base_url,
        }

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def complete(self, system: str, user: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "seed": 0,
            "max_tokens": 450,
            "response_format": {"type": "json_object"},
        }
        r = self._client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers(),
                              timeout=self.timeout)
        # Некоторые внешние провайдеры не поддерживают seed или response_format=json_object
        if r.status_code == 400 and not self.is_local:
            fallback_payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0,
                "max_tokens": 450,
            }
            r = self._client.post(f"{self.base_url}/chat/completions", json=fallback_payload, headers=self._headers(),
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


def _ensure_ollama_running(llm: ChatLLM) -> list[str] | None:
    """Пробует получить список моделей, а если локальная Ollama ещё не запущена — запускает `ollama serve` в фоне с параллельными слотами GPU."""
    try:
        return llm.models()
    except (httpx.HTTPError, ValueError):
        pass

    host = urlparse(llm.base_url)
    if host.hostname not in ("127.0.0.1", "localhost", "::1") or host.port != 11434:
        return None

    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        return None

    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        env = {
            **os.environ,
            "OLLAMA_NUM_PARALLEL": os.environ.get("OLLAMA_NUM_PARALLEL", "6"),
            "OLLAMA_FLASH_ATTENTION": os.environ.get("OLLAMA_FLASH_ATTENTION", "1"),
        }
        subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            creationflags=creationflags,
        )
        for _ in range(12):
            time.sleep(0.5)
            try:
                return llm.models()
            except (httpx.HTTPError, ValueError):
                continue
    except OSError:
        pass
    return None


def list_local_models(base_url: str = OLLAMA_URL) -> list[str]:
    """Возвращает список установленных локальных моделей (автоматически запуская Ollama при необходимости)."""
    probe = ChatLLM(base_url, DEFAULT_MODEL)
    models = _ensure_ollama_running(probe)
    return models or []


def settings_path(reports_dir: Path) -> Path:
    return reports_dir / "model_settings.json"


def load_runtime_settings(reports_dir: Path) -> dict[str, Any] | None:
    p = settings_path(reports_dir)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (ValueError, OSError):
        return None


def save_runtime_settings(reports_dir: Path, data: dict[str, Any]) -> Path:
    p = settings_path(reports_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def from_config(cfg, client: httpx.Client | None = None) -> ChatLLM | None:
    """Клиент по настройкам. None — модель не настроена или недоступна: тогда работают одни правила.

    Без настроек пробуем сохранённый выбор пользователя (reports/model_settings.json) или локальную Ollama.
    """
    saved = load_runtime_settings(cfg.reports) if (client is None and not cfg.llm_base_url and not cfg.llm_model) else None
    if saved and saved.get("mode_type") == "rules_only":
        return None

    base_url = cfg.llm_base_url or (saved.get("base_url") if saved else "") or OLLAMA_URL
    model_name = cfg.llm_model or (saved.get("model") if saved else "") or DEFAULT_MODEL
    api_key = cfg.llm_api_key or (saved.get("api_key") if saved else "") or ""
    mode_type = (saved.get("mode_type") if saved else None)
    explicit = bool(cfg.llm_base_url or (saved and saved.get("base_url")))

    llm = ChatLLM(base_url, model_name, api_key, client=client, mode_type=mode_type)
    available = _ensure_ollama_running(llm) if client is None else None
    if available is None:
        try:
            available = llm.models()
        except (httpx.HTTPError, ValueError):
            if explicit and api_key:
                return llm          # облачный провайдер может не отдавать список моделей — это не признак недоступности
            return None
    if available and not cfg.llm_model and not (saved and saved.get("model")) and llm.model not in available:
        # модель не задана, а модели по умолчанию нет — берём ту, что установлена
        family = [m for m in available if m.split(":")[0] == DEFAULT_MODEL.split(":")[0]]
        llm.model = (family or available)[0]
    return llm
