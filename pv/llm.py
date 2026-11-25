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

PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    # --- Локальные ИИ и серверы ---
    "ollama": {
        "label": "Ollama (Локальная LLM)",
        "mode_type": "local",
        "base_url": "http://127.0.0.1:11434/v1",
        "default_model": "qwen2.5:3b",
        "models": [
            "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b",
            "llama3.2:3b", "llama3.1:8b", "llama3.3:70b",
            "gemma2:9b", "gemma3:12b", "mistral:7b",
            "deepseek-r1:7b", "deepseek-r1:14b", "phi3.5:3.8b", "phi4:14b",
        ],
    },
    "lmstudio": {
        "label": "LM Studio (Локальный API сервер)",
        "mode_type": "local",
        "base_url": "http://127.0.0.1:1234/v1",
        "default_model": "qwen2.5-7b-instruct",
        "models": [
            "qwen2.5-3b-instruct", "qwen2.5-7b-instruct",
            "llama-3.2-3b-instruct", "llama-3.1-8b-instruct",
            "mistral-7b-instruct-v0.3", "gemma-2-9b-it", "deepseek-r1-distill-qwen-7b",
        ],
    },
    "vllm": {
        "label": "vLLM / llama.cpp Server (Локальный API)",
        "mode_type": "local",
        "base_url": "http://127.0.0.1:8000/v1",
        "default_model": "Qwen/Qwen2.5-7B-Instruct",
        "models": [
            "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
        ],
    },
    "jan": {
        "label": "Jan AI (Локальный API сервер)",
        "mode_type": "local",
        "base_url": "http://127.0.0.1:1337/v1",
        "default_model": "qwen2.5-7b-instruct",
        "models": ["qwen2.5-7b-instruct", "llama3.2-3b-instruct", "mistral-ins-7b-q4"],
    },
    "localai": {
        "label": "LocalAI / TextGen WebUI / KoboldCPP",
        "mode_type": "local",
        "base_url": "http://127.0.0.1:5000/v1",
        "default_model": "gpt-3.5-turbo",
        "models": ["gpt-3.5-turbo", "qwen2.5-7b", "llama-3.1-8b"],
    },
    # --- Облачные и внешние ИИ API ---
    "openrouter": {
        "label": "OpenRouter (Все мировые ИИ + бесплатные :free модели)",
        "mode_type": "custom_api",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "google/gemma-3-27b-it:free",
        "models": [
            "google/gemma-3-27b-it:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "deepseek/deepseek-chat:free",
            "deepseek/deepseek-r1:free",
            "mistralai/mistral-small-24b-instruct-2501:free",
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3.7-sonnet",
            "anthropic/claude-3.5-haiku",
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash",
            "x-ai/grok-2-1212",
        ],
    },
    "openai": {
        "label": "OpenAI API (ChatGPT / GPT-4o / GPT-4.1 / o3)",
        "mode_type": "custom_api",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "models": [
            "gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            "o3-mini", "o1", "o1-mini", "gpt-4-turbo", "gpt-3.5-turbo",
        ],
    },
    "gemini": {
        "label": "Google Gemini API (Gemini 2.5 / 2.0 / 1.5)",
        "mode_type": "custom_api",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.5-flash",
        "models": [
            "gemini-2.5-flash", "gemini-2.5-pro",
            "gemini-2.0-flash", "gemini-2.0-flash-lite",
            "gemini-1.5-pro", "gemini-1.5-flash",
        ],
    },
    "anthropic": {
        "label": "Anthropic Claude API (Claude 3.7 / 3.5 Sonnet & Haiku)",
        "mode_type": "custom_api",
        "base_url": "https://api.anthropic.com/v1",
        "default_model": "claude-3-7-sonnet-latest",
        "models": [
            "claude-3-7-sonnet-latest", "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest", "claude-3-opus-latest",
        ],
    },
    "deepseek": {
        "label": "DeepSeek API (DeepSeek-V3 / DeepSeek-R1)",
        "mode_type": "custom_api",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "grok": {
        "label": "xAI Grok API (Grok-3 / Grok-2)",
        "mode_type": "custom_api",
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-2-latest",
        "models": ["grok-3", "grok-3-mini", "grok-2-latest", "grok-beta"],
    },
    "mistral": {
        "label": "Mistral AI API (Mistral Large / Small / Codestral)",
        "mode_type": "custom_api",
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-small-latest",
        "models": [
            "mistral-large-latest", "mistral-small-latest",
            "ministral-8b-latest", "codestral-latest", "open-mistral-nemo",
        ],
    },
    "groq": {
        "label": "Groq Cloud API (Llama 3.3 / Mixtral / Qwen / Gemma)",
        "mode_type": "custom_api",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "models": [
            "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
            "qwen-2.5-32b", "deepseek-r1-distill-llama-70b",
            "mixtral-8x7b-32768", "gemma2-9b-it",
        ],
    },
    "together": {
        "label": "Together AI API (Llama / Qwen / DeepSeek Turbo)",
        "mode_type": "custom_api",
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
        ],
    },
    "fireworks": {
        "label": "Fireworks AI API",
        "mode_type": "custom_api",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "default_model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "models": [
            "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "accounts/fireworks/models/qwen2p5-72b-instruct",
            "accounts/fireworks/models/deepseek-v3",
        ],
    },
    "perplexity": {
        "label": "Perplexity AI API (Sonar / Sonar Pro)",
        "mode_type": "custom_api",
        "base_url": "https://api.perplexity.ai",
        "default_model": "sonar",
        "models": ["sonar", "sonar-pro", "sonar-reasoning", "sonar-reasoning-pro"],
    },
    "cohere": {
        "label": "Cohere AI API (Command R+ / Command A)",
        "mode_type": "custom_api",
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "default_model": "command-r-plus",
        "models": ["command-a-03-2025", "command-r-plus", "command-r"],
    },
    "qwen_cloud": {
        "label": "Alibaba Cloud Qwen (DashScope API)",
        "mode_type": "custom_api",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "models": ["qwen-max", "qwen-plus", "qwen-turbo", "qwen2.5-72b-instruct"],
    },
    "yandexgpt": {
        "label": "YandexGPT API (Яндекс Cloud)",
        "mode_type": "custom_api",
        "base_url": "https://llm.api.cloud.yandex.net/v1",
        "default_model": "yandexgpt/latest",
        "models": ["yandexgpt/latest", "yandexgpt-lite/latest", "llama/latest"],
    },
    "gigachat": {
        "label": "GigaChat API (Сбер / OpenAI-шлюз)",
        "mode_type": "custom_api",
        "base_url": "https://gigachat.devices.sberbank.ru/api/v1",
        "default_model": "GigaChat-Pro",
        "models": ["GigaChat", "GigaChat-Pro", "GigaChat-Max"],
    },
    "huggingface": {
        "label": "Hugging Face Inference API",
        "mode_type": "custom_api",
        "base_url": "https://router.huggingface.co/v1",
        "default_model": "Qwen/Qwen2.5-72B-Instruct",
        "models": [
            "Qwen/Qwen2.5-72B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "mistralai/Mistral-Small-24B-Instruct-2501",
        ],
    },
    "custom": {
        "label": "Другой произвольный OpenAI-совместимый API",
        "mode_type": "custom_api",
        "base_url": "",
        "default_model": "",
        "models": [],
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


KNOWN_MODEL_PARAMS_B: dict[str, float] = {
    "gpt-4o": 200.0, "gpt-4.1": 220.0, "gpt-4o-mini": 8.0, "gpt-4.1-mini": 14.0, "gpt-4.1-nano": 4.0,
    "o1": 300.0, "o3-mini": 40.0, "o1-mini": 32.0, "gpt-3.5-turbo": 20.0,
    "gemini-2.5-pro": 250.0, "gemini-2.5-flash": 32.0, "gemini-2.0-flash": 27.0, "gemini-2.0-flash-lite": 8.0,
    "gemini-1.5-pro": 175.0, "gemini-1.5-flash": 18.0,
    "claude-3-7-sonnet": 200.0, "claude-3-5-sonnet": 175.0, "claude-3-5-haiku": 20.0, "claude-3-opus": 300.0,
    "deepseek-chat": 671.0, "deepseek-v3": 671.0, "deepseek-reasoner": 671.0,
    "grok-3": 314.0, "grok-3-mini": 32.0, "grok-2": 140.0,
    "mistral-large": 123.0, "mistral-small": 24.0, "codestral": 22.0,
    "sonar-pro": 70.0, "sonar": 8.0, "command-r-plus": 104.0, "command-r": 35.0, "command-a": 111.0,
    "qwen-max": 200.0, "qwen-plus": 72.0, "qwen-turbo": 14.0,
    "yandexgpt-lite": 8.0, "yandexgpt": 35.0,
    "gigachat-max": 70.0, "gigachat-pro": 29.0, "gigachat": 13.0,
}


_HW_CACHE: dict[str, Any] = {"ts": 0.0, "data": None}


def detect_hardware_specs() -> dict[str, Any]:
    """Определяет технические характеристики ПК/сервера и текущую загрузку (CPU, RAM, GPU) с коротким кэшем."""
    now_t = time.monotonic()
    if _HW_CACHE["data"] is not None and (now_t - _HW_CACHE["ts"]) < 15.0:
        return dict(_HW_CACHE["data"])

    cpu_cores = os.cpu_count() or 4
    cpu_load_pct = 25.0
    ram_total_gb = 16.0
    ram_free_gb = 8.0
    gpu_name = None
    vram_gb = 0.0

    try:
        import ctypes

        class _MEMSTAT(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        if os.name == "nt":
            ms = _MEMSTAT()
            ms.dwLength = ctypes.sizeof(_MEMSTAT)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                ram_total_gb = round(ms.ullTotalPhys / (1024 ** 3), 1)
                ram_free_gb = round(ms.ullAvailPhys / (1024 ** 3), 1)
    except Exception:
        pass

    try:
        import psutil
        cpu_load_pct = round(float(psutil.cpu_percent(interval=0.05)), 1)
        vm = psutil.virtual_memory()
        ram_total_gb = round(vm.total / (1024 ** 3), 1)
        ram_free_gb = round(vm.available / (1024 ** 3), 1)
    except Exception:
        pass

    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            res = subprocess.run(
                [nvsmi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=1.5, creationflags=creationflags,
            )
            if res.returncode == 0 and res.stdout.strip():
                first = res.stdout.strip().splitlines()[0].split(",")
                gpu_name = first[0].strip()
                if len(first) > 1:
                    vram_gb = round(float(first[1].strip()) / 1024.0, 1)
        except Exception:
            pass

    is_overloaded = cpu_load_pct >= 80.0 or ram_free_gb < 2.0
    hw_tier = "high" if (vram_gb >= 8.0 or (cpu_cores >= 12 and ram_total_gb >= 24)) and not is_overloaded else (
        "low" if is_overloaded or (vram_gb == 0 and cpu_cores <= 4) else "medium"
    )
    out = {
        "cpu_cores": cpu_cores,
        "cpu_load_pct": cpu_load_pct,
        "ram_gb": ram_total_gb,
        "ram_total_gb": ram_total_gb,
        "ram_free_gb": ram_free_gb,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "hw_tier": hw_tier,
        "is_overloaded": is_overloaded,
    }
    _HW_CACHE["ts"] = now_t
    _HW_CACHE["data"] = out
    return dict(out)


def detect_model_parameters(model_name: str, base_url: str = "") -> dict[str, Any]:
    """
    Определяет количество параметров модели (в млрд — B) тремя способами:
    1) Метаданные локального сервера Ollama (/api/show → details.parameter_size / general.parameter_count);
    2) Разбор индекса параметров в названии модели (например, 3b, 3.5b, 7b, 27b, 70b, 8x7b);
    3) Встроенный справочник известных облачных моделей (KNOWN_MODEL_PARAMS_B).
    """
    import re
    raw_name = (model_name or "").strip()
    low = raw_name.lower()
    source_way = "mapping"
    params_b: float | None = None

    # Способ 1: прямой запрос метаданных у локальной Ollama (/api/show)
    if base_url and ("11434" in base_url or "localhost" in base_url or "127.0.0.1" in base_url) and raw_name:
        try:
            root_ollama = base_url.split("/v1")[0].rstrip("/")
            with httpx.Client(trust_env=False, timeout=1.5) as cl:
                r = cl.post(f"{root_ollama}/api/show", json={"name": raw_name})
                if r.status_code == 200:
                    d = r.json()
                    psize = str((d.get("details") or {}).get("parameter_size") or "")
                    m_ps = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])", psize)
                    if m_ps:
                        val = float(m_ps.group(1))
                        params_b = round(val / 1000.0, 2) if m_ps.group(2).lower() == "m" else val
                        source_way = "ollama_api_show"
                    elif isinstance(d.get("model_info"), dict):
                        cnt = d["model_info"].get("general.parameter_count")
                        if isinstance(cnt, (int, float)) and cnt > 0:
                            params_b = round(float(cnt) / 1e9, 2)
                            source_way = "gguf_metadata"
        except Exception:
            pass

    # Способ 2: регулярное выражение по названию модели (8x7b, 70b, 3.5b и т.д.)
    if params_b is None:
        moe = re.search(r"(\d+)x(\d+(?:\.\d+)?)b\b", low)
        if moe:
            params_b = round(float(moe.group(1)) * float(moe.group(2)), 1)
            source_way = "name_moe_regex"
        else:
            m_b = re.search(r"(?:^|[-_:/\s])(\d+(?:\.\d+)?)b\b", low)
            if m_b:
                params_b = float(m_b.group(1))
                source_way = "name_regex"

    # Способ 3: справочник известных ИИ-моделей
    if params_b is None:
        for key, val in KNOWN_MODEL_PARAMS_B.items():
            if key in low:
                params_b = val
                source_way = "known_catalog"
                break

    if params_b is None:
        params_b = 7.0
        source_way = "heuristic_default"

    strength = "strong" if params_b >= 65.0 else ("medium" if params_b >= 14.0 else "weak")
    return {
        "model": raw_name,
        "params_b": params_b,
        "params_label": f"~{params_b:g}B параметров",
        "detection_method": source_way,
        "strength": strength,
    }


def compute_calibration_plan(
    model_name: str,
    mode_type: str,
    user_minutes: float | None = None,
    base_url: str = "",
    measured_sec: float | None = None,
) -> dict[str, Any]:
    """
    Рассчитывает план калибровки ИИ и раздельной скрытой калибровки Системного AI:
    - Количество тестов зависит от силы модели (параметров), скорости ответа, характеристик ПК и заданного времени.
    - Верхняя граница (anti_template_cap) защищает от шаблонности (переобучения фильтра).
    - Если выбранное время превышает разумный предел (best_time_minutes), оно автоматически урезается.
    """
    hw = detect_hardware_specs()
    mp = detect_model_parameters(model_name, base_url)
    strength = mp["strength"]
    is_local = mode_type in ("local", "local_only")

    # Оценка времени ответа на 1 вакансию с учётом загрузки ПК и типа модели
    if measured_sec and measured_sec > 0:
        sec_per_vac = float(measured_sec)
    elif mode_type == "rules_only":
        sec_per_vac = 0.01
    elif is_local:
        base_sec = 2.8 if hw["vram_gb"] >= 6.0 else 14.0
        if mp["params_b"] > 8:
            base_sec *= (mp["params_b"] / 5.0)
        if hw["cpu_load_pct"] >= 85.0:
            base_sec = max(base_sec * 3.5, 45.0)  # сценарий сильной загрузки ПК (80-100 сек при троттлинге)
        sec_per_vac = round(base_sec, 1)
    else:
        sec_per_vac = 1.4 if strength != "strong" else 2.2

    # Индивидуальные границы тестов под конкретную LLM:
    # Чем быстрее и сильнее модель, тем меньше минимальный порог; верхний потолок защищает от шаблонности
    if strength == "strong":
        min_tests = 10 if sec_per_vac <= 3.0 else 14
        anti_template_cap = 34
        priority_owner = "ai"       # Мощная ИИ: приоритет отдаётся ИИ (или полное согласие с Системным AI)
    elif strength == "medium":
        min_tests = 14 if sec_per_vac <= 4.0 else 18
        anti_template_cap = 52
        priority_owner = "balanced"
    else:
        # Слабая модель (например, 3B/7B): если быстрая — минимум меньше, если медленная/загружен ПК — компенсирует Системный AI
        min_tests = 16 if sec_per_vac <= 4.0 else 12
        anti_template_cap = 68
        priority_owner = "system_ai"

    parallel_factor = 4.0 if (not is_local or (hw["vram_gb"] >= 6.0 and not hw["is_overloaded"])) else 1.0
    effective_sec_per_item = max(0.25, sec_per_vac / parallel_factor)

    best_time_minutes = round(max(1.0, (anti_template_cap * effective_sec_per_item) / 60.0), 1)
    default_minutes = round(max(1.0, min(best_time_minutes, (22 * effective_sec_per_item) / 60.0)), 1)

    requested_minutes = float(user_minutes) if (user_minutes is not None and user_minutes > 0) else default_minutes
    time_trimmed = False
    effective_minutes = requested_minutes
    if requested_minutes > best_time_minutes:
        effective_minutes = best_time_minutes
        time_trimmed = True

    affordable_by_time = int((effective_minutes * 60.0) / effective_sec_per_item)
    llm_target_tests = max(min_tests, min(anti_template_cap, affordable_by_time))

    # Пока ИИ калибруется за effective_minutes, Системный AI параллельно и скрытно калибруется на пуле портала (400+ вакансий)
    system_ai_capacity = min(2000, max(400, int((effective_minutes * 60.0) / 0.005)))

    return {
        "hardware": hw,
        "model_meta": mp,
        "sec_per_vacancy_est": sec_per_vac,
        "min_tests": min_tests,
        "llm_target_tests": llm_target_tests,
        "anti_template_cap": anti_template_cap,
        "requested_minutes": round(requested_minutes, 1),
        "effective_minutes": round(effective_minutes, 1),
        "best_time_minutes": best_time_minutes,
        "time_trimmed": time_trimmed,
        "system_ai_hidden_pool_target": system_ai_capacity,
        "priority_owner": priority_owner,
    }


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
