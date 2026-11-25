"""Настройки: значения по умолчанию ← config.toml ← .env ← переменные окружения."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    source_url: str = "http://opendata.trudvsem.ru/api/v1/vacancies/region/39"
    schedule: str = "0 6 * * *"      # cron: минута час день месяц день_недели
    max_vacancies: int = 200
    page_size: int = 100             # больше 100 API не отдаёт
    timeout: float = 60.0            # секунд на один запрос к API
    retries: int = 3
    workers: int = 4                 # потоков анализа
    reports_dir: str = "reports"
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    web_host: str = "127.0.0.1"
    web_port: int = 8085
    calib_minutes: float = 0.0       # сколько минут отдать калибровке ИИ; 0 — столько, сколько советует план

    @property
    def reports(self) -> Path:
        p = Path(self.reports_dir)
        return p if p.is_absolute() else ROOT / p


ENV_PREFIX = "PV_"


def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _cast(name: str, value, kind: type):
    try:
        if kind is int:
            return int(value)
        if kind is float:
            return float(value)
        return str(value)
    except (TypeError, ValueError):
        raise ValueError(f"Настройка {name}: ожидается {'целое число' if kind is int else 'число'}, получено «{value}»")


def load(config_path: str | Path | None = None, env: dict[str, str] | None = None,
         env_file: str | Path | None = None) -> Config:
    """Собирает настройки. `env` и пути передаются явно только в тестах."""
    kinds = {f.name: type(f.default) for f in fields(Config)}
    values: dict = {}

    path = Path(config_path) if config_path else ROOT / "config.toml"
    if path.is_file():
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        for k, v in data.items():
            if k not in kinds:
                raise ValueError(f"Неизвестная настройка в {path.name}: {k}")
            values[k] = _cast(k, v, kinds[k])
    elif config_path:
        raise FileNotFoundError(f"Файл настроек не найден: {path}")

    merged = {**_read_env_file(Path(env_file) if env_file else ROOT / ".env"), **(os.environ if env is None else env)}
    for name, kind in kinds.items():
        raw = merged.get(ENV_PREFIX + name.upper())
        if raw not in (None, ""):
            values[name] = _cast(ENV_PREFIX + name.upper(), raw, kind)

    cfg = replace(Config(), **values)
    if cfg.max_vacancies < 1:
        raise ValueError("max_vacancies должно быть не меньше 1")
    if not 1 <= cfg.page_size <= 100:
        raise ValueError("page_size должно быть от 1 до 100")
    if cfg.workers < 1:
        raise ValueError("workers должно быть не меньше 1")
    return cfg
