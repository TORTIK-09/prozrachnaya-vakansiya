"""Профиль модели: каким видам её замечаний можно доверять.

Профиль получается настройкой (pv/calibrate.py): модель прогоняется по размеченным вакансиям, и виды замечаний,
в которых она чаще ошибается, чем попадает, отключаются. Веса модели при этом не меняются — меняется то,
какую часть её ответа сервис пускает в отчёт.

Профили лежат в двух местах: pv/profiles/ — поставляются с проектом (для модели по умолчанию),
reports/profiles/ — созданы на этом компьютере. Местный профиль важнее поставляемого.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import ROOT
from .version import VERSION

SHIPPED = Path(__file__).resolve().parent.parent / "profiles"
_cache: dict[tuple[str, str], dict | None] = {}


def slug(info: dict) -> str:
    """Имя файла профиля: провайдер и модель. Одна и та же модель на другом сервере — другая модель."""
    raw = f"{info.get('provider') or 'unknown'}__{info.get('model') or 'unknown'}"
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-").lower()


def local_dir(reports_dir: Path | None = None) -> Path:
    return (reports_dir or ROOT / "reports") / "profiles"


def load(info: dict, reports_dir: Path | None = None) -> dict | None:
    """Профиль модели или None, если её ещё не настраивали."""
    name = slug(info) + ".json"
    for folder in (local_dir(reports_dir), SHIPPED):
        p = folder / name
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            # профиль от прежней версии анализатора не годится: запрос к модели с тех пор менялся
            if isinstance(data, dict) and data.get("version") == VERSION:
                data["_shipped"] = folder == SHIPPED
                return data
    return None


def cached(info: dict, reports_dir: Path | None = None) -> dict | None:
    key = (slug(info), str(reports_dir))
    if key not in _cache:
        _cache[key] = load(info, reports_dir)
    return _cache[key]


def forget() -> None:
    """После новой настройки профиль нужно перечитать."""
    _cache.clear()


def blocked(profile: dict | None) -> set[tuple[str, str]]:
    """Виды замечаний модели, которые в отчёт не пускаются."""
    if not profile:
        return set()
    return {tuple(x) for x in profile.get("blocked", []) if isinstance(x, list) and len(x) == 2}
