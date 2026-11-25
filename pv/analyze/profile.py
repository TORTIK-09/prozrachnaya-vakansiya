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
from .version import PROFILE_VERSION

SHIPPED = Path(__file__).resolve().parent.parent / "profiles"
_cache: dict[tuple[str, str], dict | None] = {}


MIN_PRECISION = 0.75    # вид замечаний модели остаётся включённым, если верных среди них не меньше трёх четвертей
MIN_CONFIRMED = 2       # …и он подтвердился хотя бы дважды: доверие нужно заслужить


def trusted(ok: int, extra: int) -> bool:
    return ok >= MIN_CONFIRMED and ok / max(ok + extra, 1) >= MIN_PRECISION


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
            if isinstance(data, dict) and data.get("version") == PROFILE_VERSION:
                data["_shipped"] = folder == SHIPPED
                return data
    return None


def every(reports_dir: Path | None = None) -> list[dict]:
    """Все годные профили: местные и поставляемые; местный важнее поставляемого для той же модели."""
    out: dict[str, dict] = {}
    for folder in (SHIPPED, local_dir(reports_dir)):
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if isinstance(data, dict) and data.get("version") == PROFILE_VERSION and isinstance(data.get("model"), dict):
                out[p.name] = data
    return list(out.values())


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
    out = {tuple(x) for x in profile.get("blocked", []) if isinstance(x, list) and len(x) == 2}
    # пороги доверия применяются и к уже сохранённым профилям: по числу верных и лишних замечаний каждого вида
    for c in profile.get("categories", []):
        if isinstance(c, dict) and not trusted(int(c.get("ok", 0)), int(c.get("extra", 0))):
            out.add((c.get("topic"), c.get("kind")))
    return out
