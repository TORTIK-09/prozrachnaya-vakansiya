"""Нормы опыта по должностям, выученные на вакансиях портала.

По каждой группе должностей считается, в какой доле вакансий требуется опыт. Это позволяет отличить
«опыт 0, потому что он не нужен» (уборщик: опыт требуют 8 % вакансий) от «опыт 0, потому что поле забыли
заполнить» (заведующий: опыт требуют все). Таблица пересчитывается командой `python -m pv learn`.

Группа — первые пять букв первого слова названия: «Повар», «повара», «Повар-универсал» попадают вместе.
Это грубо, но не требует словаря и устойчиво к падежам.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from ..config import ROOT
from ..models import Vacancy

SHIPPED = Path(__file__).resolve().parent.parent / "data" / "experience_norms.json"
MIN_VACANCIES = 10      # меньше — группа слишком мала, чтобы ей верить
MIN_SHARE = 0.75        # доля вакансий с опытом, начиная с которой «опыт 0» выглядит необычно
_WORD = re.compile(r"[а-яёa-z]+", re.I)
_cache: dict | None = None


def key(title: str | None) -> str:
    words = [w.lower().replace("ё", "е") for w in _WORD.findall(title or "") if len(w) > 2]
    return words[0][:5] if words else ""


def build(vacancies: Iterable[Vacancy]) -> dict:
    groups: dict[str, list[int]] = {}
    total = 0
    for v in vacancies:
        k = key(v.title)
        if not k or v.experience_years is None:
            continue
        total += 1
        g = groups.setdefault(k, [0, 0])
        g[0] += 1
        g[1] += int(v.experience_years > 0)
    return {"vacancies": total, "groups": {k: g for k, g in sorted(groups.items()) if g[0] >= MIN_VACANCIES}}


def local_path(reports_dir: Path | None = None) -> Path:
    return (reports_dir or ROOT / "reports") / "experience_norms.json"


def load() -> dict:
    """Таблица норм: выученная на этом компьютере, иначе поставляемая с проектом."""
    global _cache
    if _cache is None:
        _cache = {"vacancies": 0, "groups": {}}
        for p in (local_path(), SHIPPED):
            if p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    continue
                if isinstance(data, dict) and isinstance(data.get("groups"), dict):
                    _cache = data
                    break
    return _cache


def forget() -> None:
    global _cache
    _cache = None


def usually_needs_experience(title: str | None) -> tuple[int, int] | None:
    """(с опытом, всего) по группе должности, если опыт на портале требуется почти всегда; иначе None."""
    g = load()["groups"].get(key(title))
    if not g or g[0] < MIN_VACANCIES or g[1] / g[0] < MIN_SHARE:
        return None
    return g[1], g[0]
