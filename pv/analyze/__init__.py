"""Анализатор вакансии. Подпись analyze(v, llm) — граница между частями проекта, её менять нельзя.

Три способа работы:
- rules — только правила: пропуски и расхождения, видные из полей;
- llm   — только модель: она разбирает вакансию сама;
- both  — правила и модель вместе (рабочий способ): замечания правил принимаются как есть, модель добавляет
          то, чего правила не нашли. Пропуски от модели не берутся, а виды замечаний, в которых эта модель
          ненадёжна по итогам настройки, отбрасываются (см. profile.py).
"""
from __future__ import annotations

from ..models import TOPICS, Finding, LLM, Vacancy
from . import ai, profile, rules

# Версия входит в отпечаток кэша: после изменения правил или запроса к модели увеличьте её,
# иначе рабочий режим отдаст старые результаты из кэша.
VERSION = "2"
MODES = ("rules", "llm", "both")


def _sorted(found: list[Finding]) -> list[Finding]:
    order = {t: i for i, t in enumerate(TOPICS)}
    return sorted(found, key=lambda f: order.get(f.topic, 99))


def merge(rule_found: list[Finding], llm_found: list[Finding], blocked: set[tuple[str, str]]) -> list[Finding]:
    """Правила + модель. Используется и в работе, и при настройке — чтобы считалось одно и то же."""
    taken = {(f.topic, f.kind) for f in rule_found}
    extra = [f for f in llm_found
             if f.kind != "missing" and (f.topic, f.kind) not in taken and (f.topic, f.kind) not in blocked]
    return _sorted(rule_found + extra)


def analyze(v: Vacancy, llm: LLM | None = None, mode: str = "both") -> list[Finding]:
    if mode == "llm":
        return _sorted(ai.find(v, llm)) if llm is not None else []
    found = rules.check(v)
    if mode == "rules" or llm is None:
        return _sorted(found)
    try:
        llm_found = ai.find(v, llm)
    except Exception:
        return _sorted(found)      # модель недоступна или ответила мусором: остаются замечания правил
    try:
        info = llm.describe() or {}
    except Exception:
        info = {}
    return merge(found, llm_found, profile.blocked(profile.cached(info)))
