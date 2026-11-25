"""Анализатор вакансии. Подпись analyze(v, llm) — граница между частями проекта, её менять нельзя.

Три способа работы:
- rules — только правила: пропуски и расхождения, видные из полей;
- llm   — только модель: она разбирает вакансию сама;
- both  — правила и модель вместе (рабочий способ). У каждого замечания видно, кто его видит: только правило,
          только модель или оба. Если видят оба — замечание одно, с формулировкой модели. Модель добавляет и
          то, чего правила не нашли. Пропуски от модели не берутся, а виды замечаний, в которых эта модель
          ненадёжна по итогам настройки, отбрасываются (см. profile.py).
"""
from __future__ import annotations

from dataclasses import replace

from ..models import TOPICS, Finding, LLM, Vacancy
from . import ai, profile, rules
from .version import VERSION  # noqa: F401

MODES = ("rules", "llm", "both")


def _sorted(found: list[Finding]) -> list[Finding]:
    order = {t: i for i, t in enumerate(TOPICS)}
    return sorted(found, key=lambda f: order.get(f.topic, 99))


def merge(rule_found: list[Finding], llm_found: list[Finding], blocked: set[tuple[str, str]]) -> list[Finding]:
    """Правила + модель. Используется и в работе, и при настройке — чтобы считалось одно и то же."""
    by_key = {(f.topic, f.kind): f for f in llm_found}
    out = []
    for f in rule_found:
        same = by_key.get((f.topic, f.kind))
        if same is None:
            out.append(f)                       # проблему видит только правило
        else:
            # мнения совпали: одно замечание. Формулировка — модели, она написана под эту вакансию;
            # цитаты — правила, они проверены и точнее.
            out.append(replace(f, problem=same.problem, recommendation=same.recommendation, source="both",
                               rule_problem=f.problem, rule_recommendation=f.recommendation))
    taken = {(f.topic, f.kind) for f in rule_found}
    out += [f for f in llm_found                # проблему видит только модель
            if f.kind != "missing" and (f.topic, f.kind) not in taken and (f.topic, f.kind) not in blocked]
    return _sorted(out)


def analyze(v: Vacancy, llm: LLM | None = None, mode: str = "both") -> list[Finding]:
    if mode == "llm":
        return _sorted(ai.find(v, llm)) if llm is not None else []
    found = rules.check(v)
    if mode == "rules" or llm is None:
        return _sorted(found)
    try:
        info = llm.describe() or {}
    except Exception:
        info = {}
    if info.get("mode_type") in ("local_only", "custom_api_only"):
        # режим «только ИИ»: человек сам выбрал отчёт одной модели, без Системного AI
        try:
            return _sorted(ai.find(v, llm))
        except Exception:
            return []
    prof = profile.cached(info)
    if prof is None:
        # модель ещё не настроена: неизвестно, каким её замечаниям можно верить, поэтому в отчёт они не идут
        return _sorted(found)
    try:
        llm_found = ai.find(v, llm)
    except Exception:
        return _sorted(found)      # модель недоступна или ответила мусором: остаются замечания правил
    return merge(found, llm_found, profile.blocked(prof))
