"""Анализатор вакансии: правила находят пропуски и расхождения полей, модель — размытые формулировки
и смысловые противоречия. Подпись analyze — граница между частями проекта, её менять нельзя.
"""
from __future__ import annotations

from ..models import TOPICS, Finding, LLM, Vacancy
from . import ai, rules

# Версия входит в отпечаток кэша: после изменения правил или запроса к модели увеличьте её,
# иначе рабочий режим отдаст старые результаты из кэша.
VERSION = "1"


def analyze(v: Vacancy, llm: LLM | None = None) -> list[Finding]:
    found = rules.check(v)
    if llm is not None:
        try:
            found += ai.find(v, llm, found)
        except Exception:
            # модель недоступна или ответила мусором: остаются замечания правил
            pass
    order = {t: i for i, t in enumerate(TOPICS)}
    return sorted(found, key=lambda f: order.get(f.topic, 99))
