"""Анализатор вакансии. ЗАГЛУШКА части 1: замечаний не находит.

Здесь разработчик 2 размещает настоящий анализатор (см. docs/spec/02-analyzer.md).
Подпись функции — граница между частями, её менять нельзя.
"""
from __future__ import annotations

from ..models import Finding, LLM, Vacancy

# Версия входит в отпечаток кэша: после изменения правил или запросов к модели увеличьте её,
# иначе рабочий режим отдаст старые результаты из кэша.
VERSION = "0-stub"


def analyze(v: Vacancy, llm: LLM | None = None) -> list[Finding]:
    return []
