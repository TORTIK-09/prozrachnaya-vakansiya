"""Общий контракт проекта: нормализованная вакансия и замечание.

Этот файл — граница между частями команды. Менять его можно только по общему согласию
(см. docs/spec/00-contract.md): на эти типы опираются загрузка, анализатор, отчёт и проверка качества.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

# Темы замечаний — ровно семь, как в задании.
TOPICS = ("title", "duties", "requirements", "salary", "schedule", "location", "employment")
TOPIC_NAMES = {
    "title": "Название должности", "duties": "Обязанности", "requirements": "Требования",
    "salary": "Оплата", "schedule": "График", "location": "Место работы", "employment": "Тип занятости",
}
# Виды замечаний.
KINDS = ("missing", "ambiguous", "contradiction")
KIND_NAMES = {"missing": "пропуск", "ambiguous": "неоднозначность", "contradiction": "противоречие"}

# Поля источника, на которые может ссылаться замечание (пути в ответе API «Работа в России»).
SOURCE_FIELDS = (
    "job-name", "duty", "requirements", "qualification",
    "requirement.education", "requirement.experience", "requirement.qualification",
    "salary", "salary_min", "salary_max", "term.text", "benefit",
    "schedule", "scheduleTypeComment", "shift", "employment",
    "addresses.address[].location", "conditions",
)


@dataclass(frozen=True)
class Vacancy:
    """Вакансия в едином виде — одинаковом для учебного файла и для живого API.

    Тексты очищены от HTML и лишних пробелов; пустые строки и «0» в зарплате превращены в None.
    Контактные данные (телефоны, почта, контактное лицо) сюда не попадают — их нет и в `raw`.
    """
    id: str
    url: str | None            # vac_url; в учебных данных его нет
    title: str | None          # job-name
    duty: str | None           # duty
    requirements: str | None   # requirements
    qualification: str | None  # qualification (верхний уровень, в живом API)
    education: str | None      # requirement.education
    experience_years: int | None  # requirement.experience; None, если поля нет
    req_qualification: str | None  # requirement.qualification
    salary_text: str | None    # salary («от 45000»)
    salary_min: int | None     # salary_min; 0 и отсутствие → None
    salary_max: int | None     # salary_max; 0 и отсутствие → None
    currency: str | None
    term_text: str | None      # term.text (премии)
    benefit: str | None        # benefit (льготы)
    schedule: str | None       # schedule
    schedule_comment: str | None  # scheduleTypeComment
    shifts: tuple[str, ...]    # shift[]
    employment: str | None     # employment
    addresses: tuple[str, ...]  # addresses.address[].location
    conditions: str | None     # conditions
    company: str | None        # company.name — только для отчёта, в анализе не используется
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)  # исходный объект без контактов

    def text_of(self, source_field: str) -> str:
        """Значение поля источника строкой — именно в нём анализатор ищет цитату."""
        m = {
            "job-name": self.title, "duty": self.duty, "requirements": self.requirements,
            "qualification": self.qualification, "requirement.education": self.education,
            "requirement.experience": None if self.experience_years is None else str(self.experience_years),
            "requirement.qualification": self.req_qualification,
            "salary": self.salary_text,
            "salary_min": None if self.salary_min is None else str(self.salary_min),
            "salary_max": None if self.salary_max is None else str(self.salary_max),
            "term.text": self.term_text, "benefit": self.benefit,
            "schedule": self.schedule, "scheduleTypeComment": self.schedule_comment,
            "shift": "; ".join(self.shifts) or None, "employment": self.employment,
            "addresses.address[].location": "; ".join(self.addresses) or None,
            "conditions": self.conditions,
        }
        if source_field not in m:
            raise KeyError(f"Неизвестное поле источника: {source_field}")
        return m[source_field] or ""


@dataclass(frozen=True)
class Evidence:
    """Второе значение для противоречия: поле и цитата из него."""
    field: str
    quote: str


@dataclass(frozen=True)
class Finding:
    """Одно замечание к вакансии."""
    topic: str            # одна из TOPICS
    kind: str             # одна из KINDS
    field: str            # поле источника из SOURCE_FIELDS
    quote: str            # дословный фрагмент из этого поля; для пропуска — пустая строка
    problem: str          # в чём проблема, одним-двумя предложениями
    recommendation: str   # что именно уточнить у работодателя; без вымышленных фактов и выводов о законе
    conflict_with: Evidence | None = None  # только для kind == "contradiction": второе исходное значение
    rule_problem: str = ""         # при source == "both": что написало правило — для сверки с формулировкой модели
    rule_recommendation: str = ""
    source: str = "rule"  # кто видит проблему: "rule" — только правило, "llm" — только модель, "both" — оба

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        if self.conflict_with is None:
            d.pop("conflict_with")
        if not self.rule_problem:
            d.pop("rule_problem")
            d.pop("rule_recommendation")
        return d


def validate(f: Finding, v: Vacancy) -> list[str]:
    """Проверка замечания на соответствие контракту. Пустой список — замечание годится в отчёт."""
    errs: list[str] = []
    if f.topic not in TOPICS:
        errs.append(f"тема «{f.topic}» не из списка")
    if f.kind not in KINDS:
        errs.append(f"вид «{f.kind}» не из списка")
    if f.source not in ("rule", "llm", "both"):
        errs.append(f"источник «{f.source}» не из списка")
    if not f.problem.strip() or not f.recommendation.strip():
        errs.append("пустая суть или рекомендация")
    pairs = [(f.field, f.quote)] + ([(f.conflict_with.field, f.conflict_with.quote)] if f.conflict_with else [])
    for fld, quote in pairs:
        if fld not in SOURCE_FIELDS:
            errs.append(f"поле «{fld}» не из списка")
        elif quote and quote not in v.text_of(fld):
            errs.append(f"цитаты «{quote[:40]}» нет в поле {fld}")
    if f.kind == "missing" and f.quote:
        errs.append("у пропуска цитата должна быть пустой")
    if f.kind != "missing" and not f.quote:
        errs.append("у неоднозначности и противоречия цитата обязательна")
    if f.kind == "contradiction" and (f.conflict_with is None or not f.conflict_with.quote):
        errs.append("у противоречия должно быть второе значение")
    if f.kind != "contradiction" and f.conflict_with is not None:
        errs.append("второе значение допустимо только у противоречия")
    return errs


class LLM(Protocol):
    """Клиент модели. Реализует разработчик 2 (pv/llm.py), создаёт и передаёт в анализатор разработчик 1."""

    def describe(self) -> dict[str, str]:
        """Что за модель — попадает в шапку отчёта: {"provider": ..., "model": ...}."""

    def complete(self, system: str, user: str) -> str:
        """Один запрос к модели, ответ текстом. Ошибки сети — исключением."""
