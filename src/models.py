from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


IssueType = Literal["пропуск", "неоднозначность", "противоречие"]

IssueTopic = Literal[
    "Название должности",
    "Обязанности",
    "Требования",
    "Оплата",
    "График",
    "Место работы",
    "Тип занятости",
]


class VacancyIssue(BaseModel):
    """Замечание по конкретному аспекту вакансии."""

    type: IssueType = Field(
        ...,
        description="Тип проблемы: 'пропуск', 'неоднозначность' или 'противоречие'",
    )
    topic: IssueTopic = Field(
        ...,
        description="Тема проверки (Название должности, Обязанности, Требования, Оплата, График, Место работы, Тип занятости)",
    )
    field: str = Field(
        ...,
        description="Поле или поля источника в JSON вакансии (например, 'salary_min, salary_max' или 'duty')",
    )
    quote: str = Field(
        default="",
        description="Цитата или значение из поля источника. Для пропуска — пустая строка ''. Для противоречия — оба исходных значения.",
    )
    issue: str = Field(
        ...,
        description="Суть проблемы: почему это мешает соискателю понять условия или каких сведений не хватает",
    )
    recommendation: str = Field(
        ...,
        description="Конкретная рекомендация редактору/работодателю: что именно уточнить (без вымышленных фактов и выводов о нарушении закона)",
    )
    detected_by: str = Field(
        default="rules+ai",
        description="Механизм выявления/уточнения замечания ('rules', 'ai', 'rules+ai')",
    )


class VacancyReportItem(BaseModel):
    """Результат проверки одной вакансии."""

    id: str = Field(..., description="Уникальный идентификатор вакансии (vacancy.id)")
    url: Optional[str] = Field(
        default=None,
        description="Ссылка на карточку вакансии (vac_url) при наличии в источнике",
    )
    job_name: str = Field(default="", description="Название должности (job-name)")
    company_name: str = Field(
        default="", description="Название организации (company.name)"
    )
    location: str = Field(
        default="", description="Основной адрес места работы из карточки"
    )
    is_transparent: bool = Field(
        default=True,
        description="True, если описание условий полностью прозрачно и замечаний нет",
    )
    transparency_score: int = Field(
        default=100,
        description="Индекс прозрачности вакансии от 0 до 100",
    )
    content_hash: str = Field(
        default="",
        description="Хэш содержимого вакансии для идемпотентности при повторных запусках",
    )
    issues: List[VacancyIssue] = Field(
        default_factory=list,
        description="Массив замечаний. Если замечаний нет — пустой массив []",
    )


class FullReport(BaseModel):
    """Итоговый JSON-отчёт анализатора."""

    generated_at: str = Field(..., description="Время формирования отчёта (ISO 8601)")
    mode: Literal["verify", "live"] = Field(
        ..., description="Режим запуска: 'verify' (локальный JSON) или 'live' (API)"
    )
    source: str = Field(..., description="Источник данных (путь к файлу или URL API)")
    ai_provider_used: str = Field(
        ..., description="Использованный ИИ-провайдер и модель"
    )
    pages_processed: int = Field(
        default=1, description="Количество обработанных страниц источника"
    )
    total_vacancies: int = Field(
        default=0, description="Общее количество уникальных вакансий в отчёте"
    )
    clean_vacancies: int = Field(
        default=0, description="Количество вакансий без замечаний"
    )
    vacancies_with_issues: int = Field(
        default=0, description="Количество вакансий с замечаниями"
    )
    total_issues: int = Field(
        default=0, description="Общее количество выявленных замечаний"
    )
    vacancies: List[VacancyReportItem] = Field(
        default_factory=list,
        description="Массив результатов проверки по каждой вакансии (1 объект на 1 ID вакансии)",
    )


class CandidateResume(BaseModel):
    """Модель обезличенного резюме соискателя для дополнительного сценария."""

    resume_id: str
    candidate_alias: str = Field(
        ..., description="Обезличенное имя или профиль кандидата"
    )
    desired_role: str
    city: str
    education_level: str
    education_details: str
    experience_years: int
    work_history: List[Dict[str, Any]] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    preferred_schedule: str = ""


class TailoredResumeResult(BaseModel):
    """Результат работы дополнительного сценария: черновик резюме под выбранную вакансию."""

    vacancy_id: str
    vacancy_title: str
    company_name: str
    candidate_alias: str
    match_score: int = Field(..., description="Оценка соответствия резюме вакансии (0-100%)")
    match_summary: str
    matched_skills: List[str]
    missing_or_unconfirmed_requirements: List[str]
    tailored_resume_draft: Dict[str, Any] = Field(
        ...,
        description="Черновик резюме, собранный строго на основе опыта, навыков и образования из исходного резюме",
    )
    cover_letter_draft: str = Field(
        ..., description="Черновик сопроводительного письма"
    )
    questions_for_employer: List[str] = Field(
        default_factory=list,
        description="Список уточняющих вопросов работодателю на основе замечаний из базового отчёта",
    )
    ai_provider_used: str = ""
