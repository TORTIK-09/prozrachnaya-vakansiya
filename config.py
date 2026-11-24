import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


@dataclass
class AppConfig:
    """Конфигурация сервиса «Прозрачная вакансия»."""

    # Источник данных API «Работа в России» (Регион 39 — Калининградская область)
    api_url: str = os.getenv(
        "API_URL", "http://opendata.trudvsem.ru/api/v1/vacancies/region/39"
    )
    api_timeout_seconds: int = int(os.getenv("API_TIMEOUT_SECONDS", "15"))

    # Лимиты и пагинация (по умолчанию 20 вакансий по 10 на страницу = минимум 2 страницы)
    max_vacancies: int = int(os.getenv("MAX_VACANCIES", "20"))
    page_limit: int = int(os.getenv("PAGE_LIMIT", "10"))

    # Расписание автоматического запуска
    schedule_interval_minutes: int = int(os.getenv("SCHEDULE_INTERVAL_MINUTES", "60"))
    schedule_cron: str = os.getenv("SCHEDULE_CRON", "0 * * * *")

    # Пути к файлам
    sample_input_path: str = os.getenv(
        "SAMPLE_INPUT_PATH", "kantkod_sample_vacancies.json"
    )
    report_output_path: str = os.getenv(
        "REPORT_OUTPUT_PATH", "reports/latest_report.json"
    )
    sample_report_output_path: str = os.getenv(
        "SAMPLE_REPORT_OUTPUT_PATH", "reports/sample_report.json"
    )
    last_successful_report_path: str = os.getenv(
        "LAST_SUCCESSFUL_REPORT_PATH", "reports/last_successful_report.json"
    )

    # Настройки ИИ
    ai_provider: str = os.getenv("AI_PROVIDER", "auto").lower()

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv(
        "OPENROUTER_MODEL", "google/gemma-3-27b-it:free"
    )

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Веб-сервер
    web_host: str = os.getenv("WEB_HOST", "127.0.0.1")
    web_port: int = int(os.getenv("WEB_PORT", "8085"))


def get_config() -> AppConfig:
    return AppConfig()
