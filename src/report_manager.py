import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from src.models import FullReport, VacancyReportItem

logger = logging.getLogger("transparent_vacancy.report_manager")


def compute_vacancy_hash(vacancy: Dict[str, Any]) -> str:
    """Вычисляет детерминированный SHA-256 хэш содержимого карточки вакансии."""
    canonical = json.dumps(vacancy, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class ReportManager:
    """
    Управление сохранением, дедупликацией по ID и резервированием последнего успешного отчёта.
    """

    def __init__(self, last_successful_path: str | Path = "reports/last_successful_report.json"):
        self.last_successful_path = Path(last_successful_path)

    def load_report(self, path: str | Path) -> Optional[FullReport]:
        """Загружает ранее сохранённый отчёт, если он существует и валиден."""
        file_path = Path(path)
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return FullReport.model_validate(data)
        except Exception as exc:
            logger.warning("Не удалось прочитать существующий отчёт %s: %s", file_path, exc)
            return None

    def get_cached_items_by_id(
        self, path: str | Path, expected_ai_provider: Optional[str] = None
    ) -> Dict[str, VacancyReportItem]:
        """
        Возвращает словарь {vacancy_id: VacancyReportItem} из текущего отчёта
        для предотвращения дублирования и повторной обработки неизменных данных.
        Если сменился ИИ-провайдер (например, включили Ollama), кэш сбрасывается для пересчёта через новую модель.
        """
        existing = self.load_report(path)
        if not existing:
            return {}
        if expected_ai_provider and existing.ai_provider_used != expected_ai_provider:
            return {}
        return {item.id: item for item in existing.vacancies if item.id}

    def save_report(
        self,
        report: FullReport,
        output_path: str | Path,
        update_last_successful: bool = True,
    ) -> Path:
        """
        Сохраняет отчёт в JSON.
        Гарантирует отсутствие дубликатов по ID вакансии (1 результат на 1 ID вакансии).
        """
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Строгая дедупликация по ID вакансии с сохранением порядка
        deduped_map: Dict[str, VacancyReportItem] = {}
        for item in report.vacancies:
            deduped_map[item.id] = item

        deduped_list = list(deduped_map.values())
        report.vacancies = deduped_list
        report.total_vacancies = len(deduped_list)
        report.clean_vacancies = sum(1 for v in deduped_list if len(v.issues) == 0)
        report.vacancies_with_issues = sum(1 for v in deduped_list if len(v.issues) > 0)
        report.total_issues = sum(len(v.issues) for v in deduped_list)

        payload = report.model_dump()
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(
            "Отчёт успешно сохранён в %s (вакансий: %d, без замечаний: %d, с замечаниями: %d, всего замечаний: %d)",
            target_path,
            report.total_vacancies,
            report.clean_vacancies,
            report.vacancies_with_issues,
            report.total_issues,
        )

        if update_last_successful:
            self.last_successful_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.resolve() != self.last_successful_path.resolve():
                shutil.copyfile(target_path, self.last_successful_path)

        return target_path

    def handle_api_failure(self, output_path: str | Path, error_message: str) -> Optional[FullReport]:
        """
        При недоступности API сообщает об ошибке и сохраняет последний успешный отчёт, если он есть.
        """
        logger.error("ОШИБКА ДОСТУПА К API: %s", error_message)
        target_path = Path(output_path)

        if target_path.exists():
            logger.info(
                "Сохраняется текущий последний успешный отчёт: %s", target_path
            )
            return self.load_report(target_path)

        if self.last_successful_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.last_successful_path, target_path)
            logger.info(
                "Восстановлен последний успешный отчёт из резервной копии в %s",
                target_path,
            )
            return self.load_report(target_path)

        logger.warning("Последний успешный отчёт отсутствует (это первый запуск).")
        return None
