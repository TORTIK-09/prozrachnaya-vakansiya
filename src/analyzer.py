from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from config import AppConfig
from src.ai_engine import AIEngine
from src.api_client import (
    APIFetchError,
    fetch_vacancies_from_api,
    load_vacancies_from_file,
)
from src.models import FullReport, VacancyIssue, VacancyReportItem
from src.report_manager import ReportManager, compute_vacancy_hash
from src.rule_engine import RuleEngine, clean_text

logger = logging.getLogger("transparent_vacancy.analyzer")


class VacancyAnalyzer:
    """
    Ядро анализатора «Прозрачная вакансия».
    Используется единообразно как в рабочем режиме (API региона 39),
    так и в проверочном режиме (локальный JSON).
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.rule_engine = RuleEngine()
        self.ai_engine = AIEngine(config)
        self.report_manager = ReportManager(config.last_successful_report_path)

    def analyze_single_vacancy(self, vacancy: Dict[str, Any]) -> VacancyReportItem:
        """Проверяет одну вакансию: правила + ИИ-обогащение."""
        vac_id = str(vacancy.get("id", "UNKNOWN")).strip()
        vac_url = vacancy.get("vac_url") or vacancy.get("url") or None
        if isinstance(vac_url, str):
            vac_url = vac_url.strip() or None

        job_name = clean_text(vacancy.get("job-name"))
        company_obj = vacancy.get("company")
        company_name = (
            clean_text(company_obj.get("name"))
            if isinstance(company_obj, dict)
            else ""
        )

        addresses_obj = vacancy.get("addresses")
        address_list = (
            addresses_obj.get("address", [])
            if isinstance(addresses_obj, dict)
            else []
        )
        primary_location = ""
        if isinstance(address_list, list) and address_list and isinstance(address_list[0], dict):
            primary_location = clean_text(address_list[0].get("location"))

        content_hash = compute_vacancy_hash(vacancy)

        # Шаг 1: Детерминированный анализ полей и кросс-полевых правил
        rule_issues = self.rule_engine.analyze(vacancy)

        # Шаг 2: ИИ-обработка (семантический анализ и контекстное уточнение рекомендаций)
        final_issues = self.ai_engine.enrich_and_validate_issues(
            vacancy, rule_issues
        )

        score = self._calculate_transparency_score(final_issues)

        return VacancyReportItem(
            id=vac_id,
            url=vac_url,
            job_name=job_name,
            company_name=company_name,
            location=primary_location,
            is_transparent=(len(final_issues) == 0),
            transparency_score=score,
            content_hash=content_hash,
            issues=final_issues,
        )

    @staticmethod
    def _calculate_transparency_score(issues: List[VacancyIssue]) -> int:
        if not issues:
            return 100
        penalty = 0
        for item in issues:
            if item.type == "противоречие":
                penalty += 25
            elif item.type == "пропуск":
                penalty += 20
            else:
                penalty += 15
        return max(0, 100 - penalty)

    def analyze_vacancies_list(
        self,
        vacancies: List[Dict[str, Any]],
        mode: Literal["verify", "live"],
        source: str,
        pages_processed: int = 1,
        existing_cache: Optional[Dict[str, VacancyReportItem]] = None,
    ) -> FullReport:
        """
        Формирует полный отчёт по списку вакансий.
        Обеспечивает дедупликацию по ID и переиспользование результата для неизменных данных.
        """
        cache = existing_cache or {}
        results_by_id: Dict[str, VacancyReportItem] = {}

        for vac in vacancies:
            vac_id = str(vac.get("id", "")).strip()
            if not vac_id:
                continue

            current_hash = compute_vacancy_hash(vac)
            cached_item = cache.get(vac_id)
            if cached_item and cached_item.content_hash == current_hash:
                logger.debug(
                    "Вакансия %s не изменилась (hash=%s), переиспользуем актуальный результат.",
                    vac_id,
                    current_hash,
                )
                results_by_id[vac_id] = cached_item
            else:
                results_by_id[vac_id] = self.analyze_single_vacancy(vac)

        items = list(results_by_id.values())
        now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

        return FullReport(
            generated_at=now_iso,
            mode=mode,
            source=str(source),
            ai_provider_used=self.ai_engine.provider_label,
            pages_processed=pages_processed,
            total_vacancies=len(items),
            clean_vacancies=sum(1 for i in items if len(i.issues) == 0),
            vacancies_with_issues=sum(1 for i in items if len(i.issues) > 0),
            total_issues=sum(len(i.issues) for i in items),
            vacancies=items,
        )

    def run_verify_mode(
        self,
        input_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
    ) -> FullReport:
        """
        Проверочный режим: принимает локальный JSON в формате ответов API
        и формирует отчёт без обращения к порталу.
        """
        in_file = Path(input_path or self.config.sample_input_path)
        out_file = Path(output_path or self.config.sample_report_output_path)

        vacancies, pages_count = load_vacancies_from_file(in_file)
        existing_cache = self.report_manager.get_cached_items_by_id(
            out_file, expected_ai_provider=self.ai_engine.provider_label
        )

        report = self.analyze_vacancies_list(
            vacancies=vacancies,
            mode="verify",
            source=str(in_file),
            pages_processed=pages_count,
            existing_cache=existing_cache,
        )
        self.report_manager.save_report(report, out_file, update_last_successful=True)
        return report

    def run_live_mode(
        self,
        api_url: Optional[str] = None,
        max_vacancies: Optional[int] = None,
        page_limit: Optional[int] = None,
        output_path: Optional[str | Path] = None,
    ) -> FullReport:
        """
        Рабочий режим: загружает вакансии региона 39 через API «Работа в России»,
        обрабатывает минимум 2 страницы (если лимит и объём позволяют) и сохраняет отчёт.
        При недоступности API сообщает об ошибке и сохраняет последний успешный отчёт.
        """
        url = api_url or self.config.api_url
        max_v = max_vacancies if max_vacancies is not None else self.config.max_vacancies
        p_limit = page_limit if page_limit is not None else self.config.page_limit
        out_file = Path(output_path or self.config.report_output_path)

        try:
            vacancies, pages_processed = fetch_vacancies_from_api(
                api_url=url,
                max_vacancies=max_v,
                page_limit=p_limit,
                timeout_seconds=self.config.api_timeout_seconds,
            )
        except APIFetchError as exc:
            preserved = self.report_manager.handle_api_failure(out_file, str(exc))
            if preserved is not None:
                return preserved
            raise

        existing_cache = self.report_manager.get_cached_items_by_id(
            out_file, expected_ai_provider=self.ai_engine.provider_label
        )
        report = self.analyze_vacancies_list(
            vacancies=vacancies,
            mode="live",
            source=url,
            pages_processed=pages_processed,
            existing_cache=existing_cache,
        )
        self.report_manager.save_report(report, out_file, update_last_successful=True)
        return report
