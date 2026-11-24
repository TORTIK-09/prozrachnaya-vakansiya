from datetime import datetime, timedelta
import logging
import time
from typing import Optional
from config import AppConfig
from src.analyzer import VacancyAnalyzer

logger = logging.getLogger("transparent_vacancy.scheduler")


class VacancyScheduler:
    """
    Планировщик автоматического запуска анализатора вакансий в рабочем режиме.
    Загружает вакансии через API региона 39, обрабатывает их и сохраняет отчёт по расписанию.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.analyzer = VacancyAnalyzer(config)

    def run_once(self) -> None:
        """Выполняет один цикл загрузки и анализа вакансий (используется и планировщиком, и при ручном запуске)."""
        logger.info(
            "Запуск цикла анализа (URL=%s, max_vacancies=%d, page_limit=%d)...",
            self.config.api_url,
            self.config.max_vacancies,
            self.config.page_limit,
        )
        try:
            report = self.analyzer.run_live_mode()
            logger.info(
                "Цикл завершён успешно: обработано страниц=%d, вакансий=%d, замечаний=%d. Отчёт: %s",
                report.pages_processed,
                report.total_vacancies,
                report.total_issues,
                self.config.report_output_path,
            )
        except Exception as exc:
            logger.error(
                "Цикл завершился с ошибкой API: %s. Последний успешный отчёт сохранён.",
                exc,
            )

    def start_loop(
        self,
        interval_minutes: Optional[int] = None,
        max_runs: Optional[int] = None,
    ) -> None:
        """
        Запускает непрерывный цикл по расписанию.
        Параметр max_runs позволяет ограничить число итераций при тестировании.
        """
        interval = (
            interval_minutes
            if interval_minutes is not None
            else self.config.schedule_interval_minutes
        )
        interval = max(1, interval)
        runs_count = 0

        logger.info(
            "Планировщик «Прозрачная вакансия» запущен. Интервал: каждые %d мин. (cron: %s)",
            interval,
            self.config.schedule_cron,
        )

        try:
            while True:
                runs_count += 1
                self.run_once()

                if max_runs is not None and runs_count >= max_runs:
                    logger.info(
                        "Достигнуто заданное число запусков (%d), остановка планировщика.",
                        max_runs,
                    )
                    break

                next_run = datetime.now() + timedelta(minutes=interval)
                logger.info(
                    "Следующий автоматический запуск запланирован на %s (через %d мин.)",
                    next_run.strftime("%Y-%m-%d %H:%M:%S"),
                    interval,
                )
                time.sleep(interval * 60)
        except KeyboardInterrupt:
            logger.info("Планировщик остановлен пользователем.")
