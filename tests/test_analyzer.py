import json
from pathlib import Path
from typing import Any, Dict
import unittest
from unittest.mock import MagicMock
import requests

from config import AppConfig
from src.analyzer import VacancyAnalyzer
from src.api_client import fetch_vacancies_from_api
from src.resume_adapter import ResumeAdapter, load_sample_resumes


class TestTransparentVacancy(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(ai_provider="local_ai")
        self.analyzer = VacancyAnalyzer(self.config)

    def test_verify_mode_on_sample_dataset(self) -> None:
        """Проверка учебного набора kantkod_sample_vacancies.json в проверочном режиме."""
        out_path = Path("reports/test_sample_report.json")
        if out_path.exists():
            out_path.unlink()

        report = self.analyzer.run_verify_mode(
            input_path="kantkod_sample_vacancies.json",
            output_path=out_path,
        )

        self.assertEqual(report.total_vacancies, 10)
        self.assertEqual(report.clean_vacancies, 4)
        self.assertEqual(report.vacancies_with_issues, 6)

        by_id = {v.id: v for v in report.vacancies}

        # Эталонные прозрачные вакансии должны иметь пустой массив замечаний []
        for clean_id in ("DEMO-39-001", "DEMO-39-005", "DEMO-39-007", "DEMO-39-009"):
            self.assertEqual(
                len(by_id[clean_id].issues),
                0,
                f"У вакансии {clean_id} не должно быть замечаний",
            )

        # DEMO-39-002: пропуск зарплаты должен иметь пустую цитату ""
        v002_issues = by_id["DEMO-39-002"].issues
        salary_missing = [
            i for i in v002_issues if i.topic == "Оплата" and i.type == "пропуск"
        ]
        self.assertEqual(len(salary_missing), 1)
        self.assertEqual(salary_missing[0].quote, "")

        # DEMO-39-003: противоречие по опыту (0 лет vs 5 лет)
        v003_issues = by_id["DEMO-39-003"].issues
        exp_contra = [
            i for i in v003_issues if i.topic == "Требования" and i.type == "противоречие"
        ]
        self.assertEqual(len(exp_contra), 1)
        self.assertIn("0", exp_contra[0].quote)
        self.assertIn("5 лет", exp_contra[0].quote)

        # DEMO-39-004: противоречие дистанционной работы и присутствия в офисе
        v004_issues = by_id["DEMO-39-004"].issues
        loc_contra = [
            i for i in v004_issues if i.topic == "Место работы" and i.type == "противоречие"
        ]
        self.assertEqual(len(loc_contra), 1)
        self.assertIn("дистанционная", loc_contra[0].quote)
        self.assertIn("в офисе", loc_contra[0].quote)

        # DEMO-39-006: общее название, противоречие зарплаты 35000 vs 0
        v006_issues = by_id["DEMO-39-006"].issues
        topics_006 = {i.topic for i in v006_issues}
        self.assertIn("Название должности", topics_006)
        self.assertIn("Оплата", topics_006)

        # DEMO-39-008: противоречие гибкого графика и пятидневки 09:00-18:00
        v008_issues = by_id["DEMO-39-008"].issues
        sched_contra = [
            i for i in v008_issues if i.topic == "График" and i.type == "противоречие"
        ]
        self.assertEqual(len(sched_contra), 1)

        if out_path.exists():
            out_path.unlink()

    def test_idempotency_on_repeated_runs(self) -> None:
        """Повторный запуск на неизменных данных сохраняет ровно один результат на ID вакансии."""
        out_path = Path("reports/test_idempotent_report.json")
        if out_path.exists():
            out_path.unlink()

        report1 = self.analyzer.run_verify_mode(
            input_path="kantkod_sample_vacancies.json",
            output_path=out_path,
        )
        report2 = self.analyzer.run_verify_mode(
            input_path="kantkod_sample_vacancies.json",
            output_path=out_path,
        )

        ids1 = [v.id for v in report1.vacancies]
        ids2 = [v.id for v in report2.vacancies]
        self.assertEqual(len(ids2), 10)
        self.assertEqual(len(set(ids2)), 10)
        self.assertEqual(ids1, ids2)

        if out_path.exists():
            out_path.unlink()

    def test_pagination_at_least_two_pages(self) -> None:
        """Проверка обработки минимум двух страниц при загрузке через API."""
        with open("kantkod_sample_vacancies.json", "r", encoding="utf-8") as f:
            sample_data = json.load(f)

        all_vacs = sample_data["results"]["vacancies"]
        page_0_payload: Dict[str, Any] = {
            "status": "200",
            "meta": {"total": "10", "limit": "5", "offset": "0"},
            "results": {"vacancies": all_vacs[:5]},
        }
        page_1_payload: Dict[str, Any] = {
            "status": "200",
            "meta": {"total": "10", "limit": "5", "offset": "1"},
            "results": {"vacancies": all_vacs[5:10]},
        }

        mock_session = MagicMock()
        resp0 = MagicMock()
        resp0.status_code = 200
        resp0.json.return_value = page_0_payload
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = page_1_payload
        mock_session.get.side_effect = [resp0, resp1]

        vacancies, pages_processed = fetch_vacancies_from_api(
            api_url="http://opendata.trudvsem.ru/api/v1/vacancies/region/39",
            max_vacancies=10,
            page_limit=5,
            session=mock_session,
        )

        self.assertEqual(pages_processed, 2)
        self.assertEqual(len(vacancies), 10)

    def test_preserve_last_successful_report_on_api_failure(self) -> None:
        """При недоступности API сохраняется последний успешный отчёт."""
        out_path = Path("reports/test_preserved_report.json")
        backup_path = Path("reports/test_backup_report.json")
        cfg = AppConfig(
            ai_provider="local_ai",
            api_url="http://127.0.0.1:9/unavailable_api",
            api_timeout_seconds=1,
            last_successful_report_path=str(backup_path),
        )
        analyzer = VacancyAnalyzer(cfg)

        # Сначала создаём успешный отчёт
        initial_report = analyzer.run_verify_mode(
            input_path="kantkod_sample_vacancies.json",
            output_path=out_path,
        )
        self.assertEqual(initial_report.total_vacancies, 10)

        # Теперь запускаем рабочий режим с недоступным API
        preserved_report = analyzer.run_live_mode(output_path=out_path)
        self.assertIsNotNone(preserved_report)
        self.assertEqual(preserved_report.total_vacancies, 10)
        self.assertTrue(out_path.exists())

        if out_path.exists():
            out_path.unlink()
        if backup_path.exists():
            backup_path.unlink()

    def test_additional_scenario_resume_adapter(self) -> None:
        """Проверка дополнительного сценария: черновик резюме берёт только данные из исходного резюме."""
        resumes = load_sample_resumes()
        with open("kantkod_sample_vacancies.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        vacs = [item["vacancy"] for item in raw["results"]["vacancies"]]

        adapter = ResumeAdapter(self.config, self.analyzer)
        result = adapter.adapt_resume_for_vacancy(resumes[0], vacs[0])

        self.assertEqual(result.vacancy_id, "DEMO-39-001")
        self.assertGreaterEqual(result.match_score, 70)
        self.assertEqual(
            result.tailored_resume_draft["education"]["level"],
            resumes[0].education_level,
        )
        self.assertEqual(
            set(result.tailored_resume_draft["key_skills_prioritized"]),
            set(resumes[0].skills),
        )


if __name__ == "__main__":
    unittest.main()
