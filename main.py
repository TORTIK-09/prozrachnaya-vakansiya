import argparse
import json
import logging
from pathlib import Path
import sys
import uvicorn
from config import get_config
from src.analyzer import VacancyAnalyzer
from src.api_client import APIFetchError, load_vacancies_from_file
from src.models import FullReport
from src.resume_adapter import ResumeAdapter, load_sample_resumes
from src.scheduler import VacancyScheduler


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_report_summary(report: FullReport, output_path: str | Path) -> None:
    print("\n" + "=" * 78)
    print(" ОТЧЁТ СЕРВИСА «ПРОЗРАЧНАЯ ВАКАНСИЯ» (КАНТКОД • РЕГИОН 39)")
    print("=" * 78)
    print(f" Режим запуска       : {report.mode}")
    print(f" Источник данных     : {report.source}")
    print(f" ИИ-модуль           : {report.ai_provider_used}")
    print(f" Обработано страниц  : {report.pages_processed}")
    print(f" Всего вакансий (ID) : {report.total_vacancies}")
    print(f" Прозрачных (0 зам.) : {report.clean_vacancies}")
    print(f" С замечаниями       : {report.vacancies_with_issues}")
    print(f" Всего замечаний     : {report.total_issues}")
    print(f" Файл отчёта         : {output_path}")
    print("-" * 78)
    for item in report.vacancies:
        status = (
            "OK (0 замечаний)"
            if item.is_transparent
            else f"Замечаний: {len(item.issues)} (индекс: {item.transparency_score}%)"
        )
        print(f" [{item.id}] {item.job_name:<36} | {status}")
    print("=" * 78 + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Сервис «Прозрачная вакансия» (КАНТКОД) — аудит понятности условий вакансий региона 39"
    )
    parser.add_argument(
        "--ai-provider",
        type=str,
        default=None,
        help="ИИ-провайдер: auto, ollama, openrouter, gemini, openai, local_ai",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Подробный вывод логов"
    )

    subparsers = parser.add_subparsers(dest="command")

    # 1. Проверочный режим (локальный JSON)
    verify_parser = subparsers.add_parser(
        "verify",
        help="Проверочный режим: анализ локального JSON в формате ответов API без обращения к порталу",
    )
    verify_parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Путь к входному JSON-файлу (по умолчанию kantkod_sample_vacancies.json)",
    )
    verify_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Путь для сохранения JSON-отчёта (по умолчанию reports/sample_report.json)",
    )

    # 2. Рабочий режим (ручной запуск загрузки из API региона 39)
    live_parser = subparsers.add_parser(
        "live",
        help="Рабочий режим (ручной запуск): загрузка вакансий через API региона 39 и формирование отчёта",
    )
    live_parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL API источника (по умолчанию http://opendata.trudvsem.ru/api/v1/vacancies/region/39)",
    )
    live_parser.add_argument(
        "--max-vacancies",
        type=int,
        default=None,
        help="Максимальное число вакансий для обработки (по умолчанию 20)",
    )
    live_parser.add_argument(
        "--page-limit",
        type=int,
        default=None,
        help="Размер одной страницы запроса (по умолчанию 10, для обработки >= 2 страниц)",
    )
    live_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Путь для сохранения отчёта (по умолчанию reports/latest_report.json)",
    )

    # 3. Автоматический запуск по расписанию
    sched_parser = subparsers.add_parser(
        "schedule",
        help="Рабочий режим по расписанию: автоматическая загрузка из API региона 39 и обновление отчёта",
    )
    sched_parser.add_argument(
        "--interval-minutes",
        type=int,
        default=None,
        help="Интервал запуска в минутах (по умолчанию из SCHEDULE_INTERVAL_MINUTES)",
    )
    sched_parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Ограничить число циклов (для тестирования)",
    )
    sched_parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Переопределить URL API источника",
    )
    sched_parser.add_argument(
        "--max-vacancies",
        type=int,
        default=None,
        help="Переопределить максимальное число вакансий",
    )

    # 4. Дополнительный сценарий: адаптация черновика резюме под вакансию
    resume_parser = subparsers.add_parser(
        "resume",
        help="Дополнительный сценарий: подготовка черновика резюме под выбранную вакансию региона 39",
    )
    resume_parser.add_argument(
        "--resume-id",
        type=str,
        default="RES-39-01",
        help="ID обезличенного резюме из data/sample_resumes.json (RES-39-01, RES-39-02, RES-39-03)",
    )
    resume_parser.add_argument(
        "--vacancy-id",
        type=str,
        default="DEMO-39-001",
        help="ID вакансии (например, DEMO-39-001, DEMO-39-003, DEMO-39-007)",
    )
    resume_parser.add_argument(
        "--vacancies-file",
        type=str,
        default=None,
        help="Путь к JSON с вакансиями (по умолчанию kantkod_sample_vacancies.json)",
    )
    resume_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/sample_resume_draft_output.json",
        help="Путь для сохранения результата адаптации резюме",
    )

    # 5. Веб-интерфейс
    web_parser = subparsers.add_parser(
        "web",
        help="Запуск интерактивного веб-интерфейса (дашборда) для демонстрации",
    )
    web_parser.add_argument("--host", type=str, default=None)
    web_parser.add_argument("--port", type=int, default=None)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    config = get_config()
    if args.ai_provider:
        config.ai_provider = args.ai_provider.lower()

    command = args.command or "verify"

    if command == "verify":
        input_path = getattr(args, "input", None) or config.sample_input_path
        output_path = getattr(args, "output", None) or config.sample_report_output_path
        analyzer = VacancyAnalyzer(config)
        report = analyzer.run_verify_mode(input_path=input_path, output_path=output_path)
        print_report_summary(report, output_path)
        return 0

    if command == "live":
        if args.url:
            config.api_url = args.url
        if args.max_vacancies:
            config.max_vacancies = args.max_vacancies
        if args.page_limit:
            config.page_limit = args.page_limit
        output_path = args.output or config.report_output_path

        analyzer = VacancyAnalyzer(config)
        try:
            report = analyzer.run_live_mode(
                api_url=config.api_url,
                max_vacancies=config.max_vacancies,
                page_limit=config.page_limit,
                output_path=output_path,
            )
            print_report_summary(report, output_path)
            return 0
        except APIFetchError as exc:
            print(
                f"\n[ОШИБКА API] {exc}\nПоследний успешный отчёт сохранён (если присутствовал).",
                file=sys.stderr,
            )
            return 1

    if command == "schedule":
        if args.url:
            config.api_url = args.url
        if args.max_vacancies:
            config.max_vacancies = args.max_vacancies
        scheduler = VacancyScheduler(config)
        scheduler.start_loop(
            interval_minutes=args.interval_minutes,
            max_runs=args.max_runs,
        )
        return 0

    if command == "resume":
        vac_file = args.vacancies_file or config.sample_input_path
        resumes = load_sample_resumes()
        vacancies, _ = load_vacancies_from_file(vac_file)

        selected_resume = next(
            (r for r in resumes if r.resume_id == args.resume_id),
            resumes[0] if resumes else None,
        )
        selected_vac = next(
            (v for v in vacancies if str(v.get("id")) == args.vacancy_id),
            vacancies[0] if vacancies else None,
        )
        if not selected_resume or not selected_vac:
            print("Не удалось найти выбранное резюме или вакансию.", file=sys.stderr)
            return 1

        adapter = ResumeAdapter(config)
        result = adapter.adapt_resume_for_vacancy(selected_resume, selected_vac)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 78)
        print(" ДОПОЛНИТЕЛЬНЫЙ СЦЕНАРИЙ: ЧЕРНОВИК РЕЗЮМЕ ПОД ВАКАНСИЮ РЕГИОНА 39")
        print("=" * 78)
        print(f" Вакансия        : [{result.vacancy_id}] {result.vacancy_title}")
        print(f" Соискатель      : {result.candidate_alias}")
        print(f" Совместимость   : {result.match_score}% ({result.match_summary})")
        print(f" Файл результата : {out_path}")
        print("=" * 78 + "\n")
        return 0

    if command == "web":
        host = args.host or config.web_host
        port = args.port or config.web_port
        print(f"Запуск веб-интерфейса «Прозрачная вакансия» на http://{host}:{port}")
        uvicorn.run("src.web_app:app", host=host, port=port, reload=False)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
