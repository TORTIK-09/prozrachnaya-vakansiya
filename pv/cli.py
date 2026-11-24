"""Команды: run (рабочий режим), check (проверочный режим), schedule (по расписанию).

Обработка начинается здесь: `python -m pv ...` → main().
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from . import config as config_mod
from . import cron, report, source
from .models import LLM


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def get_analyzer():
    """Анализатор и его версия. Версия входит в отпечаток кэша: сменился анализатор — считаем заново."""
    from . import analyze as mod
    return mod.analyze, str(getattr(mod, "VERSION", "0"))


def make_llm(cfg: config_mod.Config, enabled: bool) -> LLM | None:
    """Клиент модели из настроек. Нет модуля, настроек или связи — работаем на одних правилах и говорим об этом."""
    if not enabled:
        log("модель отключена флагом --no-llm: работают только правила")
        return None
    try:
        from . import llm as llm_mod
    except ImportError:
        log("клиент модели (pv/llm.py) ещё не подключён: работают только правила")
        return None
    try:
        client = llm_mod.from_config(cfg)
    except Exception as e:
        log(f"модель недоступна ({type(e).__name__}: {e}): работают только правила")
        return None
    if client is None:
        log("модель не подключена: работают только правила")
        return None
    from .analyze import profile
    info = client.describe()
    if profile.load(info, cfg.reports) is None:
        log(f"модель {info.get('model')} сервису незнакома: её замечания берутся без настройки. "
            "Настроить: python -m pv calibrate (панель делает это сама)")
    return client


def _salt(version: str, llm: LLM | None) -> str:
    if llm is None:
        return f"{version}|no-llm"
    try:
        info = llm.describe() or {}
    except Exception:
        info = {}
    return f"{version}|{info.get('provider')}|{info.get('model')}"


def _summary(rep: dict, path: Path) -> None:
    s = rep["stats"]
    log(f"вакансий: {s['vacancies']}, с замечаниями: {s['with_findings']}, замечаний: {s['findings']}"
        + (f", из кэша: {s['from_cache']}" if s["from_cache"] else "")
        + (f", отброшено замечаний: {s['rejected_findings']}" if s["rejected_findings"] else "")
        + (f", сбоев анализатора: {s['analyzer_errors']}" if s["analyzer_errors"] else ""))
    log(f"отчёт: {path}")


def cmd_run(cfg: config_mod.Config, use_llm: bool = True, on_log=None, on_entry=None, on_raw=None) -> int:
    """Рабочий режим: API → анализ → reports/latest.json. Этот же код запускает расписание."""
    log(f"источник: {cfg.source_url}, не больше {cfg.max_vacancies} вакансий")
    try:
        raw = source.fetch_api(cfg, log=(lambda m: (log(m), on_log(m))) if on_log else log)
    except source.SourceError as e:
        path = report.save_error(str(e), cfg.source_url, cfg.reports)
        latest = cfg.reports / "latest.json"
        log(f"ОШИБКА: {e}")
        log(f"последний успешный отчёт сохранён: {latest}" if latest.is_file() else "успешных отчётов ещё не было")
        log(f"сведения об ошибке: {path}")
        return 2
    if not raw:
        path = report.save_error("Источник вернул пустой список вакансий", cfg.source_url, cfg.reports)
        log(f"ОШИБКА: источник вернул пустой список вакансий; прежний отчёт не тронут; сведения: {path}")
        return 2
    if on_raw is not None:
        on_raw(raw)
    analyze, version = get_analyzer()
    llm = make_llm(cfg, use_llm)
    cache = report.Cache(cfg.reports / "cache.json")
    entries, stats = report.analyze_all(raw, analyze, llm, cfg.workers, cache, _salt(version, llm), log, on_entry)
    rep = report.build(entries, stats, "api", cfg.source_url, llm)
    _summary(rep, report.save_latest(rep, cfg.reports))
    return 0


def cmd_check(cfg: config_mod.Config, file: str, out: str, use_llm: bool = True, use_cache: bool = False) -> int:
    """Проверочный режим: локальный JSON → отчёт по всем переданным вакансиям. В сеть за вакансиями не ходит."""
    try:
        raw = source.load_file(file)
    except source.SourceError as e:
        log(f"ОШИБКА: {e}")
        return 2
    analyze, version = get_analyzer()
    llm = make_llm(cfg, use_llm)
    cache = report.Cache(cfg.reports / "cache-check.json") if use_cache else None
    entries, stats = report.analyze_all(raw, analyze, llm, cfg.workers, cache, _salt(version, llm), log)
    rep = report.build(entries, stats, "file", Path(file).name, llm)
    out_path = Path(out)
    report.write_json(out_path, rep)
    _summary(rep, out_path)
    return 0


def cmd_schedule(cfg: config_mod.Config, use_llm: bool = True) -> int:
    """Работает постоянно и запускает рабочий режим по cron-выражению из настроек."""
    try:
        cron.parse(cfg.schedule)
    except ValueError as e:
        log(f"ОШИБКА в расписании «{cfg.schedule}»: {e}")
        return 2
    log(f"расписание: {cfg.schedule} (остановить — Ctrl+C)")
    try:
        while True:
            nxt = cron.next_run(cfg.schedule, datetime.now())
            log(f"следующий запуск: {nxt:%d.%m.%Y %H:%M}")
            while datetime.now() < nxt:
                time.sleep(min(30.0, max(0.5, (nxt - datetime.now()).total_seconds())))
            try:
                cmd_run(cfg, use_llm)
            except Exception as e:   # расписание переживает сбой отдельного запуска
                log(f"ОШИБКА запуска: {type(e).__name__}: {e}")
    except KeyboardInterrupt:
        log("остановлено")
        return 0


def cmd_web(cfg: config_mod.Config, port: int | None = None) -> int:
    """Веб-панель: просмотр отчёта, запуск проверок, сценарий с резюме."""
    import socket

    import uvicorn

    from . import web
    port = port or cfg.web_port
    with socket.socket() as s:
        if s.connect_ex((cfg.web_host, port)) == 0:
            log(f"ОШИБКА: порт {port} уже занят. Запустите с другим: python -m pv web --port {port + 1}")
            return 2
    log(f"панель: http://{cfg.web_host}:{port}  (остановить — Ctrl+C)")
    uvicorn.run(web.create_app(cfg), host=cfg.web_host, port=port, log_level="warning")
    return 0


def cmd_calibrate(cfg: config_mod.Config) -> int:
    """Настройка модели под задачу и сравнение способов работы (см. pv/calibrate.py)."""
    from . import calibrate
    llm = make_llm(cfg, True)
    if llm is None:
        log("ОШИБКА: модель не подключена — настраивать нечего")
        return 2
    info = llm.describe()
    log(f"настройка модели {info.get('model')} ({info.get('provider')}): один запрос на каждую размеченную вакансию")

    def progress(done: int, total: int, sec: float) -> None:
        log(f"  {done} из {total}, осталось примерно {round(sec * (total - done) / 60)} мин")

    result = calibrate.run(llm, progress)
    if result["errors"] >= result["vacancies"]:
        log("ОШИБКА: модель не ответила ни на один запрос")
        return 2
    path = calibrate.save(result, cfg.reports)
    for st in result["sets"]:
        m = st["modes"]
        log(f"{st['title']}: правила {m['rules']['must_found']}/{m['rules']['must_total']}, "
            f"модель {m['llm']['must_found']}/{m['llm']['must_total']}, вместе {m['both']['must_found']}/{m['both']['must_total']}; "
            f"лишних: {m['rules']['extra_total']} / {m['llm']['extra_total']} / {m['both']['extra_total']}")
    log(f"отключено видов замечаний модели: {len(result['blocked'])}; профиль: {path}")
    return 0


def cmd_learn(cfg: config_mod.Config, count: int) -> int:
    """Пересчитать нормы опыта по должностям на свежих вакансиях портала (pv/analyze/norms.py)."""
    from dataclasses import replace as _replace

    from .analyze import norms
    from .normalize import normalize
    try:
        raw = source.fetch_api(_replace(cfg, max_vacancies=count), log=log)
    except source.SourceError as e:
        log(f"ОШИБКА: {e}")
        return 2
    table = norms.build(v for v in map(normalize, raw) if v)
    path = norms.local_path(cfg.reports)
    report.write_json(path, table)
    norms.forget()
    log(f"нормы опыта: {len(table['groups'])} групп должностей по {table['vacancies']} вакансиям; файл: {path}")
    return 0


def cmd_forget(cfg: config_mod.Config) -> int:
    """Удалить всё, что сервис накопил на этом компьютере: профили настройки моделей, выученные нормы, кэш."""
    import shutil
    removed = []
    for p in (cfg.reports / "profiles", cfg.reports / "experience_norms.json", cfg.reports / "cache.json",
              cfg.reports / "cache-check.json"):
        if p.is_dir():
            shutil.rmtree(p)
            removed.append(p.name)
        elif p.is_file():
            p.unlink()
            removed.append(p.name)
    log("удалено: " + (", ".join(removed) if removed else "нечего удалять"))
    log("веса модели сервис не менял, поэтому в самой модели удалять нечего; "
        "данные, уже отправленные облачному провайдеру, отозвать нельзя — см. docs/spec/06-model-and-data.md")
    return 0


def cmd_resume(cfg: config_mod.Config, resume_id: str, vacancy_id: str, report_path: str, out: str | None) -> int:
    """Свой сценарий: черновик резюме под вакансию из готового отчёта."""
    import json

    from .extra import resume as resume_mod
    try:
        rep = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log(f"ОШИБКА: не удалось прочитать отчёт {report_path}: {e}")
        return 2
    entry = next((v for v in rep.get("vacancies", []) if str(v.get("id")) == vacancy_id), None)
    res = next((r for r in resume_mod.load_resumes() if r["id"] == resume_id), None)
    if entry is None or res is None:
        log("ОШИБКА: вакансия или резюме не найдены")
        return 2
    result = resume_mod.draft(res, entry)
    if out:
        report.write_json(Path(out), result)
        log(f"черновик: {out}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="python -m pv", description="Отчёт о понятности вакансий «Работа в России»")
    ap.add_argument("--config", help="путь к config.toml (по умолчанию — в корне проекта)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="рабочий режим: загрузить вакансии из API и записать отчёт")
    p_run.add_argument("--max", type=int, dest="max_vacancies", help="максимум вакансий (иначе из настроек)")
    p_run.add_argument("--url", dest="source_url", help="URL источника (иначе из настроек)")
    p_check = sub.add_parser("check", help="проверочный режим: отчёт по локальному JSON")
    p_check.add_argument("file", help="JSON в формате ответа API")
    p_check.add_argument("--out", default="reports/check.json", help="куда записать отчёт")
    p_check.add_argument("--cache", action="store_true", help="не анализировать заново неизменившиеся вакансии")
    p_sched = sub.add_parser("schedule", help="работать постоянно и запускать рабочий режим по расписанию")
    p_web = sub.add_parser("web", help="веб-панель для просмотра отчёта")
    p_web.add_argument("--port", type=int, help="порт (иначе из настроек, по умолчанию 8085)")
    sub.add_parser("calibrate", help="настроить подключённую модель под задачу и сравнить её с правилами")
    p_learn = sub.add_parser("learn", help="пересчитать нормы опыта по должностям на вакансиях портала")
    p_learn.add_argument("--count", type=int, default=400, help="сколько вакансий взять (по умолчанию 400)")
    sub.add_parser("forget", help="удалить накопленное на этом компьютере: профили моделей, нормы, кэш")
    p_res = sub.add_parser("resume", help="свой сценарий: черновик резюме под вакансию из отчёта")
    p_res.add_argument("--resume", default="RES-01", help="ID резюме из examples/sample_resumes.json")
    p_res.add_argument("--vacancy", required=True, help="ID вакансии из отчёта")
    p_res.add_argument("--report", default="reports/check.json", help="отчёт, из которого берётся вакансия")
    p_res.add_argument("--out", help="куда записать черновик (иначе — на экран)")
    for p in (p_run, p_check, p_sched):
        p.add_argument("--no-llm", action="store_true", help="не обращаться к модели, только правила")
    a = ap.parse_args(argv)

    try:
        cfg = config_mod.load(a.config)
        over = {k: getattr(a, k) for k in ("max_vacancies", "source_url") if getattr(a, k, None) is not None}
        if over:
            cfg = replace(cfg, **over)
            if cfg.max_vacancies < 1:
                raise ValueError("--max должно быть не меньше 1")
    except (ValueError, FileNotFoundError) as e:
        log(f"ОШИБКА настроек: {e}")
        return 2

    if a.cmd == "run":
        return cmd_run(cfg, not a.no_llm)
    if a.cmd == "check":
        return cmd_check(cfg, a.file, a.out, not a.no_llm, a.cache)
    if a.cmd == "web":
        return cmd_web(cfg, a.port)
    if a.cmd == "calibrate":
        return cmd_calibrate(cfg)
    if a.cmd == "learn":
        return cmd_learn(cfg, a.count)
    if a.cmd == "forget":
        return cmd_forget(cfg)
    if a.cmd == "resume":
        return cmd_resume(cfg, a.resume, a.vacancy, a.report, a.out)
    return cmd_schedule(cfg, not a.no_llm)
