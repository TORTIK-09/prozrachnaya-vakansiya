"""Веб-панель: просмотр отчёта, запуск проверок, сценарий с резюме, статистика. Запуск: `python -m pv web`.

Панель рассчитана на работу на своём компьютере: слушает 127.0.0.1, принимает запросы только с собственного
адреса и не принимает путей к файлам — только содержимое загруженного JSON.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import calibrate, cli, config as config_mod, report, source
from .analyze import profile, rules
from .config import ROOT
from .extra import resume as resume_mod

STATIC = Path(__file__).resolve().parent / "static"
SAMPLE = ROOT / "examples" / "sample_vacancies.json"
MAX_UPLOAD = 20 * 1024 * 1024
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


class Progress:
    """Ход текущей проверки для показа в панели в реальном времени."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data: dict = {"job": 0, "state": "idle"}
        self.entries: list[dict] = []

    def begin(self, kind: str, src: str) -> int | None:
        with self.lock:
            if self.data.get("state") == "running":
                return None
            self.entries = []
            self.final: dict[int, dict] = {}
            self.data = {"job": self.data["job"] + 1, "state": "running", "kind": kind, "source": src,
                         "phase": "Подготовка", "done": 0, "total": 0, "with_findings": 0, "findings": 0,
                         "awaiting_model": False, "error": None}
            return self.data["job"]

    def phase(self, text: str) -> None:
        with self.lock:
            self.data["phase"] = text[:160]

    def entry(self, index: int, entry: dict, total: int, pending: bool = False) -> None:
        """Готовая запись вакансии. pending — предварительная: правила уже отработали, ответа модели ещё нет."""
        with self.lock:
            self.entries.append({**entry, "_i": index, "pending": pending})
            if pending:
                self.data.update(total=total, awaiting_model=True, phase="Правила отработали, вакансии проверяет модель")
                return
            self.final[index] = entry
            vals = self.final.values()
            self.data.update(done=len(self.final), total=total, phase="Разбор вакансий",
                             with_findings=sum(1 for e in vals if e.get("findings")),
                             findings=sum(len(e.get("findings") or []) for e in vals))

    def pending(self, index: int, entry: dict, total: int) -> None:
        self.entry(index, entry, total, pending=True)

    def finish(self, rep: dict | None, error: str | None) -> None:
        with self.lock:
            self.data.update(state="failed" if error else "done", error=error, phase="Готово" if not error else "Сбой")

    def snapshot(self, since: int) -> dict:
        with self.lock:
            return {**self.data, "entries": self.entries[since:] if self.data.get("state") == "running" else []}


def create_app(cfg: config_mod.Config | None = None, auto_calibrate: bool = True) -> FastAPI:
    cfg = cfg or config_mod.load()
    app = FastAPI(title="Прозрачная вакансия", docs_url=None, redoc_url=None, openapi_url=None)
    lock = threading.Lock()                 # проверки идут по одной: модель и кэш общие
    state: dict = {"llm": None, "llm_ready": False}
    job = calibrate.Job(cfg.reports)
    progress = Progress()

    def llm():
        if not state["llm_ready"]:
            state["llm"] = cli.make_llm(cfg, True)
            state["llm_ready"] = True
        return state["llm"]

    def model_profile():
        model = llm()
        return profile.load(model.describe(), cfg.reports) if model else None

    def calibration() -> dict:
        """Состояние настройки модели. Незнакомая модель настраивается сама при первом обращении."""
        model = llm()
        if model is None:
            return {"state": "no_model"}
        st = job.status()
        if st["state"] in ("running", "failed", "done"):
            return st
        prof = model_profile()
        if prof is not None:
            return {"state": "ready", "model": prof.get("model"), "finished_at": prof.get("calibrated_at"),
                    "shipped": bool(prof.get("_shipped")), "blocked": len(prof.get("blocked", []))}
        if auto_calibrate:
            job.start(model)
            return job.status()
        return {"state": "needed", "model": model.describe()}

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0]
        if host not in LOCAL_HOSTS:
            return JSONResponse({"detail": "Панель доступна только с этого компьютера"}, status_code=403)
        if request.method not in ("GET", "HEAD"):
            origin = request.headers.get("origin")
            if origin and origin.split("://", 1)[-1].rsplit(":", 1)[0] not in LOCAL_HOSTS:
                return JSONResponse({"detail": "Запрос с чужого сайта отклонён"}, status_code=403)
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"     # после обновления сервиса браузер берёт свежие файлы
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; frame-ancestors 'none'")
        return response

    verdict_lock = threading.Lock()

    def verdicts() -> dict:
        """Решения человека там, где правило и ИИ разошлись: ключ «вакансия|тема|вид» → чей ответ принят."""
        try:
            data = json.loads((cfg.reports / "verdicts.json").read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (ValueError, OSError):
            return {}

    def with_verdicts(rep: dict) -> dict:
        saved = verdicts()
        if saved:
            for v in rep.get("vacancies", []):
                for f in v.get("findings", []):
                    hit = saved.get(f"{v.get('id')}|{f.get('topic')}|{f.get('kind')}")
                    if hit:
                        f["verdict"] = hit["choice"]
        return rep

    def read_report(kind: str) -> dict | None:
        path = cfg.reports / ("latest.json" if kind == "latest" else "check.json")
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def read_sample_report() -> dict | None:
        """Сохранённый отчёт по учебному набору — основа, которую не трогает загрузка пользовательского JSON."""
        p = cfg.reports / "check-sample.json"
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                return None
        cur = read_report("check")
        if cur and cur.get("source") == SAMPLE.name and not cur.get("imported_source"):
            report.write_json(p, cur)
            return cur
        return None

    def merge_with_sample(imported_rep: dict, sample_rep: dict) -> dict:
        """Объединяет загруженный JSON с сохранённым учебным набором без повторного анализа учебного набора."""
        imp_vacancies = list(imported_rep.get("vacancies") or [])
        imp_ids = [str(v.get("id")) for v in imp_vacancies]
        imp_set = set(imp_ids)
        base_vacancies = [v for v in (sample_rep.get("vacancies") or []) if str(v.get("id")) not in imp_set]
        merged = imp_vacancies + base_vacancies
        stats = dict(imported_rep.get("stats") or {})
        stats.update(
            vacancies=len(merged),
            with_findings=sum(1 for v in merged if v.get("findings")),
            findings=sum(len(v.get("findings") or []) for v in merged),
            imported=len(imp_vacancies),
        )
        src = str(imported_rep.get("source") or "")
        return {
            **imported_rep,
            "source": f"{SAMPLE.name} + {src}" if src else SAMPLE.name,
            "imported_source": src or "загруженный файл",
            "imported_ids": imp_ids,
            "stats": stats,
            "vacancies": merged,
        }

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/stats")
    def stats_page():
        return FileResponse(STATIC / "stats.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/status")
    def status():
        model = llm()
        info = (model.describe() if model else None)
        error = None
        err_path = cfg.reports / "last_error.json"
        if err_path.is_file():
            try:
                error = json.loads(err_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                error = None
        return {"source_url": cfg.source_url, "schedule": cfg.schedule, "max_vacancies": cfg.max_vacancies,
                "model": info, "has_latest": (cfg.reports / "latest.json").is_file(), "last_error": error,
                "calibration": calibration()}

    @app.get("/api/calibration")
    def get_calibration():
        return calibration()

    @app.post("/api/calibration")
    def start_calibration():
        """Настроить модель заново."""
        model = llm()
        if model is None:
            raise HTTPException(409, "Модель не подключена")
        job.start(model)
        return job.status()

    @app.get("/api/stats")
    def stats():
        """Сравнение способов работы: только правила, только модель, вместе."""
        prof = model_profile()
        data = prof if prof is not None else calibrate.rules_only()
        saved = verdicts().values()
        human = {"rule": sum(1 for x in saved if x.get("choice") == "rule"),
                 "llm": sum(1 for x in saved if x.get("choice") == "llm")}
        return {"stats": data, "calibration": calibration(), "verdicts": human}

    @app.get("/api/report")
    def get_report(kind: str = "check"):
        if kind not in ("latest", "check"):
            raise HTTPException(422, "kind: latest или check")
        rep = read_report(kind)
        if rep is None:
            raise HTTPException(404, "Такого отчёта ещё нет")
        return with_verdicts(rep)

    @app.post("/api/verdict")
    def set_verdict(vacancy_id: str, topic: str, kind: str, choice: str, source: str = ""):
        """Человек решает, чей ответ принять при расхождении: choice = rule, llm или none (снять решение)."""
        if choice not in ("rule", "llm", "none") or len(vacancy_id) > 200 or len(topic) > 40 or len(kind) > 40:
            raise HTTPException(422, "choice: rule, llm или none")
        key = f"{vacancy_id}|{topic}|{kind}"
        with verdict_lock:
            saved = verdicts()
            if choice == "none":
                saved.pop(key, None)
            else:
                saved[key] = {"choice": choice, "source": source[:10],
                              "at": datetime.now().astimezone().isoformat(timespec="seconds")}
            report.write_json(cfg.reports / "verdicts.json", saved)
        return {"ok": True, "choice": None if choice == "none" else choice}

    def preview(raw: list) -> None:
        """Мгновенный предварительный результат одних правил — пока модель думает, страница уже не пустая."""
        report.analyze_all(raw, lambda v, m=None: rules.check(v), None, cfg.workers, None, "", cli.log, progress.pending)

    def work_check(raw: list, src: str, is_sample: bool = False):
        def fn(on_entry, on_log):
            analyzer, _ = cli.get_analyzer()
            model = llm()
            if model is not None:
                preview(raw)
            entries, stats = report.analyze_all(raw, analyzer, model, cfg.workers, None, "", cli.log, on_entry)
            rep = report.build(entries, stats, "file", src, model)
            if is_sample:
                report.write_json(cfg.reports / "check-sample.json", rep)
                report.write_json(cfg.reports / "check.json", rep)
                return rep, None
            sample_rep = read_sample_report()
            if sample_rep is not None:
                rep = merge_with_sample(rep, sample_rep)
            report.write_json(cfg.reports / "check.json", rep)
            return rep, None
        return fn

    def work_run(run_cfg: config_mod.Config):
        def fn(on_entry, on_log):
            code = cli.cmd_run(run_cfg, True, on_log=on_log, on_entry=on_entry,
                               on_raw=preview if llm() is not None else None)
            rep = read_report("latest")
            if code == 0:
                return rep, None
            err = {}
            try:
                err = json.loads((cfg.reports / "last_error.json").read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
            return rep, err.get("error", "Источник недоступен")
        return fn

    def launch(kind: str, src: str, fn, background: bool):
        """Запуск проверки. В фоне — сразу отвечаем, а страница следит за ходом через /api/progress."""
        job_id = progress.begin(kind, src)
        if job_id is None:
            raise HTTPException(409, "Проверка уже идёт — дождитесь её окончания")

        def run() -> tuple[dict | None, str | None]:
            try:
                with lock:
                    rep, error = fn(progress.entry, progress.phase)
            except Exception as e:
                rep, error = None, f"{type(e).__name__}: {e}"[:300]
            progress.finish(rep, error)
            return rep, error

        if background:
            threading.Thread(target=run, name="check", daemon=True).start()
            return JSONResponse(progress.snapshot(0), status_code=202)
        rep, error = run()
        if error:
            return JSONResponse({"detail": error, "report": rep}, status_code=502)
        return rep

    @app.get("/api/progress")
    def get_progress(since: int = 0):
        """Ход текущей проверки: сколько сделано и записи вакансий, готовые после номера `since`."""
        return progress.snapshot(max(0, since))

    @app.post("/api/check-sample")
    def check_sample(background: bool = False, restore: bool = False):
        cur = read_report("check")
        sample_rep = read_sample_report()
        if sample_rep is not None and (restore or (cur and cur.get("imported_source"))):
            if progress.snapshot(0).get("state") == "running":
                raise HTTPException(409, "Проверка уже идёт — дождитесь её окончания")
            with lock:
                report.write_json(cfg.reports / "check.json", sample_rep)
            return with_verdicts(sample_rep)
        return launch("check", SAMPLE.name, work_check(source.load_file(SAMPLE), SAMPLE.name, is_sample=True), background)

    @app.post("/api/check")
    async def check_upload(request: Request, name: str = "загруженный файл", background: bool = False):
        body = await request.body()
        if len(body) > MAX_UPLOAD:
            raise HTTPException(413, "Файл больше 20 МБ")
        try:
            raw = source.extract(json.loads(body.decode("utf-8-sig")))
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(422, "Файл не является корректным JSON")
        except source.SourceError as e:
            raise HTTPException(422, str(e))
        if not raw:
            raise HTTPException(422, "В файле не найдено ни одной вакансии")
        if background:
            return launch("check", name[:120], work_check(raw, name[:120]), True)
        # разбор может идти долго: не держим им цикл обработки запросов
        return await run_in_threadpool(launch, "check", name[:120], work_check(raw, name[:120]), False)

    @app.post("/api/run")
    def run_live(max_vacancies: int | None = None, background: bool = False):
        """Ручной запуск рабочего режима — тот же код, что по расписанию."""
        run_cfg = cfg
        if max_vacancies is not None:
            if not 1 <= max_vacancies <= 5000:
                raise HTTPException(422, "Число вакансий: от 1 до 5000")
            run_cfg = replace(cfg, max_vacancies=max_vacancies)
        return launch("latest", cfg.source_url, work_run(run_cfg), background)

    @app.get("/api/resumes")
    def resumes():
        return resume_mod.load_resumes()

    @app.post("/api/resume")
    def make_resume(resume_id: str, vacancy_id: str, kind: str = "check"):
        rep = read_report(kind if kind in ("latest", "check") else "check")
        if rep is None:
            raise HTTPException(404, "Сначала соберите отчёт")
        entry = next((v for v in rep.get("vacancies", []) if str(v.get("id")) == vacancy_id), None)
        res = next((r for r in resume_mod.load_resumes() if r["id"] == resume_id), None)
        if entry is None or res is None:
            raise HTTPException(404, "Вакансия или резюме не найдены")
        return resume_mod.draft(res, entry)

    return app


def app_factory() -> FastAPI:
    return create_app()
