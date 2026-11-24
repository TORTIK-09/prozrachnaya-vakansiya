"""Веб-панель: просмотр отчёта, запуск проверок, сценарий с резюме. Запуск: `python -m pv web`.

Панель рассчитана на работу на своём компьютере: слушает 127.0.0.1, принимает запросы только с собственного
адреса и не принимает путей к файлам — только содержимое загруженного JSON.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from . import cli, config as config_mod, report, source
from .config import ROOT
from .extra import resume as resume_mod

STATIC = Path(__file__).resolve().parent / "static"
SAMPLE = ROOT / "examples" / "sample_vacancies.json"
MAX_UPLOAD = 20 * 1024 * 1024
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def create_app(cfg: config_mod.Config | None = None) -> FastAPI:
    cfg = cfg or config_mod.load()
    app = FastAPI(title="Прозрачная вакансия", docs_url=None, redoc_url=None, openapi_url=None)
    lock = threading.Lock()                 # проверки идут по одной: модель и кэш общие
    state: dict = {"llm": None, "llm_ready": False}

    def llm():
        if not state["llm_ready"]:
            state["llm"] = cli.make_llm(cfg, True)
            state["llm_ready"] = True
        return state["llm"]

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
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; frame-ancestors 'none'")
        return response

    def analyze(raw: list, mode: str, src: str) -> dict:
        analyzer, version = cli.get_analyzer()
        model = llm()
        entries, stats = report.analyze_all(raw, analyzer, model, cfg.workers, None, "", cli.log)
        return report.build(entries, stats, mode, src, model)

    def read_report(kind: str) -> dict | None:
        path = cfg.reports / ("latest.json" if kind == "latest" else "check.json")
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

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
                "model": info, "has_latest": (cfg.reports / "latest.json").is_file(), "last_error": error}

    @app.get("/api/report")
    def get_report(kind: str = "check"):
        if kind not in ("latest", "check"):
            raise HTTPException(422, "kind: latest или check")
        rep = read_report(kind)
        if rep is None:
            raise HTTPException(404, "Такого отчёта ещё нет")
        return rep

    @app.post("/api/check-sample")
    def check_sample():
        with lock:
            rep = analyze(source.load_file(SAMPLE), "file", SAMPLE.name)
            report.write_json(cfg.reports / "check.json", rep)
        return rep

    @app.post("/api/check")
    async def check_upload(request: Request, name: str = "загруженный файл"):
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
        with lock:
            rep = analyze(raw, "file", name[:120])
            report.write_json(cfg.reports / "check.json", rep)
        return rep

    @app.post("/api/run")
    def run_live(max_vacancies: int | None = None):
        """Ручной запуск рабочего режима — тот же код, что по расписанию."""
        from dataclasses import replace
        run_cfg = cfg
        if max_vacancies is not None:
            if not 1 <= max_vacancies <= 5000:
                raise HTTPException(422, "Число вакансий: от 1 до 5000")
            run_cfg = replace(cfg, max_vacancies=max_vacancies)
        with lock:
            code = cli.cmd_run(run_cfg, True)
        rep = read_report("latest")
        if code != 0:
            err = {}
            try:
                err = json.loads((cfg.reports / "last_error.json").read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
            return JSONResponse({"detail": err.get("error", "Источник недоступен"), "report": rep}, status_code=502)
        return rep

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
