"""Веб-панель: просмотр отчёта, запуск проверок, сценарий с резюме, статистика. Запуск: `python -m pv web`.

Панель рассчитана на работу на своём компьютере: слушает 127.0.0.1, принимает запросы только с собственного
адреса и не принимает путей к файлам — только содержимое загруженного JSON.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import calibrate, cli, config as config_mod, llm as llm_mod, report, source
from .analyze import profile
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
        # модель могла быть занята или ещё не запущена: если её не нашли, пробуем снова, но не чаще раза в полминуты
        if state["llm"] is None and time.monotonic() - state.get("tried", -1e9) > 30:
            state["tried"] = time.monotonic()
            state["llm"] = cli.make_llm(cfg, True)
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
            pl = plan()
            start_calibration_job(model, pl)
            return job.status()
        return {"state": "needed", "model": model.describe()}

    def start_calibration_job(model, pl: dict) -> None:
        """Калибровка по плану: скорость меряется на первых двух вакансиях, дальше — сколько поместится во время."""
        budget = pl["effective_minutes"] * 60.0
        job.start(model, limit=min(pl["anti_template_cap"], pl["available_tests"]), learn_seconds=budget, cfg=cfg,
                  budget_seconds=budget, min_tests=min(pl["min_tests"], pl["available_tests"]))

    def modes_table() -> list[dict]:
        """Показатели пяти режимов. Числа есть только у режимов, в которых модель действительно калибровалась."""
        def total(sets: list[dict], mode_key: str) -> dict | None:
            rows = [s["modes"][mode_key] for s in sets if mode_key in s.get("modes", {})]
            if not rows:
                return None
            return {"found": sum(r["must_found"] for r in rows), "must": sum(r["must_total"] for r in rows),
                    "extra": sum(r["extra_total"] for r in rows), "vacancies": sum(r["vacancies"] for r in rows)}

        base = calibrate.rules_only()
        out = [{"mode": "rules_only", "model": None, "seconds": base["seconds_per_vacancy"]["rules"],
                **(total(base["sets"], "rules") or {})}]
        by_kind: dict[str, dict] = {}
        for prof in profile.every(cfg.reports):
            kind = prof["model"].get("kind") or ("local" if prof["model"].get("provider") in ("ollama", "local") else "custom_api")
            cur = by_kind.get(kind)
            if cur is None or (prof.get("calibrated_at") or "") > (cur.get("calibrated_at") or ""):
                by_kind[kind] = prof
        for kind in ("local", "custom_api"):
            prof = by_kind.get(kind)
            for mode_name, key in ((kind, "both"), (kind + "_only", "llm")):
                row = {"mode": mode_name, "model": prof["model"].get("model") if prof else None}
                if prof:
                    row.update(total(prof["sets"], key) or {})
                    spv = prof.get("seconds_per_vacancy") or {}
                    llm_s = spv.get("llm")
                    rules_s = spv.get("rules") or 0.0
                    row["seconds"] = round(llm_s + rules_s, 1) if (llm_s is not None and key == "both") else llm_s
                    row["calibrated_at"] = prof.get("calibrated_at")
                out.append(row)
        return out

    MODES = ("rules_only", "local", "local_only", "custom_api", "custom_api_only")

    def saved_settings() -> dict:
        return llm_mod.load_runtime_settings(cfg.reports) or {}

    def mode() -> str:
        """Текущий режим работы: только Системный AI, Системный AI + ИИ или только ИИ (локальный либо по API)."""
        model = llm()
        if model is None:
            return "rules_only"
        return str(model.describe().get("mode_type") or "local")

    def meta() -> dict | None:
        """Сила модели: число параметров и класс (слабая, средняя, сильная) — по Ollama, имени или справочнику."""
        model = llm()
        if model is None:
            return None
        info = model.describe()
        key = (info.get("base_url"), info.get("model"))
        if state.get("meta_key") != key:
            found = llm_mod.detect_model_parameters(str(info.get("model") or ""), str(info.get("base_url") or ""))
            state["meta"] = llm_mod.remember_model(cfg.reports, found)
            state["meta_key"] = key
        return state["meta"]

    def plan(minutes: float | None = None) -> dict:
        """План калибровки: сколько вакансий успеет модель за отведённое время на этом компьютере."""
        model = llm()
        info = model.describe() if model else {}
        prof = model_profile() if model else None
        measured = ((prof or {}).get("seconds_per_vacancy") or {}).get("llm")
        if minutes is None:
            minutes = float(saved_settings().get("calib_minutes") or cfg.calib_minutes or 0) or None
        pl = llm_mod.compute_calibration_plan(str(info.get("model") or ""), mode(), minutes,
                                              str(info.get("base_url") or ""), measured)
        available = sum(len(st["vacancies"]) for st in calibrate._load_sets())
        pl["available_tests"] = available
        pl["llm_target_tests"] = min(pl["llm_target_tests"], available)     # размеченных вакансий больше нет
        return pl

    def with_context(st: dict) -> dict:
        """К состоянию калибровки — сила модели, режим и ход обучения Системного AI."""
        return {**st, "meta": meta(), "mode": mode(), "learning": job.learning}

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
                        if hit.get("choice"):
                            f["verdict"] = hit["choice"]
                        f["history"] = hit.get("history", [])
        return rep

    def read_report(kind: str) -> dict | None:
        path = cfg.reports / ("latest.json" if kind == "latest" else "check.json")
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

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
                "calibration": with_context(calibration()), "mode": mode(), "meta": meta()}

    @app.get("/api/calibration")
    def get_calibration():
        return with_context(calibration())

    @app.get("/api/settings")
    def get_settings():
        """Что можно выбрать: режимы, провайдеры и модели; что выбрано; что умеет этот компьютер."""
        saved = saved_settings()
        shown = {k: v for k, v in saved.items() if k != "api_key"}
        shown["has_api_key"] = bool(saved.get("api_key"))
        try:
            local = llm_mod.list_local_models()
        except Exception:
            local = []
        return {"modes": MODES, "mode": mode(), "saved": shown, "providers": llm_mod.PROVIDER_PRESETS,
                "local_models": local, "hardware": llm_mod.detect_hardware_specs(), "meta": meta(),
                "env_locked": bool(cfg.llm_base_url or cfg.llm_model)}

    @app.get("/api/plan")
    def get_plan(model: str = "", mode_type: str = "local", base_url: str = "", minutes: float | None = None):
        """Предварительный план калибровки для выбранной модели — до сохранения настроек."""
        if minutes is not None and not 0 < minutes <= 600:
            raise HTTPException(422, "Время калибровки: от 1 до 600 минут")
        pl = llm_mod.compute_calibration_plan(model[:200], mode_type if mode_type in MODES else "local", minutes, base_url[:300])
        available = sum(len(st["vacancies"]) for st in calibrate._load_sets())
        pl["available_tests"] = available
        pl["llm_target_tests"] = min(pl["llm_target_tests"], available)
        return pl

    @app.post("/api/settings")
    async def set_settings(request: Request):
        """Сохранить выбор режима и модели. Ключ хранится только на этом компьютере, в папке reports."""
        try:
            body = json.loads((await request.body()).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(422, "Ожидается JSON")
        m = str(body.get("mode_type") or "")
        if m not in MODES:
            raise HTTPException(422, "Неизвестный режим")
        base_url, model_name = str(body.get("base_url") or "").strip()[:300], str(body.get("model") or "").strip()[:200]
        if m != "rules_only" and (not base_url.startswith(("http://", "https://")) or not model_name):
            raise HTTPException(422, "Укажите адрес сервера модели и её имя")
        prev = saved_settings()
        key = str(body.get("api_key") or "").strip()[:500]
        if not key and prev.get("api_key") and prev.get("base_url") == base_url:
            key = str(prev["api_key"])          # поле ключа оставили пустым — прежний ключ для того же сервера сохраняется
        try:
            minutes = float(body.get("calib_minutes") or 0)
        except (TypeError, ValueError):
            minutes = 0.0
        llm_mod.save_runtime_settings(cfg.reports, {
            "mode_type": m, "provider": str(body.get("provider") or "")[:40], "base_url": base_url, "model": model_name,
            "api_key": key, "calib_minutes": max(0.0, min(600.0, minutes))})
        state.update(llm=None, tried=-1e9, meta_key=None)       # модель подключается заново с новыми настройками
        profile.forget()
        return {"ok": True, "mode": mode(), "calibration": with_context(calibration())}

    @app.post("/api/calibration")
    def start_calibration():
        """Настроить модель заново."""
        model = llm()
        if model is None:
            raise HTTPException(409, "Модель не подключена")
        start_calibration_job(model, plan())
        return with_context(job.status())

    @app.get("/api/stats")
    def stats():
        """Сравнение способов работы: только правила, только модель, вместе."""
        prof = model_profile()
        data = prof if prof is not None else calibrate.rules_only()
        saved = verdicts().values()
        human = {"rule": sum(1 for x in saved if x.get("choice") == "rule"),
                 "llm": sum(1 for x in saved if x.get("choice") == "llm")}
        return {"stats": data, "calibration": with_context(calibration()), "verdicts": human, "mode": mode(), "meta": meta(),
                "modes": modes_table()}

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
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            entry = saved.get(key) or {"choice": None, "source": source[:10], "history": []}
            new = None if choice == "none" else choice
            if new != entry.get("choice"):
                # история: что было выбрано до этого, что выбрано теперь и когда
                entry.setdefault("history", []).append({"at": now, "from": entry.get("choice"), "to": new})
                entry["history"] = entry["history"][-50:]
                entry.update(choice=new, at=now)
            saved[key] = entry
            report.write_json(cfg.reports / "verdicts.json", saved)
        return {"ok": True, "choice": None if choice == "none" else choice}

    def work_check(raw: list, src: str):
        def fn(on_entry, on_log):
            analyzer, _ = cli.get_analyzer()
            model = llm()
            entries, stats = report.analyze_all(raw, analyzer, model, cfg.workers, None, "", cli.log, on_entry)
            rep = report.build(entries, stats, "file", src, model)
            report.write_json(cfg.reports / "check.json", rep)
            if src == SAMPLE.name:
                report.write_json(cfg.reports / "check-sample.json", rep)     # чтобы вернуться к нему без повторного разбора
            return rep, None
        return fn

    def work_run(run_cfg: config_mod.Config):
        def fn(on_entry, on_log):
            code = cli.cmd_run(run_cfg, True, on_log=on_log, on_entry=on_entry,
                               on_raw=None)
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
        """Проверка учебного набора. restore — вернуть сохранённый отчёт по нему, если он есть, без повторного разбора."""
        saved = cfg.reports / "check-sample.json"
        if restore and saved.is_file() and progress.snapshot(0).get("state") != "running":
            try:
                rep = json.loads(saved.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                rep = None
            if rep:
                with lock:
                    report.write_json(cfg.reports / "check.json", rep)
                return with_verdicts(rep)
        return launch("check", SAMPLE.name, work_check(source.load_file(SAMPLE), SAMPLE.name), background)

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
