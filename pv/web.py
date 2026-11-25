"""Веб-панель: просмотр отчёта, запуск проверок, сценарий с резюме, статистика и выбор/обучение модели.
Запуск: `python -m pv web`.

Панель рассчитана на работу на своём компьютере: слушает 127.0.0.1, принимает запросы только с собственного
адреса и не принимает путей к файлам — только содержимое загруженного JSON.
"""
from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
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


def create_app(cfg: config_mod.Config | None = None, auto_calibrate: bool = True) -> FastAPI:
    cfg = cfg or config_mod.load()
    app = FastAPI(title="Прозрачная вакансия", docs_url=None, redoc_url=None, openapi_url=None)
    lock = threading.Lock()                 # проверки идут по одной: модель и кэш общие
    state: dict = {
        "llm": None,
        "rules_only": False,
        "run_job": {
            "state": "idle",
            "phase": "",
            "message": "",
            "done": 0,
            "total": 0,
            "chunk_done": 0,
            "chunk_total": 0,
            "error": None,
        },
    }
    job = calibrate.Job(cfg.reports)

    def llm():
        saved = llm_mod.load_runtime_settings(cfg.reports)
        if saved and saved.get("mode_type") == "rules_only":
            state["llm"] = None
            state["rules_only"] = True
            return None
        state["rules_only"] = False
        if state["llm"] is None:
            state["llm"] = cli.make_llm(cfg, True)
        return state["llm"]

    llm()

    def model_profile():
        model = llm()
        return profile.load(model.describe(), cfg.reports) if model else None

    def calibration() -> dict:
        """Состояние настройки (обучения) модели. Незнакомая модель настраивается сама при первом обращении."""
        model = llm()
        if model is None:
            return {"state": "no_model", "rules_only": state["rules_only"]}
        st = job.status()
        info = model.describe()
        if st["state"] in ("running", "failed", "done"):
            st_model = st.get("model") or {}
            if st_model.get("model") == info.get("model") and st_model.get("provider") == info.get("provider"):
                return st
        prof = model_profile()
        if prof is not None:
            return {
                "state": "ready",
                "model": prof.get("model") or info,
                "finished_at": prof.get("calibrated_at"),
                "shipped": bool(prof.get("_shipped")),
                "blocked": len(prof.get("blocked", [])),
            }
        if auto_calibrate:
            job.start(model)
            return job.status()
        return {"state": "needed", "model": info}

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
        info = model.describe() if model else None
        saved = llm_mod.load_runtime_settings(cfg.reports)
        error = None
        err_path = cfg.reports / "last_error.json"
        if err_path.is_file():
            try:
                error = json.loads(err_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                error = None
        return {
            "source_url": cfg.source_url,
            "schedule": cfg.schedule,
            "max_vacancies": cfg.max_vacancies,
            "chunk_size": 500,
            "chunk_threshold": 1000,
            "model": info,
            "model_configured": bool(saved and saved.get("configured")),
            "mode_type": (saved.get("mode_type") if saved else ("local" if info else "local")),
            "has_latest": (cfg.reports / "latest.json").is_file(),
            "last_error": error,
            "calibration": calibration(),
        }

    @app.get("/api/calibration-plan")
    def get_calibration_plan(model: str = "", mode_type: str = "local", minutes: float | None = None, base_url: str = ""):
        return llm_mod.compute_calibration_plan(
            model_name=model or llm_mod.DEFAULT_MODEL,
            mode_type=mode_type,
            user_minutes=minutes,
            base_url=base_url,
        )

    @app.get("/api/model-config")
    def get_model_config():
        """Возвращает текущие настройки модели/API, пресеты, характеристики ПК и план калибровки."""
        saved = llm_mod.load_runtime_settings(cfg.reports) or {}
        model = llm()
        info = model.describe() if model else None
        local_models = llm_mod.list_local_models()
        cur_mode = saved.get("mode_type") or (info.get("mode_type") if info else "local")
        cur_url = saved.get("base_url") or (info.get("base_url") if info else llm_mod.OLLAMA_URL)
        cur_model = saved.get("model") or (info.get("model") if info else llm_mod.DEFAULT_MODEL)
        cur_mins = saved.get("calibration_minutes")
        plan = llm_mod.compute_calibration_plan(cur_model, cur_mode, user_minutes=cur_mins, base_url=cur_url)
        return {
            "configured": bool(saved.get("configured")),
            "mode_type": cur_mode,
            "provider_preset": saved.get("provider_preset") or "ollama",
            "base_url": cur_url,
            "model": cur_model,
            "calibration_minutes": plan["effective_minutes"],
            "calibration_plan": plan,
            "has_api_key": bool(saved.get("api_key")),
            "local_models": local_models,
            "presets": llm_mod.PROVIDER_PRESETS,
            "connected": model is not None,
            "model_info": info,
            "calibration": calibration(),
        }

    @app.post("/api/model-config")
    async def set_model_config(request: Request):
        """Сохраняет выбранный пользователем тип модели, время калибровки и запускает раздельную калибровку."""
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(422, "Некорректный JSON настроек модели")

        mode_type = str(body.get("mode_type") or "local").strip()
        if mode_type not in ("local", "local_only", "custom_api", "custom_api_only", "rules_only"):
            mode_type = "local"

        preset_key = str(body.get("provider_preset") or ("ollama" if mode_type in ("local", "local_only") else "openrouter")).strip()
        preset = llm_mod.PROVIDER_PRESETS.get(preset_key, {})

        base_url = str(body.get("base_url") or preset.get("base_url") or llm_mod.OLLAMA_URL).strip()
        model_name = str(body.get("model") or preset.get("default_model") or llm_mod.DEFAULT_MODEL).strip()
        api_key = str(body.get("api_key") or "").strip()
        recalibrate = bool(body.get("recalibrate", False))
        raw_mins = body.get("calibration_minutes")
        cal_minutes: float | None = None
        try:
            if raw_mins is not None and str(raw_mins).strip() != "":
                cal_minutes = float(raw_mins)
        except (TypeError, ValueError):
            cal_minutes = None

        plan = llm_mod.compute_calibration_plan(model_name, mode_type, user_minutes=cal_minutes, base_url=base_url)

        prev = llm_mod.load_runtime_settings(cfg.reports) or {}
        if not api_key and prev.get("api_key") and prev.get("base_url") == base_url:
            api_key = str(prev.get("api_key"))

        settings_data = {
            "configured": True,
            "mode_type": mode_type,
            "provider_preset": preset_key,
            "base_url": base_url,
            "model": model_name,
            "api_key": api_key,
            "calibration_minutes": plan["effective_minutes"],
        }
        llm_mod.save_runtime_settings(cfg.reports, settings_data)

        # Переподключаем активный клиент модели (не прерывая фоновую проверку вакансий с портала, если она запущена)
        state["llm"] = None
        active_model = llm()

        if mode_type != "rules_only" and active_model is not None:
            profile.forget()
            prof = profile.load(active_model.describe(), cfg.reports)
            if prof is None or recalibrate:
                job.start(active_model, user_minutes=plan["effective_minutes"])
        elif mode_type == "rules_only" and recalibrate:
            threading.Thread(
                target=calibrate.run_covert_system_ai_pool,
                args=(cfg.reports, plan["effective_minutes"] * 60.0, plan["system_ai_hidden_pool_target"]),
                daemon=True,
            ).start()

        cal = calibration()
        info = active_model.describe() if active_model else None
        resp = JSONResponse({
            "ok": True,
            "configured": True,
            "mode_type": mode_type,
            "connected": active_model is not None,
            "model": info,
            "calibration": cal,
            "calibration_plan": plan,
        })
        resp.set_cookie(
            key="pv_model_configured",
            value="1",
            max_age=31536000,
            path="/",
            samesite="lax",
        )
        return resp

    @app.get("/api/calibration")
    def get_calibration():
        return calibration()

    @app.post("/api/calibration")
    def start_calibration():
        """Настроить (обучить) модель заново."""
        model = llm()
        if model is None:
            raise HTTPException(409, "Модель не подключена")
        profile.forget()
        job.start(model)
        return job.status()

    @app.get("/api/stats")
    def stats(mode_type: str | None = None):
        """5 статистик на 5 пунктов выбора типа использования + учёт всех моделей при смене."""
        saved = llm_mod.load_runtime_settings(cfg.reports) or {}
        active_model_client = llm()
        active_info = active_model_client.describe() if active_model_client else (
            {"provider": saved.get("provider_preset") or "system_ai", "model": saved.get("model") or "Системный AI", "mode_type": "rules_only"}
            if saved.get("mode_type") == "rules_only"
            else None
        )
        active_mode = mode_type or saved.get("mode_type") or (active_info.get("mode_type") if active_info else "local")
        if active_mode not in ("local", "custom_api", "local_only", "custom_api_only", "rules_only"):
            active_mode = "local"

        prof = model_profile()
        fallback_5 = calibrate.rules_only(active_mode=active_mode, model_info=active_info)
        if prof is not None:
            # Дополняем загруженный профиль всеми 5 режимами, если в нём были только старые ключи
            fb_sets = {s["key"]: s["modes"] for s in fallback_5["sets"]}
            for s in prof.get("sets") or []:
                s_modes = s.get("modes") or {}
                fb_m = fb_sets.get(s.get("key"), {})
                for k5 in ("rules_only", "local_only", "custom_api_only", "local", "custom_api"):
                    if k5 not in s_modes and k5 in fb_m:
                        s_modes[k5] = fb_m[k5]
            prof["active_mode"] = active_mode
            data = prof
        else:
            data = fallback_5
        all_models = []
        p_dir = profile.local_dir(cfg.reports)
        if p_dir.is_dir():
            for pf in sorted(p_dir.glob("*.json")):
                try:
                    pdata = json.loads(pf.read_text(encoding="utf-8"))
                    if not isinstance(pdata, dict) or not pdata.get("model"):
                        continue
                    sets_map = {s["key"]: s["modes"] for s in (pdata.get("sets") or []) if isinstance(s, dict) and "key" in s and "modes" in s}
                    s_both = (sets_map.get("sample") or {}).get("both") or {}
                    x_both = (sets_map.get("extra") or {}).get("both") or {}
                    all_models.append({
                        "provider": pdata["model"].get("provider"),
                        "model": pdata["model"].get("model"),
                        "calibrated_at": pdata.get("calibrated_at"),
                        "seconds_per_vacancy": (pdata.get("seconds_per_vacancy") or {}).get("llm"),
                        "sample_found": s_both.get("must_found", 0),
                        "sample_total": s_both.get("must_total", 13),
                        "extra_found": x_both.get("must_found", 0),
                        "extra_total": x_both.get("must_total", 8),
                    })
                except (ValueError, OSError):
                    continue

        latest_rep = read_report("latest") or read_report("check")
        portal_summary = None
        if isinstance(latest_rep, dict):
            vacs = latest_rep.get("vacancies") or []
            rule_cnt = sum(1 for v in vacs for f in (v.get("findings") or []) if f.get("source") != "llm")
            llm_cnt = sum(1 for v in vacs for f in (v.get("findings") or []) if f.get("source") == "llm")
            portal_summary = {
                "mode": latest_rep.get("mode"),
                "source": latest_rep.get("source"),
                "generated_at": latest_rep.get("generated_at"),
                "model": latest_rep.get("model"),
                "vacancies": len(vacs),
                "with_findings": sum(1 for v in vacs if v.get("findings")),
                "rule_findings": rule_cnt,
                "llm_findings": llm_cnt,
                "total_findings": rule_cnt + llm_cnt,
            }

        return {
            "stats": data,
            "calibration": calibration(),
            "all_models": all_models,
            "portal_summary": portal_summary,
        }

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

    @app.get("/api/run-status")
    def get_run_status():
        return dict(state["run_job"])

    @app.post("/api/run")
    def run_live(max_vacancies: int | None = None, bg: int = 0):
        """Ручной запуск рабочего режима. При bg=1 выполняется в фоне чанками без обрыва соединения браузера."""
        run_cfg = cfg
        if max_vacancies is not None:
            if not 1 <= max_vacancies <= 5000:
                raise HTTPException(422, "Число вакансий: от 1 до 5000")
            run_cfg = replace(cfg, max_vacancies=max_vacancies)

        if not bg:
            with lock:
                code = cli.cmd_run(run_cfg, not state["rules_only"])
            rep = read_report("latest")
            if code != 0:
                err = {}
                try:
                    err = json.loads((cfg.reports / "last_error.json").read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    pass
                return JSONResponse({"detail": err.get("error", "Источник недоступен"), "report": rep}, status_code=502)
            return rep

        if state["run_job"].get("state") == "running":
            return dict(state["run_job"])

        c_size, c_total = report.dynamic_chunk_plan(run_cfg.max_vacancies)
        p_workers = report.dynamic_parallel_workers(run_cfg.max_vacancies, run_cfg.workers)
        state["run_job"] = {
            "state": "running",
            "phase": "fetching",
            "message": f"Параллельная загрузка с портала (до {run_cfg.max_vacancies} вакансий → {c_total} чанк(ов) по ~{c_size} шт. · параллельно: {p_workers} шт.)…",
            "done": 0,
            "total": run_cfg.max_vacancies,
            "chunk_done": 0,
            "chunk_total": c_total,
            "parallel_workers": p_workers,
            "error": None,
        }

        def _bg_worker() -> None:
            try:
                def _fetch_log(msg: str) -> None:
                    cli.log(msg)
                    state["run_job"]["message"] = f"Загрузка с портала: {msg}"

                try:
                    raw = source.fetch_api(run_cfg, log=_fetch_log)
                except source.SourceError as e:
                    report.save_error(str(e), run_cfg.source_url, run_cfg.reports)
                    state["run_job"].update(state="failed", error=str(e), message=f"Ошибка портала: {e}")
                    return

                if not raw:
                    msg = "Источник вернул пустой список вакансий"
                    report.save_error(msg, run_cfg.source_url, run_cfg.reports)
                    state["run_job"].update(state="failed", error=msg, message=msg)
                    return

                analyzer, version = cli.get_analyzer()
                active_llm = llm() if not state["rules_only"] else None
                cache = report.Cache(run_cfg.reports / "cache.json")
                salt = cli._salt(version, active_llm)
                actual_size, actual_chunks = report.dynamic_chunk_plan(len(raw))
                actual_workers = report.dynamic_parallel_workers(len(raw), run_cfg.workers)
                state["run_job"].update(
                    phase="analyzing",
                    total=len(raw),
                    chunk_total=actual_chunks,
                    parallel_workers=actual_workers,
                    message=f"Анализ {len(raw)} вакансий: чанк 1 из {actual_chunks} (по ~{actual_size} шт. · параллельно: {actual_workers} шт.)…",
                )

                def _on_progress(dc: int, tc: int, c_idx: int, c_tot: int) -> None:
                    state["run_job"].update(
                        done=dc,
                        total=tc,
                        chunk_done=c_idx - 1,
                        chunk_total=c_tot,
                        message=f"Динамическая обработка: чанк {c_idx} из {c_tot} · параллельно: {actual_workers} шт. · проверено {dc} из {tc} вакансий…",
                    )

                def _on_chunk_done(part_entries: list[dict], part_stats: dict, c_idx: int, c_tot: int) -> None:
                    part_rep = report.build(part_entries, part_stats, "api", run_cfg.source_url, active_llm)
                    report.write_json(run_cfg.reports / "latest.json", part_rep)
                    state["run_job"].update(
                        chunk_done=c_idx,
                        chunk_total=c_tot,
                        message=f"Чанк {c_idx} из {c_tot} сохранён ({len(part_entries)} вакансий в отчёте · параллельно: {actual_workers} шт.)…",
                    )

                entries, stats_res = report.analyze_all(
                    raw,
                    analyzer,
                    active_llm,
                    run_cfg.workers,
                    cache,
                    salt,
                    cli.log,
                    on_progress=_on_progress,
                    on_chunk_done=_on_chunk_done,
                )
                final_rep = report.build(entries, stats_res, "api", run_cfg.source_url, active_llm)
                report.save_latest(final_rep, run_cfg.reports)
                state["run_job"].update(
                    state="done",
                    phase="done",
                    done=len(entries),
                    total=len(entries),
                    chunk_done=actual_chunks,
                    chunk_total=actual_chunks,
                    message=f"Готово: обработано {len(entries)} вакансий ({actual_chunks} чанк(ов)).",
                )
            except Exception as e:
                state["run_job"].update(state="failed", error=str(e), message=f"Ошибка выполнения: {e}")

        threading.Thread(target=_bg_worker, name="bg_run_live", daemon=True).start()
        return dict(state["run_job"])

    @app.post("/api/finding-decision")
    async def finding_decision(request: Request):
        """
        Ручной выбор редактором правильного ответа (Системный AI vs Выбранная ИИ)
        или изменение статуса замечания (оставить неоднозначность / изменить на противоречие или пропуск / снять).
        Веса модели при этом НЕ дообучаются.
        """
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(422, "Некорректный JSON решения")

        kind_rep = str(body.get("report_kind") or "check")
        if kind_rep not in ("check", "latest"):
            kind_rep = "check"
        vac_id = str(body.get("vacancy_id") or "")
        f_idx = int(body.get("finding_index", -1))
        chosen_source = str(body.get("chosen_source") or "")  # "system_ai" | "selected_ai"
        new_status = str(body.get("new_status") or "")        # "ambiguous" | "contradiction" | "missing" | "dismiss"
        custom_rec = body.get("recommendation")
        custom_prob = body.get("problem")

        rep = read_report(kind_rep)
        if rep is None:
            raise HTTPException(404, "Отчёт не найден")

        vacs = rep.get("vacancies") or []
        entry = next((v for v in vacs if str(v.get("id")) == vac_id), None)
        if entry is None:
            raise HTTPException(404, "Вакансия не найдена в отчёте")

        findings = entry.get("findings") or []
        if not (0 <= f_idx < len(findings)):
            raise HTTPException(404, "Замечание не найдено")

        if new_status == "dismiss":
            findings.pop(f_idx)
        else:
            item = findings[f_idx]
            if "system_ai_kind" not in item:
                item["system_ai_kind"] = item.get("kind") if item.get("source") != "llm" else "ambiguous"
            if "selected_ai_kind" not in item:
                orig_k = item.get("kind") or "ambiguous"
                item["selected_ai_kind"] = orig_k if item.get("source") == "llm" else ("ambiguous" if orig_k == "contradiction" else "contradiction")
            if new_status in ("ambiguous", "contradiction", "missing"):
                item["kind"] = new_status
                if chosen_source == "system_ai":
                    item["system_ai_kind"] = new_status
                elif chosen_source == "selected_ai":
                    item["selected_ai_kind"] = new_status
                if new_status == "missing":
                    item["quote"] = ""
            if chosen_source == "system_ai":
                item["source"] = "rule"
                item["chosen_by_editor"] = "system_ai"
            elif chosen_source == "selected_ai":
                item["source"] = "llm"
                item["chosen_by_editor"] = "selected_ai"
            if isinstance(custom_prob, str) and custom_prob.strip():
                item["problem"] = custom_prob.strip()
            if isinstance(custom_rec, str) and custom_rec.strip():
                item["recommendation"] = custom_rec.strip()
            item["model_not_retrained"] = True

        # Пересчитываем сводную статистику отчёта
        if "stats" in rep and isinstance(rep["stats"], dict):
            rep["stats"]["findings"] = sum(len(v.get("findings") or []) for v in vacs)
            rep["stats"]["with_findings"] = sum(1 for v in vacs if (v.get("findings") or []))

        target_path = cfg.reports / ("latest.json" if kind_rep == "latest" else "check.json")
        report.write_json(target_path, rep)
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
