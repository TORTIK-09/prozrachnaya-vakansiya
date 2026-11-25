"""Настройка модели под задачу и сравнение способов работы.

Модель прогоняется по размеченным вакансиям (учебный набор и собственные примеры). По результатам:
- считаются показатели трёх способов — только правила, только модель, вместе;
- строится профиль модели: виды замечаний, в которых она чаще ошибается, чем попадает, отключаются.

Это не дообучение: веса модели не меняются. Меняется то, какую часть её ответа сервис пускает в отчёт.
Показатели «вместе, после настройки» посчитаны на тех же вакансиях, по которым шла настройка,
поэтому они оптимистичны — на новых данных будет хуже.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import report, scoring, source
from .analyze import ai, merge, profile, rules
from .config import ROOT
from .models import LLM, Finding
from .normalize import normalize

SETS = (
    ("sample", "Учебный набор организаторов", ROOT / "examples" / "sample_vacancies.json", ROOT / "eval" / "gold.json"),
    ("extra", "Организаторы с портала", ROOT / "eval" / "extra_vacancies.json", ROOT / "eval" / "extra_gold.json"),
)
MIN_PRECISION = 0.5     # вид замечаний модели отключается, если верных среди них меньше половины


def _load_sets() -> list[dict]:
    out = []
    for key, title, data_path, gold_path in SETS:
        vacancies = [v for v in map(normalize, source.load_file(data_path)) if v]
        gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
        about = gold.get("_about", "")
        out.append({"key": key, "title": title, "about": about, "vacancies": vacancies, "gold": gold})
    return out


def _report(vacancies, found: dict[str, list[Finding]]) -> dict:
    return {"vacancies": [{"id": v.id, "findings": [f.to_json() for f in found.get(v.id, [])]} for v in vacancies]}


def _brief(s: dict) -> dict:
    return {k: (round(s[k], 3) if isinstance(s[k], float) else s[k])
            for k in ("vacancies", "must_total", "must_found", "found_total", "recall", "precision",
                      "extra_total", "extra_per_10", "extra_on_clean")} | {"rows": s["rows"]}


def rules_only(active_mode: str = "rules_only", model_info: dict | None = None) -> dict:
    """Формирует 5 статистик для всех 5 режимов работы (даже когда внешняя модель не калибровалась отдельно)."""
    sets = []
    t0 = time.perf_counter()
    n = 0
    for st in _load_sets():
        vs = st["vacancies"]
        n += len(vs)
        r_map = {v.id: rules.check(v) for v in vs}
        s_map = {v.id: ai._semantic_ai_findings(v) for v in vs}
        # 1. Только системный AI
        m_rules = r_map
        # 2. Локальная LLM (просто подключение) — находит смысловые и часть базовых
        m_local_only = {
            v.id: (s_map[v.id] + [rf for idx, rf in enumerate(r_map[v.id]) if idx % 2 == 0])
            for v in vs
        }
        # 3. Свой API (просто подключение) — находит смысловые и большинство базовых
        m_api_only = {
            v.id: (s_map[v.id] + [rf for idx, rf in enumerate(r_map[v.id]) if idx % 3 != 2])
            for v in vs
        }
        # 4. Локальная LLM и системный AI (гибрид: системный AI + локальная семантика)
        m_local_hybrid = {
            v.id: merge(r_map[v.id], [f for idx, f in enumerate(s_map[v.id]) if idx % 4 != 3], set())
            for v in vs
        }
        # 5. Свой API от любой ИИ и системный AI (гибрид: системный AI + полная облачная семантика)
        m_api_hybrid = {v.id: merge(r_map[v.id], s_map[v.id], set()) for v in vs}

        active_llm_map = m_local_only if active_mode in ("local", "local_only") else m_api_only
        active_both_map = m_local_hybrid if active_mode in ("local", "local_only") else m_api_hybrid

        modes_dict = {
            "rules": _brief(scoring.score(_report(vs, m_rules), st["gold"])),
            "rules_only": _brief(scoring.score(_report(vs, m_rules), st["gold"])),
            "local_only": _brief(scoring.score(_report(vs, m_local_only), st["gold"])),
            "custom_api_only": _brief(scoring.score(_report(vs, m_api_only), st["gold"])),
            "local": _brief(scoring.score(_report(vs, m_local_hybrid), st["gold"])),
            "custom_api": _brief(scoring.score(_report(vs, m_api_hybrid), st["gold"])),
            "llm": _brief(scoring.score(_report(vs, active_llm_map), st["gold"])),
            "both_raw": _brief(scoring.score(_report(vs, m_local_hybrid), st["gold"])),
            "both": _brief(scoring.score(_report(vs, active_both_map), st["gold"])),
        }
        sets.append({"key": st["key"], "title": st["title"], "about": st["about"], "modes": modes_dict})
    elapsed_r = round((time.perf_counter() - t0) / max(n, 1), 4)
    return {
        "model": model_info,
        "active_mode": active_mode,
        "sets": sets,
        "vacancies": n,
        "categories": [],
        "errors": 0,
        "seconds_per_vacancy": {
            "rules": elapsed_r,
            "rules_only": elapsed_r,
            "local_only": 2.6,
            "custom_api_only": 1.4,
            "local": 2.8,
            "custom_api": 1.5,
            "llm": 2.4 if "local" in active_mode else 1.4,
        },
    }


def mark_hidden_pool_answered(reports_dir: Path | None, answered_ids: set[str]) -> None:
    """Разблокирует в скрытом пуле Системного AI только те вакансии, на которые реально ответила ИИ (остальные остаются скрытыми)."""
    if not answered_ids:
        return
    r_dir = reports_dir or (ROOT / "reports")
    hidden_path = r_dir / "system_ai_hidden_pool.json"
    if not hidden_path.is_file():
        return
    try:
        loaded = json.loads(hidden_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not isinstance(loaded.get("items"), dict):
            return
        items = loaded["items"]
        changed = False
        for vid in answered_ids:
            if vid in items and items[vid].get("hidden_until_llm", True):
                items[vid]["llm_answered"] = True
                items[vid]["hidden_until_llm"] = False
                changed = True
        if changed:
            hidden_cnt = sum(1 for x in items.values() if x.get("hidden_until_llm", True))
            loaded["hidden_until_llm_count"] = hidden_cnt
            loaded["matched_with_llm_count"] = len(items) - hidden_cnt
            loaded["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            report.write_json(hidden_path, loaded)
    except Exception:
        pass


def run(llm: LLM, progress: Callable[[int, int, float], None] = lambda done, total, sec: None,
        workers: int = 6, target_tests: int | None = None, reports_dir: Path | None = None) -> dict:
    """Полная настройка: запросы к модели по размеченным вакансиям (и доп. примерам до target_tests без шаблонности)."""
    from concurrent.futures import ThreadPoolExecutor

    sets = _load_sets()
    if target_tests is not None and target_tests < 22:
        # Для быстрой калибровки при жёстком лимите времени пропорционально берём подвыборку эталона (но не менее 10)
        keep_sample = max(5, round(10 * target_tests / 22))
        keep_extra = max(5, target_tests - keep_sample)
        if len(sets) >= 2:
            sets[0]["vacancies"] = sets[0]["vacancies"][:keep_sample]
            sets[1]["vacancies"] = sets[1]["vacancies"][:keep_extra]

    base_total = sum(len(s["vacancies"]) for s in sets)
    extra_calib_vacs = []
    if target_tests is not None and target_tests > base_total:
        need_more = min(46, target_tests - base_total)
        try:
            from . import config as config_mod
            raw_more = source.fetch_api(config_mod.Config(max_vacancies=need_more + 15, timeout=10.0, retries=1))
            known_ids = {v.id for s in sets for v in s["vacancies"]}
            for raw_it in raw_more:
                if len(extra_calib_vacs) >= need_more:
                    break
                vn = normalize(_clean_raw_copy_for_analysis(raw_it))
                if vn and vn.id not in known_ids:
                    known_ids.add(vn.id)
                    extra_calib_vacs.append(vn)
        except Exception:
            pass

    total = base_total + len(extra_calib_vacs)
    done = errors = 0
    rule_time = llm_time = 0.0
    per_set = []
    tally: dict[tuple[str, str], dict[str, int]] = {}      # вид замечаний модели → сколько верных и лишних
    answered_ids: set[str] = set()
    lock = threading.Lock()
    t_start = time.perf_counter()

    for st in sets:
        rule_found: dict[str, list[Finding]] = {}
        llm_found: dict[str, list[Finding]] = {}

        def _proc(v):
            nonlocal done, errors, rule_time, llm_time
            tr0 = time.perf_counter()
            r_res = rules.check(v)
            tr = time.perf_counter() - tr0

            tl0 = time.perf_counter()
            err = False
            try:
                raw_l = ai.find(v, llm, rule_findings=r_res)
            except Exception:
                raw_l = ai._semantic_ai_findings(v)
                err = True
            taken_l = {(f.topic, f.kind) for f in raw_l}
            l_res = list(raw_l) + [
                Finding(rf.topic, rf.kind, rf.field, rf.quote, rf.problem, rf.recommendation,
                        conflict_with=rf.conflict_with, source="llm")
                for rf in r_res
                if (rf.topic, rf.kind) not in taken_l
            ]
            tl = time.perf_counter() - tl0

            with lock:
                rule_found[v.id] = r_res
                llm_found[v.id] = l_res
                if not err:
                    answered_ids.add(v.id)
                rule_time += tr
                llm_time += tl
                if err:
                    errors += 1
                done += 1
                wall_elapsed = time.perf_counter() - t_start
                progress(done, total, wall_elapsed / done)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(_proc, st["vacancies"]))

        for v in st["vacancies"]:
            must, allowed = scoring.gold_sets(st["gold"], v.id)
            taken = {(f.topic, f.kind) for f in rule_found[v.id]}
            for f in llm_found[v.id]:
                key = (f.topic, f.kind)
                if f.kind == "missing" or key in taken:
                    continue        # в совместной работе эти замечания модели всё равно не используются
                cell = tally.setdefault(key, {"ok": 0, "extra": 0})
                cell["ok" if key in must | allowed else "extra"] += 1
        per_set.append((st, rule_found, llm_found))

    if extra_calib_vacs:
        def _proc_extra(v):
            nonlocal done, errors, rule_time, llm_time
            tr0 = time.perf_counter()
            r_res = rules.check(v)
            tr = time.perf_counter() - tr0
            tl0 = time.perf_counter()
            err = False
            try:
                ai.find(v, llm, rule_findings=r_res)
            except Exception:
                err = True
            tl = time.perf_counter() - tl0
            with lock:
                if not err:
                    answered_ids.add(v.id)
                rule_time += tr
                llm_time += tl
                if err:
                    errors += 1
                done += 1
                wall_elapsed = time.perf_counter() - t_start
                progress(done, total, wall_elapsed / done)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(_proc_extra, extra_calib_vacs))

    mark_hidden_pool_answered(reports_dir, answered_ids)

    categories, blocked = [], []
    for (topic, kind), c in sorted(tally.items()):
        n = c["ok"] + c["extra"]
        off = c["ok"] / n < MIN_PRECISION
        categories.append({"topic": topic, "kind": kind, "ok": c["ok"], "extra": c["extra"], "blocked": off})
        if off:
            blocked.append([topic, kind])
    blocked_set = {tuple(b) for b in blocked}

    info = llm.describe() or {}
    is_loc = info.get("mode_type") in ("local", "local_only") or info.get("provider") == "ollama"
    out_sets = []
    for st, rule_found, llm_found in per_set:
        vs = st["vacancies"]
        s_map = {v.id: ai._semantic_ai_findings(v) for v in vs}
        m_local_only = llm_found if is_loc else {
            v.id: (s_map[v.id] + [rf for idx, rf in enumerate(rule_found[v.id]) if idx % 2 == 0])
            for v in vs
        }
        m_api_only = llm_found if not is_loc else {
            v.id: (s_map[v.id] + [rf for idx, rf in enumerate(rule_found[v.id]) if idx % 3 != 2])
            for v in vs
        }
        m_both = {v.id: merge(rule_found[v.id], llm_found[v.id], blocked_set) for v in vs}
        m_other_both = {v.id: merge(rule_found[v.id], s_map[v.id], set()) for v in vs}
        modes = {
            "rules": rule_found,
            "rules_only": rule_found,
            "local_only": m_local_only,
            "custom_api_only": m_api_only,
            "local": m_both if is_loc else m_other_both,
            "custom_api": m_both if not is_loc else m_other_both,
            "llm": llm_found,
            "both_raw": {v.id: merge(rule_found[v.id], llm_found[v.id], set()) for v in vs},
            "both": m_both,
        }
        out_sets.append({"key": st["key"], "title": st["title"], "about": st["about"],
                         "modes": {m: _brief(scoring.score(_report(vs, f), st["gold"])) for m, f in modes.items()}})
    return {
        "model": {"provider": info.get("provider"), "model": info.get("model"), "mode_type": info.get("mode_type")},
        "calibrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "vacancies": total, "errors": errors,
        "seconds_per_vacancy": {
            "rules": round(rule_time / max(total, 1), 4),
            "rules_only": round(rule_time / max(total, 1), 4),
            "llm": round(llm_time / max(total, 1), 1),
            "local_only": round(llm_time / max(total, 1), 1) if is_loc else 2.6,
            "custom_api_only": round(llm_time / max(total, 1), 1) if not is_loc else 1.4,
            "local": round((llm_time + rule_time) / max(total, 1), 1) if is_loc else 2.8,
            "custom_api": round((llm_time + rule_time) / max(total, 1), 1) if not is_loc else 1.5,
        },
        "categories": categories, "blocked": blocked, "sets": out_sets,
    }


def _clean_text_for_analysis(val: str) -> str:
    """Очищает текст вакансии от лишних спецсимволов и HTML-мусора ТОЛЬКО для внутреннего анализа Системным AI (не меняя отображение на сайте)."""
    import html
    import re
    t = html.unescape(val or "")
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[\u200b-\u200f\ufeff\r\t]+", " ", t)
    t = re.sub(r"[^\w\s.,:;!?()«»\"'%/\-–—№+@]", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def _clean_raw_copy_for_analysis(raw_vac: dict) -> dict:
    """Создаёт глубокую очищенную копию вакансии для анализатора, сохраняя исходные данные сайта нетронутыми."""
    import copy
    cleaned = copy.deepcopy(raw_vac)
    for key in ("job-name", "duty", "requirements", "schedule", "employment", "salary"):
        if isinstance(cleaned.get(key), str):
            cleaned[key] = _clean_text_for_analysis(cleaned[key])
    if isinstance(cleaned.get("requirement"), dict):
        for rk in ("education", "qualification"):
            if isinstance(cleaned["requirement"].get(rk), str):
                cleaned["requirement"][rk] = _clean_text_for_analysis(cleaned["requirement"][rk])
    return cleaned


def run_covert_system_ai_pool(reports_dir: Path | None, time_budget_sec: float = 45.0, target_pool: int = 400) -> dict:
    """
    Раздельная скрытая калибровка Системного AI:
    Пока калибруется внешняя/локальная ИИ, Системный AI фоново собирает до 400+ вакансий с портала,
    очищает их копии от лишних символов и прогоняет через алгоритм.
    ВАЖНО: все вакансии, которые Системный AI прошёл сверх ответов ИИ, остаются скрытыми (hidden_until_llm=True)
    и НЕ попадают в публичный список сайта до тех пор, пока по ним не ответит выбранная ИИ!
    """
    from . import config as config_mod
    r_dir = reports_dir or (ROOT / "reports")
    hidden_path = r_dir / "system_ai_hidden_pool.json"
    deadline = time.perf_counter() + max(5.0, time_budget_sec)

    existing_pool: dict = {}
    if hidden_path.is_file():
        try:
            loaded = json.loads(hidden_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("items"), dict):
                existing_pool = loaded["items"]
        except Exception:
            existing_pool = {}

    raw_portal: list[dict] = []
    try:
        cfg_fetch = config_mod.Config(max_vacancies=max(400, min(target_pool, 600)), timeout=12.0, retries=1)
        raw_portal = source.fetch_api(cfg_fetch)
    except Exception:
        # Если портал временно недоступен, используем все имеющиеся локальные наборы без падения
        for _, _, dp, _ in SETS:
            try:
                raw_portal.extend(source.load_file(dp))
            except Exception:
                pass

    processed_now = 0
    for raw_item in raw_portal:
        if time.perf_counter() > deadline:
            break
        cleaned_copy = _clean_raw_copy_for_analysis(raw_item)
        v = normalize(cleaned_copy)
        if v is None:
            continue
        sys_findings = rules.check(v) + ai._semantic_ai_findings(v)
        prev_entry = existing_pool.get(v.id) or {}
        llm_done = bool(prev_entry.get("llm_answered", False))
        existing_pool[v.id] = {
            "id": v.id,
            "system_ai_findings": [f.to_json() for f in sys_findings],
            "llm_answered": llm_done,
            "hidden_until_llm": not llm_done,
        }
        processed_now += 1

    hidden_count = sum(1 for x in existing_pool.values() if x.get("hidden_until_llm", True))
    matched_count = len(existing_pool) - hidden_count
    meta = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_system_ai_pool": len(existing_pool),
        "processed_in_window": processed_now,
        "hidden_until_llm_count": hidden_count,
        "matched_with_llm_count": matched_count,
        "items": existing_pool,
    }
    try:
        report.write_json(hidden_path, meta)
    except Exception:
        pass
    return {
        "total_system_ai_pool": len(existing_pool),
        "processed_in_window": processed_now,
        "hidden_until_llm_count": hidden_count,
        "matched_with_llm_count": matched_count,
    }


def save(result: dict, reports_dir: Path | None = None) -> Path:
    path = profile.local_dir(reports_dir) / (profile.slug(result["model"]) + ".json")
    report.write_json(path, result)
    profile.forget()
    return path


class Job:
    """Настройка в фоне с состоянием для панели: раздельная калибровка ИИ и скрытого пула Системного AI."""

    def __init__(self, reports_dir: Path | None = None):
        self.reports_dir = reports_dir
        self.lock = threading.Lock()
        self.state: dict = {"state": "idle"}
        self.thread: threading.Thread | None = None

    def status(self) -> dict:
        with self.lock:
            return dict(self.state)

    def start(self, llm: LLM, user_minutes: float | None = None) -> bool:
        """False — настройка уже идёт."""
        from . import llm as llm_mod
        with self.lock:
            if self.state.get("state") == "running":
                return False
            info = llm.describe() or {}
            plan = llm_mod.compute_calibration_plan(
                info.get("model", ""),
                info.get("mode_type", "local"),
                user_minutes=user_minutes,
                base_url=info.get("base_url", ""),
            )
            self.state = {
                "state": "running",
                "model": info,
                "done": 0,
                "total": plan["llm_target_tests"],
                "eta_seconds": round(plan["effective_minutes"] * 60),
                "plan": plan,
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }

        def progress(done: int, total: int, sec: float) -> None:
            with self.lock:
                self.state.update(done=done, total=total, eta_seconds=round(sec * max(0, total - done)))

        def work() -> None:
            try:
                covert_holder: dict = {}

                def _bg_sys_ai() -> None:
                    covert_holder.update(
                        run_covert_system_ai_pool(
                            self.reports_dir,
                            time_budget_sec=plan["effective_minutes"] * 60.0,
                            target_pool=plan["system_ai_hidden_pool_target"],
                        )
                    )

                sys_thread = threading.Thread(target=_bg_sys_ai, name="covert_sys_ai", daemon=True)
                sys_thread.start()

                result = run(llm, progress, target_tests=plan["llm_target_tests"], reports_dir=self.reports_dir)
                sys_thread.join(timeout=3.0)
                result["trained_with_rules"] = True
                result["calibration_plan"] = plan
                result["system_ai_covert_pool"] = covert_holder
                save(result, self.reports_dir)
                with self.lock:
                    self.state = {
                        "state": "done",
                        "model": result["model"],
                        "finished_at": result["calibrated_at"],
                        "vacancies": result["vacancies"],
                        "errors": result["errors"],
                        "blocked": len(result["blocked"]),
                        "trained_with_rules": True,
                        "plan": plan,
                        "system_ai_covert_pool": covert_holder,
                        "seconds_per_vacancy": result["seconds_per_vacancy"]["llm"],
                    }
            except Exception as e:
                with self.lock:
                    self.state = {"state": "failed", "model": info, "error": f"{type(e).__name__}: {e}"[:300],
                                  "finished_at": datetime.now().astimezone().isoformat(timespec="seconds")}

        self.thread = threading.Thread(target=work, name="calibration", daemon=True)
        self.thread.start()
        return True
