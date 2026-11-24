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


def rules_only() -> dict:
    """Показатели одних правил — считаются мгновенно и без модели."""
    sets = []
    t0 = time.perf_counter()
    n = 0
    for st in _load_sets():
        found = {v.id: rules.check(v) for v in st["vacancies"]}
        n += len(st["vacancies"])
        sets.append({"key": st["key"], "title": st["title"], "about": st["about"],
                     "modes": {"rules": _brief(scoring.score(_report(st["vacancies"], found), st["gold"]))}})
    return {"model": None, "sets": sets, "vacancies": n,
            "seconds_per_vacancy": {"rules": round((time.perf_counter() - t0) / max(n, 1), 4)}}


def run(llm: LLM, progress: Callable[[int, int, float], None] = lambda done, total, sec: None,
        workers: int = 6) -> dict:
    """Полная настройка: запросы к модели по размеченным вакансиям выполняются параллельно."""
    from concurrent.futures import ThreadPoolExecutor

    sets = _load_sets()
    total = sum(len(s["vacancies"]) for s in sets)
    done = errors = 0
    rule_time = llm_time = 0.0
    per_set = []
    tally: dict[tuple[str, str], dict[str, int]] = {}      # вид замечаний модели → сколько верных и лишних
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

    categories, blocked = [], []
    for (topic, kind), c in sorted(tally.items()):
        n = c["ok"] + c["extra"]
        off = c["ok"] / n < MIN_PRECISION
        categories.append({"topic": topic, "kind": kind, "ok": c["ok"], "extra": c["extra"], "blocked": off})
        if off:
            blocked.append([topic, kind])
    blocked_set = {tuple(b) for b in blocked}

    out_sets = []
    for st, rule_found, llm_found in per_set:
        vs = st["vacancies"]
        modes = {
            "rules": rule_found,
            "llm": llm_found,
            "both_raw": {v.id: merge(rule_found[v.id], llm_found[v.id], set()) for v in vs},
            "both": {v.id: merge(rule_found[v.id], llm_found[v.id], blocked_set) for v in vs},
        }
        out_sets.append({"key": st["key"], "title": st["title"], "about": st["about"],
                         "modes": {m: _brief(scoring.score(_report(vs, f), st["gold"])) for m, f in modes.items()}})
    info = llm.describe() or {}
    return {
        "model": {"provider": info.get("provider"), "model": info.get("model")},
        "calibrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "vacancies": total, "errors": errors,
        "seconds_per_vacancy": {"rules": round(rule_time / max(total, 1), 4), "llm": round(llm_time / max(total, 1), 1)},
        "categories": categories, "blocked": blocked, "sets": out_sets,
    }


def save(result: dict, reports_dir: Path | None = None) -> Path:
    path = profile.local_dir(reports_dir) / (profile.slug(result["model"]) + ".json")
    report.write_json(path, result)
    profile.forget()
    return path


class Job:
    """Настройка в фоне с состоянием для панели: идёт ли, сколько сделано, сколько осталось."""

    def __init__(self, reports_dir: Path | None = None):
        self.reports_dir = reports_dir
        self.lock = threading.Lock()
        self.state: dict = {"state": "idle"}
        self.thread: threading.Thread | None = None

    def status(self) -> dict:
        with self.lock:
            return dict(self.state)

    def start(self, llm: LLM) -> bool:
        """False — настройка уже идёт."""
        with self.lock:
            if self.state.get("state") == "running":
                return False
            info = llm.describe() or {}
            self.state = {"state": "running", "model": info, "done": 0, "total": 0, "eta_seconds": None,
                          "started_at": datetime.now().astimezone().isoformat(timespec="seconds")}

        def progress(done: int, total: int, sec: float) -> None:
            with self.lock:
                self.state.update(done=done, total=total, eta_seconds=round(sec * (total - done)))

        def work() -> None:
            try:
                result = run(llm, progress)
                result["trained_with_rules"] = True
                save(result, self.reports_dir)
                with self.lock:
                    self.state = {"state": "done", "model": result["model"], "finished_at": result["calibrated_at"],
                                  "vacancies": result["vacancies"], "errors": result["errors"],
                                  "blocked": len(result["blocked"]),
                                  "trained_with_rules": True,
                                  "seconds_per_vacancy": result["seconds_per_vacancy"]["llm"]}
            except Exception as e:
                with self.lock:
                    self.state = {"state": "failed", "model": info, "error": f"{type(e).__name__}: {e}"[:300],
                                  "finished_at": datetime.now().astimezone().isoformat(timespec="seconds")}

        self.thread = threading.Thread(target=work, name="calibration", daemon=True)
        self.thread.start()
        return True
