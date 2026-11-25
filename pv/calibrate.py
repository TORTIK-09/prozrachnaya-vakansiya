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
from .analyze.version import PROFILE_VERSION as VERSION
from .config import ROOT
from .models import LLM, Finding
from .normalize import normalize

SETS = (
    ("sample", "Учебный набор организаторов", ROOT / "examples" / "sample_vacancies.json", ROOT / "eval" / "gold.json"),
    ("extra", "Смысловые расхождения", ROOT / "eval" / "extra_vacancies.json", ROOT / "eval" / "extra_gold.json"),
    ("holdout", "Контрольный набор", ROOT / "eval" / "holdout_vacancies.json", ROOT / "eval" / "holdout_gold.json"),
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


MIN_CONFIRMED = 2       # вид замечаний модели остаётся включённым, только если подтвердился хотя бы дважды


def _limit(sets: list[dict], limit: int | None) -> list[dict]:
    """Не больше `limit` вакансий на всё: по очереди из каждого набора, чтобы в выборке были и понятные, и проблемные."""
    total = sum(len(s["vacancies"]) for s in sets)
    if not limit or limit >= total:
        return sets
    take = {s["key"]: 0 for s in sets}
    left = max(1, limit)
    while left > 0:
        moved = False
        for s in sets:
            if left > 0 and take[s["key"]] < len(s["vacancies"]):
                take[s["key"]] += 1
                left -= 1
                moved = True
        if not moved:
            break
    out = []
    for s in sets:
        vs = s["vacancies"][:take[s["key"]]]
        if vs:
            ids = {v.id for v in vs}
            # разметка — только по взятым вакансиям, иначе невзятые посчитаются пропущенными
            out.append({**s, "vacancies": vs, "gold": {k: g for k, g in s["gold"].items() if k in ids}})
    return out


def run(llm: LLM, progress: Callable[[int, int, float], None] = lambda done, total, sec: None,
        limit: int | None = None, budget_seconds: float | None = None, min_tests: int = 0) -> dict:
    """Калибровка: один запрос к модели на каждую размеченную вакансию.

    `limit` — жёсткий потолок числа вакансий. `budget_seconds` — отведённое время: скорость модели меряется
    на первых двух вакансиях, после чего берётся столько вакансий, сколько помещается во время, но не меньше
    `min_tests` и не больше `limit`. Вакансии идут по очереди из каждого набора, поэтому даже короткая
    калибровка видит и понятные, и проблемные.
    """
    all_sets = _load_sets()
    full_total = sum(len(s["vacancies"]) for s in all_sets)
    sets = _limit(all_sets, limit)
    order = []                                  # (номер набора, вакансия) — по очереди из каждого набора
    for i in range(max(len(s["vacancies"]) for s in sets)):
        order += [(k, s["vacancies"][i]) for k, s in enumerate(sets) if i < len(s["vacancies"])]
    total = len(order)
    done = errors = 0
    rule_time = llm_time = 0.0
    found_r: list[dict[str, list[Finding]]] = [{} for _ in sets]
    found_l: list[dict[str, list[Finding]]] = [{} for _ in sets]
    agree = {}
    tally: dict[tuple[str, str], dict[str, int]] = {}      # вид замечаний модели → сколько верных и лишних
    kinds_seen: dict[tuple[str, str], int] = {}            # все замечания модели по видам — признак шаблонных ответов
    llm_total = on_clean = clean_n = 0
    for k, v in order:
        st = sets[k]
        t = time.perf_counter()
        found_r[k][v.id] = rules.check(v)
        rule_time += time.perf_counter() - t
        t = time.perf_counter()
        try:
            found_l[k][v.id] = ai.find(v, llm)
        except Exception:
            found_l[k][v.id] = []
            errors += 1
        llm_time += time.perf_counter() - t
        done += 1
        if budget_seconds and done == 2:
            # скорость измерена — уточняем, сколько вакансий помещается в отведённое время
            fit = int(budget_seconds / max(llm_time / done, 1e-3))
            total = max(2, min(len(order), max(min_tests, fit)))
        progress(done, total, llm_time / done)
        must, allowed = scoring.gold_sets(st["gold"], v.id)
        taken = {(f.topic, f.kind) for f in found_r[k][v.id]}
        # кто что видит: правило, модель или оба — и сколько из этого верно по разметке
        seen_llm = {(f.topic, f.kind) for f in found_l[k][v.id]}
        a = agree.setdefault(st["key"], {x: [0, 0] for x in ("both", "rules_only", "llm_only")})
        for name, keys in (("both", taken & seen_llm), ("rules_only", taken - seen_llm), ("llm_only", seen_llm - taken)):
            a[name][0] += len(keys)
            a[name][1] += len(keys & (must | allowed))
        clean = not must and not allowed
        clean_n += clean
        for f in found_l[k][v.id]:
            key = (f.topic, f.kind)
            kinds_seen[key] = kinds_seen.get(key, 0) + 1
            llm_total += 1
            on_clean += clean
            if f.kind == "missing" or key in taken:
                continue        # в совместной работе эти замечания модели всё равно не используются
            cell = tally.setdefault(key, {"ok": 0, "extra": 0})
            cell["ok" if key in must | allowed else "extra"] += 1
        if done >= total:
            break
    total = done
    per_set = []
    for k, st in enumerate(sets):
        vs = [v for v in st["vacancies"] if v.id in found_r[k]]
        if vs:
            ids = {v.id for v in vs}
            per_set.append(({**st, "vacancies": vs, "gold": {i: g for i, g in st["gold"].items() if i in ids}},
                            found_r[k], found_l[k]))
    # шаблонность: модель выдаёт одно и то же замечание почти на каждую вакансию, в том числе на понятные
    top = max(kinds_seen.values(), default=0)
    template = {"findings": llm_total, "top_kind_share": round(top / llm_total, 2) if llm_total else 0.0,
                "on_clean_per_vacancy": round(on_clean / clean_n, 2) if clean_n else 0.0}
    template["templated"] = llm_total >= 5 and template["top_kind_share"] >= 0.5 and template["on_clean_per_vacancy"] >= 0.5

    categories, blocked = [], []
    for (topic, kind), c in sorted(tally.items()):
        n = c["ok"] + c["extra"]
        # доверие нужно заслужить: мало подтверждений или больше ошибок, чем попаданий, — вид отключается
        off = c["ok"] / n < MIN_PRECISION or c["ok"] < MIN_CONFIRMED
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
        out_sets.append({"key": st["key"], "title": st["title"], "about": st["about"], "agreement": agree.get(st["key"]),
                         "modes": {m: _brief(scoring.score(_report(vs, f), st["gold"])) for m, f in modes.items()}})
    info = llm.describe() or {}
    return {
        "model": {"provider": info.get("provider"), "model": info.get("model"),
                  "kind": "local" if str(info.get("mode_type") or "").startswith("local")
                  or info.get("provider") in ("ollama", "local") else "custom_api"},
        "version": VERSION,
        "calibrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "vacancies": total, "available": full_total, "errors": errors,
        "seconds_per_vacancy": {"rules": round(rule_time / max(total, 1), 4), "llm": round(llm_time / max(total, 1), 1)},
        "categories": categories, "blocked": blocked, "template": template,
        "examples_in_prompt": ai.examples_for(llm), "sets": out_sets,
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
        self.learning: dict = {"state": "idle", "vacancies": 0}
        self.thread: threading.Thread | None = None

    def status(self) -> dict:
        with self.lock:
            return dict(self.state)

    def start(self, llm: LLM, limit: int | None = None, learn_seconds: float = 0.0, cfg=None,
              budget_seconds: float | None = None, min_tests: int = 0) -> bool:
        """False — настройка уже идёт. `limit` — сколько вакансий взять по плану калибровки.
        `learn_seconds` — сколько времени Системный AI параллельно учится на вакансиях портала."""
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
                result = run(llm, progress, limit, budget_seconds, min_tests)
                if result["errors"] >= result["vacancies"]:
                    raise RuntimeError("модель не ответила ни на один запрос")
                save(result, self.reports_dir)
                with self.lock:
                    self.state = {"state": "done", "model": result["model"], "finished_at": result["calibrated_at"],
                                  "vacancies": result["vacancies"], "available": result["available"],
                                  "errors": result["errors"],
                                  "blocked": len(result["blocked"]),
                                  "seconds_per_vacancy": result["seconds_per_vacancy"]["llm"]}
            except Exception as e:
                with self.lock:
                    self.state = {"state": "failed", "model": info, "error": f"{type(e).__name__}: {e}"[:300],
                                  "finished_at": datetime.now().astimezone().isoformat(timespec="seconds")}

        self.thread = threading.Thread(target=work, name="calibration", daemon=True)
        self.thread.start()
        if learn_seconds > 0 and cfg is not None:
            threading.Thread(target=self._learn, args=(cfg, learn_seconds), name="system-ai-learning", daemon=True).start()
        return True

    def _learn(self, cfg, seconds: float) -> None:
        """Пока калибруется ИИ, Системный AI учится на вакансиях портала: пересчитывает нормы по должностям.

        Успевает столько страниц, сколько помещается во время калибровки, — вплоть до всего портала.
        Портал недоступен — ничего страшного: остаются нормы, поставляемые с проектом.
        """
        import httpx

        from . import source
        from .analyze import norms
        t0, raw, page = time.monotonic(), [], 0
        self.learning = {"state": "running", "vacancies": 0}
        try:
            with httpx.Client(headers={"User-Agent": "prozrachnaya-vakansiya/0.1"}, follow_redirects=True) as client:
                while time.monotonic() - t0 < seconds:
                    body = source._get_page(client, cfg, page, 100, time.sleep)
                    items = source.extract(body)
                    if not items:
                        break
                    raw.extend(items)
                    page += 1
                    self.learning = {"state": "running", "vacancies": len(raw)}
                    if len(items) < 100:
                        break
        except Exception as e:
            if not raw:
                self.learning = {"state": "failed", "vacancies": 0, "error": f"{type(e).__name__}: {e}"[:200]}
                return
        table = norms.build(v for v in map(normalize, raw) if v)
        if table["groups"]:
            report.write_json(norms.local_path(self.reports_dir), table)
            norms.forget()
        self.learning = {"state": "done", "vacancies": len(raw), "groups": len(table["groups"])}
