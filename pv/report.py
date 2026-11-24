"""Прогон анализатора по вакансиям, сборка и запись отчёта, кэш результатов."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .models import Evidence, Finding, LLM, Vacancy, validate
from .normalize import normalize

Analyze = Callable[[Vacancy, LLM | None], list[Finding]]


def fingerprint(v: Vacancy, salt: str) -> str:
    """Отпечаток содержимого вакансии. `salt` — версия анализатора и модель: сменились — кэш не годится."""
    d = asdict(v)
    d.pop("raw", None)
    blob = json.dumps([salt, d], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _finding_from_json(d: dict) -> Finding:
    c = d.get("conflict_with")
    return Finding(topic=d["topic"], kind=d["kind"], field=d["field"], quote=d["quote"], problem=d["problem"],
                   recommendation=d["recommendation"], conflict_with=Evidence(c["field"], c["quote"]) if c else None,
                   source=d.get("source", "rule"))


class Cache:
    """Результаты анализа по ID вакансии. Неизменившаяся вакансия заново не анализируется."""

    def __init__(self, path: Path | None):
        self.path = path
        self.data: dict[str, dict] = {}
        if path and path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data = loaded
            except (ValueError, OSError):
                self.data = {}   # испорченный кэш — не причина падать: просто считаем заново

    def get(self, vid: str, fp: str) -> list[Finding] | None:
        e = self.data.get(vid)
        if not isinstance(e, dict) or e.get("fp") != fp:
            return None
        try:
            return [_finding_from_json(f) for f in e["findings"]]
        except (KeyError, TypeError):
            return None

    def put(self, vid: str, fp: str, findings: list[Finding]) -> None:
        self.data[vid] = {"fp": fp, "findings": [f.to_json() for f in findings]}

    def save(self, keep: set[str]) -> None:
        if not self.path:
            return
        write_json(self.path, {k: v for k, v in self.data.items() if k in keep})


def _dedup(findings: list[Finding]) -> list[Finding]:
    seen, out = set(), []
    for f in findings:
        key = (f.topic, f.kind, f.field, f.quote, f.conflict_with)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def analyze_all(raw_items: list[Any], analyze: Analyze, llm: LLM | None, workers: int = 4,
                cache: Cache | None = None, salt: str = "",
                log: Callable[[str], None] = lambda s: None) -> tuple[list[dict], dict]:
    """Возвращает записи отчёта (по одной на ID, в порядке входных данных) и статистику."""
    stats = {"input": len(raw_items), "vacancies": 0, "with_findings": 0, "findings": 0, "skipped_no_id": 0,
             "duplicates": 0, "from_cache": 0, "rejected_findings": 0, "analyzer_errors": 0}
    vacancies: list[Vacancy] = []
    seen: set[str] = set()
    for raw in raw_items:
        v = normalize(raw)
        if v is None:
            stats["skipped_no_id"] += 1
        elif v.id in seen:
            stats["duplicates"] += 1
        else:
            seen.add(v.id)
            vacancies.append(v)
    if stats["skipped_no_id"]:
        log(f"пропущено записей без ID: {stats['skipped_no_id']}")

    def one(v: Vacancy) -> tuple[list[Finding], str | None, bool, int]:
        fp = fingerprint(v, salt)
        if cache is not None:
            hit = cache.get(v.id, fp)
            if hit is not None:
                return hit, None, True, 0
        try:
            result = analyze(v, llm)
            found = list(result) if result is not None else []
        except Exception as e:  # сбой на одной вакансии не должен ронять весь прогон
            return [], f"{type(e).__name__}: {e}"[:300], False, 0
        good = [f for f in found if isinstance(f, Finding) and not validate(f, v)]
        good = _dedup(good)
        if cache is not None:
            cache.put(v.id, fp, good)
        return good, None, False, len(found) - len(good)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(one, vacancies))

    entries = []
    for v, (found, error, cached, rejected) in zip(vacancies, results):
        entry: dict[str, Any] = {"id": v.id, "url": v.url, "title": v.title,
                                 "findings": [f.to_json() for f in found]}
        if error:
            entry["error"] = error
            stats["analyzer_errors"] += 1
        stats["from_cache"] += int(cached)
        stats["rejected_findings"] += rejected
        stats["findings"] += len(found)
        stats["with_findings"] += int(bool(found))
        entries.append(entry)
    stats["vacancies"] = len(entries)
    if cache is not None:
        cache.save(keep=seen)
    return entries, stats


def build(entries: list[dict], stats: dict, mode: str, source: str, llm: LLM | None,
          now: datetime | None = None) -> dict:
    model = {"provider": None, "model": None, "used": False}
    if llm is not None:
        try:
            info = llm.describe() or {}
        except Exception:
            info = {}
        model = {"provider": info.get("provider"), "model": info.get("model"), "used": True}
    return {
        "generated_at": (now or datetime.now().astimezone()).isoformat(timespec="seconds"),
        "mode": mode, "source": source, "model": model, "stats": stats, "vacancies": entries,
    }


def write_json(path: Path, data: Any) -> None:
    """Запись через временный файл и переименование: наполовину записанного файла не бывает."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _content(report: dict) -> dict:
    """Отчёт без времени создания и счётчика кэша — для сравнения «изменилось ли что-то»."""
    r = {k: v for k, v in report.items() if k != "generated_at"}
    r["stats"] = {k: v for k, v in report.get("stats", {}).items() if k != "from_cache"}
    return r


def save_latest(report: dict, reports_dir: Path) -> Path:
    """Текущий отчёт рабочего режима. Прежний уходит в history, если содержимое изменилось."""
    latest = reports_dir / "latest.json"
    if latest.is_file():
        try:
            old = json.loads(latest.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            old = None
        if isinstance(old, dict) and _content(old) != _content(report):
            stamp = str(old.get("generated_at") or "unknown").replace(":", "-")
            write_json(reports_dir / "history" / f"{stamp}.json", old)
    write_json(latest, report)
    error = reports_dir / "last_error.json"
    if error.is_file():
        error.unlink()   # успешный запуск снимает прошлую ошибку
    return latest


def save_error(message: str, source: str, reports_dir: Path, now: datetime | None = None) -> Path:
    """Сбой загрузки: записываем ошибку, latest.json не трогаем."""
    latest = reports_dir / "latest.json"
    path = reports_dir / "last_error.json"
    write_json(path, {
        "failed_at": (now or datetime.now().astimezone()).isoformat(timespec="seconds"),
        "source": source, "error": message,
        "last_successful_report": str(latest) if latest.is_file() else None,
    })
    return path
