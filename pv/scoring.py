"""Сравнение замечаний с эталоном. Совпадение — пара (тема, вид) у одной вакансии."""
from __future__ import annotations

MIN_RECALL = 0.9          # доля обязательных замечаний, которые найдены
MAX_EXTRA_PER_10 = 1.0    # лишних замечаний на десять вакансий


def gold_sets(gold: dict, vid: str) -> tuple[set, set]:
    g = gold.get(vid) or {}
    return {tuple(x) for x in g.get("must", [])}, {tuple(x) for x in g.get("allowed", [])}


def score(report: dict, gold: dict) -> dict:
    by_id = {str(v["id"]): v for v in report.get("vacancies", [])}
    must_total = must_found = found_total = found_ok = extra_clean = 0
    rows = []
    for vid in gold:
        if vid.startswith("_"):
            continue
        must, allowed = gold_sets(gold, vid)
        found = {(f["topic"], f["kind"]) for f in by_id.get(vid, {}).get("findings", [])}
        missed, extra = must - found, found - must - allowed
        must_total += len(must)
        must_found += len(must & found)
        found_total += len(found)
        found_ok += len(found & (must | allowed))
        if not must and not allowed:
            extra_clean += len(extra)
        rows.append({"id": vid, "missed": sorted(missed), "extra": sorted(extra), "absent": vid not in by_id})
    n = len(rows)
    extra_total = found_total - found_ok
    return {
        "vacancies": n, "must_total": must_total, "must_found": must_found, "found_total": found_total,
        "recall": must_found / must_total if must_total else 1.0,
        "precision": found_ok / found_total if found_total else 1.0,
        "extra_total": extra_total,
        "extra_per_10": extra_total / n * 10 if n else 0.0,
        "extra_on_clean": extra_clean,
        "rows": rows,
    }


def passed(s: dict) -> bool:
    return s["recall"] >= MIN_RECALL and s["extra_per_10"] <= MAX_EXTRA_PER_10 and not any(r["absent"] for r in s["rows"])
