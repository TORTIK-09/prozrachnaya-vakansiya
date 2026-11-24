"""Сравнение отчёта с эталоном.

    uv run python eval/score.py reports/check.json [eval/gold.json]

Совпадение — пара (тема, вид) у одной вакансии. Код возврата 1, если показатели ниже порогов.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

MIN_RECALL = 0.9          # доля обязательных замечаний, которые найдены
MAX_EXTRA_PER_10 = 1.0    # лишних замечаний на десять вакансий


def score(report: dict, gold: dict) -> dict:
    by_id = {str(v["id"]): v for v in report.get("vacancies", [])}
    must_total = must_found = found_total = found_ok = extra_clean = 0
    rows = []
    for vid, g in gold.items():
        if vid.startswith("_"):
            continue
        must = {tuple(x) for x in g.get("must", [])}
        allowed = {tuple(x) for x in g.get("allowed", [])}
        found = {(f["topic"], f["kind"]) for f in by_id.get(vid, {}).get("findings", [])}
        missed, extra = must - found, found - must - allowed
        must_total += len(must)
        must_found += len(must & found)
        found_total += len(found)
        found_ok += len(found & (must | allowed))
        if not must and not allowed:
            extra_clean += len(extra)
        rows.append({"id": vid, "missed": sorted(missed), "extra": sorted(extra),
                     "absent": vid not in by_id})
    n = len(rows)
    extra_total = found_total - found_ok
    return {
        "vacancies": n,
        "recall": must_found / must_total if must_total else 1.0,
        "precision": found_ok / found_total if found_total else 1.0,
        "extra_total": extra_total,
        "extra_per_10": extra_total / n * 10 if n else 0.0,
        "extra_on_clean": extra_clean,
        "rows": rows,
    }


def passed(s: dict) -> bool:
    return s["recall"] >= MIN_RECALL and s["extra_per_10"] <= MAX_EXTRA_PER_10 and not any(r["absent"] for r in s["rows"])


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if not argv:
        print(__doc__)
        return 2
    report = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    gold_path = Path(argv[1]) if len(argv) > 1 else Path(__file__).with_name("gold.json")
    s = score(report, json.loads(gold_path.read_text(encoding="utf-8")))
    model = report.get("model", {})
    print(f"модель: {model.get('model') if model.get('used') else 'не использовалась'}")
    print(f"полнота:  {s['recall']:.2f}  (порог {MIN_RECALL})")
    print(f"точность: {s['precision']:.2f}")
    print(f"лишних замечаний: {s['extra_total']}, на 10 вакансий: {s['extra_per_10']:.1f}  (порог {MAX_EXTRA_PER_10})")
    print(f"лишних на понятных вакансиях: {s['extra_on_clean']}")
    for r in s["rows"]:
        if r["absent"]:
            print(f"  {r['id']}: нет в отчёте")
        if r["missed"]:
            print(f"  {r['id']}: пропущено {r['missed']}")
        if r["extra"]:
            print(f"  {r['id']}: лишнее {r['extra']}")
    ok = passed(s)
    print("ИТОГ:", "пороги пройдены" if ok else "НИЖЕ ПОРОГОВ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
