"""Сравнение отчёта с эталоном.

    uv run python eval/score.py reports/check.json [eval/gold.json]

Код возврата 1, если показатели ниже порогов. Сам подсчёт — в pv/scoring.py.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pv.scoring import MAX_EXTRA_PER_10, MIN_RECALL, passed, score  # noqa: E402


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
