"""Тесты скрипта оценки качества (eval/score.py) и подсчёта метрик (pv/scoring.py)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pv import scoring, source
from pv.config import ROOT
from pv.models import KINDS, TOPICS

SCORE_SCRIPT = ROOT / "eval" / "score.py"
GOLD_PATH = ROOT / "eval" / "gold.json"
EXTRA_GOLD_PATH = ROOT / "eval" / "extra_gold.json"
SAMPLE_VACANCIES_PATH = ROOT / "examples" / "sample_vacancies.json"
EXTRA_VACANCIES_PATH = ROOT / "eval" / "extra_vacancies.json"
SAMPLE_REPORT_PATH = ROOT / "examples" / "sample_report.json"


def _report(items: dict[str, list[tuple[str, str]]], model: dict | None = None) -> dict:
    return {
        "model": model or {"provider": None, "model": None, "used": False},
        "vacancies": [
            {
                "id": vid,
                "findings": [{"topic": t, "kind": k} for t, k in pairs],
            }
            for vid, pairs in items.items()
        ],
    }


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCORE_SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(ROOT),
    )


def test_perfect_match_passes():
    gold = {
        "_about": "тестовый эталон",
        "V1": {"must": [], "allowed": []},
        "V2": {"must": [["salary", "missing"], ["schedule", "ambiguous"]], "allowed": [["location", "ambiguous"]]},
    }
    rep = _report({"V1": [], "V2": [("salary", "missing"), ("schedule", "ambiguous")]})
    s = scoring.score(rep, gold)
    assert s["vacancies"] == 2
    assert s["must_total"] == 2 and s["must_found"] == 2
    assert s["found_total"] == 2
    assert s["recall"] == 1.0 and s["precision"] == 1.0
    assert s["extra_total"] == 0 and s["extra_per_10"] == 0.0 and s["extra_on_clean"] == 0
    assert all(not r["missed"] and not r["extra"] and not r["absent"] for r in s["rows"])
    assert scoring.passed(s)


def test_allowed_finding_is_neither_penalized_nor_required():
    gold = {
        "V1": {"must": [["salary", "missing"]], "allowed": [["location", "ambiguous"]]},
        "V2": {"must": [], "allowed": [["schedule", "ambiguous"]]},
    }
    # допустимые пропущены — recall и precision всё равно 1.0
    s_without = scoring.score(_report({"V1": [("salary", "missing")], "V2": []}), gold)
    assert s_without["recall"] == 1.0 and s_without["precision"] == 1.0 and s_without["extra_total"] == 0
    assert scoring.passed(s_without)

    # допустимые найдены — тоже не штрафуются и не считаются лишними на чистой
    s_with = scoring.score(
        _report({"V1": [("salary", "missing"), ("location", "ambiguous")], "V2": [("schedule", "ambiguous")]}),
        gold,
    )
    assert s_with["recall"] == 1.0 and s_with["precision"] == 1.0
    assert s_with["found_total"] == 3 and s_with["extra_total"] == 0 and s_with["extra_on_clean"] == 0
    assert scoring.passed(s_with)


def test_missed_must_reduces_recall_and_fails_threshold():
    gold = {
        "V1": {"must": [["salary", "missing"], ["duties", "ambiguous"]], "allowed": []},
        "V2": {"must": [], "allowed": []},
    }
    rep = _report({"V1": [("salary", "missing")], "V2": []})
    s = scoring.score(rep, gold)
    assert s["must_total"] == 2 and s["must_found"] == 1
    assert s["recall"] == pytest.approx(0.5)
    assert s["precision"] == 1.0
    assert s["rows"][0]["missed"] == [("duties", "ambiguous")]
    assert not scoring.passed(s)


def test_extra_findings_on_clean_and_problematic_vacancies():
    gold = {
        "CLEAN-1": {"must": [], "allowed": []},
        "PROB-1": {"must": [["schedule", "contradiction"]], "allowed": []},
    }
    rep = _report({
        "CLEAN-1": [("title", "ambiguous"), ("duties", "ambiguous")],
        "PROB-1": [("schedule", "contradiction"), ("salary", "missing")],
    })
    s = scoring.score(rep, gold)
    assert s["recall"] == 1.0
    assert s["found_total"] == 4
    assert s["extra_total"] == 3
    assert s["extra_on_clean"] == 2
    assert s["precision"] == pytest.approx(1 / 4)
    assert s["extra_per_10"] == pytest.approx(15.0)
    rows = {r["id"]: r for r in s["rows"]}
    assert rows["CLEAN-1"]["extra"] == [("duties", "ambiguous"), ("title", "ambiguous")]
    assert rows["PROB-1"]["extra"] == [("salary", "missing")]
    assert not scoring.passed(s)


def test_exact_threshold_boundaries():
    # 10 вакансий, 10 обязательных замечаний: 9 найдено (recall = 0.9), 1 лишнее (extra_per_10 = 1.0) → проходит
    gold = {f"V{i}": {"must": [["schedule", "ambiguous"]], "allowed": []} for i in range(10)}
    items = {f"V{i}": [("schedule", "ambiguous")] for i in range(9)}
    items["V9"] = [("title", "ambiguous")]  # 1 пропуск и 1 лишнее
    s_edge = scoring.score(_report(items), gold)
    assert s_edge["recall"] == pytest.approx(scoring.MIN_RECALL)
    assert s_edge["extra_per_10"] == pytest.approx(scoring.MAX_EXTRA_PER_10)
    assert scoring.passed(s_edge)

    # ещё одно лишнее → extra_per_10 = 2.0 > 1.0 → провал
    items["V0"].append(("salary", "missing"))
    assert not scoring.passed(scoring.score(_report(items), gold))


def test_absent_vacancy_fails_even_if_clean():
    gold = {
        "V1": {"must": [["salary", "missing"]], "allowed": []},
        "V2": {"must": [], "allowed": []},
    }
    # V2 отсутствует в отчёте: recall всё равно 1.0, но absent=True должен провалить passed()
    s = scoring.score(_report({"V1": [("salary", "missing")]}), gold)
    rows = {r["id"]: r for r in s["rows"]}
    assert rows["V1"]["absent"] is False and rows["V2"]["absent"] is True
    assert s["recall"] == 1.0
    assert not scoring.passed(s)


def test_duplicate_pairs_in_same_vacancy_are_deduplicated_and_all_clean_gold():
    gold = {"V1": {"must": [], "allowed": []}, "V2": {"must": [["duties", "ambiguous"]], "allowed": []}}
    rep = _report({"V1": [], "V2": [("duties", "ambiguous"), ("duties", "ambiguous")]})
    s = scoring.score(rep, gold)
    assert s["found_total"] == 1 and s["recall"] == 1.0 and s["precision"] == 1.0

    # только чистые вакансии и 0 замечаний → recall = 1.0, precision = 1.0
    s_clean = scoring.score(_report({"V1": []}), {"V1": {"must": [], "allowed": []}})
    assert s_clean["recall"] == 1.0 and s_clean["precision"] == 1.0 and scoring.passed(s_clean)


def test_cli_no_args_returns_2():
    res = _run_cli()
    assert res.returncode == 2
    assert "eval/score.py" in res.stdout


def test_cli_pass_and_fail_exit_codes_and_output(tmp_path: Path):
    gold = {
        "_about": "тест",
        "V1": {"must": [], "allowed": []},
        "V2": {"must": [["salary", "missing"]], "allowed": []},
    }
    gold_file = tmp_path / "gold.json"
    gold_file.write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")

    ok_report = tmp_path / "ok.json"
    ok_report.write_text(
        json.dumps(
            _report({"V1": [], "V2": [("salary", "missing")]}, model={"provider": "ollama", "model": "qwen2.5:3b", "used": True}),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    res_ok = _run_cli(str(ok_report), str(gold_file))
    assert res_ok.returncode == 0
    assert "модель: qwen2.5:3b" in res_ok.stdout
    assert "полнота:  1.00" in res_ok.stdout
    assert "точность: 1.00" in res_ok.stdout
    assert "лишних на понятных вакансиях: 0" in res_ok.stdout
    assert "ИТОГ: пороги пройдены" in res_ok.stdout

    bad_report = tmp_path / "bad.json"
    bad_report.write_text(
        json.dumps(_report({"V1": [("title", "ambiguous")]}), ensure_ascii=False),
        encoding="utf-8",
    )
    res_bad = _run_cli(str(bad_report), str(gold_file))
    assert res_bad.returncode == 1
    assert "модель: не использовалась" in res_bad.stdout
    assert "лишних на понятных вакансиях: 1" in res_bad.stdout
    assert "V1: лишнее [('title', 'ambiguous')]" in res_bad.stdout
    assert "V2: нет в отчёте" in res_bad.stdout
    assert "V2: пропущено [('salary', 'missing')]" in res_bad.stdout
    assert "ИТОГ: НИЖЕ ПОРОГОВ" in res_bad.stdout


def test_cli_on_sample_report_with_default_gold():
    res = _run_cli(str(SAMPLE_REPORT_PATH))
    assert res.returncode == 0
    assert "полнота:  1.00" in res.stdout
    assert "лишних на понятных вакансиях: 0" in res.stdout
    assert "ИТОГ: пороги пройдены" in res.stdout


@pytest.mark.parametrize(
    ("vacancies_path", "gold_path"),
    [
        (SAMPLE_VACANCIES_PATH, GOLD_PATH),
        (EXTRA_VACANCIES_PATH, EXTRA_GOLD_PATH),
    ],
)
def test_gold_files_match_vacancies_and_contract(vacancies_path: Path, gold_path: Path):
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    raw = source.load_file(vacancies_path)
    vacancy_ids = {str(v["id"]) for v in raw}
    gold_ids = {k for k in gold if not k.startswith("_")}
    assert gold_ids == vacancy_ids
    for vid in gold_ids:
        entry = gold[vid]
        assert isinstance(entry.get("why"), str) and entry["why"].strip()
        for bucket in ("must", "allowed"):
            pairs = entry.get(bucket, [])
            assert isinstance(pairs, list)
            for topic, kind in pairs:
                assert topic in TOPICS, f"{vid}: неизвестная тема {topic}"
                assert kind in KINDS, f"{vid}: неизвестный вид {kind}"
