import json
import threading

from pv import report
from pv.models import Evidence, Finding

RAW = [{"vacancy": {"id": f"V{i}", "job-name": "Специалист", "schedule": "Гибкий график",
                    "duty": "Работа с документами."}} for i in range(5)]


def good(v, llm=None):
    return [Finding("schedule", "ambiguous", "schedule", "Гибкий график", "Нет дней и часов.", "Уточните дни и часы.")]


def test_one_entry_per_id_in_input_order():
    raw = RAW + [RAW[0], {"vacancy": {}}, "мусор"]
    entries, stats = report.analyze_all(raw, good, None, workers=3)
    assert [e["id"] for e in entries] == ["V0", "V1", "V2", "V3", "V4"]
    assert stats["vacancies"] == 5 and stats["duplicates"] == 1 and stats["skipped_no_id"] == 2
    assert stats["findings"] == 5 and stats["with_findings"] == 5
    assert entries[0]["url"] is None and entries[0]["findings"][0]["quote"] == "Гибкий график"


def test_invalid_findings_rejected_and_duplicates_merged():
    def messy(v, llm=None):
        f = good(v)[0]
        return [f, f, Finding("schedule", "ambiguous", "schedule", "Свободный график", "п", "р"),
                Finding("pay", "ambiguous", "schedule", "Гибкий график", "п", "р"), "не замечание"]

    entries, stats = report.analyze_all(RAW[:1], messy, None)
    assert len(entries[0]["findings"]) == 1 and stats["rejected_findings"] == 4


def test_analyzer_crash_keeps_vacancy_and_run_continues():
    def flaky(v, llm=None):
        if v.id == "V2":
            raise RuntimeError("модель упала")
        return good(v)

    entries, stats = report.analyze_all(RAW, flaky, None, workers=2)
    assert len(entries) == 5 and stats["analyzer_errors"] == 1
    bad = next(e for e in entries if e["id"] == "V2")
    assert bad["findings"] == [] and "модель упала" in bad["error"]


def test_none_result_is_empty_list():
    entries, _ = report.analyze_all(RAW[:1], lambda v, llm=None: None, None)
    assert entries[0]["findings"] == []


def test_cache_skips_unchanged_and_reruns_changed(tmp_path):
    calls, lock = [], threading.Lock()

    def counting(v, llm=None):
        with lock:
            calls.append(v.id)
        return good(v)

    path = tmp_path / "cache.json"
    e1, s1 = report.analyze_all(RAW, counting, None, cache=report.Cache(path), salt="v1")
    e2, s2 = report.analyze_all(RAW, counting, None, cache=report.Cache(path), salt="v1")
    assert len(calls) == 5 and s2["from_cache"] == 5 and e1 == e2
    changed = json.loads(json.dumps(RAW))
    changed[3]["vacancy"]["schedule"] = "Гибкий график, 5/2"
    report.analyze_all(changed, counting, None, cache=report.Cache(path), salt="v1")
    assert calls[5:] == ["V3"]
    report.analyze_all(changed, counting, None, cache=report.Cache(path), salt="v2")   # сменился анализатор
    assert len(calls) == 11


def test_failed_analysis_is_not_cached(tmp_path):
    state = {"fail": True}

    def flaky(v, llm=None):
        if state["fail"]:
            raise RuntimeError("сбой")
        return good(v)

    path = tmp_path / "c.json"
    report.analyze_all(RAW[:1], flaky, None, cache=report.Cache(path))
    state["fail"] = False
    entries, stats = report.analyze_all(RAW[:1], flaky, None, cache=report.Cache(path))
    assert stats["from_cache"] == 0 and len(entries[0]["findings"]) == 1


def test_cache_keeps_contradiction_and_survives_garbage(tmp_path):
    raw = [{"id": "C", "duty": "Работа дистанционная.", "requirement": {"qualification": "Присутствие в офисе."}}]

    def an(v, llm=None):
        return [Finding("location", "contradiction", "duty", "дистанционная", "п", "р",
                        conflict_with=Evidence("requirement.qualification", "в офисе"), source="llm")]

    path = tmp_path / "c.json"
    e1, _ = report.analyze_all(raw, an, None, cache=report.Cache(path))
    e2, s2 = report.analyze_all(raw, an, None, cache=report.Cache(path))
    assert e1 == e2 and s2["from_cache"] == 1 and e2[0]["findings"][0]["conflict_with"]["quote"] == "в офисе"
    path.write_text("{испорчено", encoding="utf-8")
    e3, s3 = report.analyze_all(raw, an, None, cache=report.Cache(path))
    assert e3 == e1 and s3["from_cache"] == 0


def test_cache_drops_vacancies_gone_from_source(tmp_path):
    path = tmp_path / "c.json"
    report.analyze_all(RAW, good, None, cache=report.Cache(path))
    report.analyze_all(RAW[:2], good, None, cache=report.Cache(path))
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {"V0", "V1"}


class FakeLLM:
    def describe(self):
        return {"provider": "тест", "model": "m1"}

    def complete(self, system, user):
        return ""


def test_build_shape():
    entries, stats = report.analyze_all(RAW[:2], good, None)
    r = report.build(entries, stats, "file", "sample.json", FakeLLM())
    assert list(r) == ["generated_at", "mode", "source", "model", "stats", "vacancies"]
    assert r["model"] == {"provider": "тест", "model": "m1", "used": True, "calibrated": False}
    assert report.build(entries, stats, "api", "u", None)["model"]["used"] is False


def test_latest_history_and_error(tmp_path):
    entries, stats = report.analyze_all(RAW[:2], good, None)
    r1 = report.build(entries, stats, "api", "u", None)
    report.save_latest(r1, tmp_path)
    # тот же результат ещё раз: history не растёт
    r1b = dict(r1, generated_at="2026-11-25T10:00:00+02:00")
    report.save_latest(r1b, tmp_path)
    assert not (tmp_path / "history").exists()
    # сбой загрузки: latest прежний, появляется last_error
    err = report.save_error("API не ответил", "u", tmp_path)
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8")) == r1b
    e = json.loads(err.read_text(encoding="utf-8"))
    assert e["error"] == "API не ответил" and e["last_successful_report"].endswith("latest.json")
    # изменившийся результат: прежний уходит в history, ошибка снимается
    entries2, stats2 = report.analyze_all(RAW[:3], good, None)
    report.save_latest(report.build(entries2, stats2, "api", "u", None), tmp_path)
    assert len(list((tmp_path / "history").iterdir())) == 1 and not err.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_error_without_previous_report(tmp_path):
    e = json.loads(report.save_error("сбой", "u", tmp_path).read_text(encoding="utf-8"))
    assert e["last_successful_report"] is None and not (tmp_path / "latest.json").exists()
