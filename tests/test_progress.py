"""Показ хода проверки в панели: записи отдаются по мере готовности."""
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from pv import cli, config, report, source, web
from pv.models import Finding

RAW = [{"vacancy": {"id": f"V{i}", "job-name": "Специалист", "schedule": "Гибкий график"}} for i in range(6)]


def test_on_entry_called_once_per_vacancy_with_index_and_total():
    got, lock = [], threading.Lock()

    def on_entry(i, entry, total):
        with lock:
            got.append((i, entry["id"], total))

    def an(v, llm=None):
        return [Finding("schedule", "ambiguous", "schedule", "Гибкий график", "п", "р")]

    entries, _ = report.analyze_all(RAW, an, None, workers=3, on_entry=on_entry)
    assert sorted(got) == [(i, f"V{i}", 6) for i in range(6)]
    assert [e["id"] for e in entries] == [f"V{i}" for i in range(6)]


def test_broken_progress_callback_does_not_break_analysis():
    def boom(i, entry, total):
        raise RuntimeError("панель упала")

    entries, stats = report.analyze_all(RAW, lambda v, llm=None: [], None, on_entry=boom)
    assert len(entries) == 6 and stats["analyzer_errors"] == 0


@pytest.fixture
def slow(tmp_path, monkeypatch):
    """Панель с анализатором, который ждёт разрешения: так виден промежуточный ход."""
    gate = threading.Semaphore(0)

    def analyzer(v, llm=None):
        gate.acquire(timeout=5)
        return [Finding("schedule", "ambiguous", "schedule", "Гибкий график", "п", "р")] if v.schedule else []

    monkeypatch.setattr(cli, "get_analyzer", lambda: (analyzer, "t"))
    monkeypatch.setattr(cli, "make_llm", lambda c, enabled: None)
    monkeypatch.setattr(cli, "log", lambda m: None)
    cfg = config.Config(reports_dir=str(tmp_path / "reports"), workers=1)
    c = TestClient(web.create_app(cfg, auto_calibrate=False), base_url="http://127.0.0.1:8085")
    return c, gate, cfg


def wait(c, cond):
    for _ in range(200):
        p = c.get("/api/progress").json()
        if cond(p):
            return p
        time.sleep(0.02)
    raise AssertionError("не дождались: " + json.dumps(p, ensure_ascii=False)[:200])


def test_background_check_reports_entries_as_they_complete(slow):
    c, gate, cfg = slow
    assert c.get("/api/progress").json() == {"job": 0, "state": "idle", "entries": []}
    r = c.post("/api/check?name=my.json&background=1", content=json.dumps(RAW), headers={"content-type": "application/json"})
    assert r.status_code == 202 and r.json()["state"] == "running" and r.json()["job"] == 1
    # второй запуск, пока идёт первый, отклоняется
    assert c.post("/api/check-sample?background=1").status_code == 409
    gate.release(); gate.release()
    p = wait(c, lambda p: p["done"] == 2)
    assert p["total"] == 6 and p["with_findings"] == 2 and [e["_i"] for e in p["entries"]] == [0, 1]
    assert p["entries"][0]["findings"][0]["quote"] == "Гибкий график"
    assert c.get("/api/progress?since=2").json()["entries"] == []          # уже полученное заново не отдаётся
    assert not (cfg.reports / "check.json").exists()                       # отчёт пишется только целиком
    for _ in range(4):
        gate.release()
    p = wait(c, lambda p: p["state"] == "done")
    assert p["done"] == 6 and p["findings"] == 6 and p["entries"] == []
    rep = c.get("/api/report?kind=check").json()
    assert [v["id"] for v in rep["vacancies"]] == [f"V{i}" for i in range(6)] and rep["source"] == "my.json"
    # следующая проверка получает новый номер
    for _ in range(10):
        gate.release()
    assert c.post("/api/check-sample?background=1").json()["job"] == 2
    wait(c, lambda p: p["state"] == "done" and p["job"] == 2)


def test_background_run_failure_keeps_last_report(slow, monkeypatch):
    c, gate, cfg = slow
    for _ in range(10):
        gate.release()
    monkeypatch.setattr(source, "fetch_api", lambda cfg, log=None: (log("страница 1: получено 2 из 2"), RAW[:2])[1])
    c.post("/api/run?background=1")
    p = wait(c, lambda p: p["state"] == "done")
    assert p["kind"] == "latest" and p["done"] == 2

    def down(cfg, log=None):
        raise source.SourceError("API не ответил")

    monkeypatch.setattr(source, "fetch_api", down)
    c.post("/api/run?background=1")
    p = wait(c, lambda p: p["state"] == "failed")
    assert "API не ответил" in p["error"]
    assert len(c.get("/api/report?kind=latest").json()["vacancies"]) == 2


def test_synchronous_call_still_returns_report(slow):
    c, gate, _ = slow
    for _ in range(20):
        gate.release()
    rep = c.post("/api/check-sample").json()
    assert len(rep["vacancies"]) == 10
    assert c.get("/api/progress").json()["state"] == "done"


def test_pages_ship_live_scripts(slow):
    c, _, _ = slow
    assert "live.js" in c.get("/").text and "startJob" in c.get("/static/live.js").text
    assert c.get("/static/live.js").headers["cache-control"] == "no-cache"


def test_rules_preview_comes_first_when_model_is_slow(tmp_path, monkeypatch):
    """С моделью страница сразу получает замечания правил, а потом — уточнённые записи."""
    gate = threading.Semaphore(0)

    class Slow:
        def describe(self):
            return {"provider": "test", "model": "slow"}

        def complete(self, system, user):
            gate.acquire(timeout=5)
            return '{"findings": []}'

    from pv.analyze import profile
    monkeypatch.setattr(profile, "cached", lambda info, reports_dir=None: {"blocked": []})
    monkeypatch.setattr(cli, "make_llm", lambda c, enabled: Slow())
    monkeypatch.setattr(cli, "log", lambda m: None)
    cfg = config.Config(reports_dir=str(tmp_path / "reports"), workers=1)
    c = TestClient(web.create_app(cfg, auto_calibrate=False), base_url="http://127.0.0.1:8085")
    c.post("/api/check?background=1", content=json.dumps(RAW[:3]), headers={"content-type": "application/json"})
    p = wait(c, lambda p: len(p["entries"]) == 3)
    assert p["done"] == 0 and p["awaiting_model"] is True and all(e["pending"] for e in p["entries"])
    assert p["entries"][0]["findings"][0]["source"] == "rule"
    for _ in range(3):
        gate.release()
    p = wait(c, lambda p: p["state"] == "done")
    assert p["done"] == 3 and p["with_findings"] == 3
