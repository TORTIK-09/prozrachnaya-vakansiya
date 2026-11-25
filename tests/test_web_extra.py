"""Веб-панель и сценарий с резюме."""
import json

import pytest
from fastapi.testclient import TestClient

from pv import cli, config, source, web
from pv.extra import resume as resume_mod


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "make_llm", lambda c, enabled: None)
    monkeypatch.setattr(cli, "log", lambda m: None)
    cfg = config.Config(reports_dir=str(tmp_path / "reports"))
    return TestClient(web.create_app(cfg, auto_calibrate=False), base_url="http://127.0.0.1:8085"), cfg


def test_page_and_status(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200 and "Прозрачная вакансия" in r.text
    assert r.headers["x-frame-options"] == "DENY" and "default-src 'self'" in r.headers["content-security-policy"]
    s = c.get("/api/status").json()
    assert s["model"] is None and s["has_latest"] is False
    assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404


def test_report_absent_then_sample_check(client):
    c, cfg = client
    assert c.get("/api/report?kind=check").status_code == 404
    assert c.get("/api/report?kind=other").status_code == 422
    rep = c.post("/api/check-sample").json()
    assert len(rep["vacancies"]) == 10 and rep["mode"] == "file"
    assert rep["vacancies"][0]["fields"]["job-name"] == "Специалист по документообороту"
    assert c.get("/api/report?kind=check").json()["vacancies"] == rep["vacancies"]
    assert not (cfg.reports / "latest.json").exists()


def test_upload(client):
    c, _ = client
    body = json.dumps({"results": {"vacancies": [{"vacancy": {"id": "U1", "job-name": "Специалист"}}]}})
    rep = c.post("/api/check?name=my.json", content=body, headers={"content-type": "application/json"}).json()
    assert rep["source"] == "my.json" and [v["id"] for v in rep["vacancies"]] == ["U1"]
    assert c.post("/api/check", content="{не json").status_code == 422
    assert c.post("/api/check", content='{"results": {"vacancies": []}}').status_code == 422
    assert c.post("/api/check", content='{"что-то": 1}').status_code == 422


def test_upload_report_has_only_uploaded_vacancies_and_sample_returns_without_reanalysis(tmp_path, monkeypatch):
    """Отчёт по своему JSON содержит только переданные вакансии; возврат к учебному набору — из сохранённого отчёта."""
    calls = []

    def analyzer(v, llm=None):
        calls.append(v.id)
        return []

    monkeypatch.setattr(cli, "get_analyzer", lambda: (analyzer, "t"))
    monkeypatch.setattr(cli, "make_llm", lambda c, enabled: None)
    monkeypatch.setattr(cli, "log", lambda m: None)
    cfg = config.Config(reports_dir=str(tmp_path / "reports"))
    c = TestClient(web.create_app(cfg, auto_calibrate=False), base_url="http://127.0.0.1:8085")
    assert len(c.post("/api/check-sample").json()["vacancies"]) == 10 and len(calls) == 10
    body = json.dumps({"results": {"vacancies": [{"vacancy": {"id": "IMP-1", "job-name": "Сторож"}}]}})
    rep = c.post("/api/check?name=first.json", content=body, headers={"content-type": "application/json"}).json()
    assert [v["id"] for v in rep["vacancies"]] == ["IMP-1"] and rep["source"] == "first.json"      # учебные не подмешиваются
    back = c.post("/api/check-sample?restore=1").json()
    assert len(back["vacancies"]) == 10 and back["source"] == "sample_vacancies.json" and len(calls) == 11   # без повторного разбора
    assert len(c.get("/api/report?kind=check").json()["vacancies"]) == 10
    c.post("/api/check-sample")                       # без restore — честная повторная проверка
    assert len(calls) == 21


def test_foreign_host_and_origin_rejected(client):
    c, _ = client
    assert c.get("/api/status", headers={"host": "evil.example"}).status_code == 403
    assert c.post("/api/check-sample", headers={"origin": "http://evil.example"}).status_code == 403
    assert c.post("/api/check-sample", headers={"origin": "http://127.0.0.1:8085"}).status_code == 200


def test_run_failure_keeps_report(client, monkeypatch):
    c, cfg = client
    monkeypatch.setattr(source, "fetch_api", lambda cfg, log=None: [{"vacancy": {"id": "L1", "job-name": "Сторож"}}])
    assert [v["id"] for v in c.post("/api/run?max_vacancies=5").json()["vacancies"]] == ["L1"]

    def down(cfg, log=None):
        raise source.SourceError("API не ответил")

    monkeypatch.setattr(source, "fetch_api", down)
    r = c.post("/api/run")
    assert r.status_code == 502 and "API не ответил" in r.json()["detail"]
    assert [v["id"] for v in r.json()["report"]["vacancies"]] == ["L1"]
    assert c.get("/api/status").json()["last_error"]["error"] == "API не ответил"
    assert c.post("/api/run?max_vacancies=0").status_code == 422


def test_resume_endpoint(client):
    c, _ = client
    assert c.post("/api/resume?resume_id=RES-01&vacancy_id=DEMO-39-001").status_code == 404   # отчёта ещё нет
    c.post("/api/check-sample")
    d = c.post("/api/resume?resume_id=RES-03&vacancy_id=DEMO-39-003").json()
    assert d["vacancy"]["id"] == "DEMO-39-003" and "SQL" in d["draft"]["skills_first"]
    assert c.post("/api/resume?resume_id=НЕТ&vacancy_id=DEMO-39-003").status_code == 404
    assert len(c.get("/api/resumes").json()) == 3


def entry(**fields):
    return {"id": "V", "title": "Аналитик", "company": "Организация", "url": None, "fields": fields, "findings": []}


def test_draft_never_invents():
    resumes = resume_mod.load_resumes()
    entries = [entry(**{"job-name": "Аналитик данных", "duty": "SQL-отчёты, очистка таблиц", "requirement.experience": "5",
                        "requirement.education": "Высшее"}),
               entry(**{"job-name": "Сварщик", "duty": "Сварка металлоконструкций"}), entry()]
    for r in resumes:
        for e in entries:
            d = resume_mod.draft(r, e)
            assert resume_mod.invented(d, r) == []
            assert sorted(d["draft"]["skills_first"] + d["draft"]["skills_other"]) == sorted(r["skills"])


def test_draft_orders_by_relevance_and_reports_gaps():
    r = next(x for x in resume_mod.load_resumes() if x["id"] == "RES-01")
    e = entry(**{"job-name": "Аналитик", "duty": "Работа с электронным архивом", "requirement.experience": "5",
                 "requirement.education": "Высшее"})
    e["findings"] = [{"topic": "schedule", "kind": "ambiguous", "problem": "п"},
                     {"topic": "schedule", "kind": "contradiction", "problem": "Гибкий график против пятидневки."},
                     {"topic": "salary", "kind": "missing", "problem": "п"}]
    d = resume_mod.draft(r, e)
    assert d["draft"]["skills_first"][0] == "Работа с электронным архивом"
    assert len(d["not_confirmed"]) == 2
    assert len(d["questions_for_employer"]) == 2            # по одному вопросу на тему
    assert d["cautions"] == ["Гибкий график против пятидневки."]


def test_resume_command(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "log", lambda m: None)
    monkeypatch.setattr(cli, "make_llm", lambda c, enabled: None)
    cfg = config.Config(reports_dir=str(tmp_path))
    rep = tmp_path / "check.json"
    assert cli.cmd_check(cfg, "examples/sample_vacancies.json", str(rep)) == 0
    out = tmp_path / "draft.json"
    assert cli.cmd_resume(cfg, "RES-02", "DEMO-39-007", str(rep), str(out)) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["resume_id"] == "RES-02"
    assert cli.cmd_resume(cfg, "RES-02", "НЕТ", str(rep), None) == 2


def test_human_verdict_on_disagreement(client):
    c, cfg = client
    c.post("/api/check-sample")
    q = "/api/verdict?vacancy_id=DEMO-39-004&topic=location&kind=contradiction&source=rule&choice="
    assert c.post(q + "llm").json() == {"ok": True, "choice": "llm"}
    f = next(v for v in c.get("/api/report?kind=check").json()["vacancies"] if v["id"] == "DEMO-39-004")["findings"][0]
    assert f["verdict"] == "llm"
    assert c.get("/api/stats").json()["verdicts"] == {"rule": 0, "llm": 1}
    c.post(q + "rule")
    assert c.get("/api/stats").json()["verdicts"] == {"rule": 1, "llm": 0}
    c.post(q + "none")
    f = next(v for v in c.get("/api/report?kind=check").json()["vacancies"] if v["id"] == "DEMO-39-004")["findings"][0]
    assert "verdict" not in f and c.post(q + "кто-то").status_code == 422
    # история: каждое изменение с прежним и новым выбором и временем; повтор того же выбора в неё не пишется
    assert [(h["from"], h["to"]) for h in f["history"]] == [(None, "llm"), ("llm", "rule"), ("rule", None)]
    assert all(h["at"].startswith("20") for h in f["history"])
    c.post(q + "none")
    f = next(v for v in c.get("/api/report?kind=check").json()["vacancies"] if v["id"] == "DEMO-39-004")["findings"][0]
    assert len(f["history"]) == 3 and c.get("/api/stats").json()["verdicts"] == {"rule": 0, "llm": 0}
    # сам файл отчёта решения человека не меняют
    import json as _json
    assert "verdict" not in (cfg.reports / "check.json").read_text(encoding="utf-8")


def test_sample_resume_draft_is_fresh_and_has_no_invented_facts():
    sample_path = config.ROOT / "examples" / "sample_resume_draft.json"
    assert sample_path.is_file()
    saved = json.loads(sample_path.read_text(encoding="utf-8"))
    rep = json.loads((config.ROOT / "examples" / "sample_report.json").read_text(encoding="utf-8"))
    res = next(r for r in resume_mod.load_resumes() if r["id"] == saved["resume_id"])
    vac = next(v for v in rep["vacancies"] if str(v["id"]) == str(saved["vacancy"]["id"]))
    assert saved == resume_mod.draft(res, vac)
    assert resume_mod.invented(saved, res) == []


def test_not_confirmed_checks_text_requirements_and_text_experience():
    rep = json.loads((config.ROOT / "examples" / "sample_report.json").read_text(encoding="utf-8"))
    by_id = {v["id"]: v for v in rep["vacancies"]}
    resumes = {r["id"]: r for r in resume_mod.load_resumes()}

    # подходящие кандидаты не получают ложных пробелов по своим навыкам
    assert resume_mod.draft(resumes["RES-01"], by_id["DEMO-39-001"])["not_confirmed"] == []
    assert resume_mod.draft(resumes["RES-02"], by_id["DEMO-39-007"])["not_confirmed"] == []

    # неподходящий кандидат получает конкретные неподтверждённые навыки из текста
    gaps_01_07 = resume_mod.draft(resumes["RES-01"], by_id["DEMO-39-007"])["not_confirmed"]
    assert any("Windows" in g for g in gaps_01_07)

    # опыт из текста (5 лет при 0 в поле) и SQL попадают в not_confirmed
    gaps_02_03 = resume_mod.draft(resumes["RES-02"], by_id["DEMO-39-003"])["not_confirmed"]
    assert any("5 лет" in g for g in gaps_02_03)
    assert any("SQL" in g for g in gaps_02_03)

    # общие личные качества и «приветствуется» не считаются неподтверждёнными требованиями
    assert resume_mod.draft(resumes["RES-03"], by_id["DEMO-39-002"])["not_confirmed"] == []
    assert resume_mod.draft(resumes["RES-03"], by_id["DEMO-39-006"])["not_confirmed"] == []


