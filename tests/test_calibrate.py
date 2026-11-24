"""Настройка модели, профиль доверия и страница статистики — на подставной модели."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from pv import calibrate, cli, config, web
from pv.analyze import analyze, merge, profile
from pv.models import Finding
from pv.normalize import normalize


class Fake:
    """Находит одно верное смысловое противоречие и придирается к понятным вакансиям."""

    def __init__(self, name="fake-1", fail=False):
        self.name, self.fail, self.calls = name, fail, 0

    def describe(self):
        return {"provider": "test", "model": self.name}

    def complete(self, system, user):
        self.calls += 1
        if self.fail:
            raise RuntimeError("нет связи")
        fields = json.loads(user[user.index("{"):user.rindex("}") + 1])
        out = []
        if fields.get("job-name") == "Водитель автомобиля":       # то, чего правила не видят: должность и обязанности о разном
            out.append({"topic": "title", "kind": "contradiction", "field": "job-name", "quote": "Водитель автомобиля",
                        "problem": "п", "recommendation": "р", "conflict_field": "duty", "conflict_quote": "Ведение бухгалтерского учёта"})
        if fields.get("job-name") in ("Библиотекарь", "Повар горячего цеха"):
            out.append({"topic": "requirements", "kind": "ambiguous", "field": "requirements",
                        "quote": fields["requirements"].rstrip("."), "problem": "п", "recommendation": "р"})
        return json.dumps({"findings": out}, ensure_ascii=False)


@pytest.fixture(autouse=True)
def fresh_profiles():
    profile.forget()
    yield
    profile.forget()


def modes(result, key):
    return next(s for s in result["sets"] if s["key"] == key)["modes"]


def test_rules_only_stats_need_no_model():
    r = calibrate.rules_only()
    assert r["model"] is None and [s["key"] for s in r["sets"]] == ["sample", "extra", "holdout"]
    sample, extra, hold = (x["modes"]["rules"] for x in r["sets"])
    assert sample["recall"] == 1.0 and sample["extra_total"] == 0
    assert (extra["must_found"], extra["must_total"], extra["extra_total"]) == (7, 8, 0)   # кроме «должность против обязанностей»
    assert hold["recall"] == 1.0 and hold["extra_total"] == 0 and hold["extra_on_clean"] == 0


def test_calibration_measures_three_ways_and_blocks_unreliable_kind():
    llm = Fake()
    seen = []
    r = calibrate.run(llm, lambda done, total, sec: seen.append((done, total)))
    assert llm.calls == r["vacancies"] == 32 and seen[-1] == (32, 32) and r["errors"] == 0
    assert r["blocked"] == [["requirements", "ambiguous"]]             # две придирки, ни одного попадания
    cats = {(c["topic"], c["kind"]): c for c in r["categories"]}
    assert cats[("title", "contradiction")] == {"topic": "title", "kind": "contradiction", "ok": 1, "extra": 0, "blocked": False}
    extra, sample = modes(r, "extra"), modes(r, "sample")
    assert extra["rules"]["must_found"] == 7 and extra["llm"]["must_found"] == 1 and extra["both"]["must_found"] == 8
    assert extra["both_raw"]["extra_total"] == 1 and extra["both"]["extra_total"] == 0
    assert sample["both_raw"]["extra_on_clean"] == 1 and sample["both"]["extra_on_clean"] == 0
    assert sample["both"]["recall"] == 1.0 and sample["llm"]["recall"] < 1.0


def test_saved_profile_is_applied_in_analysis(tmp_path):
    llm = Fake()
    v = normalize({"id": "B", "job-name": "Библиотекарь", "duty": "Выдача и приём книг, ведение формуляров.",
                   "requirements": "Работа с каталогом", "salary_min": 38000, "salary_max": 38000,
                   "schedule": "Понедельник–пятница, 10:00–19:00", "employment": "Полная занятость",
                   "requirement": {"education": "Среднее", "experience": 0},
                   "addresses": {"address": [{"location": "г. Светлогорск, ул. Примерная, д. 3"}]}})
    assert analyze(v, llm) == [] and llm.calls == 0                     # до настройки модели не верим вовсе
    path = calibrate.save(calibrate.run(llm), tmp_path)
    assert path.name == "test__fake-1.json" and profile.load(llm.describe(), tmp_path)["blocked"]
    blocked = profile.blocked(profile.load(llm.describe(), tmp_path))
    assert merge([], [Finding("requirements", "ambiguous", "requirements", "Работа с каталогом", "п", "р", source="llm")], blocked) == []
    assert profile.load({"provider": "test", "model": "другая"}, tmp_path) is None      # другая модель — другой профиль
    assert profile.slug({"provider": "ollama", "model": "qwen2.5:3b"}) == "ollama__qwen2.5-3b"


def test_modes_of_analyze(monkeypatch):
    llm = Fake()
    v = normalize({"id": "K", "job-name": "Водитель автомобиля", "duty": "Ведение бухгалтерского учёта, сверка расчётов.",
                   "requirements": "Знание складского учёта", "salary_min": 1, "salary_max": 2, "schedule": "Гибкий график",
                   "employment": "Полная занятость", "requirement": {"experience": 1},
                   "addresses": {"address": [{"location": "г. Калининград, ул. Учебная, д. 7"}]}})
    assert {(f.topic, f.source) for f in analyze(v, llm, "rules")} == {("schedule", "rule")}
    assert {(f.topic, f.source) for f in analyze(v, llm, "llm")} == {("title", "llm")}
    assert {(f.topic, f.source) for f in analyze(v, llm, "both")} == {("schedule", "rule")}       # не настроена
    monkeypatch.setattr(profile, "cached", lambda info, reports_dir=None: {"blocked": []})
    assert {(f.topic, f.source) for f in analyze(v, llm, "both")} == {("schedule", "rule"), ("title", "llm")}
    assert analyze(v, None, "llm") == []


def test_calibration_with_dead_model_counts_errors():
    r = calibrate.run(Fake(fail=True))
    assert r["errors"] == 32 and r["blocked"] == [] and modes(r, "sample")["both"]["recall"] == 1.0


def client_with(llm, tmp_path, monkeypatch, auto=True):
    monkeypatch.setattr(cli, "make_llm", lambda c, enabled: llm)
    monkeypatch.setattr(cli, "log", lambda m: None)
    cfg = config.Config(reports_dir=str(tmp_path / "reports"))
    return TestClient(web.create_app(cfg, auto_calibrate=auto), base_url="http://127.0.0.1:8085")


def wait_done(c):
    for _ in range(200):
        st = c.get("/api/calibration").json()
        if st["state"] != "running":
            return st
        time.sleep(0.05)
    raise AssertionError("настройка не закончилась")


def test_panel_without_model(tmp_path, monkeypatch):
    c = client_with(None, tmp_path, monkeypatch)
    assert c.get("/api/calibration").json() == {"state": "no_model"}
    s = c.get("/api/stats").json()
    assert s["stats"]["model"] is None and set(s["stats"]["sets"][0]["modes"]) == {"rules"}
    assert c.post("/api/calibration").status_code == 409
    assert "Статистика" in c.get("/stats").text and c.get("/static/app.css").status_code == 200


def test_unknown_model_is_calibrated_automatically_then_recognised(tmp_path, monkeypatch):
    c = client_with(Fake(), tmp_path, monkeypatch)
    first = c.get("/api/calibration").json()
    assert first["state"] == "running" and first["model"]["model"] == "fake-1"
    done = wait_done(c)
    assert done["state"] == "done" and done["vacancies"] == 32 and done["blocked"] == 1
    stats = c.get("/api/stats").json()["stats"]
    assert stats["model"]["model"] == "fake-1" and "both" in stats["sets"][0]["modes"]
    # после перезапуска панели та же модель узнаётся и заново не настраивается
    llm2 = Fake()
    c2 = client_with(llm2, tmp_path, monkeypatch)
    st = c2.get("/api/calibration").json()
    assert st["state"] == "ready" and st["blocked"] == 1 and llm2.calls == 0
    # а модель с другим именем — незнакомая
    c3 = client_with(Fake("fake-2"), tmp_path, monkeypatch)
    assert c3.get("/api/calibration").json()["state"] == "running"
    wait_done(c3)


def test_failed_calibration_is_reported(tmp_path, monkeypatch):
    c = client_with(Fake(fail=True), tmp_path, monkeypatch)
    c.get("/api/calibration")
    st = wait_done(c)
    assert st["state"] == "failed" and "ни на один запрос" in st["error"]


def test_manual_mode_does_not_start_by_itself(tmp_path, monkeypatch):
    llm = Fake()
    c = client_with(llm, tmp_path, monkeypatch, auto=False)
    assert c.get("/api/calibration").json()["state"] == "needed" and llm.calls == 0
    assert c.post("/api/calibration").json()["state"] == "running"
    assert wait_done(c)["state"] == "done"


def test_agreement_takes_model_wording_and_rule_quotes():
    rule = Finding("schedule", "ambiguous", "schedule", "Гибкий график", "текст правила", "совет правила")
    other = Finding("salary", "missing", "salary_min", "", "п", "р")
    llm_same = Finding("schedule", "ambiguous", "schedule", "Гибкий", "текст ИИ", "совет ИИ", source="llm")
    llm_new = Finding("duties", "ambiguous", "duty", "Работа", "п", "р", source="llm")
    out = {f.topic: f for f in merge([rule, other], [llm_same, llm_new], set())}
    assert (out["schedule"].source, out["schedule"].problem, out["schedule"].quote) == ("both", "текст ИИ", "Гибкий график")
    assert out["salary"].source == "rule" and out["duties"].source == "llm" and len(out) == 3
    # ненадёжный вид отключается только там, где ИИ один; совпадение с правилом остаётся
    blocked = {("schedule", "ambiguous"), ("duties", "ambiguous")}
    assert sorted(f.source for f in merge([rule, other], [llm_same, llm_new], blocked)) == ["both", "rule"]


def test_calibration_counts_who_sees_what():
    r = calibrate.run(Fake())
    a = {s["key"]: s["agreement"] for s in r["sets"]}
    assert a["extra"]["llm_only"] == [2, 1] and a["extra"]["both"] == [0, 0] and a["extra"]["rules_only"] == [7, 7]
    assert a["sample"]["rules_only"][0] == a["sample"]["rules_only"][1] > 10
