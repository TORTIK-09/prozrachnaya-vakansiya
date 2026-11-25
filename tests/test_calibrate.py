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
    # придирки отключены; единственное верное замечание тоже — одного подтверждения мало, доверие нужно заслужить
    assert r["blocked"] == [["requirements", "ambiguous"], ["title", "contradiction"]]
    cats = {(c["topic"], c["kind"]): c for c in r["categories"]}
    assert cats[("title", "contradiction")] == {"topic": "title", "kind": "contradiction", "ok": 1, "extra": 0, "blocked": True}
    extra, sample = modes(r, "extra"), modes(r, "sample")
    assert extra["rules"]["must_found"] == 7 and extra["llm"]["must_found"] == 1 and extra["both"]["must_found"] == 7 and extra["both_raw"]["must_found"] == 8
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
    st = c.get("/api/calibration").json()
    assert st["state"] == "no_model" and st["mode"] == "rules_only" and st["meta"] is None
    s = c.get("/api/stats").json()
    assert s["stats"]["model"] is None and set(s["stats"]["sets"][0]["modes"]) == {"rules"}
    assert c.post("/api/calibration").status_code == 409
    assert "Статистика" in c.get("/stats").text and c.get("/static/app.css").status_code == 200


def test_unknown_model_is_calibrated_automatically_then_recognised(tmp_path, monkeypatch):
    c = client_with(Fake(), tmp_path, monkeypatch)
    first = c.get("/api/calibration").json()
    assert first["state"] == "running" and first["model"]["model"] == "fake-1"
    done = wait_done(c)
    assert done["state"] == "done" and 10 <= done["vacancies"] <= done["available"] == 32     # число вакансий — по плану калибровки
    stats = c.get("/api/stats").json()["stats"]
    assert stats["model"]["model"] == "fake-1" and "both" in stats["sets"][0]["modes"]
    # после перезапуска панели та же модель узнаётся и заново не настраивается
    llm2 = Fake()
    c2 = client_with(llm2, tmp_path, monkeypatch)
    st = c2.get("/api/calibration").json()
    assert st["state"] == "ready" and llm2.calls == 0 and st["mode"] == "local" and st["meta"]["strength"] == "weak"
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


def test_calibration_limit_takes_from_every_set_and_trust_must_be_earned():
    llm = Fake()
    r = calibrate.run(llm, limit=9)
    assert r["vacancies"] == 9 and llm.calls == 9 and r["available"] == 32
    assert [s["modes"]["rules"]["vacancies"] for s in r["sets"]] == [3, 3, 3]
    assert all(c["blocked"] for c in r["categories"] if c["ok"] < 2)


def test_settings_modes_and_plan(tmp_path, monkeypatch):
    from pv import llm as llm_mod
    monkeypatch.setattr(cli, "log", lambda m: None)
    monkeypatch.setattr(llm_mod, "list_local_models", lambda base_url="": ["qwen2.5:3b"])
    seen = {}

    def make(cfg, enabled):
        saved = llm_mod.load_runtime_settings(cfg.reports) or {}
        seen["saved"] = saved
        if saved.get("mode_type", "rules_only") == "rules_only":
            return None
        m = Fake(saved["model"])
        m.describe = lambda: {"provider": "test", "model": saved["model"], "mode_type": saved["mode_type"], "base_url": saved["base_url"]}
        return m

    monkeypatch.setattr(cli, "make_llm", make)
    cfg = config.Config(reports_dir=str(tmp_path / "reports"))
    c = TestClient(web.create_app(cfg, auto_calibrate=False), base_url="http://127.0.0.1:8085")
    s = c.get("/api/settings").json()
    assert s["mode"] == "rules_only" and "ollama" in s["providers"] and s["local_models"] == ["qwen2.5:3b"] and s["hardware"]["cpu_cores"] >= 1
    body = {"mode_type": "custom_api_only", "provider": "custom", "base_url": "https://api.example/v1", "model": "big-70b", "api_key": "k-123", "calib_minutes": 5}
    r = c.post("/api/settings", content=json.dumps(body)).json()
    assert r["mode"] == "custom_api_only" and r["calibration"]["meta"]["strength"] == "strong"
    s = c.get("/api/settings").json()
    assert s["saved"]["has_api_key"] is True and "api_key" not in s["saved"] and "k-123" not in json.dumps(s)
    # ключ не теряется, если поле оставили пустым для того же сервера
    c.post("/api/settings", content=json.dumps({**body, "api_key": "", "mode_type": "custom_api"}))
    assert seen["saved"]["api_key"] == "k-123" and c.get("/api/calibration").json()["mode"] == "custom_api"
    assert c.post("/api/settings", content=json.dumps({"mode_type": "чужой"})).status_code == 422
    assert c.post("/api/settings", content=json.dumps({"mode_type": "local", "base_url": "ftp://x", "model": "m"})).status_code == 422
    c.post("/api/settings", content=json.dumps({"mode_type": "rules_only"}))
    assert c.get("/api/calibration").json()["state"] == "no_model"
    p = c.get("/api/plan?model=qwen2.5:3b&mode_type=local&minutes=600").json()
    assert p["model_meta"]["strength"] == "weak" and p["time_trimmed"] is True and p["llm_target_tests"] <= p["available_tests"] == 32
    assert c.get("/api/plan?minutes=0").status_code == 422


def test_ai_only_mode_skips_system_ai(monkeypatch):
    llm = Fake()
    llm.describe = lambda: {"provider": "test", "model": "fake-1", "mode_type": "local_only"}
    v = normalize({"id": "K", "job-name": "Водитель автомобиля", "duty": "Ведение бухгалтерского учёта, сверка расчётов.",
                   "schedule": "Гибкий график", "requirement": {"experience": 1}})
    assert {(f.topic, f.source) for f in analyze(v, llm)} == {("title", "llm")}


def test_budget_measures_speed_then_fits_time():
    llm = Fake()
    r = calibrate.run(llm, budget_seconds=0.0001, min_tests=7)        # времени нет совсем — берётся минимум
    assert r["vacancies"] == 7 and llm.calls == 7
    assert sum(s["modes"]["rules"]["vacancies"] for s in r["sets"]) == 7 and len(r["sets"]) == 3
    r = calibrate.run(Fake(), budget_seconds=3600, min_tests=5, limit=12)   # времени много — упирается в потолок
    assert r["vacancies"] == 12
    assert set(r["template"]) == {"findings", "top_kind_share", "on_clean_per_vacancy", "templated"} and r["examples_in_prompt"] == 0


def test_prompt_examples_depend_on_model_strength():
    from pv.analyze import ai

    class M:
        def __init__(self, s): self.strength = s
    weak, mid, strong = (ai.system_prompt(M(x)) for x in ("weak", "medium", "strong"))
    assert "Продавец-кассир" not in weak and "Продавец-кассир" in mid and "Специалист отдела сопровождения" not in mid
    assert "Специалист отдела сопровождения" in strong and "{examples}" not in weak + mid + strong
    assert ai.system_prompt(None) == weak          # неизвестная модель — как слабая


def test_five_modes_table_has_numbers_only_for_calibrated_modes(tmp_path, monkeypatch):
    c = client_with(Fake(), tmp_path, monkeypatch)
    c.get("/api/calibration"); wait_done(c)
    rows = {r["mode"]: r for r in c.get("/api/stats").json()["modes"]}
    assert list(rows) == ["rules_only", "local", "local_only", "custom_api", "custom_api_only"]
    assert rows["rules_only"]["found"] == 27 and rows["rules_only"]["extra"] == 0
    # подставная модель не локальная: её числа — в строках «по API», а локальные режимы не измерялись
    assert rows["custom_api"]["model"] == "fake-1" and rows["custom_api"]["found"] >= rows["custom_api_only"]["found"]
    # локальная строка — из профиля, поставляемого с проектом, если он есть; иначе «не измерялось»
    assert rows["local"]["model"] in (None, "qwen2.5:3b") and ("found" in rows["local"]) == (rows["local"]["model"] is not None)


def test_unknown_model_is_remembered_in_local_table(tmp_path):
    from pv import llm as llm_mod
    seen = llm_mod.remember_model(tmp_path, {"model": "Mystery", "strength": "medium", "params_b": 20.0, "params_label": "~20", "detection_method": "ollama_api_show"})
    assert seen["strength"] == "medium" and "mystery" in llm_mod.load_known_models(tmp_path)
    again = llm_mod.remember_model(tmp_path, {"model": "mystery", "strength": "weak", "params_b": None, "params_label": "?", "detection_method": "unknown"})
    assert again["strength"] == "medium" and again["detection_method"] == "local_table"
    fresh = llm_mod.remember_model(tmp_path, {"model": "never-seen", "strength": "weak", "params_b": None, "params_label": "?", "detection_method": "unknown"})
    assert fresh["strength"] == "weak" and "never-seen" not in llm_mod.load_known_models(tmp_path)
