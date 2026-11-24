import json
from datetime import datetime

import pytest

from pv import cli, config, cron, source
from pv.models import Finding


def cfg(tmp_path, **kw):
    return config.Config(source_url="http://api.test/v", reports_dir=str(tmp_path / "reports"), **kw)


def raw(n):
    return [{"vacancy": {"id": f"V{i}", "schedule": "Гибкий график"}} for i in range(n)]


def finder(v, llm=None):
    return [Finding("schedule", "ambiguous", "schedule", "Гибкий график", "п", "р")]


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(cli, "get_analyzer", lambda: (finder, "t1"))
    monkeypatch.setattr(cli, "make_llm", lambda c, enabled: None)
    monkeypatch.setattr(cli, "log", lambda m: None)


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def test_check_sample_gives_ten_entries(tmp_path, patched):
    out = tmp_path / "check.json"
    assert cli.cmd_check(cfg(tmp_path), "examples/sample_vacancies.json", str(out)) == 0
    r = read(out)
    assert r["mode"] == "file" and len(r["vacancies"]) == 10
    assert all(set(v) >= {"id", "url", "findings"} for v in r["vacancies"])
    assert not (tmp_path / "reports" / "latest.json").exists()      # проверочный режим не трогает рабочий отчёт


def test_check_missing_file_is_error(tmp_path, patched):
    assert cli.cmd_check(cfg(tmp_path), str(tmp_path / "none.json"), str(tmp_path / "o.json")) == 2
    assert not (tmp_path / "o.json").exists()


def test_run_twice_same_result_then_api_failure_keeps_report(tmp_path, patched, monkeypatch):
    c = cfg(tmp_path)
    monkeypatch.setattr(source, "fetch_api", lambda cfg, log=None: raw(3))
    assert cli.cmd_run(c) == 0
    latest = c.reports / "latest.json"
    first = read(latest)
    assert cli.cmd_run(c) == 0
    second = read(latest)
    assert [v["id"] for v in second["vacancies"]] == ["V0", "V1", "V2"]
    assert second["vacancies"] == first["vacancies"] and second["stats"]["from_cache"] == 3
    assert not (c.reports / "history").exists()

    def down(cfg, log=None):
        raise source.SourceError("API не ответил (страница 0, попыток: 3): ConnectError")

    monkeypatch.setattr(source, "fetch_api", down)
    assert cli.cmd_run(c) == 2
    assert read(latest) == second
    assert "API не ответил" in read(c.reports / "last_error.json")["error"]


def test_run_empty_source_is_error_and_keeps_report(tmp_path, patched, monkeypatch):
    c = cfg(tmp_path)
    monkeypatch.setattr(source, "fetch_api", lambda cfg, log=None: raw(2))
    cli.cmd_run(c)
    monkeypatch.setattr(source, "fetch_api", lambda cfg, log=None: [])
    assert cli.cmd_run(c) == 2
    assert len(read(c.reports / "latest.json")["vacancies"]) == 2


def test_main_bad_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "log", lambda m: None)
    assert cli.main(["--config", str(tmp_path / "none.toml"), "check", "x.json"]) == 2
    assert cli.main(["run", "--max", "0"]) == 2


def test_config_layers(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text('max_vacancies = 50\nschedule = "*/30 * * * *"\n', encoding="utf-8")
    envf = tmp_path / ".env"
    envf.write_text("# ключ\nPV_LLM_API_KEY='secret'\nPV_MAX_VACANCIES=70\n", encoding="utf-8")
    c = config.load(toml, env={}, env_file=envf)
    assert (c.max_vacancies, c.schedule, c.llm_api_key) == (70, "*/30 * * * *", "secret")
    c = config.load(toml, env={"PV_MAX_VACANCIES": "90", "PV_SOURCE_URL": "http://x"}, env_file=envf)
    assert (c.max_vacancies, c.source_url) == (90, "http://x")
    with pytest.raises(ValueError):
        config.load(toml, env={"PV_MAX_VACANCIES": "много"}, env_file=envf)
    bad = tmp_path / "bad.toml"
    bad.write_text("unknown = 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        config.load(bad, env={}, env_file=envf)


def test_cron():
    t = datetime(2026, 11, 24, 17, 20)                      # вторник
    assert cron.next_run("0 6 * * *", t) == datetime(2026, 11, 25, 6, 0)
    assert cron.next_run("*/15 * * * *", t) == datetime(2026, 11, 24, 17, 30)
    assert cron.next_run("30 9 * * 1-5", datetime(2026, 11, 27, 10, 0)) == datetime(2026, 11, 30, 9, 30)
    assert cron.next_run("0 0 1 * *", t) == datetime(2026, 12, 1, 0, 0)
    assert cron.next_run("0 12 * * 7", t) == datetime(2026, 11, 29, 12, 0)      # 7 — тоже воскресенье
    for bad in ("0 6 * *", "61 * * * *", "a * * * *", "*/0 * * * *", "5-1 * * * *", "0 0 31 2 *"):
        with pytest.raises(ValueError):
            cron.next_run(bad, t)
