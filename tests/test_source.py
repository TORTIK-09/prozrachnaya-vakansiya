import json

import httpx
import pytest

from pv import source
from pv.config import Config

URL = "http://api.test/v"
CFG = Config(source_url=URL, max_vacancies=150, page_size=100, retries=3)


def page(ids, total):
    return {"status": "200", "meta": {"total": total, "limit": len(ids)},
            "results": {"vacancies": [{"vacancy": {"id": i}} for i in ids]}}


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def nosleep(_):
    pass


def test_two_pages_until_limit():
    calls = []

    def h(req):
        p = int(req.url.params["offset"])
        calls.append((p, int(req.url.params["limit"])))
        return httpx.Response(200, json=page([f"{p}-{i}" for i in range(100)], 4232))

    out = source.fetch_api(CFG, client(h), sleep=nosleep)
    assert len(out) == 150 and calls == [(0, 100), (1, 100)]
    assert out[0]["id"] == "0-0" and out[-1]["id"] == "1-49"


def test_stops_when_page_is_short():
    def h(req):
        p = int(req.url.params["offset"])
        return httpx.Response(200, json=page([f"{p}-{i}" for i in range(100 if p == 0 else 30)], 9999))

    assert len(source.fetch_api(Config(source_url=URL, max_vacancies=1000), client(h), sleep=nosleep)) == 130


def test_stops_by_total_without_extra_request():
    calls = []

    def h(req):
        calls.append(int(req.url.params["offset"]))
        return httpx.Response(200, json=page([f"{calls[-1]}-{i}" for i in range(100)], 200))

    out = source.fetch_api(Config(source_url=URL, max_vacancies=1000), client(h), sleep=nosleep)
    assert len(out) == 200 and calls == [0, 1]


def test_duplicates_across_pages_counted_once():
    def h(req):
        p = int(req.url.params["offset"])
        ids = [f"a{i}" for i in range(100)] if p == 0 else ["a0", "a1"] + [f"b{i}" for i in range(98)]
        return httpx.Response(200, json=page(ids, 198))

    out = source.fetch_api(Config(source_url=URL, max_vacancies=500), client(h), sleep=nosleep)
    ids = [v["id"] for v in out]
    assert len(ids) == len(set(ids)) == 198


def test_retry_then_success():
    n = {"calls": 0}

    def h(req):
        n["calls"] += 1
        if n["calls"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=page(["x"], 1))

    waits = []
    assert len(source.fetch_api(CFG, client(h), sleep=waits.append)) == 1
    assert n["calls"] == 3 and len(waits) == 2


@pytest.mark.parametrize("status,body", [
    (500, "err"),
    (200, "<html>not json</html>"),
    (200, json.dumps({"status": "500", "meta": {"error": "сбой"}})),
    (200, "[1, 2]"),
])
def test_bad_answers_raise_source_error(status, body):
    with pytest.raises(source.SourceError) as e:
        source.fetch_api(CFG, client(lambda req: httpx.Response(status, text=body)), sleep=nosleep)
    assert "API не ответил" in str(e.value)


def test_network_error_raises_source_error():
    def h(req):
        raise httpx.ConnectError("нет связи")

    with pytest.raises(source.SourceError):
        source.fetch_api(CFG, client(h), sleep=nosleep)


def test_file_formats(tmp_path):
    one = page(["a", "b"], 2)
    variants = {
        "api.json": one,
        "pages.json": [one, page(["c"], 1)],
        "bare.json": [{"id": "a"}, {"vacancy": {"id": "b"}}],
        "vacancies.json": {"vacancies": [{"vacancy": {"id": "a"}}]},
        "empty.json": {"status": "200", "results": {}},
    }
    for name, data in variants.items():
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "bom.json").write_bytes(bytes([0xEF, 0xBB, 0xBF]) + json.dumps(one).encode())
    assert len(source.load_file(tmp_path / "api.json")) == 2
    assert len(source.load_file(tmp_path / "pages.json")) == 3
    assert len(source.load_file(tmp_path / "bare.json")) == 2
    assert len(source.load_file(tmp_path / "vacancies.json")) == 1
    assert source.load_file(tmp_path / "empty.json") == []
    assert len(source.load_file(tmp_path / "bom.json")) == 2


def test_file_errors(tmp_path):
    with pytest.raises(source.SourceError):
        source.load_file(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{не json", encoding="utf-8")
    with pytest.raises(source.SourceError):
        source.load_file(bad)
    odd = tmp_path / "odd.json"
    odd.write_text('{"что-то": 1}', encoding="utf-8")
    with pytest.raises(source.SourceError):
        source.load_file(odd)
