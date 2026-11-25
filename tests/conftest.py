"""Тесты не зависят от того, что сервис накопил на этом компьютере: нормы берутся только поставляемые с проектом."""
import pytest

from pv.analyze import norms


@pytest.fixture(autouse=True)
def shipped_norms_only(tmp_path_factory, monkeypatch):
    missing = tmp_path_factory.mktemp("no-local-norms") / "experience_norms.json"
    monkeypatch.setattr(norms, "local_path", lambda reports_dir=None: missing if reports_dir is None else reports_dir / "experience_norms.json")
    norms.forget()
    yield
    norms.forget()
