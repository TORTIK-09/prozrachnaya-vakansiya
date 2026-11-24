"""Контракт: что считается годным замечанием."""
from pv.models import Evidence, Finding, Vacancy, validate


def vac(**kw) -> Vacancy:
    base = dict(
        id="T-1", url=None, title="Специалист", duty="Работа с документами.", requirements=None,
        qualification=None, education=None, experience_years=0,
        req_qualification="Требуется не менее 5 лет опыта.", salary_text=None, salary_min=None,
        salary_max=None, currency="RUB", term_text=None, benefit=None, schedule="Гибкий график",
        schedule_comment=None, shifts=(), employment=None, addresses=("г. Калининград",),
        conditions=None, company=None,
    )
    return Vacancy(**{**base, **kw})


def finding(**kw) -> Finding:
    base = dict(topic="schedule", kind="ambiguous", field="schedule", quote="Гибкий график",
                problem="Нет дней и часов.", recommendation="Уточните рабочие дни и время.")
    return Finding(**{**base, **kw})


def test_good_ambiguous_passes():
    assert validate(finding(), vac()) == []


def test_quote_must_be_in_field():
    assert validate(finding(quote="Свободный график"), vac())


def test_missing_has_empty_quote():
    assert validate(finding(topic="salary", kind="missing", field="salary_min", quote=""), vac()) == []
    assert validate(finding(topic="salary", kind="missing", field="salary_min", quote="0"), vac())


def test_ambiguous_needs_quote():
    assert validate(finding(quote=""), vac())


def test_contradiction_needs_both_values():
    ok = finding(topic="requirements", kind="contradiction", field="requirement.experience", quote="0",
                 conflict_with=Evidence("requirement.qualification", "не менее 5 лет"))
    assert validate(ok, vac()) == []
    assert validate(finding(kind="contradiction"), vac())
    bad = finding(topic="requirements", kind="contradiction", field="requirement.experience", quote="0",
                  conflict_with=Evidence("requirement.qualification", "не менее 3 лет"))
    assert validate(bad, vac())


def test_unknown_topic_kind_field_rejected():
    assert validate(finding(topic="pay"), vac())
    assert validate(finding(kind="vague"), vac())
    assert validate(finding(field="description"), vac())


def test_conflict_only_for_contradiction():
    assert validate(finding(conflict_with=Evidence("duty", "Работа")), vac())


def test_to_json_omits_empty_conflict():
    assert "conflict_with" not in finding().to_json()
    j = finding(kind="contradiction", conflict_with=Evidence("duty", "Работа")).to_json()
    assert j["conflict_with"] == {"field": "duty", "quote": "Работа"}
