from pv import source
from pv.normalize import normalize, number, text

SAMPLE = "examples/sample_vacancies.json"


def test_sample_all_ten_normalized():
    vs = [normalize(r) for r in source.load_file(SAMPLE)]
    assert len(vs) == 10 and all(vs)
    assert [v.id for v in vs][:2] == ["DEMO-39-001", "DEMO-39-002"]
    v1, v2, v6 = vs[0], vs[1], vs[5]
    assert v1.title == "Специалист по документообороту" and v1.url is None
    assert (v1.salary_min, v1.salary_max) == (45000, 55000)
    assert (v2.salary_min, v2.salary_max) == (None, None)        # 0 → не указано
    assert (v6.salary_min, v6.salary_max) == (35000, None)
    assert v1.experience_years == 0                                # ноль опыта — это значение, не пропуск
    assert v1.addresses == ("г. Калининград, ул. Учебная, д. 1",)
    assert v1.text_of("schedule").startswith("Полный рабочий день")


def test_wrapped_and_bare_object():
    assert normalize({"vacancy": {"id": "A"}}).id == "A"
    assert normalize({"id": 7}).id == "7"


def test_no_id_or_garbage_gives_none():
    for bad in ({}, {"vacancy": {}}, {"id": ""}, {"id": "  "}, None, "x", 5, []):
        assert normalize(bad) is None


def test_html_and_spaces_cleaned():
    assert text("<p>Первое&nbsp;дело</p><ul><li>раз</li><li>два</li></ul>") == "Первое дело\nраз\nдва"
    assert text("  a \t b  ") == "a b"
    assert text("") is None and text("   ") is None and text(None) is None and text({"a": 1}) is None


def test_numbers():
    assert number("45 000", True) == 45000
    assert number(0, True) is None and number(0, False) == 0
    assert number("abc", True) is None and number(None, True) is None and number(-5, False) is None
    assert number(True, False) is None


def test_live_api_shape():
    raw = {
        "id": "x1", "vac_url": "https://trudvsem.ru/vacancy/card/1/x1", "job-name": "Инженер",
        "salary": "от 45000", "salary_min": 45000, "salary_max": 66000,
        "requirement": {"education": "Высшее", "experience": 1},
        "addresses": {"address": [{"location": "г Калининград, Больничная улица", "lat": "54.7"}]},
        "term": {"text": "QUARTERLY 100 %"}, "shift": ["Дневная смена"], "scheduleTypeComment": "2/2",
        "contact_list": [{"contact_type": "Телефон", "contact_value": "+7(000) 000-00-00"}],
        "contact_person": "Иванов И.И.", "company": {"name": "Организация", "email": "hr@example.org", "inn": "1"},
    }
    v = normalize(raw)
    assert v.url.endswith("/x1") and v.salary_text == "от 45000" and v.term_text == "QUARTERLY 100 %"
    assert v.shifts == ("Дневная смена",) and v.schedule_comment == "2/2" and v.employment is None
    assert v.company == "Организация"
    # контакты не попадают в Vacancy, исходный объект при этом не меняется
    assert "contact_list" not in v.raw and "contact_person" not in v.raw and "email" not in v.raw["company"]
    assert "contact_list" in raw and "email" in raw["company"]


def test_broken_nested_types_do_not_crash():
    v = normalize({"id": "b", "requirement": "нет", "addresses": [], "term": 5, "company": "ООО", "shift": "ночь"})
    assert v.id == "b" and v.addresses == () and v.education is None and v.shifts == ("ночь",)
