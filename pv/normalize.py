"""Сырой объект вакансии из API → Vacancy. Одинаково для учебного файла и живого API."""
from __future__ import annotations

import copy
import html
import re
from typing import Any

from .models import Vacancy

_TAG = re.compile(r"<[^>]{0,200}>")
_BREAK = re.compile(r"<\s*(br|/p|/li|/div|/ul|/ol)\s*/?\s*>", re.I)
_SPACE = re.compile(r"[ \t\r\f\v ]+")
_NEWLINES = re.compile(r"\s*\n\s*")

# Контактные данные в Vacancy не попадают: в модель и в репозиторий они не уходят.
_CONTACT_KEYS = ("contact_list", "contact_person")
_COMPANY_CONTACT_KEYS = ("email", "phone", "fax", "contact_person")


def text(value: Any) -> str | None:
    """Строка без HTML и лишних пробелов; пустое → None."""
    if value is None or isinstance(value, (dict, list, bool)):
        return None
    s = str(value)
    s = _BREAK.sub("\n", s)
    s = _TAG.sub(" ", s)
    s = html.unescape(s)
    s = _SPACE.sub(" ", s)
    s = _NEWLINES.sub("\n", s).strip()
    return s or None


def number(value: Any, zero_is_none: bool) -> int | None:
    """Целое из числа или строки. Для зарплаты 0 означает «не указано»."""
    if value is None or isinstance(value, bool):
        return None
    try:
        n = int(float(str(value).replace(" ", "").replace(",", ".")))
    except ValueError:
        return None
    if n < 0 or (n == 0 and zero_is_none):
        return None
    return n


def _texts(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    items = value if isinstance(value, list) else [value]
    out = []
    for it in items:
        t = text(it.get("location") or it.get("name") or it.get("text") if isinstance(it, dict) else it)
        if t:
            out.append(t)
    return tuple(out)


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _strip_contacts(raw: dict) -> dict:
    out = copy.deepcopy(raw)
    for k in _CONTACT_KEYS:
        out.pop(k, None)
    if isinstance(out.get("company"), dict):
        for k in _COMPANY_CONTACT_KEYS:
            out["company"].pop(k, None)
    return out


def normalize(raw: Any) -> Vacancy | None:
    """None — если это не вакансия или у неё нет ID."""
    if isinstance(raw, dict) and isinstance(raw.get("vacancy"), dict):
        raw = raw["vacancy"]
    if not isinstance(raw, dict):
        return None
    vid = text(raw.get("id"))
    if not vid:
        return None
    req = _dict(raw.get("requirement"))
    addresses = _dict(raw.get("addresses")).get("address")
    return Vacancy(
        id=vid,
        url=text(raw.get("vac_url")),
        title=text(raw.get("job-name")),
        duty=text(raw.get("duty")),
        requirements=text(raw.get("requirements")),
        qualification=text(raw.get("qualification")),
        education=text(req.get("education")),
        experience_years=number(req.get("experience"), zero_is_none=False),
        req_qualification=text(req.get("qualification")),
        salary_text=text(raw.get("salary")),
        salary_min=number(raw.get("salary_min"), zero_is_none=True),
        salary_max=number(raw.get("salary_max"), zero_is_none=True),
        currency=text(raw.get("currency")),
        term_text=text(_dict(raw.get("term")).get("text")),
        benefit=text(raw.get("benefit")),
        schedule=text(raw.get("schedule")),
        schedule_comment=text(raw.get("scheduleTypeComment")),
        shifts=_texts(raw.get("shift")),
        employment=text(raw.get("employment")),
        addresses=_texts(addresses),
        conditions=text(raw.get("conditions")),
        company=text(_dict(raw.get("company")).get("name")),
        raw=_strip_contacts(raw),
    )
