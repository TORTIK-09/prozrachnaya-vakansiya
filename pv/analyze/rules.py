"""Проверки правилами: пропуски в полях и расхождения, которые видны без понимания смысла текста.

Каждое правило возвращает замечание с дословной цитатой из поля источника. Правила намеренно осторожны:
спорные и смысловые случаи оставлены модели (ai.py).
"""
from __future__ import annotations

import re
from typing import Iterable

from ..models import Evidence, Finding, Vacancy
from . import norms

I = re.IGNORECASE

# Поля со свободным текстом, в которых работодатель может написать что угодно
TEXT_FIELDS = ("duty", "requirements", "requirement.qualification", "qualification")
REQ_FIELDS = ("requirement.qualification", "requirements", "qualification")
SCHEDULE_FIELDS = ("schedule", "scheduleTypeComment", "shift", *TEXT_FIELDS)

GENERIC_TITLES = {
    "специалист", "сотрудник", "рабочий", "работник", "менеджер", "помощник", "ассистент", "эксперт",
    "служащий", "исполнитель", "персонал", "кадры", "вакансия", "стажер", "стажёр",
}

TIME_RANGE = re.compile(r"(?:с\s*)?\d{1,2}[:.]\d{2}\s*(?:[-–—]|до)\s*\d{1,2}[:.]\d{2}", I)
DAYS = re.compile(
    r"понедельник\w*|вторник\w*|сред[аыу]\b|четверг\w*|пятниц\w*|суббот\w*|воскресень\w*|"
    r"пятидневн\w+(?:\s+рабоч\w+\s+недел\w+)?|шестидневн\w+(?:\s+рабоч\w+\s+недел\w+)?|"
    r"\b[56][\s-]*(?:ти|и)?[\s-]*дневн\w+(?:\s+рабоч\w+\s+недел\w+)?|"
    r"\b[1-7]\s*/\s*[1-7]\b|сутки\s+через\s+\w+|\bпн\b|\bпт\b", I)
# Портал отдаёт график значением из справочника: «Полный рабочий день», «Сменная работа»,
# «Режим гибкого рабочего времени», «Неполный рабочий день/неполная рабочая неделя». Шаблоны покрывают и их.
FLEXIBLE = re.compile(r"гибк\w+\s+график\w*|свободн\w+\s+график\w*|режим\w*\s+гибк\w+\s+рабоч\w+\s+времен\w+|"
                      r"гибк\w+\s+рабоч\w+\s+времен\w+|"
                      r"график\w*\s+по\s+(?:согласовани\w+|договор[её]нност\w+)|гибк\w+\s+режим\w*", I)
SHIFT = re.compile(r"сменн\w+\s+(?:график|режим|работ)\w*|работа\s+по\s+сменам|посменн\w+", I)
SHIFT_SCHEME = re.compile(r"\b[1-7]\s*/\s*[1-7]\b|сутки\s+через\s+\w+|\d+\s*смен[аы]?\b", I)

REMOTE = re.compile(r"(?:полностью\s+)?дистанционн\w+(?:\s+из\s+любого\s+города)?|удал[её]нн\w+\s+работ\w+|"
                    r"работ\w+\s+удал[её]нно|работ\w+\s+из\s+дома|из\s+любого\s+города", I)
OFFICE = re.compile(r"(?:обязательн\w+\s+)?(?:ежедневн\w+\s+)?присутстви\w+\s+в\s+офисе|работ\w+\s+в\s+офисе|"
                    r"явк\w+\s+в\s+офис|только\s+в\s+офисе", I)

DEFERRED = re.compile(
    r"[^.;!?\n]*?(?:обсужда\w+|уточня\w+|оговарива\w+|определя\w+)\s+(?:на|при|в\s+ходе|по\s+итогам)\s+собеседовани\w+|"
    r"[^.;!?\n]*?по\s+(?:результатам|итогам)\s+собеседовани\w+", I)
PLACE_WORDS = re.compile(r"адрес|мест\w+\s+работы|рабоч\w+\s+мест\w+|формат\b", I)
VAGUE_DUTY = re.compile(
    r"(?:выполнение\s+)?различн\w+\s+поручени\w+(?:\s+руководител\w+)?|"
    r"(?:и\s+)?(?:други\w+|ины\w+|прочи\w+)\s+(?:задач\w*|поручени\w+|обязанност\w+)(?:\s+руководител\w+)?"
    r"(?:\s+по\s+необходимости)?|"
    r"(?:согласно|в\s+соответствии\s+с)\s+должностн\w+\s+инструкци\w+|по\s+необходимости", I)

EXP_YEARS = re.compile(
    r"(?:не\s+менее|от|свыше|более)\s+(\d{1,2})\s*(?:-?х\s+)?(?:лет|года|год)\w*(?:\s+опыта(?:\s+работы)?)?|"
    r"опыт\w*(?:\s+работы)?\s*(?:[-–—:]\s*)?(?:не\s+менее\s+|от\s+)?(\d{1,2})\s*(?:\+\s*)?(?:лет|года|год)\w*|"
    r"стаж\w*(?:\s+работы)?\s*(?:[-–—:]\s*)?(?:не\s+менее\s+|от\s+)?(\d{1,2})\s*(?:лет|года|год)\w*", I)
EXP_REQUIRED = re.compile(
    r"(?:обязателен\s+|требуется\s+|необходим\s+)?опыт\w*\s+(?:работы|разработки|ведения|управления|проведения|"
    r"составления|продаж|руководства|вождения|преподавания)(?:\s+[\w-]+){0,2}|"
    r"(?:требовани\w+\s+к\s+)?стаж\w*\s+(?:работы|муниципальн\w+|государственн\w+|по\s+специальности)(?:\s+[\w-]+){0,2}", I)
# оговорки, при которых упоминание опыта не делает его обязательным
EXP_OPTIONAL = re.compile(r"\bбез\b|\bне\b|\bнет\b|приветству\w+|желател\w+|плюс\w*|преимуществ\w+|\bили\b|\bлибо\b", I)
# Признаки того, что опыт действительно не нужен: сказано прямо либо должность начальная и учат на месте
EXP_NOT_NEEDED = re.compile(
    r"без\s+опыта|опыт\w*(?:\s+работы)?\s+не\s+(?:требуется|обязателен|нужен|имеет\s+значения)|не\s+требу\w+\s+опыт\w*|"
    r"обучени\w+|обучаем|научим|стажировк\w+|наставник\w*|без\s+предъявления\s+требований|"
    r"требовани\w+\s+к\s+(?:стажу|опыту)[^.;]*не\s+предъявля\w+", I)
# прямое «опыт не нужен» где угодно в требованиях снимает и упоминание опыта рядом
EXP_WAIVED = re.compile(r"без\s+опыта|нет\s+опыта|научим|опыт\w*(?:\s+работы)?\s+не\s+(?:требуется|обязателен|нужен|имеет\s+значения)", I)
PAY_WORDS = re.compile(r"доплат\w*|надбавк\w*|выплат\w*|преми\w+|компенсаци\w+|\bза\s+стаж", I)
JUNIOR_TITLE = re.compile(r"стаж[её]р|ученик|младш\w+|помощник|подсобн\w+|разнорабоч\w+|начинающ\w+", I)
SENIOR_TITLE = re.compile(r"ведущ\w+|главн\w+|старш\w+|руководител\w+|начальник\w*|заведующ\w+|директор\w*|"
                          r"заместител\w+|управляющ\w+", I)
ALTERNATIVE = re.compile(r"\bили\b|\bлибо\b", I)
NO_EXP = re.compile(r"без\s+опыта(?:\s+работы)?|опыт\w*(?:\s+работы)?\s+не\s+(?:требуется|обязателен|нужен)", I)

HARD_SKILL = re.compile(
    r"требовани|знани|знать|уметь|аккредитац|лиценз|специальност|професси|владени|навык|опыт|образовани|умени|\bпк\b|"
    r"компьютер|программ|права|категори|удостоверени|допуск|разряд|обучени|работ\w*\s+с\b|диплом|сертификат|"
    r"квалификаци|медицинск|книжк|справк|стаж|язык", I)
NO_EDUCATION = ("не указано", "не указан", "не имеет значения")

STREET = re.compile(
    r"\bул\b|\bулица\b|пр-кт|проспект|\bпер\b|переулок|\bпл\b|площадь|б-р|бульвар|шоссе|\bнаб\b|набережная|"
    r"проезд|тупик|аллея|мкр|микрорайон|\bд\.?\s*\d|\bдом\b|,\s*\d+\s*[а-я]?\s*(?:;|,|$)", I)
CITY = re.compile(r"\bг\b\.?|\bгород\b", I)
VILLAGE = re.compile(r"\bп\b\.?|\bпос\b\.?|пос[её]лок|\bпгт\b|\bс\b\.|\bсело\b|деревня|\bдер\b|хутор|станица", I)

PART_TIME_TEXT = re.compile(
    r"частичн\w+\s+занятост\w+|неполн\w+\s+(?:рабоч\w+\s+день|занятост\w+|рабоч\w+\s+недел\w+)|"
    r"0[.,]5\s*ставки|полставки|пол\s+ставки|подработк\w+", I)
FULL_DAY = re.compile(r"\bполн\w+\s+рабоч\w+\s+день", I)
# «Ненормированный рабочий день» — режим из Трудового кодекса: обычная рабочая неделя, это не неоднозначность
STANDARD_DAY = re.compile(r"\bполн\w+\s+рабоч\w+\s+день|ненормированн\w+\s+рабоч\w+\s+день", I)
PART_DAY = re.compile(r"неполн\w+\s+рабоч\w+\s+(?:день|недел\w+)|частичн\w+\s+занятост\w+", I)

_LOOKALIKE = str.maketrans("aAcCeEoOpPxXyHKMTB", "аАсСеЕоОрРхХуНКМТВ")
_SENTENCE_END = ".;!?\n"


def norm(text: str) -> str:
    """Копия текста той же длины, где латинские буквы-двойники заменены кириллицей.

    В вакансиях встречаются слова, набранные вперемешку («oтветcтвeннoe»): шаблоны ищут по этой копии,
    а цитата вырезается из исходного текста по тем же позициям.
    """
    return text.translate(_LOOKALIKE)


def search(pattern: re.Pattern, text: str) -> str:
    """Дословный фрагмент исходного текста, совпавший с шаблоном; пустая строка — совпадения нет."""
    m = pattern.search(norm(text))
    return text[m.start():m.end()].strip(" ,;:") if m else ""


def _first(v: Vacancy, fields: Iterable[str], pattern: re.Pattern, skip: str | None = None) -> tuple[str, str] | None:
    """Первое совпадение шаблона: (поле, дословный фрагмент)."""
    for f in fields:
        if f == skip:
            continue
        hit = search(pattern, v.text_of(f))
        if hit:
            return f, hit
    return None


def _has(v: Vacancy, fields: Iterable[str], pattern: re.Pattern) -> bool:
    return _first(v, fields, pattern) is not None


def _sentence(text: str, fragment: str) -> str:
    """Предложение, в котором стоит фрагмент."""
    i = text.find(fragment)
    if i < 0:
        return fragment
    start = max(text.rfind(c, 0, i) for c in _SENTENCE_END) + 1
    ends = [e for e in (text.find(c, i) for c in _SENTENCE_END) if e >= 0]
    return text[start:min(ends) if ends else len(text)]


def _title(v: Vacancy) -> list[Finding]:
    if not v.title:
        return [Finding("title", "missing", "job-name", "", "Не указано название должности.",
                        "Уточните у работодателя название должности, на которую открыта вакансия.")]
    if norm(v.title).lower().strip(" .,;-") in GENERIC_TITLES:
        return [Finding("title", "ambiguous", "job-name", v.title,
                        "Название должности слишком общее: по нему нельзя понять специализацию и направление работы.",
                        "Уточните у работодателя специализацию должности: направление работы или подразделение.")]
    return []


def _duties(v: Vacancy) -> list[Finding]:
    if not v.duty:
        return [Finding("duties", "missing", "duty", "", "Обязанности не описаны.",
                        "Уточните у работодателя, какие задачи сотрудник выполняет регулярно.")]
    hits = list(VAGUE_DUTY.finditer(norm(v.duty)))
    if not hits:
        return []           # короткие, но конкретные обязанности («Уборка помещений») — не замечание
    rest = v.duty
    for m in reversed(hits):
        rest = rest[:m.start()] + " " + rest[m.end():]
    if len(re.sub(r"[^\w]+", "", rest)) >= 40:
        return []           # рядом с общей фразой есть содержательный перечень
    quote = v.duty[hits[0].start():hits[0].end()].strip(" ,;:")
    return [Finding("duties", "ambiguous", "duty", quote,
                    "Обязанности сформулированы общими словами: из описания нельзя понять, чем именно предстоит заниматься.",
                    "Уточните у работодателя перечень конкретных регулярных задач по этой должности.")]


def _nothing(value: str | None) -> bool:
    """Поле пустое или в нём «не указано». «Требования не предъявляются» — это ответ, а не пустота."""
    return not value or norm(value).lower().strip(" .") in NO_EDUCATION


def _explicitly_none(value: str | None) -> bool:
    """В поле прямо сказано, что требований нет."""
    return bool(value) and bool(re.search(r"^(?:нет|без\s+\w+|не\s+требу\w+|требовани\w+\s+не\s+предъявля\w+)[\s.]*$",
                                           norm(value).strip(), I))


def _zero_experience(v: Vacancy) -> list[Finding]:
    """Опыт 0 и в тексте про опыт ничего: это «опыт не нужен» или поле забыли заполнить?

    Считаем, что опыт действительно не нужен, если так сказано в любом поле или должность начальная.
    Вопрос работодателю задаём в двух случаях: должность руководящая или старшая; либо на портале в этой
    профессии опыт требуют почти всегда (norms.py). В остальных случаях ноль принимаем как есть.
    """
    texts = " ".join(v.text_of(f) for f in (*REQ_FIELDS, "duty"))
    if EXP_NOT_NEEDED.search(norm(texts)) or JUNIOR_TITLE.search(norm(v.title or "")):
        return []
    if _explicitly_none(v.qualification) and not search(SENIOR_TITLE, v.title or ""):
        return []           # в квалификации стоит «нет» или «не требуется»: ноль в опыте поставлен осознанно
    senior = search(SENIOR_TITLE, v.title or "")
    if senior:
        return [Finding("requirements", "ambiguous", "requirement.experience", "0",
                        f"В поле опыта указано 0, хотя должность «{v.title}» старшая или руководящая, а в тексте вакансии "
                        "про опыт ничего не сказано: непонятно, рассматривают ли кандидатов без опыта.",
                        "Уточните у работодателя, рассматриваются ли кандидаты без опыта работы; если нет — какой стаж нужен.")]
    usual = norms.usually_needs_experience(v.title)
    if usual:
        with_exp, total = usual
        return [Finding("requirements", "ambiguous", "requirement.experience", "0",
                        f"В поле опыта указано 0, и в тексте про опыт ничего не сказано. При этом в похожих вакансиях портала "
                        f"опыт требуется в {with_exp} из {total}: возможно, поле не заполнили.",
                        "Уточните у работодателя, действительно ли рассматриваются кандидаты без опыта работы.")]
    return []


def _requirements(v: Vacancy) -> list[Finding]:
    out: list[Finding] = []
    texts = [v.text_of(f) for f in REQ_FIELDS]
    exp = v.experience_years

    # опыт: число в поле против текста
    years = None
    if exp is not None:
        for f in (*REQ_FIELDS, "duty"):
            t = v.text_of(f)
            m = EXP_YEARS.search(norm(t))
            if m and int(next(g for g in m.groups() if g)) > exp:
                quote = t[m.start():m.end()].strip(" ,;:")
                if ALTERNATIVE.search(norm(_sentence(t, quote))):
                    continue        # стаж нужен только в одном из вариантов, а не всем
                years = (f, quote, int(next(g for g in m.groups() if g)))
                break
    if years:
        f, quote, n = years
        out.append(Finding(
            "requirements", "contradiction", "requirement.experience", str(exp),
            f"В поле опыта указано {exp}, а в тексте требуется опыт от {n} лет.",
            "Уточните у работодателя, какой опыт работы действительно обязателен, и приведите поле и текст к одному значению.",
            conflict_with=Evidence(f, quote)))
    elif exp == 0:
        for f in REQ_FIELDS:
            t = v.text_of(f)
            hit = search(EXP_REQUIRED, t)
            # оговорки «без опыта», «не обязателен», «приветствуется» в том же предложении снимают противоречие
            # ...а «доплата за стаж» — это льгота, не требование; «нет опыта — научим» снимает вопрос целиком
            sentence = norm(_sentence(t, hit)) if hit else ""
            if hit and (PAY_WORDS.search(sentence) or EXP_WAIVED.search(norm(" ".join(texts)))):
                continue
            if hit and not EXP_OPTIONAL.search(sentence):
                out.append(Finding(
                    "requirements", "contradiction", "requirement.experience", "0",
                    "В поле опыта указано 0, то есть опыт не нужен, а в тексте требований упомянут опыт без оговорки, что он необязателен.",
                    "Уточните у работодателя, обязателен ли опыт работы; если да — какой минимальный стаж нужен.",
                    conflict_with=Evidence(f, hit)))
                break
        else:
            out.extend(_zero_experience(v))
    elif exp is not None and exp >= 1:
        hit = _first(v, REQ_FIELDS, NO_EXP)
        if hit:
            out.append(Finding(
                "requirements", "contradiction", "requirement.experience", str(exp),
                f"В поле опыта указано {exp}, а в тексте сказано, что опыт не требуется.",
                "Уточните у работодателя, нужен ли опыт работы, и приведите поле и текст к одному значению.",
                conflict_with=Evidence(hit[0], hit[1])))

    if not any(texts):
        if _nothing(v.education):
            out.append(Finding("requirements", "missing", "requirements", "",
                               "Требования к кандидату не указаны: нет ни уровня образования, ни навыков, ни квалификации.",
                               "Уточните у работодателя требования к образованию, навыкам и квалификации кандидата."))
        return out

    # требования отложены до собеседования или состоят из одних личных качеств
    deferred = _first(v, REQ_FIELDS, DEFERRED)
    if deferred and not PLACE_WORDS.search(norm(deferred[1])):
        out.append(Finding("requirements", "ambiguous", deferred[0], deferred[1],
                           "Условия отложены до собеседования: из вакансии нельзя понять, что требуется от кандидата и что ему предлагают.",
                           "Уточните у работодателя, какие именно условия и требования обсуждаются на собеседовании, и укажите их в вакансии."))
    elif (_nothing(v.education) and _nothing(v.qualification) and not exp
          and sum(len(t) for t in texts) <= 200 and not any(HARD_SKILL.search(norm(t)) for t in texts if t)):
        # образование, квалификация или опыт в полях — уже требования: повторять их в тексте работодатель не обязан
        f = next(f for f in REQ_FIELDS if v.text_of(f))
        out.append(Finding("requirements", "ambiguous", f, v.text_of(f),
                           "В требованиях перечислены только общие личные качества: непонятно, какие знания, навыки или образование нужны.",
                           "Уточните у работодателя требования к образованию и конкретным навыкам либо то, что обучение проводится на месте."))
    return out


def _salary(v: Vacancy) -> list[Finding]:
    lo, hi = v.salary_min, v.salary_max
    if lo is None and hi is None:
        if v.salary_text and not re.search(r"\d", v.salary_text):
            return [Finding("salary", "ambiguous", "salary", v.salary_text,
                            "Размер оплаты не назван: вместо суммы указана общая формулировка.",
                            "Уточните у работодателя размер оплаты: сумму или диапазон.")]
        if not v.salary_text:
            return [Finding("salary", "missing", "salary_min", "",
                            "Размер оплаты не указан: в полях зарплаты нет ни минимальной, ни максимальной суммы.",
                            "Уточните у работодателя размер оплаты: сумму или диапазон.")]
        return []
    if lo is not None and hi is not None and lo > hi:
        return [Finding("salary", "contradiction", "salary_min", str(lo),
                        "Минимальная зарплата больше максимальной.",
                        "Уточните у работодателя верные границы оплаты.",
                        conflict_with=Evidence("salary_max", str(hi)))]
    if lo is not None and hi is None:
        return [Finding("salary", "ambiguous", "salary_min", str(lo),
                        "Указана только нижняя граница оплаты: непонятно, это фиксированная сумма или начало диапазона.",
                        "Уточните у работодателя, фиксированная ли это сумма, и если нет — верхнюю границу и от чего зависит размер оплаты.")]
    return []


def _schedule(v: Vacancy) -> list[Finding]:
    has_time = _has(v, SCHEDULE_FIELDS, TIME_RANGE)
    has_days = _has(v, SCHEDULE_FIELDS, DAYS)
    if not v.schedule:
        if has_time or has_days or v.schedule_comment or v.shifts:
            return []
        return [Finding("schedule", "missing", "schedule", "", "График работы не указан.",
                        "Уточните у работодателя рабочие дни и время начала и окончания работы.")]
    flexible = search(FLEXIBLE, v.schedule)
    if flexible:
        # гибкий график и «5/2» совместимы: дни известны, часы плавают. Противоречие — только жёсткие часы.
        fixed = _first(v, SCHEDULE_FIELDS, TIME_RANGE, skip="schedule")
        if fixed:
            days = _first(v, (fixed[0],), DAYS)
            if days and days[1] in _sentence(v.text_of(fixed[0]), fixed[1]):
                fixed = (fixed[0], _sentence(v.text_of(fixed[0]), fixed[1]).strip(" .;"))
            return [Finding("schedule", "contradiction", "schedule", flexible,
                            "В одном поле заявлен гибкий график, в другом — фиксированное расписание.",
                            "Уточните у работодателя, какой режим действует на самом деле: фиксированные дни и часы или гибкий график, и по каким правилам.",
                            conflict_with=Evidence(fixed[0], fixed[1]))]
        if not has_time and not has_days:
            return [Finding("schedule", "ambiguous", "schedule", flexible,
                            "Заявлен гибкий график, но конкретного расписания нет ни в одном поле.",
                            "Уточните у работодателя рабочие дни, количество часов в неделю и то, кто и как определяет расписание.")]
        return []
    shift = search(SHIFT, v.schedule)
    if shift:
        if not (_has(v, SCHEDULE_FIELDS, SHIFT_SCHEME) or v.shifts) or not (has_time or v.shifts):
            return [Finding("schedule", "ambiguous", "schedule", shift,
                            "Указан сменный график без схемы чередования смен и времени их начала и окончания.",
                            "Уточните у работодателя схему смен (сколько рабочих и выходных дней подряд) и время начала и окончания смены.")]
        return []
    if search(STANDARD_DAY, v.schedule) and not search(PART_TIME_TEXT, v.schedule):
        # «Полный рабочий день» — общепринятое обозначение пятидневки с восьмичасовым днём: это не неоднозначность
        return []
    part = search(PART_DAY, v.schedule)
    if part and not has_time and not has_days and not v.schedule_comment and not v.shifts:
        return [Finding("schedule", "ambiguous", "schedule", v.schedule,
                        "Указан неполный рабочий день или неполная неделя, но нигде не сказано, сколько часов и в какие дни нужно работать.",
                        "Уточните у работодателя количество рабочих часов в день или в неделю и рабочие дни.")]
    if not has_time and not has_days and not v.schedule_comment and not v.shifts:
        return [Finding("schedule", "ambiguous", "schedule", v.schedule,
                        "По указанному режиму нельзя понять, в какие дни и сколько часов предстоит работать.",
                        "Уточните у работодателя рабочие дни недели и время начала и окончания работы.")]
    return []


def _location(v: Vacancy) -> list[Finding]:
    remote = _first(v, TEXT_FIELDS, REMOTE)
    if remote:
        office = _first(v, TEXT_FIELDS, OFFICE)
        if office:
            return [Finding("location", "contradiction", remote[0], remote[1],
                            "В описании одновременно заявлены дистанционная работа и обязательное присутствие в офисе.",
                            "Уточните у работодателя формат работы: дистанционный, в офисе или смешанный, и в какие дни нужно присутствие.",
                            conflict_with=Evidence(office[0], office[1]))]
    deferred = _first(v, TEXT_FIELDS, DEFERRED)
    if deferred and PLACE_WORDS.search(norm(deferred[1])):
        return [Finding("location", "ambiguous", deferred[0], deferred[1],
                        "Место и формат работы отложены до собеседования.",
                        "Уточните у работодателя адрес рабочего места и формат работы: в офисе, дистанционно или смешанно.")]
    if remote:
        return []
    if not v.addresses:
        return [Finding("location", "missing", "addresses.address[].location", "", "Адрес места работы не указан.",
                        "Уточните у работодателя адрес рабочего места: населённый пункт, улицу и номер дома.")]
    addr = v.addresses[0]
    if STREET.search(addr) or VILLAGE.search(addr):
        return []           # в сёлах и посёлках улицы и номера дома может не быть вовсе
    if CITY.search(addr):
        return [Finding("location", "ambiguous", "addresses.address[].location", addr,
                        "В адресе указан только населённый пункт, без улицы и дома.",
                        "Уточните у работодателя точный адрес рабочего места: улицу и номер дома.")]
    return [Finding("location", "ambiguous", "addresses.address[].location", addr,
                    "В адресе указан только регион: непонятно, в каком населённом пункте находится рабочее место.",
                    "Уточните у работодателя населённый пункт, улицу и номер дома рабочего места.")]


def _employment(v: Vacancy) -> list[Finding]:
    if not v.employment:
        if v.schedule:
            # график уже говорит о занятости: «полный рабочий день» — полная, «неполный рабочий день» — частичная
            return []
        return [Finding("employment", "missing", "employment", "", "Тип занятости не указан.",
                        "Уточните у работодателя тип занятости: полная, частичная, временная или иная.")]
    emp = v.employment.lower()
    if "полн" in emp and "неполн" not in emp:
        hit = _first(v, ("schedule", "scheduleTypeComment", *TEXT_FIELDS), PART_TIME_TEXT)
        if hit:
            return [Finding("employment", "contradiction", "employment", v.employment,
                            "В поле указана полная занятость, а в тексте — частичная занятость или неполный день.",
                            "Уточните у работодателя тип занятости и приведите поле и описание к одному значению.",
                            conflict_with=Evidence(hit[0], hit[1]))]
    if "частич" in emp or "неполн" in emp:
        hit = _first(v, ("schedule", "scheduleTypeComment"), FULL_DAY)
        if hit:
            return [Finding("employment", "contradiction", "employment", v.employment,
                            "В поле указана частичная занятость, а в графике — полный рабочий день.",
                            "Уточните у работодателя тип занятости и продолжительность рабочего дня.",
                            conflict_with=Evidence(hit[0], hit[1]))]
    return []


CHECKS = (_title, _duties, _requirements, _salary, _schedule, _location, _employment)


def check(v: Vacancy) -> list[Finding]:
    out: list[Finding] = []
    for fn in CHECKS:
        out.extend(fn(v))
    return out
