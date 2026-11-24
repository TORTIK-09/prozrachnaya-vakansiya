import html
import re
from typing import Any, Dict, List
from src.models import VacancyIssue


GENERIC_JOB_TITLES = {
    "специалист",
    "сотрудник",
    "рабочий",
    "менеджер",
    "инженер",
    "консультант",
    "оператор",
    "помощник",
    "ассистент",
    "эксперт",
    "мастер",
    "техник",
    "администратор",
    "руководитель",
    "служащий",
    "работник",
}

VAGUE_DUTY_PATTERNS = [
    r"различн\w+\s+поручен",
    r"други\w+\s+задач\w*\s+по\s+необходимости",
    r"иные\s+поручения\s+руководителя",
    r"согласно\s+должностной\s+инструкции",
    r"обсужда\w+\s+на\s+собеседовании",
    r"уточня\w+\s+на\s+собеседовании",
    r"по\s+договоренности",
]

STREET_OR_HOUSE_PATTERN = re.compile(
    r"(\bул\.|\bулица\b|\bпр-кт\b|\bпроспект\b|\bпер\.|\bпереулок\b|\bпл\.|\bплощадь\b|"
    r"\bб-р\b|\bбульвар\b|\bш\.|\bшоссе\b|\bнаб\.|\bнабережная\b|\bмкр\.|\bмикрорайон\b|"
    r"\bд\.\s*\d+|\bдом\s*\d+|,\s*\d+[а-яА-Яa-zA-Z]?\b)",
    re.IGNORECASE,
)

TIME_RANGE_PATTERN = re.compile(
    r"(\d{1,2}[:.]\d{2}\s*[-–—]\s*\d{1,2}[:.]\d{2}|с\s*\d{1,2}[:.]\d{2}\s*до\s*\d{1,2}[:.]\d{2})",
    re.IGNORECASE,
)

DAYS_OR_SHIFT_PATTERN = re.compile(
    r"(понедельник|вторник|сред|четверг|пятниц|суббот|воскресень|"
    r"пятиднев|шестиднев|\b5/2\b|\b2/2\b|\b1/3\b|\b3/3\b|\b6/1\b|\b4/2\b|\b1/2\b)",
    re.IGNORECASE,
)


def clean_text(raw: Any) -> str:
    """Очищает строку от HTML-тегов, сущностей и лишних пробелов."""
    if raw is None:
        return ""
    text = str(raw)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_number(val: Any) -> int:
    """Преобразует значение зарплаты или опыта в целое число."""
    if val is None or val == "":
        return 0
    if isinstance(val, (int, float)):
        return int(val)
    digits = re.findall(r"\d+", str(val))
    return int(digits[0]) if digits else 0


class RuleEngine:
    """
    Детерминированный анализатор структуры и кросс-полевых правил вакансии.
    Проверяет 7 ключевых аспектов: Название должности, Обязанности, Требования,
    Оплата, График, Место работы, Тип занятости.
    """

    def analyze(self, vacancy: Dict[str, Any]) -> List[VacancyIssue]:
        issues: List[VacancyIssue] = []

        job_name = clean_text(vacancy.get("job-name"))
        duty = clean_text(vacancy.get("duty"))
        employment = clean_text(vacancy.get("employment"))
        schedule = clean_text(vacancy.get("schedule"))
        salary_str = clean_text(vacancy.get("salary"))
        salary_min = parse_number(vacancy.get("salary_min"))
        salary_max = parse_number(vacancy.get("salary_max"))

        req_obj = vacancy.get("requirement")
        if not isinstance(req_obj, dict):
            req_obj = {}
        education = clean_text(req_obj.get("education"))
        exp_raw = req_obj.get("experience")
        experience_years = parse_number(exp_raw)
        qualification = clean_text(req_obj.get("qualification"))
        requirements_text = clean_text(vacancy.get("requirements"))

        addresses_obj = vacancy.get("addresses")
        address_list = (
            addresses_obj.get("address", [])
            if isinstance(addresses_obj, dict)
            else []
        )
        locations: List[str] = []
        if isinstance(address_list, list):
            for addr in address_list:
                if isinstance(addr, dict):
                    loc = clean_text(addr.get("location"))
                    if loc:
                        locations.append(loc)

        combined_description = " ".join(
            part for part in [duty, qualification, requirements_text] if part
        )

        # 1. Проверка названия должности
        self._check_job_name(job_name, issues)

        # 2. Проверка оплаты труда
        self._check_salary(salary_min, salary_max, salary_str, issues)

        # 3. Проверка типа занятости
        self._check_employment(employment, schedule, combined_description, issues)

        # 4. Проверка графика работы
        self._check_schedule(schedule, qualification, duty, employment, issues)

        # 5. Проверка обязанностей
        self._check_duty(duty, job_name, issues)

        # 6. Проверка требований (образование, опыт, квалификация)
        self._check_requirements(
            job_name,
            education,
            experience_years,
            exp_raw,
            qualification,
            requirements_text,
            duty,
            issues,
        )

        # 7. Проверка места и формата работы
        self._check_workplace(locations, duty, qualification, requirements_text, issues)

        return issues

    def _check_job_name(self, job_name: str, issues: List[VacancyIssue]) -> None:
        if not job_name:
            issues.append(
                VacancyIssue(
                    type="пропуск",
                    topic="Название должности",
                    field="job-name",
                    quote="",
                    issue="Не указано название должности.",
                    recommendation="Указать конкретное наименование должности в соответствии со штатным расписанием и профилем деятельности.",
                    detected_by="rules",
                )
            )
            return

        normalized = job_name.lower().strip(" .,;-")
        if normalized in GENERIC_JOB_TITLES:
            issues.append(
                VacancyIssue(
                    type="неоднозначность",
                    topic="Название должности",
                    field="job-name",
                    quote=job_name,
                    issue="Название должности слишком общее и не отражает специализацию, профиль или сферу работы сотрудника.",
                    recommendation=f"Уточнить специализацию в поле job-name (например, вместо «{job_name}» указать конкретное направление: по документообороту, по закупкам, по работе с клиентами и т.п.).",
                    detected_by="rules+ai",
                )
            )

    def _check_salary(
        self,
        salary_min: int,
        salary_max: int,
        salary_str: str,
        issues: List[VacancyIssue],
    ) -> None:
        if salary_min <= 0 and salary_max <= 0 and not salary_str:
            issues.append(
                VacancyIssue(
                    type="пропуск",
                    topic="Оплата",
                    field="salary_min, salary_max",
                    quote="",
                    issue="Не указан размер оплаты труда (минимальное и максимальное значения равны 0).",
                    recommendation="Указать конкретный размер заработной платы в полях salary_min и salary_max (фиксированную сумму или зарплатную вилку в рублях).",
                    detected_by="rules",
                )
            )
            return

        if salary_min > 0 and salary_max == 0:
            issues.append(
                VacancyIssue(
                    type="противоречие",
                    topic="Оплата",
                    field="salary_min, salary_max",
                    quote=f"salary_min: {salary_min} | salary_max: {salary_max}",
                    issue=f"Максимальная граница оплаты труда (salary_max = 0) меньше минимальной (salary_min = {salary_min}), из-за чего непонятно, является ли сумма фиксированной или указан только нижний порог.",
                    recommendation=f"Согласовать значения полей salary_min и salary_max: если оплата фиксированная — указать {salary_min} в обоих полях, если предусмотрена вилка — указать верхнюю границу в salary_max.",
                    detected_by="rules",
                )
            )
        elif salary_min > 0 and salary_max > 0 and salary_max < salary_min:
            issues.append(
                VacancyIssue(
                    type="противоречие",
                    topic="Оплата",
                    field="salary_min, salary_max",
                    quote=f"salary_min: {salary_min} | salary_max: {salary_max}",
                    issue="Верхняя граница заработной платы (salary_max) меньше нижней границы (salary_min).",
                    recommendation="Исправить значения в полях salary_min и salary_max так, чтобы максимальная зарплата была не меньше минимальной.",
                    detected_by="rules",
                )
            )
        elif salary_min > 0 and salary_max >= int(salary_min * 2.5):
            issues.append(
                VacancyIssue(
                    type="неоднозначность",
                    topic="Оплата",
                    field="salary_min, salary_max",
                    quote=f"salary_min: {salary_min}, salary_max: {salary_max}",
                    issue="Слишком широкий диапазон заработной платы без пояснения условий получения верхней границы.",
                    recommendation="Пояснить в описании вакансии, от чего зависит итоговый уровень дохода (квалификационная категория, выполнение KPI, объём часов или стимулирующие выплаты).",
                    detected_by="rules+ai",
                )
            )

    def _check_employment(
        self,
        employment: str,
        schedule: str,
        combined_description: str,
        issues: List[VacancyIssue],
    ) -> None:
        if not employment:
            issues.append(
                VacancyIssue(
                    type="пропуск",
                    topic="Тип занятости",
                    field="employment",
                    quote="",
                    issue="Не указан тип занятости.",
                    recommendation="Заполнить поле employment (например: «Полная занятость», «Частичная занятость», «Временная»). ",
                    detected_by="rules",
                )
            )
            return

        emp_lower = employment.lower()
        other_text = f"{schedule} {combined_description}".lower()
        if "полная" in emp_lower and re.search(
            r"\b(частичная\s+занятость|неполный\s+рабочий\s+день|0[.,]5\s*ставки|полставки)\b",
            other_text,
        ):
            issues.append(
                VacancyIssue(
                    type="противоречие",
                    topic="Тип занятости",
                    field="employment, schedule",
                    quote=f"employment: «{employment}» | описание: «{schedule or combined_description}»",
                    issue="В поле employment указана полная занятость, однако в тексте упоминается частичная занятость или неполный рабочий день.",
                    recommendation="Привести поле employment и описание режима работы к единому типу занятости.",
                    detected_by="rules+ai",
                )
            )

    def _check_schedule(
        self,
        schedule: str,
        qualification: str,
        duty: str,
        employment: str,
        issues: List[VacancyIssue],
    ) -> None:
        if not schedule:
            issues.append(
                VacancyIssue(
                    type="пропуск",
                    topic="График",
                    field="schedule",
                    quote="",
                    issue="Не указан график работы.",
                    recommendation="Указать в поле schedule рабочие дни недели (или схему смен) и часы начала и окончания рабочего дня.",
                    detected_by="rules",
                )
            )
            return

        sched_lower = schedule.lower()
        qual_lower = qualification.lower()
        all_schedule_context = f"{schedule} {qualification} {duty}"

        # Проверка противоречия: в schedule "Гибкий график", а в qualification фиксированная пятидневка 09:00-18:00 или "без уточнения правил"
        if "гибкий" in sched_lower and (
            TIME_RANGE_PATTERN.search(qualification)
            or "пятидневная" in qual_lower
            or "без уточнения правил" in qual_lower
        ):
            issues.append(
                VacancyIssue(
                    type="противоречие",
                    topic="График",
                    field="schedule, requirement.qualification",
                    quote=f"schedule: «{schedule}» | requirement.qualification: «{qualification}»",
                    issue="Противоречие в описании графика: в поле schedule заявлен «Гибкий график», а в requirement.qualification указана фиксированная пятидневная рабочая неделя с 09:00 до 18:00 и оговорка о гибком графике без уточнения правил.",
                    recommendation="Согласовать поля schedule и requirement.qualification: указать либо стандартную пятидневную неделю с фиксированными часами, либо конкретные правила гибкого графика (диапазон прихода/ухода, обязательные часы присутствия и норму часов).",
                    detected_by="rules+ai",
                )
            )
            return

        has_time = bool(TIME_RANGE_PATTERN.search(all_schedule_context))
        has_days_or_shifts = bool(DAYS_OR_SHIFT_PATTERN.search(all_schedule_context))

        if "гибкий" in sched_lower and not has_time and not has_days_or_shifts:
            issues.append(
                VacancyIssue(
                    type="неоднозначность",
                    topic="График",
                    field="schedule",
                    quote=schedule,
                    issue="Указана формулировка «Гибкий график», но ни в поле schedule, ни в описании вакансии нет конкретных сведений о рабочих днях, продолжительности смен или норме часов в неделю.",
                    recommendation="Уточнить параметры гибкого графика: количество рабочих часов в неделю, ожидаемые дни работы и допустимый интервал рабочих часов.",
                    detected_by="rules+ai",
                )
            )
        elif "сменный" in sched_lower and not (has_time and has_days_or_shifts):
            issues.append(
                VacancyIssue(
                    type="неоднозначность",
                    topic="График",
                    field="schedule",
                    quote=schedule,
                    issue="Указан «Сменный график», но не приведены схема чередования смен (например, 2/2, 1/3) и часы начала и окончания смены.",
                    recommendation="Дополнить поле schedule схемой чередования смен и точным временем работы (например: «Сменный график: 2/2, 08:00–20:00»).",
                    detected_by="rules+ai",
                )
            )
        elif not has_time and not has_days_or_shifts:
            issues.append(
                VacancyIssue(
                    type="неоднозначность",
                    topic="График",
                    field="schedule",
                    quote=schedule,
                    issue="В графике указан только общий режим («Полный рабочий день») без уточнения рабочих дней недели и часов начала и окончания работы.",
                    recommendation="Указать в поле schedule рабочие дни недели (например, 5/2, понедельник–пятница) и время рабочего дня (например, 09:00–18:00).",
                    detected_by="rules+ai",
                )
            )
        elif not has_time:
            issues.append(
                VacancyIssue(
                    type="неоднозначность",
                    topic="График",
                    field="schedule",
                    quote=schedule,
                    issue="В графике работы указана пятидневная неделя, но не уточнены часы начала и окончания рабочего дня.",
                    recommendation="Дополнить поле schedule временем начала и окончания рабочего дня (например, «09:00–18:00»).",
                    detected_by="rules+ai",
                )
            )

    def _check_duty(
        self, duty: str, job_name: str, issues: List[VacancyIssue]
    ) -> None:
        if not duty:
            issues.append(
                VacancyIssue(
                    type="пропуск",
                    topic="Обязанности",
                    field="duty",
                    quote="",
                    issue="Отсутствует описание должностных обязанностей.",
                    recommendation="Перечислить в поле duty конкретные регулярные задачи и функции сотрудника.",
                    detected_by="rules",
                )
            )
            return

        is_vague = any(
            re.search(pat, duty, re.IGNORECASE) for pat in VAGUE_DUTY_PATTERNS
        )
        if is_vague or len(duty) < 30:
            issues.append(
                VacancyIssue(
                    type="неоднозначность",
                    topic="Обязанности",
                    field="duty",
                    quote=duty,
                    issue="Обязанности сформулированы слишком обобщённо и не дают соискателю понимания конкретных ежедневных задач и объёма работы.",
                    recommendation="Заменить размытые формулировки («различные поручения», «другие задачи по необходимости») на конкретный перечень трудовых функций и видов обрабатываемых документов или процессов.",
                    detected_by="rules+ai",
                )
            )

    def _check_requirements(
        self,
        job_name: str,
        education: str,
        experience_years: int,
        exp_raw: Any,
        qualification: str,
        requirements_text: str,
        duty: str,
        issues: List[VacancyIssue],
    ) -> None:
        combined_req = f"{qualification} {requirements_text}".strip()

        # 1. Противоречие: experience == 0 (или "Младший"), а в тексте "не менее X лет опыта" / "опыт работы от X лет"
        years_in_text_match = re.search(
            r"(?:не\s+менее|от)\s+(\d+)\s*(?:лет|года|год)\s+опыта|опыт\s+работы\s+(?:не\s+менее|от)\s+(\d+)\s*(?:лет|года|год)",
            combined_req,
            re.IGNORECASE,
        )
        if years_in_text_match:
            req_years_text = int(
                years_in_text_match.group(1) or years_in_text_match.group(2)
            )
            is_junior = bool(
                re.search(r"\b(младший|junior|стажер|стажёр|начинающий)\b", job_name, re.IGNORECASE)
            )
            if experience_years == 0 or (is_junior and req_years_text >= 3):
                field_str = (
                    "job-name, requirement.experience, requirement.qualification"
                    if is_junior
                    else "requirement.experience, requirement.qualification"
                )
                quote_str = (
                    f"job-name: «{job_name}», requirement.experience: {experience_years} | requirement.qualification: «{qualification or requirements_text}»"
                    if is_junior
                    else f"requirement.experience: {experience_years} | requirement.qualification: «{qualification or requirements_text}»"
                )
                issues.append(
                    VacancyIssue(
                        type="противоречие",
                        topic="Требования",
                        field=field_str,
                        quote=quote_str,
                        issue=f"Противоречие в требованиях к стажу: для должности «{job_name}» в поле requirement.experience указано {experience_years} лет (без опыта), но в текстовом описании квалификации требуется не менее {req_years_text} лет опыта работы.",
                        recommendation="Согласовать уровень должности и требования к стажу: для младшей позиции без опыта скорректировать текст requirement.qualification, либо изменить уровень должности и значение поля requirement.experience.",
                        detected_by="rules+ai",
                    )
                )
        elif experience_years == 0:
            # Проверим, не требуется ли явно профильный опыт в requirements / qualification (кроме "приветствуется" / "не обязателен" / "без опыта")
            has_mandatory_exp_phrase = bool(
                re.search(
                    r"\bопыт\s+(?:разработки|работы|ведения|управления|проведения|составления)\b",
                    combined_req,
                    re.IGNORECASE,
                )
            )
            is_optional_exp = bool(
                re.search(
                    r"опыт\s+(?:работы\s+)?(?:не\s+обязателен|приветствуется|желателен|не\s+требуется)",
                    combined_req,
                    re.IGNORECASE,
                )
            )
            if has_mandatory_exp_phrase and not is_optional_exp:
                target_field = (
                    "requirement.experience, requirements"
                    if "опыт" in requirements_text.lower()
                    else "requirement.experience, requirement.qualification"
                )
                target_quote_text = (
                    requirements_text
                    if "опыт" in requirements_text.lower()
                    else qualification
                )
                issues.append(
                    VacancyIssue(
                        type="противоречие",
                        topic="Требования",
                        field=target_field,
                        quote=f"requirement.experience: {experience_years} | requirements: «{target_quote_text}»",
                        issue="В структурированном поле requirement.experience указано 0 (опыт не требуется), однако в текстовых требованиях указано наличие профильного опыта.",
                        recommendation="Уточнить обязательность опыта работы: если опыт обязателен — указать минимальный стаж (в годах) в поле requirement.experience; если рассматриваются кандидаты без опыта — добавить пояснение, что опыт желателен, но не обязателен.",
                        detected_by="rules+ai",
                    )
                )

        # 2. Проверка на размытые или отложенные до собеседования требования
        if re.search(
            r"условия\s+обсуждаются\s+на\s+собеседовании",
            combined_req,
            re.IGNORECASE,
        ):
            issues.append(
                VacancyIssue(
                    type="неоднозначность",
                    topic="Требования",
                    field="requirement.education, requirement.qualification, requirements",
                    quote=f"education: «{education}», qualification: «{qualification}», requirements: «{requirements_text}»",
                    issue="Квалификационные требования и уровень образования не конкретизированы и отложены до собеседования.",
                    recommendation="Указать в карточке вакансии требуемый уровень образования и конкретные профессиональные навыки, необходимые для выполнения работы.",
                    detected_by="rules+ai",
                )
            )
        elif (
            education.lower() in ("", "не указано")
            and "обучение" not in combined_req.lower()
            and not re.search(
                r"(знание|владение|навык|умение|компьютер|программ|sql|windows|каталог|переписк|речь)",
                combined_req,
                re.IGNORECASE,
            )
        ):
            issues.append(
                VacancyIssue(
                    type="неоднозначность",
                    topic="Требования",
                    field="requirement.education, requirement.qualification, requirements",
                    quote=f"education: «{education}», qualification: «{qualification}», requirements: «{requirements_text}»",
                    issue="В требованиях указаны только общие личные качества без конкретных профессиональных навыков и без требований к уровню образования.",
                    recommendation="Уточнить минимальные требования к образованию и практическим навыкам кандидата (например, владение ПК или офисными программами) либо указать, что предусмотрено обучение на месте.",
                    detected_by="rules+ai",
                )
            )

    def _check_workplace(
        self,
        locations: List[str],
        duty: str,
        qualification: str,
        requirements_text: str,
        issues: List[VacancyIssue],
    ) -> None:
        all_text = f"{duty} {qualification} {requirements_text}"

        # 1. Противоречие: дистанционная работа vs обязательное присутствие в офисе
        has_remote = bool(
            re.search(
                r"(дистанционн\w+|удаленн\w+|удалённ\w+|из\s+любого\s+города|работа\s+из\s+дома)",
                all_text,
                re.IGNORECASE,
            )
        )
        has_office_mandatory = bool(
            re.search(
                r"(присутствие\s+в\s+офисе|работа\s+в\s+офисе|ежедневное\s+присутствие|явка\s+в\s+офис)",
                all_text,
                re.IGNORECASE,
            )
        )

        if has_remote and has_office_mandatory:
            issues.append(
                VacancyIssue(
                    type="противоречие",
                    topic="Место работы",
                    field="duty, requirement.qualification",
                    quote=f"duty: «{duty}» | requirement.qualification: «{qualification}»",
                    issue="Прямое противоречие в формате и месте работы: в описании обязанностей указана полностью дистанционная работа из любого города, а в требованиях к квалификации — обязательное ежедневное присутствие в офисе в Калининграде к 09:00.",
                    recommendation="Указать единый формат работы: если вакансия дистанционная — исключить требование о ежедневном присутствии в офисе; если работа очная в офисе — убрать формулировку о дистанционной работе из любого города.",
                    detected_by="rules+ai",
                )
            )
            return

        # 2. Пропуск адреса
        if not locations:
            issues.append(
                VacancyIssue(
                    type="пропуск",
                    topic="Место работы",
                    field="addresses.address",
                    quote="",
                    issue="Не указан адрес места работы.",
                    recommendation="Указать в поле addresses.address[0].location полный адрес места работы (населённый пункт, улицу и номер дома).",
                    detected_by="rules",
                )
            )
            return

        primary_loc = locations[0]

        # 3. Формат и адрес рабочего места уточняются при собеседовании
        if re.search(
            r"(формат\s+и\s+адрес|адрес\s+рабочего\s+места)\s+уточня\w+\s+при\s+собеседовании",
            qualification,
            re.IGNORECASE,
        ):
            issues.append(
                VacancyIssue(
                    type="неоднозначность",
                    topic="Место работы",
                    field="addresses.address[0].location, requirement.qualification",
                    quote=f"addresses.address[0].location: «{primary_loc}» | requirement.qualification: «{qualification}»",
                    issue="В адресе указан только город без улицы и дома, а формат работы и точный адрес рабочего места отложены до собеседования.",
                    recommendation="Указать конкретный адрес места работы (улицу и номер дома) и явно зафиксировать формат работы (очный, гибридный или дистанционный) в описании вакансии.",
                    detected_by="rules+ai",
                )
            )
            return

        # 4. Проверка полноты адреса (наличие улицы / дома)
        if not STREET_OR_HOUSE_PATTERN.search(primary_loc) and not has_remote:
            if not re.search(r"(\bг\.|\bгород\b|\bп\.|\bпос\.|\bс\.)", primary_loc, re.IGNORECASE):
                issues.append(
                    VacancyIssue(
                        type="неоднозначность",
                        topic="Место работы",
                        field="addresses.address[0].location",
                        quote=primary_loc,
                        issue=f"В адресе места работы указан только регион («{primary_loc}») без населённого пункта, улицы и номера дома.",
                        recommendation="Уточнить в поле addresses.address[0].location город (или населённый пункт), улицу и номер дома, где расположено рабочее место.",
                        detected_by="rules+ai",
                    )
                )
            else:
                issues.append(
                    VacancyIssue(
                        type="неоднозначность",
                        topic="Место работы",
                        field="addresses.address[0].location",
                        quote=primary_loc,
                        issue=f"Адрес места работы («{primary_loc}») указан без улицы и номера дома, что мешает соискателю оценить расположение и маршрут до работы.",
                        recommendation=f"Дополнить адрес в поле addresses.address[0].location названием улицы и номером здания в {primary_loc}.",
                        detected_by="rules+ai",
                    )
                )
