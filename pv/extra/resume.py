"""Свой сценарий: черновик резюме под выбранную вакансию.

Кому помогает: соискателю и карьерному консультанту центра занятости.
Что берёт: запись вакансии из готового отчёта (поля источника и найденные замечания) и резюме соискателя.
Что выдаёт: черновик резюме, в котором навыки и опыт переставлены по близости к вакансии; список требований
вакансии, которые резюме не подтверждает; вопросы работодателю по замечаниям отчёта.

Главное правило: в черновик попадает только то, что есть в исходном резюме. Сценарий ничего не дописывает —
он отбирает и переставляет. Это проверяется функцией `invented` и тестами.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..config import ROOT

RESUMES_PATH = ROOT / "examples" / "sample_resumes.json"
VACANCY_TEXT_FIELDS = ("job-name", "duty", "requirements", "requirement.qualification", "qualification")
_WORD = re.compile(r"[а-яёa-z0-9]+", re.I)
_STOP = {"работа", "работы", "работе", "опыт", "опыта", "навыки", "навык", "знание", "знания", "основы", "ведение",
         "подготовка", "учёт", "учет", "базовый", "базовые", "установка", "настройка", "оформление"}
_EDU_RANK = {"среднее": 1, "среднее профессиональное": 2, "высшее": 3}

_REQ_SPLIT = re.compile(r"(?:[;\n•]+|(?<=[а-яёa-z0-9])\.\s+)", re.I)
_REQ_SKIP = re.compile(
    r"(?:приветствуется|желательн|преимуществ|не\s+обязател|не\s+требует|не\s+предъявля|"
    r"без\s+опыта|остальному\s+науч|обучим|обучение\s+(?:на\s+месте|предоставля)|"
    r"на\s+собеседован|график\s+работ|заработн\w*\s+плат|выплат\w*\s+з|условия:)",
    re.I,
)
_REQ_TEXT_YEARS = re.compile(r"(?:от|не\s+менее|минимум|более)\s+(\d{1,2})\s*(?:лет|года|год)\b", re.I)
_SOFT_OR_GENERIC_STEMS = {
    "ответ", "актив", "внима", "аккур", "чисто", "обуча", "пункт", "добро", "комму", "дисци",
    "стрес", "иници", "испол", "орган", "креат", "коман", "колле", "вредн", "привы", "желан",
    "готов", "увере", "владе", "умени", "уметь", "спосо", "требу", "обяза", "необх", "должн",
    "уровн", "профи", "специ", "образ", "высше", "средн", "класс", "стажа", "более", "менее",
    "рабоч", "места", "месте", "своем", "людей", "сотру", "динам", "резул", "посто", "разв",
    "также", "будет", "иметь", "налиц", "налич", "очень", "хорош", "высок", "низк", "разли",
}
_MAX_TEXT_GAPS = 5


def _resume_stems(resume: dict) -> set[str]:
    parts: list[str] = [
        str(resume.get("desired_role") or ""),
        str(resume.get("education_level") or ""),
        str(resume.get("education") or ""),
    ]
    parts.extend(str(s) for s in resume.get("skills") or [])
    for job in resume.get("work") or []:
        parts.append(str(job.get("role") or ""))
        parts.extend(str(d) for d in job.get("duties") or [])
    return _stems(" ".join(parts))


def _text_experience_years(fields: dict) -> int:
    best = 0
    for key in ("requirements", "requirement.qualification", "qualification"):
        raw = str(fields.get(key) or "")
        for part in _REQ_SPLIT.split(raw):
            if _REQ_SKIP.search(part):
                continue
            for m in _REQ_TEXT_YEARS.finditer(part):
                years = int(m.group(1))
                if 1 <= years <= 30:
                    best = max(best, years)
    return best


def _unconfirmed_text_requirements(fields: dict, resume_stems: set[str]) -> list[str]:
    raw = str(fields.get("requirements") or "").strip()
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in _REQ_SPLIT.split(raw):
        item = re.sub(r"^[\s\-–—•*]+", "", part).strip(" .;,:")
        if len(item) < 4 or _REQ_SKIP.search(item):
            continue
        if _REQ_TEXT_YEARS.search(item):
            continue
        item_stems = {s for s in _stems(item) if s not in _SOFT_OR_GENERIC_STEMS and not s.isdigit()}
        if not item_stems:
            continue
        if item_stems & resume_stems:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(f"В резюме не подтверждено требование: «{item}».")
        if len(out) >= _MAX_TEXT_GAPS:
            break
    return out


def load_resumes(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path else RESUMES_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    return [r for r in data if isinstance(r, dict) and r.get("id")]


def _stems(text: str) -> set[str]:
    """Основы слов: первые пять букв — грубо, но для русского текста без словаря этого достаточно."""
    return {w[:5] for w in (m.group(0).lower() for m in _WORD.finditer(text or "")) if len(w) >= 3 and w not in _STOP}


def _overlap(text: str, vacancy_stems: set[str]) -> int:
    return len(_stems(text) & vacancy_stems)


def _edu_rank(level: str) -> int:
    level = (level or "").lower()
    best = 0
    for name, rank in _EDU_RANK.items():
        if name in level:
            best = max(best, rank)
    return best


QUESTIONS = {
    "title": "Какая специализация у этой должности и в каком подразделении она открыта?",
    "duties": "Какие задачи предстоит выполнять регулярно?",
    "requirements": "Какие требования к опыту, образованию и навыкам обязательны, а какие желательны?",
    "salary": "Каков размер оплаты — сумма или диапазон — и от чего он зависит?",
    "schedule": "Какие рабочие дни и часы и кто определяет расписание?",
    "location": "По какому адресу находится рабочее место и какой формат работы — в офисе или дистанционно?",
    "employment": "Какой тип занятости — полная или частичная?",
}


def _questions(findings: list[dict]) -> list[str]:
    """Замечания отчёта → вопросы соискателя работодателю, по одному на тему."""
    out: list[str] = []
    for f in findings:
        q = QUESTIONS.get(f.get("topic"))
        if q and q not in out:
            out.append(q)
    return out


def draft(resume: dict, entry: dict) -> dict[str, Any]:
    """Черновик резюме под вакансию. `entry` — запись вакансии из отчёта (с полями `fields` и `findings`)."""
    fields = entry.get("fields") or {}
    vacancy_text = " ".join(str(fields.get(f, "")) for f in VACANCY_TEXT_FIELDS)
    stems = _stems(vacancy_text)

    skills = [str(s) for s in resume.get("skills") or []]
    matched = [s for s in skills if _overlap(s, stems)]
    other = [s for s in skills if s not in matched]
    matched.sort(key=lambda s: -_overlap(s, stems))

    work = []
    relevant_duties = 0
    for job in resume.get("work") or []:
        duties = [str(d) for d in job.get("duties") or []]
        close = sorted((d for d in duties if _overlap(d, stems)), key=lambda d: -_overlap(d, stems))
        relevant_duties += len(close)
        work.append({"period": job.get("period", ""), "company": job.get("company", ""), "role": job.get("role", ""),
                     "duties": close + [d for d in duties if d not in close], "relevant": len(close)})

    # чего вакансия требует, а резюме не подтверждает — в черновик это не дописывается
    gaps: list[str] = []
    try:
        need_years = int(fields.get("requirement.experience", 0) or 0)
    except ValueError:
        need_years = 0
    have_years = int(resume.get("experience_years") or 0)
    text_years = _text_experience_years(fields)
    if need_years > have_years:
        gaps.append(f"В вакансии указан опыт от {need_years} лет, в резюме — {have_years}.")
    elif text_years > have_years:
        gaps.append(f"В тексте вакансии указан опыт от {text_years} лет, в резюме — {have_years}.")
    need_edu, have_edu = str(fields.get("requirement.education", "")), str(resume.get("education_level", ""))
    if _edu_rank(need_edu) > _edu_rank(have_edu):
        gaps.append(f"В вакансии указано образование «{need_edu}», в резюме — «{have_edu}».")
    gaps.extend(_unconfirmed_text_requirements(fields, _resume_stems(resume)))

    findings = entry.get("findings") or []
    questions = _questions(findings)
    cautions = [str(f.get("problem", "")) for f in findings if f.get("kind") == "contradiction"]

    title = entry.get("title") or "вакансию"
    summary_parts = [f"Образование: {resume.get('education', '')}".rstrip(": ")]
    if have_years:
        summary_parts.append(f"опыт работы: {have_years} г.")
    if matched:
        summary_parts.append("ключевые навыки: " + ", ".join(matched[:4]))
    letter = (f"Здравствуйте! Меня заинтересовала вакансия «{title}». "
              + (f"Из моих навыков к ней ближе всего: {', '.join(matched[:4])}. " if matched else "")
              + (f"В работе у меня были похожие задачи: {work[0]['duties'][0][:1].lower() + work[0]['duties'][0][1:]}. "
                 if work and work[0]["relevant"] else "")
              + "Буду рад(а) обсудить вакансию и ответить на вопросы.")
    return {
        "vacancy": {"id": entry.get("id"), "title": entry.get("title"), "company": entry.get("company"),
                    "url": entry.get("url")},
        "resume_id": resume.get("id"),
        "match": {"skills_matched": len(matched), "skills_total": len(skills), "duties_matched": relevant_duties},
        "draft": {
            "target_position": title,
            "city": resume.get("city", ""),
            "summary": "; ".join(summary_parts) + ".",
            "skills_first": matched,
            "skills_other": other,
            "education": {"level": have_edu, "details": resume.get("education", "")},
            "experience_years": have_years,
            "work": work,
        },
        "cover_letter": letter,
        "not_confirmed": gaps,
        "questions_for_employer": questions,
        "cautions": cautions,
    }


def invented(result: dict, resume: dict) -> list[str]:
    """Навыки и обязанности черновика, которых нет в исходном резюме. Должен быть пустым списком."""
    skills = set(map(str, resume.get("skills") or []))
    duties = {str(d) for job in resume.get("work") or [] for d in job.get("duties") or []}
    d = result["draft"]
    bad = [s for s in d["skills_first"] + d["skills_other"] if s not in skills]
    bad += [x for job in d["work"] for x in job["duties"] if x not in duties]
    if d["experience_years"] != int(resume.get("experience_years") or 0):
        bad.append("experience_years")
    if d["education"]["details"] != resume.get("education", ""):
        bad.append("education")
    return bad
