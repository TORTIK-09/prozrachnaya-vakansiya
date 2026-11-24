import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from config import AppConfig
from src.analyzer import VacancyAnalyzer
from src.models import CandidateResume, TailoredResumeResult, VacancyIssue
from src.rule_engine import clean_text, parse_number


def load_sample_resumes(path: str | Path = "data/sample_resumes.json") -> List[CandidateResume]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [CandidateResume.model_validate(item) for item in raw]


class ResumeAdapter:
    """
    Дополнительный сценарий на данных API региона 39:
    Формирует адаптированный черновик резюме, сопроводительное письмо и список уточняющих вопросов
    работодателю под выбранную вакансию строго на основе опыта, навыков и образования из исходного резюме.
    """

    def __init__(self, config: AppConfig, analyzer: Optional[VacancyAnalyzer] = None):
        self.config = config
        self.analyzer = analyzer or VacancyAnalyzer(config)

    def adapt_resume_for_vacancy(
        self,
        resume: CandidateResume,
        vacancy: Dict[str, Any],
    ) -> TailoredResumeResult:
        vac_id = str(vacancy.get("id", "UNKNOWN")).strip()
        job_name = clean_text(vacancy.get("job-name"))
        company_obj = vacancy.get("company")
        company_name = (
            clean_text(company_obj.get("name"))
            if isinstance(company_obj, dict)
            else "Работодатель региона 39"
        )
        duty = clean_text(vacancy.get("duty"))
        schedule = clean_text(vacancy.get("schedule"))
        employment = clean_text(vacancy.get("employment"))
        req_obj = (
            vacancy.get("requirement")
            if isinstance(vacancy.get("requirement"), dict)
            else {}
        )
        req_education = clean_text(req_obj.get("education"))
        req_experience = parse_number(req_obj.get("experience"))
        qualification = clean_text(req_obj.get("qualification"))
        requirements_text = clean_text(vacancy.get("requirements"))

        vacancy_full_text = f"{job_name} {duty} {qualification} {requirements_text}".lower()

        # 1. Сопоставление навыков строго из исходного резюме
        matched_skills: List[str] = []
        other_skills: List[str] = []
        for skill in resume.skills:
            tokens = [
                t
                for t in re.findall(r"[а-яА-Яa-zA-Z0-9]+", skill.lower())
                if len(t) >= 3
            ]
            is_match = any(
                tok[:4] in vacancy_full_text or tok in vacancy_full_text
                for tok in tokens
            )
            if is_match:
                matched_skills.append(skill)
            else:
                other_skills.append(skill)

        prioritized_skills = matched_skills + other_skills

        # 2. Выделение релевантных обязанностей из исходного опыта работы (без выдумывания новых фактов)
        highlighted_history: List[Dict[str, Any]] = []
        relevant_duties_count = 0
        for job in resume.work_history:
            duties_list = job.get("duties", [])
            relevant_duties: List[str] = []
            additional_duties: List[str] = []
            for d in duties_list:
                d_tokens = [
                    t
                    for t in re.findall(r"[а-яА-Яa-zA-Z0-9]+", d.lower())
                    if len(t) >= 4
                ]
                if any(tok[:5] in vacancy_full_text for tok in d_tokens):
                    relevant_duties.append(d)
                    relevant_duties_count += 1
                else:
                    additional_duties.append(d)

            highlighted_history.append(
                {
                    "period": job.get("period", ""),
                    "company": job.get("company", ""),
                    "role": job.get("role", ""),
                    "prioritized_duties": relevant_duties + additional_duties,
                }
            )

        # 3. Проверка недостающих или требующих уточнения требований вакансии
        missing_reqs: List[str] = []

        # Проверка стажа
        years_in_text_match = re.search(
            r"(?:не\s+менее|от)\s+(\d+)\s*(?:лет|года|год)\s+опыта",
            f"{qualification} {requirements_text}",
            re.IGNORECASE,
        )
        effective_req_exp = req_experience
        if years_in_text_match:
            effective_req_exp = max(effective_req_exp, int(years_in_text_match.group(1)))

        if resume.experience_years < effective_req_exp:
            missing_reqs.append(
                f"Требуемый стаж в описании вакансии ({effective_req_exp} г./лет) превышает фактический стаж в исходном резюме ({resume.experience_years} г./лет)."
            )

        # Проверка уровня образования
        if (
            "высшее" in req_education.lower()
            and "высшее" not in resume.education_level.lower()
        ):
            missing_reqs.append(
                f"В вакансии указано образование «{req_education}», в резюме — «{resume.education_level}»."
            )

        # 4. Расчёт оценки совместимости (Match Score)
        skill_ratio = (
            len(matched_skills) / max(1, len(resume.skills))
            if resume.skills
            else 0.5
        )
        score = int(45 + min(35, skill_ratio * 45) + min(20, relevant_duties_count * 7))
        if missing_reqs:
            score = max(25, score - 15 * len(missing_reqs))
        score = min(100, score)

        # 5. Анализ самой вакансии базовым анализатором для генерации вопросов работодателю
        base_issues = self.analyzer.rule_engine.analyze(vacancy)
        questions = self._generate_questions_from_issues(base_issues)

        # 6. Формирование блока «О себе» строго по данным исходного резюме
        top_skills_str = ", ".join(prioritized_skills[:4])
        prof_summary = (
            f"Соискатель из {resume.city} с образованием уровня «{resume.education_level}» "
            f"({resume.education_details}) и подтверждённым опытом работы ({resume.experience_years} г./лет). "
            f"Ключевые компетенции из резюме для позиции «{job_name}»: {top_skills_str}."
        )

        tailored_draft = {
            "target_position": job_name,
            "candidate_alias": resume.candidate_alias,
            "city": resume.city,
            "desired_employment": employment or "Полная занятость",
            "preferred_schedule": resume.preferred_schedule or schedule,
            "education": {
                "level": resume.education_level,
                "details": resume.education_details,
            },
            "experience_years": resume.experience_years,
            "professional_summary": prof_summary,
            "key_skills_prioritized": prioritized_skills,
            "work_experience": highlighted_history,
        }

        cover_letter = (
            f"Здравствуйте! Меня заинтересовала вакансия «{job_name}» (ID: {vac_id}) в организации «{company_name}». "
            f"Моё образование ({resume.education_details}) и практический опыт ({resume.experience_years} г./лет) "
            f"включают навыки: {top_skills_str}, что напрямую соотносится с задачами вашей вакансии ({duty}). "
            f"Буду рад(а) обсудить сотрудничество на собеседовании."
        )

        match_summary = (
            f"Совпадение {score}%: подтверждено {len(matched_skills)} из {len(resume.skills)} навыков резюме "
            f"и {relevant_duties_count} релевантных блок(а/ов) практического опыта."
        )

        return TailoredResumeResult(
            vacancy_id=vac_id,
            vacancy_title=job_name,
            company_name=company_name,
            candidate_alias=resume.candidate_alias,
            match_score=score,
            match_summary=match_summary,
            matched_skills=matched_skills,
            missing_or_unconfirmed_requirements=missing_reqs,
            tailored_resume_draft=tailored_draft,
            cover_letter_draft=cover_letter,
            questions_for_employer=questions,
            ai_provider_used=self.analyzer.ai_engine.provider_label,
        )

    @staticmethod
    def _generate_questions_from_issues(issues: List[VacancyIssue]) -> List[str]:
        """Преобразует замечания базового отчёта в вежливые вопросы соискателя работодателю."""
        questions: List[str] = []
        for item in issues:
            if item.topic == "Оплата":
                questions.append(
                    "Подскажите, пожалуйста, каков ориентир или вилка заработной платы на данной позиции?"
                )
            elif item.topic == "График":
                questions.append(
                    "Уточните, пожалуйста, точный график работы: рабочие дни недели и часы начала и окончания смены?"
                )
            elif item.topic == "Место работы":
                questions.append(
                    "По какому адресу в Калининградской области находится рабочее место и предусмотрен ли очный или дистанционный формат?"
                )
            elif item.topic == "Обязанности":
                questions.append(
                    "Какие основные ежедневные задачи и виды документов будут входить в зону ответственности сотрудника?"
                )
            elif item.topic == "Требования":
                questions.append(
                    "Рассматриваются ли кандидаты без длительного стажа и предусмотрено ли вводное обучение?"
                )
            elif item.topic == "Название должности":
                questions.append(
                    "В каком отделе открыта вакансия и какова основная специализация должности?"
                )
        return questions
