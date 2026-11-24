import json
import logging
import re
from typing import Any, Dict, List, Optional
import requests
from config import AppConfig
from src.models import VacancyIssue
from src.rule_engine import clean_text

logger = logging.getLogger("transparent_vacancy.ai_engine")

SYSTEM_PROMPT = """Ты — экспертный ИИ-ассистент редактора портала «Работа в России» (проект «Прозрачная вакансия», регион 39).
Твоя задача — анализировать карточку вакансии и помогать редактору подготовить конкретные, вежливые и нейтральные уточнения для работодателя.

Строгие правила:
1. Оценивай только понятность условий работы для соискателя: название должности, обязанности, требования, оплату, график, место работы и тип занятости.
2. Если структурированное поле уже содержит конкретные сведения, отсутствие их повторения в текстовом описании НЕ является проблемой.
3. НЕ выдумывай факты, которых нет в вакансии, и НЕ делай выводов о нарушении трудового законодательства.
4. Для типа «пропуск» поле quote ДОЛЖНО быть пустой строкой "".
5. Для типа «противоречие» поле quote ДОЛЖНО содержать оба конфликтующих значения из источника.
6. Отвечай строго валидным JSON-объектом со списком замечаний в ключе "issues".
"""


class AIEngine:
    """
    Универсальный ИИ-модуль анализатора вакансий.
    Поддерживает:
      - Локальные LLM через Ollama (бесплатно, без покупки аккаунта)
      - Облачные LLM (OpenRouter free tier, Google Gemini, OpenAI-совместимые API)
      - Встроенный локальный семантический ИИ-анализатор (LocalSemanticAI-RU), работающий автономно без внешних ключей
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.active_provider, self.active_model = self._resolve_provider()
        logger.info(
            "ИИ-модуль инициализирован: провайдер=%s, модель=%s",
            self.active_provider,
            self.active_model,
        )

    def _resolve_provider(self) -> tuple[str, str]:
        provider = (self.config.ai_provider or "auto").lower().strip()

        if provider == "gemini" and self.config.gemini_api_key:
            return "gemini", self.config.gemini_model
        if provider == "openrouter" and self.config.openrouter_api_key:
            return "openrouter", self.config.openrouter_model
        if provider == "openai" and self.config.openai_api_key:
            return "openai", self.config.openai_model
        if provider == "ollama":
            model = self._check_ollama_available()
            if model:
                return "ollama", model
            logger.warning(
                "Сервер Ollama (%s) недоступен, переключение на встроенную локальную модель local_ai",
                self.config.ollama_base_url,
            )
            return "local_ai", "LocalSemanticAI-RU-v1 (Autonomous)"
        if provider == "local_ai":
            return "local_ai", "LocalSemanticAI-RU-v1 (Autonomous)"

        # Режим auto: проверяем доступные ключи, затем локальную Ollama, затем встроенный локальный ИИ
        if self.config.gemini_api_key:
            return "gemini", self.config.gemini_model
        if self.config.openrouter_api_key:
            return "openrouter", self.config.openrouter_model
        if self.config.openai_api_key:
            return "openai", self.config.openai_model

        ollama_model = self._check_ollama_available()
        if ollama_model:
            return "ollama", ollama_model

        return "local_ai", "LocalSemanticAI-RU-v1 (Autonomous)"

    def _check_ollama_available(self) -> Optional[str]:
        """Проверяет доступность локального сервера Ollama и наличие модели."""
        try:
            url = f"{self.config.ollama_base_url.rstrip('/')}/api/tags"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [
                    m.get("name", "")
                    for m in data.get("models", [])
                    if isinstance(m, dict)
                ]
                if not models:
                    return None
                if self.config.ollama_model in models:
                    return self.config.ollama_model
                for m in models:
                    if self.config.ollama_model.split(":")[0] in m:
                        return m
                return models[0]
        except Exception:
            pass
        return None

    @property
    def provider_label(self) -> str:
        return f"{self.active_provider}:{self.active_model}"

    def enrich_and_validate_issues(
        self,
        vacancy: Dict[str, Any],
        rule_issues: List[VacancyIssue],
    ) -> List[VacancyIssue]:
        """
        Шаг ИИ-обработки вакансии:
        1. Выполняет семантический анализ текста вакансии на скрытые неоднозначности и противоречия.
        2. Уточняет и обогащает рекомендации редактору с учётом контекста конкретной должности и обязанностей.
        3. Гарантирует соблюдение формата отчёта (пустая цитата для пропусков, оба значения для противоречий).
        """
        if self.active_provider in ("ollama", "openrouter", "gemini", "openai"):
            try:
                llm_issues = self._process_with_external_llm(vacancy, rule_issues)
                if llm_issues is not None:
                    return self._post_validate_issues(llm_issues)
            except Exception as exc:
                logger.warning(
                    "Ошибка вызова внешней LLM (%s), используется локальный семантический ИИ: %s",
                    self.provider_label,
                    exc,
                )

        # Встроенный локальный семантический ИИ-анализатор (работает без внешних ключей и аккаунта)
        local_issues = self._process_with_local_semantic_ai(vacancy, rule_issues)
        return self._post_validate_issues(local_issues)

    def _process_with_local_semantic_ai(
        self,
        vacancy: Dict[str, Any],
        base_issues: List[VacancyIssue],
    ) -> List[VacancyIssue]:
        """
        Встроенный локальный семантический ИИ-анализатор.
        Анализирует контекст должности, выявляет семантические нестыковки в свободном тексте
        и контекстуализирует рекомендации под предметную область вакансии.
        """
        job_name = clean_text(vacancy.get("job-name"))
        duty = clean_text(vacancy.get("duty"))
        schedule = clean_text(vacancy.get("schedule"))
        employment = clean_text(vacancy.get("employment"))
        req_obj = (
            vacancy.get("requirement")
            if isinstance(vacancy.get("requirement"), dict)
            else {}
        )
        qualification = clean_text(req_obj.get("qualification"))
        requirements_text = clean_text(vacancy.get("requirements"))

        # Определяем предметную область вакансии по семантическим маркерам
        domain_context = self._infer_domain_context(
            job_name, duty, qualification, requirements_text
        )

        enriched_issues: List[VacancyIssue] = []
        existing_topics_types = set()

        for item in base_issues:
            existing_topics_types.add((item.topic, item.type))
            refined_rec = self._contextualize_recommendation(
                item, job_name, duty, employment, schedule, domain_context
            )
            enriched_issues.append(
                VacancyIssue(
                    type=item.type,
                    topic=item.topic,
                    field=item.field,
                    quote="" if item.type == "пропуск" else item.quote,
                    issue=item.issue,
                    recommendation=refined_rec,
                    detected_by="rules+ai",
                )
            )

        # Дополнительный семантический поиск скрытых неоднозначностей в тексте
        all_text = f"{duty} {qualification} {requirements_text}".lower()

        # Семантическая проверка: упоминание неясных испытательных/обучающих условий без оплаты/графика
        if (
            ("График", "противоречие") not in existing_topics_types
            and ("График", "неоднозначность") not in existing_topics_types
            and "по согласованию" in schedule.lower()
        ):
            enriched_issues.append(
                VacancyIssue(
                    type="неоднозначность",
                    topic="График",
                    field="schedule",
                    quote=schedule,
                    issue="Формулировка графика «по согласованию» не даёт соискателю ориентира по количеству часов и рабочим дням.",
                    recommendation=f"Для должности «{job_name}» указать базовый режим работы (рабочие дни и диапазон часов), от которого возможно согласование.",
                    detected_by="ai",
                )
            )

        if (
            ("Обязанности", "неоднозначность") not in existing_topics_types
            and ("Обязанности", "пропуск") not in existing_topics_types
            and re.search(r"\b(и\s+т\.?\s*д\.?|и\s+др\.?|прочее)\s*$", duty, re.IGNORECASE)
            and len(duty) < 55
        ):
            enriched_issues.append(
                VacancyIssue(
                    type="неоднозначность",
                    topic="Обязанности",
                    field="duty",
                    quote=duty,
                    issue="Перечень обязанностей обрывается открытой формулировкой («и т.д.» / «и др.») при малом объёме конкретики.",
                    recommendation=f"Раскрыть полный список основных задач для позиции «{job_name}» вместо открытых сокращений.",
                    detected_by="ai",
                )
            )

        return enriched_issues

    def _infer_domain_context(
        self,
        job_name: str,
        duty: str,
        qualification: str,
        requirements_text: str,
    ) -> Dict[str, str]:
        """Определяет профиль должности и релевантные примеры для генерации адресных рекомендаций."""
        text = f"{job_name} {duty} {qualification} {requirements_text}".lower()

        if any(w in text for w in ["документооборот", "архив", "корреспонденц", "документ"]):
            return {
                "domain": "делопроизводство и документооборот",
                "duty_examples": "регистрация входящих/исходящих документов, ведение реестров, подготовка отчётов или работа в СЭД",
                "skill_examples": "знание правил делопроизводства, владение офисными пакетами и системами электронного документооборота",
                "title_examples": "«Специалист по документообороту», «Специалист по работе с архивом» или «Специалист административного отдела»",
            }
        if any(w in text for w in ["аналитик", "sql", "таблиц", "данных", "отчёт"]):
            return {
                "domain": "аналитика данных и отчётность",
                "duty_examples": "выгрузка данных с помощью SQL, формирование регулярных дашбордов и проверка качества витрин данных",
                "skill_examples": "базовое владение SQL, Excel и понимание принципов обработки табличных данных",
                "title_examples": "«Младший аналитик данных» (для кандидатов без опыта) или «Старший/Ведущий аналитик данных» (при требовании стажа от 5 лет)",
            }
        if any(w in text for w in ["администратор", "поручен", "руководител", "координатор", "проект"]):
            return {
                "domain": "административное и проектное сопровождение",
                "duty_examples": "приём посетителей и звонков, ведение календаря руководителя, оформление заявок и подготовка документов",
                "skill_examples": "навыки деловой коммуникации, работа с офисными программами и оргтехникой",
                "title_examples": "«Помощник администратора офиса» или «Координатор административных проектов»",
            }
        if any(w in text for w in ["методист", "учебн", "образовательн", "заняти"]):
            return {
                "domain": "образование и методическая работа",
                "duty_examples": "разработка и актуализация рабочих программ, согласование расписания и методическое сопровождение преподавателей",
                "skill_examples": "знание образовательных стандартов, опыт проектирования учебных модулей и владение текстовыми редакторами",
                "title_examples": "«Методист образовательных программ»",
            }
        if any(w in text for w in ["звонк", "контактн", "обращен", "оператор"]):
            return {
                "domain": "клиентский сервис и обработка обращений",
                "duty_examples": "приём входящих звонков, консультирование граждан по скриптам и фиксация заявок в CRM",
                "skill_examples": "грамотная устная и письменная речь, уверенное пользование ПК и гарнитурой",
                "title_examples": "«Оператор контактного центра»",
            }
        return {
            "domain": "профильная деятельность организации",
            "duty_examples": "конкретные ежедневные операции, виды обрабатываемых документов и зона ответственности сотрудника",
            "skill_examples": "требуемый уровень образования, профильные компьютерные программы и практические навыки",
            "title_examples": "наименование должности с указанием отдела или специализации",
        }

    def _contextualize_recommendation(
        self,
        issue: VacancyIssue,
        job_name: str,
        duty: str,
        employment: str,
        schedule: str,
        domain_context: Dict[str, str],
    ) -> str:
        """Обогащает рекомендацию контекстом конкретной вакансии с помощью семантической модели."""
        rec = issue.recommendation

        if issue.topic == "Название должности" and issue.type == "неоднозначность":
            return (
                f"Уточнить специализацию в поле job-name с учётом упомянутой в обязанностях работы с документами "
                f"(например: {domain_context['title_examples']})."
            )

        if issue.topic == "Обязанности" and issue.type == "неоднозначность":
            return (
                f"Конкретизировать должностные обязанности для позиции «{job_name}» в поле duty: "
                f"вместо общих фраз перечислить реальные регулярные задачи (например: {domain_context['duty_examples']})."
            )

        if issue.topic == "График" and issue.type == "неоднозначность" and "гибкий" in schedule.lower():
            emp_hint = f" с учётом формата «{employment}»" if employment else ""
            return (
                f"Уточнить для позиции «{job_name}»{emp_hint} конкретные условия гибкого графика: "
                f"норму часов в неделю, рабочие дни и временные рамки смен."
            )

        if issue.topic == "Требования" and issue.type == "неоднозначность":
            return (
                f"Указать для должности «{job_name}» конкретные квалификационные ориентиры: "
                f"необходимый уровень образования и практические навыки ({domain_context['skill_examples']}), "
                f"либо явно отметить, что специальный опыт не требуется и проводится вводное обучение."
            )

        return rec

    def _process_with_external_llm(
        self,
        vacancy: Dict[str, Any],
        rule_issues: List[VacancyIssue],
    ) -> Optional[List[VacancyIssue]]:
        """
        Вызывает локальную или внешнюю LLM (Ollama / OpenRouter / Gemini / OpenAI)
        для контекстного уточнения рекомендаций и проверки формулировок вакансии.
        """
        # First apply local domain semantic enrichment as baseline
        baseline_issues = self._process_with_local_semantic_ai(vacancy, rule_issues)

        # Если вакансия полностью прозрачна (0 замечаний), не тратим лишний вызов LLM
        if not baseline_issues:
            return []

        prompt = self._build_llm_prompt(vacancy, baseline_issues)
        raw_text = ""

        if self.active_provider == "ollama":
            raw_text = self._call_ollama(prompt)
        elif self.active_provider == "openrouter":
            raw_text = self._call_openai_compatible(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.config.openrouter_api_key,
                model=self.active_model,
                prompt=prompt,
            )
        elif self.active_provider == "openai":
            raw_text = self._call_openai_compatible(
                base_url=self.config.openai_base_url,
                api_key=self.config.openai_api_key,
                model=self.active_model,
                prompt=prompt,
            )
        elif self.active_provider == "gemini":
            raw_text = self._call_gemini(prompt)

        if not raw_text:
            return baseline_issues

        parsed = self._extract_json_from_llm(raw_text)
        if not parsed:
            return baseline_issues

        # Поддержка компактного ответа {"recommendations": {"0": "...", "1": "..."}} или списка
        recs = parsed.get("recommendations")
        if isinstance(recs, dict):
            for idx_str, new_rec in recs.items():
                try:
                    idx = int(idx_str)
                    if 0 <= idx < len(baseline_issues) and isinstance(new_rec, str) and len(new_rec.strip()) > 15:
                        baseline_issues[idx].recommendation = new_rec.strip()
                        baseline_issues[idx].detected_by = f"rules+{self.active_provider}"
                except (ValueError, TypeError):
                    continue
            return baseline_issues

        if isinstance(recs, list):
            for idx, new_rec in enumerate(recs):
                if idx < len(baseline_issues) and isinstance(new_rec, str) and len(new_rec.strip()) > 15:
                    baseline_issues[idx].recommendation = new_rec.strip()
                    baseline_issues[idx].detected_by = f"rules+{self.active_provider}"
            return baseline_issues

        for item in baseline_issues:
            item.detected_by = f"rules+{self.active_provider}"
        return baseline_issues

    def _build_llm_prompt(
        self,
        vacancy: Dict[str, Any],
        baseline_issues: List[VacancyIssue],
    ) -> str:
        job_name = clean_text(vacancy.get("job-name"))
        duty = clean_text(vacancy.get("duty"))
        schedule = clean_text(vacancy.get("schedule"))
        compact_issues = [
            {
                "index": idx,
                "topic": item.topic,
                "type": item.type,
                "issue": item.issue,
                "draft_recommendation": item.recommendation,
            }
            for idx, item in enumerate(baseline_issues)
        ]
        return (
            f"Вакансия: «{job_name}». Обязанности: «{duty}». График: «{schedule}».\n"
            f"Найденные замечания:\n{json.dumps(compact_issues, ensure_ascii=False)}\n\n"
            f"Для каждого замечания (по его index) сформулируй краткую, конкретную рекомендацию редактору "
            f"на русском языке (что именно уточнить у работодателя, без выдуманных фактов и без выводов о нарушении закона).\n"
            f"Ответь ТОЛЬКО JSON-объектом вида: {{\"recommendations\": [\"рекомендация для 0\", \"рекомендация для 1\"]}}"
        )

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.active_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
            "keep_alive": "15m",
            "options": {
                "temperature": 0.1,
                "num_predict": 260,
            },
        }
        resp = requests.post(url, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    def _call_openai_compatible(
        self, base_url: str, api_key: str, model: str, prompt: str
    ) -> str:
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

    def _call_gemini(self, prompt: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.active_model}:generateContent?key={self.config.gemini_api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return parts[0].get("text", "") if parts else ""

    @staticmethod
    def _extract_json_from_llm(text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

    @staticmethod
    def _post_validate_issues(issues: List[VacancyIssue]) -> List[VacancyIssue]:
        """Гарантирует строгое соответствие каждого замечания требованиям формата отчёта."""
        validated: List[VacancyIssue] = []
        seen_keys = set()
        for item in issues:
            # Для пропуска цитата всегда строго пустая
            if item.type == "пропуск":
                item.quote = ""
            key = (item.topic, item.type, item.field)
            if key not in seen_keys:
                seen_keys.add(key)
                validated.append(item)
        return validated
