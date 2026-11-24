import json
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from config import get_config
from src.analyzer import VacancyAnalyzer
from src.api_client import (
    APIFetchError,
    extract_vacancies_from_payload,
    load_vacancies_from_file,
)
from src.resume_adapter import ResumeAdapter, load_sample_resumes

app = FastAPI(
    title="Прозрачная вакансия — КАНТКОД (Регион 39)",
    description="Автоматический аудит понятности условий работы в вакансиях Калининградской области и ИИ-адаптация резюме",
    version="1.0.0",
)


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    cfg = get_config()
    analyzer = VacancyAnalyzer(cfg)
    return {
        "status": "ok",
        "api_url": cfg.api_url,
        "max_vacancies": cfg.max_vacancies,
        "page_limit": cfg.page_limit,
        "schedule_interval_minutes": cfg.schedule_interval_minutes,
        "schedule_cron": cfg.schedule_cron,
        "ai_provider": analyzer.ai_engine.provider_label,
    }


@app.post("/api/verify")
def run_verify(
    input_path: Optional[str] = Query(default=None),
    output_path: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    cfg = get_config()
    analyzer = VacancyAnalyzer(cfg)
    try:
        report = analyzer.run_verify_mode(
            input_path=input_path or cfg.sample_input_path,
            output_path=output_path or cfg.sample_report_output_path,
        )
        return report.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/upload-verify")
async def upload_and_verify(file: UploadFile = File(...)) -> Dict[str, Any]:
    cfg = get_config()
    analyzer = VacancyAnalyzer(cfg)
    try:
        raw_bytes = await file.read()
        payload = json.loads(raw_bytes.decode("utf-8"))
        vacancies = extract_vacancies_from_payload(payload)
        report = analyzer.analyze_vacancies_list(
            vacancies=vacancies,
            mode="verify",
            source=file.filename or "uploaded_vacancies.json",
            pages_processed=1,
        )
        analyzer.report_manager.save_report(
            report, cfg.sample_report_output_path, update_last_successful=True
        )
        return report.model_dump()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Ошибка анализа загруженного JSON: {exc}"
        ) from exc


@app.post("/api/live")
def run_live(
    api_url: Optional[str] = Query(default=None),
    max_vacancies: Optional[int] = Query(default=None),
    page_limit: Optional[int] = Query(default=None),
) -> Dict[str, Any]:
    cfg = get_config()
    analyzer = VacancyAnalyzer(cfg)
    try:
        report = analyzer.run_live_mode(
            api_url=api_url or cfg.api_url,
            max_vacancies=max_vacancies or cfg.max_vacancies,
            page_limit=page_limit or cfg.page_limit,
        )
        return report.model_dump()
    except APIFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/report")
def get_current_report(mode: str = Query(default="sample")) -> Dict[str, Any]:
    cfg = get_config()
    analyzer = VacancyAnalyzer(cfg)
    target = (
        cfg.sample_report_output_path
        if mode == "sample"
        else cfg.report_output_path
    )
    report = analyzer.report_manager.load_report(target)
    if not report:
        report = analyzer.run_verify_mode()
    else:
        report.ai_provider_used = analyzer.ai_engine.provider_label
    return report.model_dump()


@app.get("/api/resumes")
def get_resumes_and_vacancies() -> Dict[str, Any]:
    cfg = get_config()
    resumes = load_sample_resumes()
    vacancies, _ = load_vacancies_from_file(cfg.sample_input_path)
    return {
        "resumes": [r.model_dump() for r in resumes],
        "vacancies": [
            {
                "id": v.get("id"),
                "job_name": v.get("job-name"),
                "company": (v.get("company") or {}).get("name", ""),
                "location": (
                    ((v.get("addresses") or {}).get("address") or [{}])[0].get(
                        "location", ""
                    )
                ),
            }
            for v in vacancies
        ],
    }


@app.post("/api/resume-adapt")
def adapt_resume(
    resume_id: str = Query(default="RES-39-01"),
    vacancy_id: str = Query(default="DEMO-39-001"),
) -> Dict[str, Any]:
    cfg = get_config()
    resumes = load_sample_resumes()
    vacancies, _ = load_vacancies_from_file(cfg.sample_input_path)

    selected_resume = next(
        (r for r in resumes if r.resume_id == resume_id),
        resumes[0] if resumes else None,
    )
    if not selected_resume:
        raise HTTPException(status_code=404, detail="Резюме не найдено")

    selected_vac = next(
        (v for v in vacancies if str(v.get("id")) == vacancy_id),
        vacancies[0] if vacancies else None,
    )
    if not selected_vac:
        raise HTTPException(status_code=404, detail="Вакансия не найдена")

    adapter = ResumeAdapter(cfg)
    result = adapter.adapt_resume_for_vacancy(selected_resume, selected_vac)
    out_path = Path("data/sample_resume_draft_output.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    return result.model_dump()


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Прозрачная вакансия — КАНТКОД (Регион 39)</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #1e293b;
      --card: #334155;
      --border: #475569;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --green: #22c55e;
      --amber: #f59e0b;
      --red: #ef4444;
      --purple: #a855f7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    header {
      background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
      border-bottom: 1px solid var(--border);
      padding: 20px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }
    .brand h1 {
      margin: 0;
      font-size: 22px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge-region {
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid var(--accent);
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }
    .ai-pill {
      background: rgba(168, 85, 247, 0.15);
      border: 1px solid var(--purple);
      color: #e9d5ff;
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 13px;
    }
    .container {
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px 32px;
    }
    .tabs {
      display: flex;
      gap: 12px;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
    }
    .tab-btn {
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 10px 20px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      font-size: 14px;
      transition: all 0.15s;
    }
    .tab-btn.active {
      background: var(--accent);
      color: #0f172a;
      border-color: var(--accent);
    }
    .controls-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
    }
    .panel h3 {
      margin-top: 0;
      margin-bottom: 12px;
      font-size: 16px;
      color: var(--accent);
    }
    .form-row {
      display: flex;
      gap: 10px;
      margin-bottom: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    input[type="text"], input[type="number"], select {
      background: #0f172a;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 8px 12px;
      border-radius: 6px;
      flex: 1;
      font-size: 13px;
    }
    button.primary {
      background: var(--accent);
      color: #0f172a;
      border: none;
      padding: 9px 16px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
    }
    button.secondary {
      background: var(--card);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 9px 16px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
    }
    .stats-bar {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .stat-card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      text-align: center;
    }
    .stat-val {
      font-size: 28px;
      font-weight: 700;
      color: var(--accent);
    }
    .stat-label {
      font-size: 13px;
      color: var(--muted);
    }
    .vac-card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 16px;
    }
    .vac-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      padding-bottom: 12px;
      margin-bottom: 14px;
    }
    .vac-title {
      font-size: 18px;
      font-weight: 700;
    }
    .vac-meta {
      font-size: 13px;
      color: var(--muted);
    }
    .status-clean {
      background: rgba(34, 197, 94, 0.15);
      color: #4ade80;
      border: 1px solid var(--green);
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }
    .status-issues {
      background: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
      border: 1px solid var(--amber);
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }
    .issue-item {
      background: rgba(15, 23, 42, 0.65);
      border-left: 4px solid var(--amber);
      border-radius: 6px;
      padding: 12px 16px;
      margin-bottom: 10px;
      font-size: 14px;
    }
    .issue-item.type-противоречие { border-left-color: var(--red); }
    .issue-item.type-пропуск { border-left-color: var(--purple); }
    .issue-tags {
      display: flex;
      gap: 8px;
      margin-bottom: 6px;
      flex-wrap: wrap;
      font-size: 12px;
    }
    .tag {
      background: var(--card);
      padding: 2px 8px;
      border-radius: 4px;
      font-weight: 600;
    }
    .quote-box {
      background: rgba(255,255,255,0.04);
      padding: 6px 10px;
      border-radius: 4px;
      font-family: monospace;
      font-size: 12px;
      color: #cbd5e1;
      margin: 6px 0;
    }
    .rec-box {
      color: #93c5fd;
      margin-top: 6px;
    }
    pre.json-view {
      background: #090d16;
      color: #a5f3fc;
      padding: 16px;
      border-radius: 8px;
      overflow-x: auto;
      font-size: 12px;
      max-height: 500px;
    }
    .hidden { display: none; }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h1>Прозрачная вакансия <span class="badge-region">КАНТКОД • Регион 39</span></h1>
      <div style="font-size:13px; color: var(--muted);">Автоматический контроль понятности условий труда («Работа в России»)</div>
    </div>
    <div class="ai-pill" id="aiProviderBadge">ИИ-модуль: загрузка...</div>
  </header>

  <div class="container">
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('audit')">1. Базовый сценарий: Аудит вакансий</button>
      <button class="tab-btn" onclick="switchTab('resume')">2. Доп. сценарий: ИИ-Адаптация черновика резюме</button>
      <button class="tab-btn" onclick="switchTab('raw')">3. JSON-отчёт (формат для жюри)</button>
    </div>

    <!-- ВКЛАДКА 1: АУДИТ ВАКАНСИЙ -->
    <div id="tab-audit">
      <div class="controls-grid">
        <div class="panel">
          <h3>Проверочный режим (Локальный JSON без обращения к порталу)</h3>
          <div class="form-row">
            <input type="text" id="localFilePath" value="kantkod_sample_vacancies.json" />
            <button class="primary" onclick="runVerifyMode()">Запустить проверку</button>
          </div>
          <div class="form-row" style="margin-top:8px;">
            <input type="file" id="uploadJsonFile" accept=".json" style="font-size:12px;" />
            <button class="secondary" onclick="uploadAndVerify()">Проверить свой JSON</button>
          </div>
        </div>

        <div class="panel">
          <h3>Рабочий режим (API «Работа в России» Регион 39 + Расписание)</h3>
          <div class="form-row">
            <input type="text" id="liveApiUrl" value="http://opendata.trudvsem.ru/api/v1/vacancies/region/39" />
          </div>
          <div class="form-row">
            <label style="font-size:12px; color:var(--muted);">Макс. вакансий:</label>
            <input type="number" id="liveMaxVac" value="20" style="max-width:80px;" />
            <label style="font-size:12px; color:var(--muted);">Стр. limit:</label>
            <input type="number" id="livePageLimit" value="10" style="max-width:70px;" />
            <button class="primary" onclick="runLiveMode()">Ручной запуск из API</button>
          </div>
        </div>
      </div>

      <div id="alertBox" class="panel hidden" style="border-color: var(--amber); margin-bottom: 20px;"></div>

      <div class="stats-bar">
        <div class="stat-card">
          <div class="stat-val" id="statTotal">0</div>
          <div class="stat-label">Всего вакансий (уникальных ID)</div>
        </div>
        <div class="stat-card">
          <div class="stat-val" id="statClean" style="color: var(--green);">0</div>
          <div class="stat-label">Прозрачных (без замечаний)</div>
        </div>
        <div class="stat-card">
          <div class="stat-val" id="statIssuesVac" style="color: var(--amber);">0</div>
          <div class="stat-label">Требуют уточнения редактора</div>
        </div>
        <div class="stat-card">
          <div class="stat-val" id="statTotalIssues" style="color: var(--red);">0</div>
          <div class="stat-label">Всего замечаний найдено</div>
        </div>
        <div class="stat-card">
          <div class="stat-val" id="statPages">1</div>
          <div class="stat-label">Обработано страниц</div>
        </div>
      </div>

      <div id="vacanciesContainer"></div>
    </div>

    <!-- ВКЛАДКА 2: ДОП. СЦЕНАРИЙ (АДАПТАЦИЯ РЕЗЮМЕ) -->
    <div id="tab-resume" class="hidden">
      <div class="panel" style="margin-bottom:20px;">
        <h3>Дополнительный сценарий: Подготовка черновика резюме под вакансию региона 39</h3>
        <p style="font-size:13px; color:var(--muted); margin-top:0;">
          Сервис формирует черновик целевого резюме, сопроводительное письмо и список уточняющих вопросов работодателю
          <strong>строго на основе опыта, навыков и образования из исходного резюме соискателя</strong>.
        </p>
        <div class="form-row">
          <select id="selectResume"></select>
          <select id="selectVacancy"></select>
          <button class="primary" onclick="runResumeAdapt()">Сформировать черновик резюме</button>
        </div>
      </div>
      <div id="resumeResultContainer"></div>
    </div>

    <!-- ВКЛАДКА 3: СЫРОЙ JSON -->
    <div id="tab-raw" class="hidden">
      <div class="panel">
        <h3>JSON-отчёт (соответствует требованиям ТЗ)</h3>
        <pre class="json-view" id="rawJsonPre">Загрузка...</pre>
      </div>
    </div>
  </div>

  <script>
    function switchTab(tab) {
      ['audit', 'resume', 'raw'].forEach(t => {
        document.getElementById('tab-' + t).classList.toggle('hidden', t !== tab);
      });
      const btns = document.querySelectorAll('.tab-btn');
      btns[0].classList.toggle('active', tab === 'audit');
      btns[1].classList.toggle('active', tab === 'resume');
      btns[2].classList.toggle('active', tab === 'raw');
    }

    function showAlert(msg, isError=false) {
      const box = document.getElementById('alertBox');
      box.classList.remove('hidden');
      box.style.borderColor = isError ? 'var(--red)' : 'var(--accent)';
      box.innerHTML = msg;
    }

    function renderReport(report) {
      document.getElementById('aiProviderBadge').textContent = 'ИИ-модуль: ' + (report.ai_provider_used || 'auto');
      document.getElementById('statTotal').textContent = report.total_vacancies;
      document.getElementById('statClean').textContent = report.clean_vacancies;
      document.getElementById('statIssuesVac').textContent = report.vacancies_with_issues;
      document.getElementById('statTotalIssues').textContent = report.total_issues;
      document.getElementById('statPages').textContent = report.pages_processed || 1;
      document.getElementById('rawJsonPre').textContent = JSON.stringify(report, null, 2);

      const container = document.getElementById('vacanciesContainer');
      container.innerHTML = '';

      (report.vacancies || []).forEach(v => {
        const card = document.createElement('div');
        card.className = 'vac-card';
        const hasIssues = v.issues && v.issues.length > 0;
        const badgeHtml = hasIssues
          ? `<span class="status-issues">Замечаний: ${v.issues.length} • Прозрачность: ${v.transparency_score}%</span>`
          : `<span class="status-clean">Условия прозрачны (0 замечаний) • 100%</span>`;

        const urlHtml = v.url ? ` • <a href="${v.url}" target="_blank" style="color:var(--accent);">Карточка вакансии</a>` : '';

        let issuesHtml = '';
        if (!hasIssues) {
          issuesHtml = `<div style="color:#4ade80; font-size:13px;">Все ключевые поля (должность, оплата, график с часами, обязанности, требования, полный адрес) заполнены конкретно и без противоречий.</div>`;
        } else {
          v.issues.forEach(iss => {
            const quoteLine = iss.type === 'пропуск'
              ? `<div class="quote-box">Цитата: "" (пропуск сведений в поле ${iss.field})</div>`
              : `<div class="quote-box">Значение в источнике: ${iss.quote}</div>`;
            issuesHtml += `
              <div class="issue-item type-${iss.type}">
                <div class="issue-tags">
                  <span class="tag">${iss.type.toUpperCase()}</span>
                  <span class="tag">Тема: ${iss.topic}</span>
                  <span class="tag">Поле: ${iss.field}</span>
                  <span class="tag">Детектор: ${iss.detected_by}</span>
                </div>
                <div><strong>Суть проблемы:</strong> ${iss.issue}</div>
                ${quoteLine}
                <div class="rec-box"><strong>Рекомендация редактору:</strong> ${iss.recommendation}</div>
              </div>
            `;
          });
        }

        card.innerHTML = `
          <div class="vac-header">
            <div>
              <div class="vac-title">${v.id} — ${v.job_name || '(Без названия)'}</div>
              <div class="vac-meta">${v.company_name || ''} • ${v.location || 'Адрес не указан'}${urlHtml}</div>
            </div>
            <div>${badgeHtml}</div>
          </div>
          <div>${issuesHtml}</div>
        `;
        container.appendChild(card);
      });
    }

    async function runVerifyMode() {
      const path = document.getElementById('localFilePath').value;
      showAlert('Выполняется анализ локального JSON-файла в проверочном режиме...');
      const res = await fetch('/api/verify?input_path=' + encodeURIComponent(path), { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        showAlert('Ошибка: ' + (data.detail || 'не удалось выполнить проверку'), true);
        return;
      }
      showAlert(`Проверочный режим выполнен успешно. Источник: <strong>${data.source}</strong>`);
      renderReport(data);
    }

    async function uploadAndVerify() {
      const fileInput = document.getElementById('uploadJsonFile');
      if (!fileInput.files.length) {
        showAlert('Выберите JSON-файл для загрузки', true);
        return;
      }
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      const res = await fetch('/api/upload-verify', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) {
        showAlert('Ошибка загрузки: ' + (data.detail || ''), true);
        return;
      }
      showAlert(`Загруженный файл <strong>${data.source}</strong> успешно проверен!`);
      renderReport(data);
    }

    async function runLiveMode() {
      const url = document.getElementById('liveApiUrl').value;
      const maxV = document.getElementById('liveMaxVac').value;
      const pLim = document.getElementById('livePageLimit').value;
      showAlert('Запрос к API «Работа в России» (Регион 39)...');
      const res = await fetch(`/api/live?api_url=${encodeURIComponent(url)}&max_vacancies=${maxV}&page_limit=${pLim}`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        showAlert('API недоступен: ' + (data.detail || '') + '. Сохранён последний успешный отчёт.', true);
        return;
      }
      showAlert(`Рабочий режим: загружено вакансий: ${data.total_vacancies} (страниц: ${data.pages_processed}) из ${data.source}`);
      renderReport(data);
    }

    async function initResumeTab() {
      const res = await fetch('/api/resumes');
      const data = await res.json();
      const rSel = document.getElementById('selectResume');
      const vSel = document.getElementById('selectVacancy');
      rSel.innerHTML = data.resumes.map(r => `<option value="${r.resume_id}">${r.candidate_alias} — ${r.desired_role}</option>`).join('');
      vSel.innerHTML = data.vacancies.map(v => `<option value="${v.id}">${v.id}: ${v.job_name} (${v.location})</option>`).join('');
    }

    async function runResumeAdapt() {
      const rId = document.getElementById('selectResume').value;
      const vId = document.getElementById('selectVacancy').value;
      const res = await fetch(`/api/resume-adapt?resume_id=${encodeURIComponent(rId)}&vacancy_id=${encodeURIComponent(vId)}`, { method: 'POST' });
      const data = await res.json();
      const box = document.getElementById('resumeResultContainer');
      const draft = data.tailored_resume_draft;
      box.innerHTML = `
        <div class="vac-card">
          <div class="vac-header">
            <div>
              <div class="vac-title">Черновик резюме под вакансию «${data.vacancy_title}» (${data.vacancy_id})</div>
              <div class="vac-meta">${data.candidate_alias} • ${data.match_summary}</div>
            </div>
            <span class="status-clean">Совместимость: ${data.match_score}%</span>
          </div>
          <p><strong>Профильное резюме (О себе):</strong> ${draft.professional_summary}</p>
          <p><strong>Приоритетные навыки (из исходного резюме):</strong> ${draft.key_skills_prioritized.join(' • ')}</p>
          <p><strong>Образование (из исходного резюме):</strong> ${draft.education.level} — ${draft.education.details}</p>
          ${data.missing_or_unconfirmed_requirements.length ? `<div class="issue-item type-противоречие"><strong>Расхождения с требованиями вакансии (не добавлены в резюме во избежание вымысла):</strong><br/>${data.missing_or_unconfirmed_requirements.join('<br/>')}</div>` : ''}
          <div class="issue-item">
            <strong>Черновик сопроводительного письма:</strong><br/>${data.cover_letter_draft}
          </div>
          ${data.questions_for_employer.length ? `<div class="issue-item type-неоднозначность"><strong>Рекомендуемые вопросы работодателю (на основе аудита прозрачности этой вакансии):</strong><ul>${data.questions_for_employer.map(q => `<li>${q}</li>`).join('')}</ul></div>` : ''}
        </div>
      `;
    }

    window.addEventListener('DOMContentLoaded', async () => {
      const res = await fetch('/api/report?mode=sample');
      const data = await res.json();
      renderReport(data);
      await initResumeTab();
      await runResumeAdapt();
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return DASHBOARD_HTML
