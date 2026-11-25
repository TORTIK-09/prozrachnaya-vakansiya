"use strict";
// Уведомление о настройке (обучении) модели и окно выбора типа модели / API при первом запуске.

function calibDuration(sec) {
  if (sec === null || sec === undefined) return "оцениваю время…";
  if (sec < 60) return "меньше минуты";
  const m = Math.round(sec / 60);
  if (m < 60) return "примерно " + m + " мин";
  return "примерно " + Math.floor(m / 60) + " ч " + (m % 60) + " мин";
}
function calibModel(c) { return c.model && c.model.model ? "«" + c.model.model + "»" : ""; }
function calibKey(c) { return "pv-calib-closed:" + (c.model ? c.model.provider + "/" + c.model.model : "") + ":" + (c.finished_at || ""); }

function hasModelCookie() {
  return document.cookie.split(";").some(c => c.trim().startsWith("pv_model_configured=1"));
}

function setModelCookie() {
  document.cookie = "pv_model_configured=1; path=/; max-age=31536000; SameSite=Lax";
  try { localStorage.setItem("pv_model_configured", "1"); } catch { /* без хранилища */ }
}

window.assessModelIntelligence = function(modelName, modeType) {
  if (modeType === "rules_only") return {isWeak: false, reason: ""};
  const m = String(modelName || "").toLowerCase().trim();
  if (!m) return {isWeak: true, reason: "модель не указана"};
  const weakPatterns = [
    /\b(0\.5b|1b|1\.5b|2b|3b|3\.8b|4b|7b|8b|9b)\b/,
    /[-_:](0\.5b|1b|1\.5b|2b|3b|3\.8b|4b|7b|8b|9b)\b/,
    /\b(mini|nano|lite|tiny|small|micro|haiku|flash-lite|gpt-3\.5|turbo|gemma-2-9b|gemma2:9b|phi3|phi-3)\b/,
    /:free\b/,
  ];
  const isWeak = weakPatterns.some(rx => rx.test(m)) || modeType === "local" || modeType === "local_only";
  return {
    isWeak,
    model: modelName || "выбранная модель",
    warning: `⚠ Внимание: модель «${modelName || "выбранная ИИ"}» имеет ограниченный объём параметров / базовый уровень рассуждений (не является сильной флагманской моделью) и может ошибаться или галлюцинировать. Будьте осторожны с её ответами — обязательно сверяйте их с Системным AI в блоках сравнения внизу каждой карточки!`,
  };
};

window.weakModelBox = function(info) {
  if (!info || !info.model) return null;
  const check = window.assessModelIntelligence(info.model, info.mode_type);
  if (!check.isWeak) return null;
  const box = document.createElement("div");
  box.className = "weak-model-warn";
  const b = document.createElement("b");
  b.textContent = "⚠ Модель «" + info.model + "» — не из сильных";
  const t = document.createElement("div");
  t.textContent = "Она может ошибаться и придумывать проблемы, которых нет. Относитесь к её ответам осторожно и сверяйте их с ответом Системного AI.";
  box.append(b, t);
  return box;
};

function updateHeaderChip(info, rulesOnly, calState) {
  const chip = document.getElementById("modelChip");
  if (!chip) return;
  chip.title = "Нажмите, чтобы выбрать модель, локальную LLM или свой API";
  if (rulesOnly) {
    chip.lastElementChild.textContent = "режим: только системный AI ⚙";
    chip.classList.remove("on");
    return;
  }
  if (info && info.model) {
    let modeLabel = "локальная LLM + системный AI";
    if (info.mode_type === "local_only") modeLabel = "локальная LLM";
    else if (info.mode_type === "custom_api") modeLabel = (info.provider || "свой API") + " + системный AI";
    else if (info.mode_type === "custom_api_only") modeLabel = (info.provider || "свой API") + " (прямой)";
    const calSuffix = calState === "running" ? " · обучается…" : "";
    const weak = window.assessModelIntelligence(info.model, info.mode_type).isWeak;
    chip.lastElementChild.textContent = (weak ? "⚠ " : "") + "модель: " + info.model + " (" + modeLabel + calSuffix + ") ⚙";
    if (weak) chip.title = "Модель не из сильных — будьте осторожны с её ответами. Нажмите, чтобы сменить модель.";
    chip.classList.add("on");
  } else {
    chip.lastElementChild.textContent = (calState === "running" ? "модель обучается…" : "модель не подключена — системный AI") + " ⚙";
    chip.classList.remove("on");
  }
}

function showCalibration(c, onDone) {
  const box = document.getElementById("calib");
  if (!box) return;
  const title = document.getElementById("calibTitle"), sub = document.getElementById("calibSub");
  const bar = document.getElementById("calibBar"), close = document.getElementById("calibClose"), link = document.getElementById("calibLink");
  let closed = false;
  try { closed = localStorage.getItem(calibKey(c)) === "1"; } catch { /* без хранилища */ }
  box.className = "notice";
  close.hidden = true; if (link) link.hidden = true;
  const isLocal = c.model && (c.model.mode_type === "local" || c.model.provider === "ollama" || c.model.provider === "local");
  const trainKind = isLocal ? "Раздельная калибровка локальной ИИ " : "Раздельная калибровка внешнего API ";
  const poolInfo = c.system_ai_covert_pool;
  const poolNote = poolInfo && poolInfo.total_system_ai_pool
    ? ` Системный AI отдельно обработал пул ${poolInfo.total_system_ai_pool} вакансий с портала (${poolInfo.hidden_until_llm_count} скрыты до ответа ИИ).`
    : " Параллельно Системный AI скрытно калибруется на пуле портала (вакансии сверх ответов ИИ остаются скрытыми до ответа ИИ).";
  if (c.state === "running") {
    box.classList.add("show");
    title.textContent = trainKind + calibModel(c) + " — подождите, " + calibDuration(c.eta_seconds);
    sub.textContent = (c.total ? "Проверено примеров ИИ: " + c.done + " из " + c.total + " (без шаблонности). " : "Модель загружается. ") + poolNote;
    bar.style.width = (c.total ? Math.round(c.done / c.total * 100) : 3) + "%";
  } else if (c.state === "done" && !closed) {
    box.classList.add("show", "done");
    title.textContent = (isLocal ? "Локальная модель " : "Модель API ") + calibModel(c) + " откалибрована";
    sub.textContent = "Проверено примеров ИИ: " + c.vacancies + (c.errors ? ", резерв Системного AI: " + c.errors : "") + ". Отключено ненадёжных видов замечаний: " + c.blocked + ". В среднем " + c.seconds_per_vacancy + " с/вак." + poolNote;
    close.hidden = false; if (link) link.hidden = false;
  } else if (c.state === "failed" && !closed) {
    box.classList.add("show", "failed");
    title.textContent = "Откалибровать модель " + calibModel(c) + " не удалось";
    sub.textContent = (c.error || "") + " Сервис продолжает работать: замечания модели используются под контролем Системного AI.";
    close.hidden = false;
  }
  close.onclick = () => { try { localStorage.setItem(calibKey(c), "1"); } catch { /* без хранилища */ } box.className = "notice"; };
  if (c.state === "done" && onDone) onDone(c);
}

function openModelSetupModal(cfgData, isFirstRun, onSaved) {
  const existing = document.getElementById("modelSetupBackdrop");
  if (existing) existing.remove();

  const presets = (cfgData && cfgData.presets) || {};
  const localModels = (cfgData && cfgData.local_models) || ["qwen2.5:3b"];
  let modeType = (cfgData && cfgData.mode_type) || "local";
  let presetKey = (cfgData && cfgData.provider_preset) || (modeType === "local" ? "ollama" : "openrouter");
  let baseUrl = (cfgData && cfgData.base_url) || "http://127.0.0.1:11434/v1";
  let modelName = (cfgData && cfgData.model) || (localModels[0] || "qwen2.5:3b");
  let calibPlan = (cfgData && cfgData.calibration_plan) || null;
  let calibMinutes = (cfgData && cfgData.calibration_minutes) || (calibPlan ? calibPlan.effective_minutes : 2.0);
  let planTimer = null;

  const backdrop = document.createElement("div");
  backdrop.id = "modelSetupBackdrop";
  backdrop.className = "modal-backdrop";

  function renderPlanDetails(plan) {
    const host = backdrop.querySelector("#calibPlanDetails");
    if (!host) return;
    if (!plan) {
      host.innerHTML = `<div class="note">Рассчитываю параметры ПК/сервера и индивидуальный лимит тестов для выбранной LLM…</div>`;
      return;
    }
    const hw = plan.hardware || {};
    const mp = plan.model_meta || {};
    const gpuStr = hw.gpu_name ? `${hw.gpu_name} (${hw.vram_gb || 0} ГБ VRAM)` : "без дискретной GPU (CPU)";
    const loadWarn = hw.is_overloaded ? ` · ⚠ высокая загрузка ${hw.cpu_load_pct}%` : ` · загрузка ${hw.cpu_load_pct || 15}%`;
    const strengthMap = {
      weak: "базовая / малая модель",
      medium: "средняя модель",
      strong: "сильная флагманская модель",
    };
    const prioText = plan.priority_owner === "ai"
      ? "Приоритет при калибровке отдаётся сильной ИИ (или полному согласию с Системным AI)."
      : plan.priority_owner === "system_ai"
        ? "Из-за малой мощности модели / нагрузки ПК упор делается на глубокую калибровку Системного AI."
        : "Сбалансированное согласование Системного AI и выбранной модели.";

    host.innerHTML = `
      <div class="calib-metrics">
        <div class="calib-metric">
          <b>Тестов для этой LLM: ${plan.llm_target_tests} шт.</b>
          <span>Мин. порог: ${plan.min_tests} · потолок без шаблонности: ${plan.anti_template_cap} примеров</span>
        </div>
        <div class="calib-metric">
          <b>Модель ~${mp.params_b || "?"}B (${strengthMap[mp.strength] || "LLM"})</b>
          <span>Оценка отклика: ~${plan.sec_per_vacancy_est} с/вак. · лучшее время: до ${plan.best_time_minutes} мин</span>
        </div>
        <div class="calib-metric">
          <b>ПК / Сервер: ${hw.cpu_cores || "?"} ядер · ${hw.ram_gb || "?"} ГБ ОЗУ</b>
          <span>${gpuStr}${loadWarn}</span>
        </div>
      </div>
      ${plan.time_trimmed ? `
        <div class="calib-trim-warn">
          ⏱ Введённое время (${plan.requested_minutes} мин) превышает предел разумности для «${modelName}» — автоматически урезано до оптимального времени <b>${plan.effective_minutes} мин</b> (${plan.anti_template_cap} примеров), чтобы большое число прогонов не уводило профиль в шаблонность.
        </div>
      ` : ""}
      <div class="note" style="line-height:1.4;">
        <b>Раздельная калибровка Системного AI и ИИ:</b> пока калибруется выбранная ИИ (${plan.llm_target_tests} примеров за ~${plan.effective_minutes} мин), Системный AI отдельно собирает и очищает пул до <b>${plan.system_ai_hidden_pool_target}+</b> вакансий с портала. Все вакансии, обработанные Системным AI сверх ответов ИИ, <b>остаются скрытыми на сайте, пока на них не ответит ИИ</b>. ${prioText}
      </div>
    `;
  }

  async function refreshCalibrationPlan(applyTrimToInput) {
    try {
      const qs = new URLSearchParams({
        model: modelName || "",
        mode_type: modeType || "local",
        minutes: String(calibMinutes || ""),
        base_url: baseUrl || "",
      });
      const r = await fetch("/api/calibration-plan?" + qs.toString());
      if (!r.ok) return;
      const plan = await r.json();
      calibPlan = plan;
      if (applyTrimToInput && plan.time_trimmed) {
        calibMinutes = plan.effective_minutes;
        const minInp = backdrop.querySelector("#mCalibMinutes");
        if (minInp) minInp.value = String(plan.effective_minutes);
      }
      renderPlanDetails(plan);
    } catch { /* фоновый пересчёт */ }
  }

  function renderModalContent() {
    const isConnected = cfgData && cfgData.connected;
    const calState = cfgData && cfgData.calibration ? cfgData.calibration.state : "idle";
    const statusText = isConnected
      ? "Автопроверка состояния: подключена модель «" + (cfgData.model_info ? cfgData.model_info.model : modelName) + "» (" + (calState === "ready" || calState === "done" ? "профиль обучен" : calState === "running" ? "идёт обучение…" : "готова к обучению") + ")."
      : "Автопроверка состояния: выберите локальную LLM или укажите свой API-ключ для подключения и обучения.";

    const localKeys = ["ollama", "lmstudio", "vllm", "jan", "localai"];
    const cloudKeys = [
      "openrouter", "openai", "gemini", "anthropic", "deepseek", "grok",
      "mistral", "groq", "together", "fireworks", "perplexity", "cohere",
      "qwen_cloud", "yandexgpt", "gigachat", "huggingface", "custom"
    ];

    const localPresetsHtml = localKeys.map(k => {
      const p = presets[k] || {label: k};
      return `<option value="${k}" ${presetKey === k ? "selected" : ""}>${p.label}</option>`;
    }).join("");

    const cloudPresetsHtml = cloudKeys.map(k => {
      const p = presets[k] || {label: k};
      return `<option value="${k}" ${presetKey === k ? "selected" : ""}>${p.label}</option>`;
    }).join("");

    const activePresetObj = presets[presetKey] || {};
    const presetModels = Array.isArray(activePresetObj.models) ? activePresetObj.models : [];
    const combinedLocalModels = Array.from(new Set([...localModels, ...presetModels]));
    const localModelsDatalist = combinedLocalModels.map(m => `<option value="${m}">${m}</option>`).join("");
    const cloudModelsDatalist = presetModels.map(m => `<option value="${m}">${m}</option>`).join("");

    backdrop.innerHTML = `
      <div class="modal-box" role="dialog" aria-modal="true" aria-labelledby="modalTitle">
        <div class="modal-head">
          <h3 id="modalTitle">${isFirstRun ? "Выбор модели и типа ИИ (первый запуск)" : "Настройка модели и API ИИ"}</h3>
          <p>Полный маппинг всех известных локальных LLM и облачных ИИ-провайдеров. После сохранения это окно больше не будет появляться автоматически.</p>
        </div>

        <div class="modal-status ${isConnected ? "" : "warn"}">${statusText}</div>

        <div class="modal-warning">
          <b>⚠ Важное предупреждение о работе ИИ и отсутствии дообучения модели</b>
          <ul>
            <li><b>Слабая ИИ-модель может ошибаться:</b> при малой мощности выбранной LLM возможны неточности или расхождения в оценке условий вакансии.</li>
            <li><b>Два блока сравнения в неоднозначностях:</b> на случай расхождения между <b>Системным AI</b> и <b>выбранной вами ИИ</b> в карточке замечания выводятся два блока — «Ответ Системного AI» и «Ответ выбранной вами ИИ».</li>
            <li><b>Выбор за вами:</b> если ответы расходятся, решение принимаете вы. Нажмите «Принять этот ответ» в нужном блоке — замечание примет его статус и формулировку. Можно и просто поменять статус вручную или снять замечание.</li>
            <li><b>Модель после получения информации не дообучается:</b> веса нейросети не изменяются, а полученные данные и ваши решения не используются для дообучения модели.</li>
          </ul>
        </div>

        <div class="mode-cards" style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;">
          <button type="button" class="mode-card ${modeType === "local" ? "active" : ""}" data-mode="local">
            <span class="card-badge">Гибрид · Локально</span>
            <b>Локальная LLM и системный AI</b>
            <span>Ollama / LM Studio / vLLM / Jan + системный AI и локальная калибровка.</span>
          </button>
          <button type="button" class="mode-card ${modeType === "custom_api" ? "active" : ""}" data-mode="custom_api">
            <span class="card-badge">Гибрид · Облако</span>
            <b>Свой API от любой ИИ и системный AI</b>
            <span>OpenAI, Gemini, Claude, DeepSeek, Grok, OpenRouter, GigaChat, YandexGPT + системный AI.</span>
          </button>
          <button type="button" class="mode-card ${modeType === "local_only" ? "active" : ""}" data-mode="local_only">
            <span class="card-badge">Только локальная ИИ</span>
            <b>Локальная LLM (просто подключение)</b>
            <span>Прямая работа через локальную модель Ollama / LM Studio без системного AI.</span>
          </button>
          <button type="button" class="mode-card ${modeType === "custom_api_only" ? "active" : ""}" data-mode="custom_api_only">
            <span class="card-badge">Только внешний API</span>
            <b>Свой API (просто подключение)</b>
            <span>Прямое подключение своего API от любой ИИ — анализ самой моделью.</span>
          </button>
          <button type="button" class="mode-card full-span ${modeType === "rules_only" ? "active" : ""}" data-mode="rules_only" style="grid-column:1 / -1;display:flex;flex-direction:row;align-items:center;justify-content:space-between;gap:14px;">
            <div>
              <b>Только системный AI</b>
              <span>Мгновенная автономная проверка встроенным системным AI без внешних вызовов.</span>
            </div>
            <span class="card-badge" style="align-self:center;white-space:nowrap;">Быстрый режим</span>
          </button>
        </div>

        <div class="calib-block" id="calibrationBlock">
          <div class="calib-block-head">
            <b>Блок «Калибровка» (индивидуально под выбранную LLM и характеристики ПК/сервера)</b>
            <span class="pill ok">Без шаблонности · Веса не меняются</span>
          </div>
          <div class="form-row" style="grid-template-columns:1fr auto;align-items:center;gap:12px;">
            <label for="mCalibMinutes" style="margin:0;">
              Количество времени, которое вы готовы потратить на калибровку ИИ (в минутах):
              <div class="note" style="font-weight:400;">Число вакансий на калибровку = характеристики ПК/сервера + время. Если введено больше предела разумности — время автоматически урезается под лучшее время обучения.</div>
            </label>
            <input type="number" id="mCalibMinutes" min="0.5" max="120" step="0.5" value="${calibMinutes}" style="width:110px;text-align:right;font-weight:650;">
          </div>
          <div id="calibPlanDetails"></div>
        </div>

        <div id="modeFormContainer">
          ${(modeType === "local" || modeType === "local_only") ? `
            <div class="form-grid">
              <div class="form-row">
                <label>Тип локального сервера (Локальный API)</label>
                <select id="mPreset">${localPresetsHtml}</select>
              </div>
              <div class="form-row">
                <label>Адрес локального API (Base URL)</label>
                <input type="text" id="mBaseUrl" value="${baseUrl}" placeholder="http://127.0.0.1:11434/v1">
              </div>
              <div class="form-row">
                <label>Локальная модель (доступные и известные модели в выпадающем списке)</label>
                <input type="text" id="mModel" list="localModelsList" value="${modelName}" placeholder="qwen2.5:3b">
                <datalist id="localModelsList">${localModelsDatalist}</datalist>
              </div>
              <label style="display:flex;gap:8px;align-items:center;font-size:13px;cursor:pointer;">
                <input type="checkbox" id="mRecalibrate" checked>
                <span>${modeType === "local" ? "Откалибровать профиль локального API и системного AI (модель не дообучается)" : "Проверить подключение локальной LLM и учесть в статистике (модель не дообучается)"}</span>
              </label>
            </div>
          ` : (modeType === "custom_api" || modeType === "custom_api_only") ? `
            <div class="form-grid">
              <div class="form-row">
                <label>Провайдер ИИ API (все известные ИИ)</label>
                <select id="mPreset">${cloudPresetsHtml}</select>
              </div>
              <div class="form-row">
                <label>Адрес API (OpenAI-совместимый Base URL)</label>
                <input type="text" id="mBaseUrl" value="${baseUrl}" placeholder="https://openrouter.ai/api/v1">
              </div>
              <div class="form-row">
                <label>Название модели (выберите из списка известных моделей провайдера или введите своё)</label>
                <input type="text" id="mModel" list="cloudModelsList" value="${modelName}" placeholder="google/gemma-3-27b-it:free или gemini-2.5-flash">
                <datalist id="cloudModelsList">${cloudModelsDatalist}</datalist>
              </div>
              <div class="form-row">
                <label>Ваш API-ключ (хранится только локально вне репозитория)</label>
                <input type="password" id="mApiKey" value="" placeholder="${cfgData && cfgData.has_api_key ? "•••••••• (ключ сохранён, введите новый для замены)" : "sk-..."}">
              </div>
              <label style="display:flex;gap:8px;align-items:center;font-size:13px;cursor:pointer;">
                <input type="checkbox" id="mRecalibrate" checked>
                <span>${modeType === "custom_api" ? "Откалибровать профиль под этот API с системным AI (модель не дообучается)" : "Проверить подключение своего API и учесть в статистике (модель не дообучается)"}</span>
              </label>
            </div>
          ` : `
            <div class="form-grid">
              <div style="font-size:13.5px;color:var(--ink-2);">
                В режиме «Только системный AI» сервис проверяет все вакансии встроенным системным AI за доли секунды. Модель после получения информации не дообучается.
              </div>
            </div>
          `}
        </div>

        <div id="bottomWeakModelWarn" class="modal-warning" style="border-color:var(--con-line);background:var(--con-bg);color:var(--con); ${window.assessModelIntelligence(modelName, modeType).isWeak ? "" : "display:none;"}">
          <b style="color:var(--con);">⚠ Осторожно: выбранная модель обладает базовым уровнем рассуждений</b>
          <div>${window.assessModelIntelligence(modelName, modeType).warning}</div>
        </div>

        <div class="modal-foot">
          <span class="note">После получения информации модель не дообучается · Чанки рассчитываются динамически.</span>
          <div style="display:flex;gap:8px;">
            ${!isFirstRun ? `<button type="button" class="btn" id="mCancelBtn">Закрыть</button>` : ""}
            <button type="button" class="btn primary" id="mSaveBtn"><span class="spin"></span>Сохранить и продолжить</button>
          </div>
        </div>
      </div>
    `;

    renderPlanDetails(calibPlan);
    refreshCalibrationPlan(false);

    const calibMinInput = backdrop.querySelector("#mCalibMinutes");
    if (calibMinInput) {
      calibMinInput.oninput = () => {
        const val = parseFloat(calibMinInput.value);
        if (!Number.isNaN(val) && val > 0) calibMinutes = val;
        if (planTimer) clearTimeout(planTimer);
        planTimer = setTimeout(() => refreshCalibrationPlan(false), 220);
      };
      calibMinInput.onchange = () => {
        const val = parseFloat(calibMinInput.value);
        if (!Number.isNaN(val) && val > 0) calibMinutes = val;
        refreshCalibrationPlan(true);
      };
    }

    const urlInputEl = backdrop.querySelector("#mBaseUrl");
    if (urlInputEl) {
      urlInputEl.oninput = () => {
        baseUrl = urlInputEl.value.trim();
        if (planTimer) clearTimeout(planTimer);
        planTimer = setTimeout(() => refreshCalibrationPlan(false), 300);
      };
    }

    const modInputEl = backdrop.querySelector("#mModel");
    if (modInputEl) {
      modInputEl.oninput = () => {
        modelName = modInputEl.value.trim();
        const bw = backdrop.querySelector("#bottomWeakModelWarn");
        if (bw) {
          const check = window.assessModelIntelligence(modelName, modeType);
          bw.style.display = check.isWeak ? "grid" : "none";
          const div = bw.querySelector("div");
          if (div) div.textContent = check.warning;
        }
        if (planTimer) clearTimeout(planTimer);
        planTimer = setTimeout(() => refreshCalibrationPlan(false), 220);
      };
    }

    backdrop.querySelectorAll(".mode-card").forEach(btn => {
      btn.onclick = () => {
        modeType = btn.dataset.mode;
        if (modeType === "local" || modeType === "local_only") {
          if (!localKeys.includes(presetKey)) presetKey = "ollama";
          const p = presets[presetKey] || presets.ollama || {};
          baseUrl = p.base_url || "http://127.0.0.1:11434/v1";
          modelName = localModels[0] || p.default_model || "qwen2.5:3b";
        } else if (modeType === "custom_api" || modeType === "custom_api_only") {
          if (!cloudKeys.includes(presetKey)) presetKey = "openrouter";
          const p = presets[presetKey] || presets.openrouter || {};
          baseUrl = p.base_url || "https://openrouter.ai/api/v1";
          modelName = p.default_model || "google/gemma-3-27b-it:free";
        }
        renderModalContent();
      };
    });

    const presetSel = backdrop.querySelector("#mPreset");
    if (presetSel) {
      presetSel.onchange = () => {
        presetKey = presetSel.value;
        const p = presets[presetKey];
        if (p) {
          if (p.base_url !== undefined) baseUrl = p.base_url;
          if (p.default_model !== undefined) modelName = p.default_model;
        }
        renderModalContent();
      };
    }

    const cancelBtn = backdrop.querySelector("#mCancelBtn");
    if (cancelBtn) {
      cancelBtn.onclick = () => {
        setModelCookie();
        backdrop.remove();
      };
    }

    const saveBtn = backdrop.querySelector("#mSaveBtn");
    saveBtn.onclick = async () => {
      saveBtn.classList.add("busy");
      saveBtn.disabled = true;
      const urlInp = backdrop.querySelector("#mBaseUrl");
      const modInp = backdrop.querySelector("#mModel");
      const keyInp = backdrop.querySelector("#mApiKey");
      const recalInp = backdrop.querySelector("#mRecalibrate");
      const minInp = backdrop.querySelector("#mCalibMinutes");

      const payload = {
        mode_type: modeType,
        provider_preset: presetKey,
        base_url: urlInp ? urlInp.value.trim() : baseUrl,
        model: modInp ? modInp.value.trim() : modelName,
        api_key: keyInp ? keyInp.value.trim() : "",
        recalibrate: recalInp ? recalInp.checked : false,
        calibration_minutes: minInp ? parseFloat(minInp.value) || calibMinutes : calibMinutes,
      };

      try {
        const r = await fetch("/api/model-config", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        const res = await r.json();
        setModelCookie();
        updateHeaderChip(res.model, res.mode_type === "rules_only", res.calibration ? res.calibration.state : "idle");
        if (res.calibration) showCalibration(res.calibration, onSaved);
        backdrop.remove();
        if (onSaved) onSaved(res);
      } catch {
        saveBtn.classList.remove("busy");
        saveBtn.disabled = false;
      }
    };
  }

  renderModalContent();
  document.body.appendChild(backdrop);
}

function watchCalibration(onDone) {
  let last = null;
  let initialized = false;

  async function checkStateAndMaybeShowFirstRun() {
    try {
      const r = await fetch("/api/model-config");
      if (!r.ok) return;
      const cfgData = await r.json();
      const calState = cfgData.calibration ? cfgData.calibration.state : "idle";
      updateHeaderChip(cfgData.model_info, cfgData.mode_type === "rules_only", calState);

      const chip = document.getElementById("modelChip");
      if (chip) {
        chip.onclick = async () => {
          try {
            const fresh = await (await fetch("/api/model-config")).json();
            openModelSetupModal(fresh, false, onDone);
          } catch {
            openModelSetupModal(cfgData, false, onDone);
          }
        };
      }

      // Если это первый запуск сайта или пользователь очистил куки — показываем окно выбора модели,
      // но при этом сервис в фоне уже проверил и запустил активную модель!
      if (!hasModelCookie()) {
        openModelSetupModal(cfgData, true, onDone);
      }
    } catch { /* сервер ещё загружается */ }
  }

  async function tick() {
    if (!initialized) {
      initialized = true;
      await checkStateAndMaybeShowFirstRun();
    }
    let c = null;
    try { const r = await fetch("/api/calibration"); if (r.ok) c = await r.json(); } catch { /* сервер недоступен — попробуем позже */ }
    if (c) {
      updateHeaderChip(c.model, Boolean(c.rules_only), c.state);
      showCalibration(c, last === "running" ? onDone : null);
      last = c.state;
    }
    setTimeout(tick, c && c.state === "running" ? 3000 : 15000);
  }
  tick();
}
