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
    chip.lastElementChild.textContent = "модель: " + info.model + " (" + modeLabel + calSuffix + ") ⚙";
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
  const trainKind = isLocal ? "Локальное обучение (калибровка) локального API " : "Обучение (калибровка) под внешний API ";
  if (c.state === "running") {
    box.classList.add("show");
    title.textContent = trainKind + calibModel(c) + " — подождите, " + calibDuration(c.eta_seconds);
    sub.textContent = (c.total ? "Проверено вакансий: " + c.done + " из " + c.total + " (параллельно). " : "Модель загружается. ") + "Сервис сверяет ответы модели с эталонной разметкой и обучает профиль фильтрации. Отчёты можно собирать уже сейчас.";
    bar.style.width = (c.total ? Math.round(c.done / c.total * 100) : 3) + "%";
  } else if (c.state === "done" && !closed) {
    box.classList.add("show", "done");
    title.textContent = (isLocal ? "Локальная модель " : "Модель API ") + calibModel(c) + " обучена и настроена";
    sub.textContent = "Проверено вакансий: " + c.vacancies + (c.errors ? ", без ответа: " + c.errors : "") + ". Отключено ненадёжных видов замечаний: " + c.blocked + ". В среднем " + c.seconds_per_vacancy + " с на вакансию.";
    close.hidden = false; if (link) link.hidden = false;
  } else if (c.state === "failed" && !closed) {
    box.classList.add("show", "failed");
    title.textContent = "Обучить модель " + calibModel(c) + " не удалось";
    sub.textContent = (c.error || "") + " Сервис продолжает работать: замечания модели используются без фильтра.";
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

  const backdrop = document.createElement("div");
  backdrop.id = "modelSetupBackdrop";
  backdrop.className = "modal-backdrop";

  function renderModalContent() {
    const isConnected = cfgData && cfgData.connected;
    const calState = cfgData && cfgData.calibration ? cfgData.calibration.state : "idle";
    const statusText = isConnected
      ? "Автопроверка состояния: подключена модель «" + (cfgData.model_info ? cfgData.model_info.model : modelName) + "» (" + (calState === "ready" || calState === "done" ? "профиль обучен" : calState === "running" ? "идёт обучение…" : "готова к обучению") + ")."
      : "Автопроверка состояния: выберите локальную LLM или укажите свой API-ключ для подключения и обучения.";

    const localPresetsHtml = ["ollama", "lmstudio"].map(k => {
      const p = presets[k] || {label: k};
      return `<option value="${k}" ${presetKey === k ? "selected" : ""}>${p.label}</option>`;
    }).join("");

    const cloudPresetsHtml = ["openrouter", "openai", "gemini", "deepseek", "groq", "custom"].map(k => {
      const p = presets[k] || {label: k};
      return `<option value="${k}" ${presetKey === k ? "selected" : ""}>${p.label}</option>`;
    }).join("");

    const localModelsDatalist = localModels.map(m => `<option value="${m}">${m}</option>`).join("");

    backdrop.innerHTML = `
      <div class="modal-box" role="dialog" aria-modal="true" aria-labelledby="modalTitle">
        <div class="modal-head">
          <h3 id="modalTitle">${isFirstRun ? "Выбор модели и типа ИИ (первый запуск)" : "Настройка модели и API ИИ"}</h3>
          <p>Выберите, какую модель использовать для проверки вакансий и автоматического обучения профиля. После сохранения это окно больше не будет появляться автоматически.</p>
        </div>

        <div class="modal-status ${isConnected ? "" : "warn"}">${statusText}</div>

        <div class="mode-cards" style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;">
          <button type="button" class="mode-card ${modeType === "local" ? "active" : ""}" data-mode="local">
            <span class="card-badge">Гибрид · Локально</span>
            <b>Локальная LLM и системный AI</b>
            <span>Ollama / LM Studio на вашем ПК + системный AI и локальное обучение.</span>
          </button>
          <button type="button" class="mode-card ${modeType === "custom_api" ? "active" : ""}" data-mode="custom_api">
            <span class="card-badge">Гибрид · Облако</span>
            <b>Свой API от любой ИИ и системный AI</b>
            <span>OpenRouter, OpenAI, Gemini, DeepSeek, Groq + обучение на системном AI.</span>
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
                <label>Локальная модель (найдено в Ollama: ${localModels.length ? localModels.join(", ") : "укажите имя"})</label>
                <input type="text" id="mModel" list="localModelsList" value="${modelName}" placeholder="qwen2.5:3b">
                <datalist id="localModelsList">${localModelsDatalist}</datalist>
              </div>
              <label style="display:flex;gap:8px;align-items:center;font-size:13px;cursor:pointer;">
                <input type="checkbox" id="mRecalibrate" checked>
                <span>${modeType === "local" ? "Обучить локальный API и системный AI и учесть в статистике" : "Проверить подключение локальной LLM и учесть в статистике"}</span>
              </label>
            </div>
          ` : (modeType === "custom_api" || modeType === "custom_api_only") ? `
            <div class="form-grid">
              <div class="form-row">
                <label>Провайдер ИИ API</label>
                <select id="mPreset">${cloudPresetsHtml}</select>
              </div>
              <div class="form-row">
                <label>Адрес API (OpenAI-совместимый Base URL)</label>
                <input type="text" id="mBaseUrl" value="${baseUrl}" placeholder="https://openrouter.ai/api/v1">
              </div>
              <div class="form-row">
                <label>Название модели (Model ID)</label>
                <input type="text" id="mModel" value="${modelName}" placeholder="google/gemma-3-27b-it:free или gemini-2.5-flash">
              </div>
              <div class="form-row">
                <label>Ваш API-ключ (хранится только локально вне репозитория)</label>
                <input type="password" id="mApiKey" value="" placeholder="${cfgData && cfgData.has_api_key ? "•••••••• (ключ сохранён, введите новый для замены)" : "sk-..."}">
              </div>
              <label style="display:flex;gap:8px;align-items:center;font-size:13px;cursor:pointer;">
                <input type="checkbox" id="mRecalibrate" checked>
                <span>${modeType === "custom_api" ? "Обучить сервис под этот API на базе системного AI и учесть в статистике" : "Проверить подключение своего API и учесть модель в статистике"}</span>
              </label>
            </div>
          ` : `
            <div class="form-grid">
              <div style="font-size:13.5px;color:var(--ink-2);">
                В режиме «Только системный AI» сервис проверяет все вакансии встроенным системным AI за доли секунды. Вы всегда сможете переключиться на локальную LLM или свой API кликом по плашке модели в шапке сайта.
              </div>
            </div>
          `}
        </div>

        <div class="modal-foot">
          <span class="note">Количество чанков рассчитывается динамически под общий объём вакансий.</span>
          <div style="display:flex;gap:8px;">
            ${!isFirstRun ? `<button type="button" class="btn" id="mCancelBtn">Закрыть</button>` : ""}
            <button type="button" class="btn primary" id="mSaveBtn"><span class="spin"></span>Сохранить и продолжить</button>
          </div>
        </div>
      </div>
    `;

    backdrop.querySelectorAll(".mode-card").forEach(btn => {
      btn.onclick = () => {
        modeType = btn.dataset.mode;
        if (modeType === "local" || modeType === "local_only") {
          presetKey = "ollama";
          baseUrl = (presets.ollama && presets.ollama.base_url) || "http://127.0.0.1:11434/v1";
          modelName = localModels[0] || "qwen2.5:3b";
        } else if (modeType === "custom_api" || modeType === "custom_api_only") {
          if (presetKey === "ollama" || presetKey === "lmstudio") {
            presetKey = "openrouter";
            baseUrl = (presets.openrouter && presets.openrouter.base_url) || "https://openrouter.ai/api/v1";
            modelName = (presets.openrouter && presets.openrouter.default_model) || "google/gemma-3-27b-it:free";
          }
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
          if (p.base_url) baseUrl = p.base_url;
          if (p.default_model) modelName = p.default_model;
          const urlInp = backdrop.querySelector("#mBaseUrl");
          const modInp = backdrop.querySelector("#mModel");
          if (urlInp && p.base_url) urlInp.value = p.base_url;
          if (modInp && p.default_model) modInp.value = p.default_model;
        }
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

      const payload = {
        mode_type: modeType,
        provider_preset: presetKey,
        base_url: urlInp ? urlInp.value.trim() : baseUrl,
        model: modInp ? modInp.value.trim() : modelName,
        api_key: keyInp ? keyInp.value.trim() : "",
        recalibrate: recalInp ? recalInp.checked : false,
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
