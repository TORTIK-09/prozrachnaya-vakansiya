"use strict";
// Настройки ИИ: режим работы, калибровка, провайдер и модель. Таблица провайдеров, определение силы модели
// и план калибровки считаются на сервере (pv/llm.py); здесь только форма.

const MODE_NAMES = {
  rules_only: ["Только Системный AI", "Встроенный алгоритм на правилах, без языковой модели. Мгновенно и одинаково на любом компьютере."],
  local: ["Системный AI + локальная ИИ", "Замечания Системного AI плюс мнение модели на этом компьютере (Ollama, LM Studio и другие)."],
  local_only: ["Только локальная ИИ", "Отчёт собирает одна локальная модель, без Системного AI."],
  custom_api: ["Системный AI + ИИ по API", "Замечания Системного AI плюс мнение облачной модели. Тексты вакансий уходят провайдеру."],
  custom_api_only: ["Только ИИ по API", "Отчёт собирает одна облачная модель, без Системного AI."],
};
const STRENGTH = {
  weak: ["небольшая", "Модель небольшая: её ответы могут быть неточными, проверяйте их. ИИ может ошибаться заметно чаще Системного AI."],
  medium: ["средняя", "Точность ИИ и Системного AI примерно одинакова."],
  strong: ["сильная", "Модель сильная: при расхождении предпочтение стоит отдать ИИ."],
};
const SET = {data: null, mode: "rules_only", open: false};

// Предупреждение о силе модели — одно и то же во всех местах, где показан выбор модели или её ответ.
function strengthNote(meta, short) {
  if (!meta || !STRENGTH[meta.strength]) return null;
  const node = document.createElement("div");
  node.className = "strength " + meta.strength;
  node.textContent = short && meta.strength !== "weak" ? STRENGTH[meta.strength][1] : STRENGTH[meta.strength][1];
  return node;
}

function setEl(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) n.append(c);
  return n;
}

const putKids = (node, ...kids) => node.replaceChildren(...kids.flat().filter(k => k !== null && k !== undefined && k !== false));

async function loadSettings() {
  const r = await fetch("/api/settings", {cache: "no-store"});
  SET.data = await r.json();
  SET.mode = SET.data.saved.mode_type || SET.data.mode;
}

function providersFor(mode) {
  const kind = mode.startsWith("local") ? "local" : "custom_api";
  return Object.entries(SET.data.providers).filter(([, p]) => p.mode_type === kind);
}

async function previewPlan() {
  const box = document.getElementById("setPlan");
  if (!box) return;
  if (SET.mode === "rules_only") { box.textContent = "Языковая модель не используется — калибровать нечего. Системный AI учится на вакансиях портала отдельно."; return; }
  const model = document.getElementById("setModel").value.trim(), url = document.getElementById("setUrl").value.trim();
  const min = parseFloat(document.getElementById("setMinutes").value);
  if (!model) { box.textContent = "Выберите модель — покажу, сколько вакансий она успеет разобрать."; return; }
  try {
    const q = "/api/plan?model=" + encodeURIComponent(model) + "&mode_type=" + SET.mode + "&base_url=" + encodeURIComponent(url) + (min > 0 ? "&minutes=" + min : "");
    const r = await fetch(q); const p = await r.json();
    if (!r.ok) { box.textContent = p.detail || "Не удалось посчитать план"; return; }
    const hw = p.hardware, m = p.model_meta;
    putKids(box, 
      setEl("div", {}, setEl("b", {text: "Модель: " + (STRENGTH[m.strength] || ["?"])[0] + " · " + m.params_label}), "  ·  Компьютер: " + hw.cpu_cores + " ядер, " + hw.ram_total_gb + " ГБ памяти" + (hw.gpu_name ? ", " + hw.gpu_name + " " + hw.vram_gb + " ГБ" : ", без видеокарты") + (hw.is_overloaded ? " — сейчас загружен" : "")),
      setEl("div", {text: "Калибровка на " + p.llm_target_tests + " из " + p.available_tests + " размеченных вакансий, примерно " + p.effective_minutes + " мин (оценка: " + p.sec_per_vacancy_est + " с на вакансию; минимум для этой модели — " + Math.min(p.min_tests, p.available_tests) + ")."}),
      p.time_trimmed ? setEl("div", {class: "trim", text: "Вы указали " + p.requested_minutes + " мин — это больше, чем нужно: калибровка займёт " + p.effective_minutes + " мин."}) : null,
      setEl("div", {class: "note", text: "Пока ИИ калибруется, Системный AI за то же время учится на вакансиях портала — на стольких, сколько успеет."}),
      strengthNote(m));
  } catch { box.textContent = "Не удалось посчитать план"; }
}

function renderSettings() {
  const box = document.getElementById("settings");
  if (!box || !SET.data) return;
  const d = SET.data, saved = d.saved || {};
  const needModel = SET.mode !== "rules_only", needKey = SET.mode.startsWith("custom_api");
  const provs = needModel ? providersFor(SET.mode) : [];
  const provKey = provs.some(([k]) => k === saved.provider) ? saved.provider : (provs[0] || [""])[0];
  const prov = d.providers[provKey] || {models: [], base_url: "", default_model: ""};
  const sameKind = saved.mode_type && saved.mode_type.startsWith(SET.mode.split("_only")[0].replace("_api", "_api")) && saved.provider === provKey;
  const models = provKey === "ollama" && d.local_models.length ? d.local_models : prov.models;

  const modeList = setEl("div", {class: "modes", role: "radiogroup", "aria-label": "Режим работы"}, Object.keys(MODE_NAMES).map(m =>
    setEl("label", {class: "mode" + (m === SET.mode ? " on" : "")},
      setEl("input", {type: "radio", name: "mode", value: m, checked: m === SET.mode, onchange: () => { SET.mode = m; renderSettings(); }}),
      setEl("span", {}, setEl("b", {text: MODE_NAMES[m][0]}), setEl("small", {text: MODE_NAMES[m][1]})))));

  const calib = setEl("fieldset", {class: "set-block", disabled: !needModel},
    setEl("legend", {text: "Калибровка ИИ"}),
    setEl("label", {class: "field"}, "Готов потратить на калибровку ", setEl("input", {type: "number", id: "setMinutes", min: "1", max: "600", step: "1", value: saved.calib_minutes > 0 ? saved.calib_minutes : "", placeholder: "авто", oninput: previewPlan}), " мин"),
    setEl("div", {class: "plan", id: "setPlan"}));

  const provider = setEl("fieldset", {class: "set-block", disabled: !needModel},
    setEl("legend", {text: "Провайдер ИИ"}),
    setEl("label", {class: "field wide"}, "Провайдер", setEl("select", {id: "setProv", onchange: e => { saved.provider = e.target.value; saved.base_url = ""; saved.model = ""; renderSettings(); }},
      provs.map(([k, p]) => setEl("option", {value: k, selected: k === provKey, text: p.label})))),
    setEl("label", {class: "field wide"}, "Адрес сервера", setEl("input", {type: "url", id: "setUrl", value: (sameKind && saved.base_url) || prov.base_url, placeholder: "https://…/v1", onchange: previewPlan})),
    setEl("label", {class: "field wide"}, "Модель", setEl("input", {type: "text", id: "setModel", list: "setModels", value: (sameKind && saved.model) || (models.includes(prov.default_model) ? prov.default_model : models[0]) || prov.default_model, onchange: previewPlan}),
      setEl("datalist", {id: "setModels"}, models.map(m => setEl("option", {value: m})))),
    needKey ? setEl("label", {class: "field wide"}, "Ключ API", setEl("input", {type: "password", id: "setKey", autocomplete: "off", placeholder: saved.has_api_key && sameKind ? "сохранён — оставьте пустым, чтобы не менять" : "ключ провайдера"})) : null,
    needKey ? setEl("div", {class: "note", text: "Ключ хранится только на этом компьютере, в папке reports. При работе через API тексты вакансий и методика проверки передаются провайдеру."}) : null);

  putKids(box, 
    setEl("h3", {text: "Настройки ИИ"}),
    d.env_locked ? setEl("div", {class: "strength weak", text: "Модель задана переменными окружения PV_LLM_… — они важнее выбора в этой форме."}) : null,
    modeList, calib, provider,
    setEl("div", {class: "set-actions"}, setEl("button", {class: "btn primary", id: "setSave", onclick: saveSettings, text: "Сохранить"}), setEl("button", {class: "btn", onclick: recalibrate, disabled: SET.data.mode === "rules_only", title: "Заново проверить подключённую модель на размеченных вакансиях", text: "Перекалибровать ИИ"}), setEl("button", {class: "btn", onclick: () => toggleSettings(false), text: "Закрыть"}), setEl("span", {class: "note", id: "setMsg"})));
  previewPlan();
}

async function saveSettings() {
  const msg = document.getElementById("setMsg"), val = id => { const n = document.getElementById(id); return n ? n.value.trim() : ""; };
  const body = {mode_type: SET.mode, provider: val("setProv"), base_url: val("setUrl"), model: val("setModel"), api_key: val("setKey"), calib_minutes: parseFloat(val("setMinutes")) || 0};
  msg.textContent = "Сохраняю…";
  try {
    const r = await fetch("/api/settings", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    const data = await r.json();
    if (!r.ok) { msg.textContent = data.detail || "Не удалось сохранить"; return; }
    msg.textContent = "Сохранено. " + (data.calibration.state === "running" ? "Началась калибровка ИИ." : data.calibration.state === "no_model" && body.mode_type !== "rules_only" ? "Модель по этому адресу пока не отвечает — сервис будет пробовать снова." : "");
    seenSettings();
    await loadSettings();
    if (typeof showCalibration === "function") showCalibration(data.calibration);
  } catch (e) { msg.textContent = "Ошибка: " + e.message; }
}

function seenSettings() { try { localStorage.setItem("pv-settings-seen", "1"); } catch { /* без хранилища */ } }

// Настройки — модальное окно. Закрывается кнопкой, клавишей Esc и щелчком по затемнению.
async function toggleSettings(force) {
  const box = document.getElementById("settings");
  if (!box) return;
  const open = force === undefined ? !box.open : force;
  if (!open) { if (box.open) box.close(); return; }
  if (!box.open) box.showModal();
  box.textContent = "Загружаю…";
  try { await loadSettings(); renderSettings(); } catch (e) { box.textContent = "Не удалось загрузить настройки: " + e.message; }
}

// Первый запуск сайта: выбора ещё нет — окно открывается само. Дальше — только по кнопке.
async function autoOpenSettings() {
  const box = document.getElementById("settings");
  if (!box) return;
  box.addEventListener("close", seenSettings);
  box.addEventListener("click", e => { if (e.target === box) box.close(); });
  let seen = false;
  try { seen = localStorage.getItem("pv-settings-seen") === "1"; } catch { /* без хранилища */ }
  if (seen) return;
  try {
    await loadSettings();
    if (SET.data.saved && SET.data.saved.mode_type) { seenSettings(); return; }     // выбор уже сделан раньше
    await toggleSettings(true);
  } catch { /* настройки необязательны для работы сайта */ }
}

// Калибровка заново — даже если модель уже знакома сервису.
async function recalibrate() {
  const msg = document.getElementById("setMsg");
  try {
    const r = await fetch("/api/calibration", {method: "POST"}); const c = await r.json();
    if (!r.ok) { msg.textContent = c.detail || "Не удалось запустить калибровку"; return; }
    toggleSettings(false);
    if (typeof showCalibration === "function") showCalibration(c);
  } catch (e) { msg.textContent = "Ошибка: " + e.message; }
}
