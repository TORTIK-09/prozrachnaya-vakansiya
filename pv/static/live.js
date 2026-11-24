"use strict";
// Показ проверки в реальном времени. Сервер разбирает вакансии в фоне, страница раз в секунду забирает готовые
// записи и дорисовывает их: числа плавно доходят до новых значений, полосы растут, строки списка появляются.
// Если проверку запустили в другой вкладке или по расписанию — страница подхватит её сама.

const LIVE = {job: 0, following: false, since: 0, seen: new Set()};
const reduceMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// Число плавно меняется от прежнего значения к новому.
function setNum(id, value) {
  const node = document.getElementById(id);
  if (!node) return;
  if (value === null || value === undefined) { node.textContent = "—"; delete node.dataset.v; return; }
  const from = node.dataset.v === undefined ? value : Number(node.dataset.v);
  node.dataset.v = String(value);
  if (from === value || reduceMotion()) { node.textContent = String(value); return; }
  const start = performance.now(), dur = 450;
  node.classList.remove("bump"); void node.offsetWidth; node.classList.add("bump");
  function step(now) {
    if (Number(node.dataset.v) !== value) return;            // пришло более новое значение — эту анимацию бросаем
    const k = Math.min(1, (now - start) / dur), eased = 1 - Math.pow(1 - k, 3);
    node.textContent = String(Math.round(from + (value - from) * eased));
    if (k < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function runLine(p) {
  const box = document.getElementById("runline");
  if (!box) return;
  if (!p || p.state !== "running") { box.classList.remove("show"); return; }
  box.classList.add("show");
  const where = p.kind === "latest" ? "Вакансии с портала" : "Проверочный набор";
  document.getElementById("runText").textContent = where + ": " + (p.total ? "разобрано " + p.done + " из " + p.total : p.phase || "подготовка");
  document.getElementById("runSub").textContent = p.total ? "с замечаниями — " + p.with_findings + ", замечаний — " + p.findings : (p.source || "");
  document.getElementById("runBar").style.width = (p.total ? Math.max(2, Math.round(p.done / p.total * 100)) : 2) + "%";
}

function liveButtons(disabled) {
  for (const id of ["btnSample", "btnUpload", "btnRun"]) {
    const b = document.getElementById(id);
    if (b) { b.disabled = disabled; b.classList.toggle("busy", disabled); }
  }
}

async function followJob(first) {
  if (LIVE.following) return;
  LIVE.following = true; LIVE.since = 0; LIVE.seen = new Set();
  liveButtons(true);
  let p = first, shown = false;
  try {
    while (true) {
      if (!p) { const r = await fetch("/api/progress?since=" + LIVE.since, {cache: "no-store"}); p = await r.json(); }
      LIVE.job = p.job;
      if (!shown && p.state === "running") {
        shown = true;
        if (p.kind && p.kind !== S.kind) await switchKind(p.kind, true);
        S.report = {partial: true, vacancies: []}; S.sel = null;
        banner("");
        render();
      }
      runLine(p);
      const fresh = p.entries || [];
      if (fresh.length && S.report && S.report.partial) {
        LIVE.since += fresh.length;
        S.report.vacancies.push(...fresh);
        S.report.vacancies.sort((a, b) => a._i - b._i);
        const hadSel = !!S.sel;
        if (!S.sel) S.sel = (S.report.vacancies.find(v => (v.findings || []).length) || {}).id || null;
        renderSummary(); renderList();
        if (!hadSel && S.sel) renderDetail();
      }
      if (p.state !== "running") break;
      p = null;
      await new Promise(r => setTimeout(r, 900));
    }
    runLine(null);
    if (p.state === "failed") banner((p.error || "Проверка не удалась") + (p.kind === "latest" ? ". Показан последний успешный отчёт." : ""));
    try { const rep = await api("/api/report?kind=" + S.kind); const keep = S.sel; setReport(rep); if (keep && rep.vacancies.some(v => v.id === keep) && keep !== S.sel) { S.sel = keep; renderList(); renderDetail(); } }
    catch { if (S.report && S.report.partial) { S.report = null; S.sel = null; render(); } }
  } catch (e) {
    runLine(null); banner("Связь с сервисом потеряна: " + e.message);
  } finally {
    LIVE.following = false; liveButtons(false);
  }
}

// Запуск проверки кнопкой: сервер отвечает сразу, дальше страница следит за ходом.
async function startJob(path, opts) {
  if (LIVE.following) return;
  banner("");
  liveButtons(true);
  try {
    const r = await fetch(path + (path.includes("?") ? "&" : "?") + "background=1", opts);
    let data = null;
    try { data = await r.json(); } catch { /* пустой ответ */ }
    if (r.status === 202 && data) { liveButtons(false); return followJob(data); }
    if (!r.ok) throw new Error((data && data.detail) || ("Ошибка " + r.status));
    if (data && data.vacancies) setReport(data);          // сервис без показа хода вернул готовый отчёт
  } catch (e) { banner(e.message); }
  liveButtons(false);
}

// Фоновое наблюдение: проверку могли начать в другой вкладке, по расписанию или из командной строки.
function watchJobs() {
  async function tick() {
    if (!LIVE.following) {
      try {
        const r = await fetch("/api/progress?since=0", {cache: "no-store"});
        if (r.ok) {
          const p = await r.json();
          if (p.state === "running") followJob(p);
          else if (p.job && p.job !== LIVE.job) { const firstLook = LIVE.job === 0; LIVE.job = p.job; if (!firstLook && p.kind === S.kind) loadReport(); }
          else if (!p.job) LIVE.job = 0;
        }
      } catch { /* сервер недоступен — попробуем позже */ }
    }
    setTimeout(tick, 3000);
  }
  tick();
}
