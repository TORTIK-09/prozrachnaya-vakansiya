"use strict";
// Уведомление о настройке модели: общее для страниц «Отчёт» и «Статистика».
// Пока настройка идёт — полоса с оценкой оставшегося времени; по окончании её сменяет сообщение об итоге,
// которое не исчезает само: его закрывает человек кнопкой.
function calibDuration(sec) {
  if (sec === null || sec === undefined) return "оцениваю время…";
  if (sec < 60) return "меньше минуты";
  const m = Math.round(sec / 60);
  if (m < 60) return "примерно " + m + " мин";
  return "примерно " + Math.floor(m / 60) + " ч " + (m % 60) + " мин";
}
function calibModel(c) { return c.model && c.model.model ? "«" + c.model.model + "»" : ""; }
function calibKey(c) { return "pv-calib-closed:" + (c.model ? c.model.provider + "/" + c.model.model : "") + ":" + (c.finished_at || ""); }

function showCalibration(c, onDone) {
  const box = document.getElementById("calib");
  if (!box) return;
  const title = document.getElementById("calibTitle"), sub = document.getElementById("calibSub");
  const bar = document.getElementById("calibBar"), close = document.getElementById("calibClose"), link = document.getElementById("calibLink");
  let closed = false;
  try { closed = localStorage.getItem(calibKey(c)) === "1"; } catch { /* без хранилища */ }
  box.className = "notice";
  close.hidden = true; if (link) link.hidden = true;
  if (c.state === "running") {
    box.classList.add("show");
    title.textContent = "Модель " + calibModel(c) + " настраивается под задачу — подождите, " + calibDuration(c.eta_seconds);
    sub.textContent = (c.total ? "Проверено вакансий: " + c.done + " из " + c.total + ". " : "Модель загружается. ") + "Сервис встретил эту модель впервые и сверяет её ответы с размеченными вакансиями. Отчёты можно собирать уже сейчас.";
    bar.style.width = (c.total ? Math.round(c.done / c.total * 100) : 3) + "%";
  } else if (c.state === "done" && !closed) {
    box.classList.add("show", "done");
    title.textContent = "Модель " + calibModel(c) + " настроена";
    sub.textContent = "Проверено вакансий: " + c.vacancies + (c.errors ? ", без ответа: " + c.errors : "") + ". Отключено ненадёжных видов замечаний: " + c.blocked + ". В среднем " + c.seconds_per_vacancy + " с на вакансию.";
    close.hidden = false; if (link) link.hidden = false;
  } else if (c.state === "failed" && !closed) {
    box.classList.add("show", "failed");
    title.textContent = "Настроить модель " + calibModel(c) + " не удалось";
    sub.textContent = (c.error || "") + " Сервис продолжает работать: замечания модели используются без настройки.";
    close.hidden = false;
  }
  close.onclick = () => { try { localStorage.setItem(calibKey(c), "1"); } catch { /* без хранилища */ } box.className = "notice"; };
  if (c.state === "done" && onDone) onDone(c);
}

function watchCalibration(onDone) {
  let last = null;
  async function tick() {
    let c = null;
    try { const r = await fetch("/api/calibration"); if (r.ok) c = await r.json(); } catch { /* сервер недоступен — попробуем позже */ }
    if (c) {
      showCalibration(c, last === "running" ? onDone : null);
      last = c.state;
    }
    setTimeout(tick, c && c.state === "running" ? 4000 : 20000);
  }
  tick();
}
