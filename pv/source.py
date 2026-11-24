"""Получение сырых вакансий: из API «Работа в России» или из локального JSON того же формата."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import Config


class SourceError(Exception):
    """Источник недоступен или отдал не то, что ожидалось. Текст — для человека."""


def extract(payload: Any) -> list[dict]:
    """Достаёт объекты вакансий из ответа API. Принимает также список страниц и «голый» список вакансий."""
    if isinstance(payload, dict):
        if "results" in payload:
            results = payload["results"]
            if not isinstance(results, dict):   # пустая выдача может прийти без словаря
                return []
            return extract(results.get("vacancies") or [])
        if "vacancies" in payload:
            return extract(payload["vacancies"])
        if "vacancy" in payload:
            v = payload["vacancy"]
            return [v] if isinstance(v, dict) else []
        if "id" in payload:                    # вакансия без обёртки
            return [payload]
        raise SourceError("В JSON нет ни results.vacancies, ни списка вакансий")
    if isinstance(payload, list):
        out: list[dict] = []
        for item in payload:
            if isinstance(item, (dict, list)):
                out.extend(extract(item))
        return out
    raise SourceError("Ожидался JSON-объект или список, получено другое")


def load_file(path: str | Path) -> list[dict]:
    """Проверочный режим: вакансии из файла, без обращения к сети."""
    p = Path(path)
    if not p.is_file():
        raise SourceError(f"Файл не найден: {p}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeDecodeError) as e:
        raise SourceError(f"Файл {p.name} не является корректным JSON: {e}")
    return extract(payload)


def _get_page(client: httpx.Client, cfg: Config, page: int, limit: int, sleep: Callable[[float], None]) -> dict:
    last = ""
    for attempt in range(1, cfg.retries + 1):
        try:
            r = client.get(cfg.source_url, params={"offset": page, "limit": limit}, timeout=cfg.timeout)
            if r.status_code != 200:
                last = f"HTTP {r.status_code}"
            else:
                try:
                    body = r.json()
                except ValueError:
                    last = "ответ не является JSON"
                else:
                    if not isinstance(body, dict):
                        last = "ответ не является JSON-объектом"
                    elif str(body.get("status", "200")) != "200":
                        err = (body.get("meta") or {}).get("error") if isinstance(body.get("meta"), dict) else None
                        last = f"статус {body.get('status')}" + (f": {err}" if err else "")
                    else:
                        return body
        except httpx.HTTPError as e:
            last = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        if attempt < cfg.retries:
            sleep(2.0 * attempt)
    raise SourceError(f"API не ответил (страница {page}, попыток: {cfg.retries}): {last}")


def fetch_api(cfg: Config, client: httpx.Client | None = None,
              sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = lambda s: None) -> list[dict]:
    """Рабочий режим: страницы API, пока не набран лимит или не кончились данные.

    `offset` в этом API — номер страницы, `limit` — её размер (не больше 100).
    """
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": "prozrachnaya-vakansiya/0.1"}, follow_redirects=True)
    try:
        out: list[dict] = []
        seen: set[str] = set()
        limit = min(cfg.page_size, 100)
        page, total = 0, None
        while len(out) < cfg.max_vacancies:
            if total is not None and page * limit >= total:
                break
            body = _get_page(client, cfg, page, limit, sleep)
            meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
            try:
                total = int(meta.get("total")) if meta.get("total") is not None else total
            except (TypeError, ValueError):
                pass
            items = extract(body)
            log(f"страница {page + 1}: получено {len(items)}" + (f" из {total}" if total is not None else ""))
            if not items:
                break
            for v in items:
                vid = str(v.get("id") or "")
                if vid and vid in seen:
                    continue
                seen.add(vid)
                out.append(v)
                if len(out) >= cfg.max_vacancies:
                    break
            if len(items) < limit:
                break
            page += 1
        return out
    finally:
        if own:
            client.close()
