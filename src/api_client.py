import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import requests

logger = logging.getLogger("transparent_vacancy.api_client")


class APIFetchError(Exception):
    """Исключение при недоступности или ошибке API «Работа в России»."""


def extract_vacancies_from_payload(payload: Any) -> List[Dict[str, Any]]:
    """
    Извлекает список объектов vacancy из структуры ответа API «Работа в России».
    Поддерживает стандартный формат {"results": {"vacancies": [{"vacancy": {...}}]}},
    а также массив страниц или прямой список вакансий.
    """
    extracted: List[Dict[str, Any]] = []

    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, dict):
            raw_list = results.get("vacancies", [])
            if isinstance(raw_list, list):
                for item in raw_list:
                    if isinstance(item, dict) and "vacancy" in item and isinstance(item["vacancy"], dict):
                        extracted.append(item["vacancy"])
                    elif isinstance(item, dict) and "id" in item:
                        extracted.append(item)
        elif "vacancies" in payload and isinstance(payload["vacancies"], list):
            for item in payload["vacancies"]:
                if isinstance(item, dict) and "vacancy" in item:
                    extracted.append(item["vacancy"])
                elif isinstance(item, dict):
                    extracted.append(item)
        elif "vacancy" in payload and isinstance(payload["vacancy"], dict):
            extracted.append(payload["vacancy"])
    elif isinstance(payload, list):
        for elem in payload:
            if isinstance(elem, dict) and "results" in elem:
                extracted.extend(extract_vacancies_from_payload(elem))
            elif isinstance(elem, dict) and "vacancy" in elem:
                extracted.append(elem["vacancy"])
            elif isinstance(elem, dict) and "id" in elem:
                extracted.append(elem)

    return extracted


def load_vacancies_from_file(file_path: str | Path) -> Tuple[List[Dict[str, Any]], int]:
    """
    Проверочный режим: читает локальный JSON в формате ответов API без обращения к порталу.
    Возвращает кортеж (список уникальных вакансий по ID, число обработанных страниц/файлов).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Локальный JSON-файл не найден: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_vacancies = extract_vacancies_from_payload(payload)
    # Сохраняем порядок и устраняем дубликаты по ID внутри входного набора
    unique_by_id: Dict[str, Dict[str, Any]] = {}
    for vac in raw_vacancies:
        vac_id = str(vac.get("id", "")).strip()
        if vac_id:
            unique_by_id[vac_id] = vac

    pages_count = len(payload) if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "results" in payload[0] else 1
    logger.info(
        "Загружено %d уникальных вакансий из локального файла %s",
        len(unique_by_id),
        path,
    )
    return list(unique_by_id.values()), pages_count


def fetch_vacancies_from_api(
    api_url: str,
    max_vacancies: int = 20,
    page_limit: int = 10,
    timeout_seconds: int = 15,
    session: requests.Session | None = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Рабочий режим: загружает вакансии с портала «Работа в России» (регион 39).
    Обеспечивает пагинацию (обрабатывает минимум 2 страницы, если лимит и объём данных позволяют)
    и дедупликацию по ID вакансии.
    """
    http = session or requests.Session()
    headers = {
        "Accept": "application/json",
        "User-Agent": "KantKod-TransparentVacancy/1.0 (Region 39 Analyzer)",
    }

    # Чтобы обработать хотя бы 2 страницы при max_vacancies >= 2,
    # ограничиваем размер страницы не более чем половиной max_vacancies
    if max_vacancies >= 2 and page_limit >= max_vacancies:
        effective_limit = max(1, max_vacancies // 2)
    else:
        effective_limit = max(1, min(page_limit, 100))

    unique_by_id: Dict[str, Dict[str, Any]] = {}
    pages_processed = 0
    page_index = 0
    offset_step = 1  # В API trudvsem.ru offset — индекс страницы (0, 1, 2...). При необходимости адаптируется.

    while len(unique_by_id) < max_vacancies:
        current_offset = page_index * offset_step
        params = {
            "offset": current_offset,
            "limit": effective_limit,
        }

        try:
            logger.info(
                "Запрос к API «Работа в России»: %s (offset=%s, limit=%s)",
                api_url,
                current_offset,
                effective_limit,
            )
            response = http.get(
                api_url,
                params=params,
                headers=headers,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise APIFetchError(
                f"Ошибка подключения к API «Работа в России» ({api_url}): {exc}"
            ) from exc
        except ValueError as exc:
            raise APIFetchError(
                f"API «Работа в России» ({api_url}) вернул некорректный JSON: {exc}"
            ) from exc

        status_code = str(payload.get("status", "200"))
        if status_code != "200":
            raise APIFetchError(
                f"API «Работа в России» вернул статус {status_code} вместо 200"
            )

        page_vacancies = extract_vacancies_from_payload(payload)
        pages_processed += 1

        if not page_vacancies:
            logger.info("Страница %d пуста, загрузка завершена.", pages_processed)
            break

        new_count = 0
        for vac in page_vacancies:
            vac_id = str(vac.get("id", "")).strip()
            if not vac_id:
                continue
            if vac_id not in unique_by_id:
                unique_by_id[vac_id] = vac
                new_count += 1
                if len(unique_by_id) >= max_vacancies:
                    break

        # Если на второй странице (offset=1) сервер вернул те же записи (считает offset в записях, а не в страницах),
        # переключаемся на шаг offset = effective_limit
        if page_index == 1 and new_count == 0 and offset_step == 1 and len(unique_by_id) < max_vacancies:
            offset_step = effective_limit
            params["offset"] = page_index * offset_step
            try:
                response = http.get(
                    api_url,
                    params=params,
                    headers=headers,
                    timeout=timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                page_vacancies = extract_vacancies_from_payload(payload)
                for vac in page_vacancies:
                    vac_id = str(vac.get("id", "")).strip()
                    if vac_id and vac_id not in unique_by_id:
                        unique_by_id[vac_id] = vac
                        new_count += 1
                        if len(unique_by_id) >= max_vacancies:
                            break
            except Exception:
                pass

        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        try:
            total_available = int(meta.get("total", 0))
        except (TypeError, ValueError):
            total_available = 0

        logger.info(
            "Страница %d обработана: получено %d вакансий (новых: %d, всего собрано: %d/%d)",
            pages_processed,
            len(page_vacancies),
            new_count,
            len(unique_by_id),
            max_vacancies,
        )

        if new_count == 0:
            break

        if total_available > 0 and len(unique_by_id) >= total_available:
            break

        if len(page_vacancies) < effective_limit:
            break

        page_index += 1

    return list(unique_by_id.values())[:max_vacancies], pages_processed
