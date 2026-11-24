"""Разбор cron-выражения из пяти полей и поиск следующего времени запуска."""
from __future__ import annotations

from datetime import datetime, timedelta

_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))   # минута, час, день, месяц, день недели (0 и 7 — воскресенье)
_NAMES = ("минута", "час", "день месяца", "месяц", "день недели")


def _field(expr: str, lo: int, hi: int, name: str) -> set[int]:
    out: set[int] = set()
    for part in expr.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/", 1)
            if not s.isdigit() or int(s) < 1:
                raise ValueError(f"cron, {name}: неверный шаг «{s}»")
            step = int(s)
        if part in ("*", ""):
            a, b = lo, hi
        elif "-" in part:
            x, y = part.split("-", 1)
            if not (x.isdigit() and y.isdigit()):
                raise ValueError(f"cron, {name}: неверный диапазон «{part}»")
            a, b = int(x), int(y)
        elif part.isdigit():
            a = b = int(part)
        else:
            raise ValueError(f"cron, {name}: непонятное значение «{part}»")
        if not (lo <= a <= hi and lo <= b <= hi and a <= b):
            raise ValueError(f"cron, {name}: значение вне диапазона {lo}–{hi}: «{part}»")
        out.update(range(a, b + 1, step))
    return out


def parse(expr: str) -> list[set[int]]:
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError("cron: нужно пять полей — минута, час, день месяца, месяц, день недели")
    sets = [_field(p, lo, hi, n) for p, (lo, hi), n in zip(parts, _RANGES, _NAMES)]
    sets[4] = {0 if d == 7 else d for d in sets[4]}
    return sets


def next_run(expr: str, after: datetime) -> datetime:
    """Ближайшее время запуска строго после `after` (с точностью до минуты)."""
    minutes, hours, days, months, weekdays = parse(expr)
    dom_any, dow_any = expr.split()[2] == "*", expr.split()[4] == "*"
    t = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(366 * 24 * 60 * 5):
        dow = (t.weekday() + 1) % 7
        if dom_any or dow_any:
            day_ok = t.day in days and dow in weekdays
        else:                       # как в cron: если заданы оба поля, достаточно совпадения одного
            day_ok = t.day in days or dow in weekdays
        if t.month in months and day_ok and t.hour in hours and t.minute in minutes:
            return t
        t += timedelta(minutes=1)
    raise ValueError("cron: по этому выражению запуск никогда не наступит")
