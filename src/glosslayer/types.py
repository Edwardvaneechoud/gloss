"""Map catalog type names (Polars, Arrow, Delta, SQL) to Ossie logical datatypes."""

from __future__ import annotations

import re

_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(u?int\d*|bigint|smallint|tinyint|integer|long|short|byte)$"), "Integer"),
    (re.compile(r"^(float\d*|double|real|float)$"), "Float"),
    (re.compile(r"^(decimal|numeric|number)(\(.*\))?$"), "Decimal"),
    (re.compile(r"^(utf8|string|str|varchar|char|text|large_string|large_utf8|categorical|enum)(\(.*\))?$"), "String"),
    (re.compile(r"^(bool|boolean)$"), "Boolean"),
    (re.compile(r"^date\d*$"), "Date"),
    (re.compile(r"^time\d*(\(.*\))?$"), "Time"),
]

_DATETIME = re.compile(r"^(datetime|timestamp)(\[|\()?(?P<args>.*?)(\]|\))?$")


def to_ossie_datatype(dtype: str | None) -> str | None:
    """Best-effort mapping; returns ``"Opaque"`` for known-but-unportable types and ``None`` if empty."""
    if not dtype:
        return None
    t = dtype.strip().lower().replace(" ", "")
    m = _DATETIME.match(t)
    if m:
        args = m.group("args") or ""
        # Polars: Datetime(time_unit='us', time_zone='UTC'); Arrow: timestamp[us, tz=UTC]
        has_tz = bool(re.search(r"(time_zone|tz)=['\"]?(?!none)[a-z/_+0-9-]+", args)) or "withtimezone" in t
        return "DateTimeTz" if has_tz else "DateTime"
    for pattern, ossie in _RULES:
        if pattern.match(t):
            return ossie
    return "Opaque"
