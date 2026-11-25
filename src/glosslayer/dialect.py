"""The SQL text Gloss writes for a host to run, and how the host hands back the answer.

Gloss never connects to a warehouse. Code that measures data (``glosslayer.joins``, ``glosslayer.rules``) or
dry-runs a metric (``glosslayer.query``) is given ``execute``: a callable that runs one statement where the data
lives and returns its rows in select-list order, as any DB-API ``fetchall()`` does.

    def execute(sql):
        return duckdb_connection.execute(sql).fetchall()

``execute`` raises :class:`QueryFailed` when the engine refused that one statement and :class:`EngineUnavailable`
when there is no engine to ask. Callers treat them differently: a failed statement is about one table, while an
unavailable engine means every further statement would fail too.

Dialects are named as in ``glosslayer.checks.to_sql``: ``ansi`` (the default), ``postgres``, ``duckdb``,
``snowflake`` and ``bigquery``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

Execute = Callable[[str], Sequence[Sequence[Any]]]
DIALECTS = frozenset({"ansi", "postgres", "duckdb", "snowflake", "bigquery"})


class QueryFailed(Exception):
    """The engine refused or failed one statement; the message is its reason."""


class EngineUnavailable(Exception):
    """There is no engine to run statements on; the next one would fail the same way."""


def _known(dialect: str) -> str:
    if dialect not in DIALECTS:
        raise ValueError(f"Unknown SQL dialect {dialect!r} (use one of {sorted(DIALECTS)})")
    return dialect


def quote_char(dialect: str = "ansi") -> str:
    """The character that quotes an identifier: a backtick for BigQuery, a double quote elsewhere."""
    return "`" if _known(dialect) == "bigquery" else '"'


def quote(name: str, dialect: str = "ansi") -> str:
    """``name`` as one quoted identifier, embedded quote characters doubled; always quoted, whatever it holds."""
    q = quote_char(dialect)
    return q + str(name).replace(q, q + q) + q


def literal(value: Any) -> str:
    """A JSON value (or a list of them, as a parenthesised list) as a SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{value!r} is not a number a filter can compare with")
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, list):
        return "(" + ", ".join(literal(v) for v in value) + ")"
    raise ValueError(f"A filter value must be text, a number, a boolean, a date or a list, not {type(value).__name__}")


def date_trunc(grain: str, expression: str, dialect: str = "ansi") -> str:
    """``expression`` truncated to ``grain`` (``day``, ``week``, ``month``, ``quarter`` or ``year``)."""
    if _known(dialect) == "bigquery":
        return f"DATE_TRUNC({expression}, {grain.upper()})"
    return f"DATE_TRUNC({literal(grain)}, {expression})"
