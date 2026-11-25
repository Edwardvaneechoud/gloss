"""Measuring a join against the data: its *to* side must hold one row per key.

A relationship runs from its many side (``from``) to its one side (``to``). A query that LEFT JOINs the *to* table
multiplies every aggregate over the base table when that side repeats its key, so a host measures a join before it
trusts it. :func:`check` runs one aggregate statement per side through the host's ``execute`` (see
:mod:`glosslayer.dialect`): the rows that have a key and the distinct keys among them. Rows whose key holds a null
are ignored; a key of several columns counts distinct combinations.

    found = joins.check("shop.orders", ["customer_id"], "shop.customers", ["customer_id"], execute)
    found.verdict     # "ok", "reversed", "many_to_many" or "unchecked"
    found.message     # one sentence a person can act on

What to do with each verdict is the host's call. Flowfile refuses ``reversed`` and ``many_to_many`` proposals and
stores ``unchecked`` ones with the reason.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .dialect import EngineUnavailable, Execute, QueryFailed, quote
from .references import short_name

log = logging.getLogger("gloss")

VERDICTS = ("ok", "reversed", "many_to_many", "unchecked")


@dataclass(frozen=True)
class JoinCheck:
    """A verdict (one of :data:`VERDICTS`) and the sentence that explains it."""

    verdict: str
    message: str


@dataclass(frozen=True)
class Side:
    """One side as measured: its label, its key columns, the rows that have a key and the distinct keys."""

    table: str
    columns: tuple[str, ...]
    rows: int
    keys: int

    @property
    def unique(self) -> bool:
        return self.keys == self.rows

    @property
    def names(self) -> str:
        return ", ".join(self.columns)

    def phrase(self) -> str:
        if self.unique:
            return f"is unique in {self.table} ({self.keys} of {self.rows} rows)"
        return f"repeats in {self.table} ({self.keys} distinct in {self.rows} rows)"


class _Unreadable(Exception):
    pass


def validate(
    from_table: str,
    from_columns: Sequence[str],
    to_table: str,
    to_columns: Sequence[str],
    columns: Mapping[str, Iterable[str]],
    *,
    labels: Mapping[str, str] | None = None,
) -> None:
    """Refuse a join that cannot be one, without asking the data: a ``ValueError`` its author can act on.

    Both sides need the same number of columns, at least one, that their tables have (``columns`` per dataset key),
    and a table joined to itself needs different columns on each side.
    """
    def label(key: str) -> str:
        return (labels or {}).get(key) or short_name(key)

    if not from_columns or len(from_columns) != len(to_columns):
        raise ValueError("A join needs the same number of columns on both sides, at least one")
    for key, names in ((from_table, from_columns), (to_table, to_columns)):
        missing = [c for c in names if c not in columns.get(key, ())]
        if missing:
            raise ValueError(f"Table {label(key)!r} has no column {missing[0]!r}")
    if from_table == to_table and list(from_columns) == list(to_columns):
        raise ValueError(
            f"A join of {label(from_table)} to itself needs different columns on each side, such as a parent column "
            f"pointing at the table's key; the same column on both sides joins every row to itself"
        )


def unchecked(why: str) -> JoinCheck:
    """The verdict for a join that could not be measured, and why."""
    return JoinCheck("unchecked", f"Direction not checked: {why}.")


def measure_sql(table: str, columns: Sequence[str], *, table_name: str | None = None, dialect: str = "ansi") -> str:
    """The statement measuring one side: ``keys`` (distinct keys) and ``rows`` (rows with a key), in that order.

    The table is written as one quoted identifier holding ``table_name``, else the dataset key.
    """
    def q(name: str) -> str:
        return quote(name, dialect)

    cols = ", ".join(q(c) for c in columns)
    not_null = " AND ".join(f"{q(c)} IS NOT NULL" for c in columns)
    return (
        f"SELECT COUNT(*) AS {q('keys')}, COALESCE(SUM({q('n')}), 0) AS {q('rows')} FROM "
        f"(SELECT {cols}, COUNT(*) AS {q('n')} FROM {q(table_name or table)} WHERE {not_null} GROUP BY {cols}) "
        f"AS {q('g')}"
    )


def _measure(
    table: str, columns: Sequence[str], execute: Execute, label: str, table_name: str | None, dialect: str
) -> Side:
    try:
        rows = execute(measure_sql(table, columns, table_name=table_name, dialect=dialect))
    except QueryFailed as exc:
        raise _Unreadable(f"{label} could not be read ({str(exc) or 'no answer'})") from None
    if not rows:
        raise _Unreadable(f"{label} could not be read (no answer)")
    keys, count = rows[0][0], rows[0][1]
    side = Side(label, tuple(columns), int(count or 0), int(keys or 0))
    if side.rows == 0:
        raise _Unreadable(f"{label} has no rows with a key (an empty table)")
    return side


def check(
    from_table: str,
    from_columns: Sequence[str],
    to_table: str,
    to_columns: Sequence[str],
    execute: Execute,
    *,
    labels: Mapping[str, str] | None = None,
    table_names: Mapping[str, str] | None = None,
    dialect: str = "ansi",
) -> JoinCheck:
    """Measure both sides of a join (dataset keys and their key columns) and say whether it runs many to one.

    ``labels`` names tables in messages (default: the key's last part); ``table_names`` maps a key to the name the
    engine knows it by. An unavailable engine, an empty side or a side the engine cannot read is ``unchecked``.
    """
    labels, table_names = labels or {}, table_names or {}

    def side(key: str, columns: Sequence[str]) -> Side:
        return _measure(key, columns, execute, labels.get(key) or short_name(key), table_names.get(key), dialect)

    try:
        to = side(to_table, to_columns)
        fr = side(from_table, from_columns)
    except EngineUnavailable:
        return unchecked("compute was not available")
    except _Unreadable as exc:
        return unchecked(str(exc))
    from_cols = "" if [c.lower() for c in fr.columns] == [c.lower() for c in to.columns] else f"{fr.names} "
    if to.unique and fr.unique:
        both = f"{to.names} {to.phrase()} and {from_cols}in {fr.table} ({fr.keys} of {fr.rows} rows)"
        return JoinCheck("ok", f"{both}: one to one.")
    if to.unique:
        return JoinCheck("ok", f"{to.names} {to.phrase()} and {from_cols}{fr.phrase()}.")
    if fr.unique:
        return JoinCheck(
            "reversed",
            f"This join is the wrong way round: {to.names} {to.phrase()}, while {fr.names} {fr.phrase()}. The from "
            f"side is the one with many rows: swap the two sides so the join goes from {to.table} to {fr.table}.",
        )
    return JoinCheck(
        "many_to_many",
        f"This join would multiply rows: {fr.names} {fr.phrase()} and {to.names} {to.phrase()}. "
        f"The to side needs columns that identify one row of {to.table}.",
    )


def check_all(
    joins: Iterable[Any],
    columns: Mapping[str, Iterable[str]],
    execute: Execute,
    *,
    labels: Mapping[str, str] | None = None,
    table_names: Mapping[str, str] | None = None,
    dialect: str = "ansi",
) -> list[tuple[Any, JoinCheck]]:
    """Every join (anything with ``from_table``, ``to_table``, ``from_columns``, ``to_columns``) with its check.

    ``columns`` maps each dataset key to its column names; a join whose table or column is not there any more is
    ``unchecked`` without a query, and so is one whose measurement raised.
    """
    out = []
    for j in joins:
        f, t = columns.get(j.from_table), columns.get(j.to_table)
        if f is None or t is None or len(j.from_columns) != len(j.to_columns):
            found = unchecked("a table or a column of this join is missing")
        elif not set(j.from_columns) <= set(f) or not set(j.to_columns) <= set(t):
            found = unchecked("a column of this join is no longer there")
        else:
            try:
                found = check(j.from_table, j.from_columns, j.to_table, j.to_columns, execute,
                              labels=labels, table_names=table_names, dialect=dialect)
            except Exception as exc:  # noqa: BLE001 - one join's failure must not hide the others' verdicts
                log.warning("checking join %s failed", getattr(j, "name", j), exc_info=True)
                found = JoinCheck("unchecked", f"Direction not checked: {exc}")
        out.append((j, found))
    return out
