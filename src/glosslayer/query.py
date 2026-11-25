"""Compiling "metric X by these dimensions with these filters" into one SQL statement.

Pure: a :class:`QueryModel` (the metrics, the joins and each dataset's columns) and a :class:`MetricQuery` become SQL
plus what it used. Nothing is run here; a host runs the statement where its data lives.

    model = QueryModel.read(g)
    q = compile_query(model, MetricQuery("revenue", ["customers.country"], grain="month"))
    q.sql, q.tables, q.metric

The emitted shape is fixed::

    SELECT <dimensions>, <metric expression> AS "<metric>"
    FROM (SELECT "<base>".*, <joined columns AS safe aliases>
          FROM "<base table>" AS "<base>" LEFT JOIN ... WHERE <query filters>) AS "<base>"
    WHERE (<metric filter>) GROUP BY <dimensions> ORDER BY ... LIMIT ...

The inner query keeps every base column under its own name and the base table's short name, so a stored expression
that says ``amount`` or ``orders.amount`` resolves as written. Joins are followed only from their many side to their
one side, so each base row meets at most one row of a joined table and the aggregate never fans out (see
``glosslayer.joins`` for measuring that a join really is many to one).

The expression is not parsed as SQL: :func:`column_references` tokenizes out its ``[schema.]table.column`` chains,
and a chain naming a joined table is projected by the inner query under a safe alias and rewritten to read it
(``products.price`` becomes ``"orders"."__gloss_d1"``). Everything else in it is left as written.

Free SQL comes only from the stored metric (its ``expression`` and ``filter``); everything taken from a query is an
identifier checked against the model and quoted, or a value rendered by ``dialect.literal``.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from .dialect import EngineUnavailable, Execute, QueryFailed, date_trunc, literal, quote, quote_char
from .references import short_name

if TYPE_CHECKING:
    from .ledger import Gloss

log = logging.getLogger("gloss")

OPS = ("=", "!=", "<", "<=", ">", ">=", "in", "not in", "between", "is null", "is not null")
GRAINS = ("day", "week", "month", "quarter", "year")
MAX_LIMIT = 10_000

# Words of SQL itself: a chain starting with one (unquoted) is syntax, not a table.
_KEYWORDS = frozenset(
    "SELECT FROM WHERE AND OR NOT CASE WHEN THEN ELSE END AS IN IS NULL LIKE ILIKE BETWEEN DISTINCT FILTER OVER "
    "PARTITION BY ORDER INTERVAL DATE TIMESTAMP TIME CAST TRUE FALSE".split()
)


def _one_of(what: str, value: Any, allowed: Collection[str]) -> None:
    if value not in allowed:
        raise ValueError(f"{what} must be one of {', '.join(allowed)}, got {value!r}")


# ------------------------------------------------------------------ what is asked
@dataclass(frozen=True)
class Filter:
    """``field`` is ``"table.column"`` or a bare column of the metric's table; ``value`` is a list for ``in`` /
    ``not in`` (any length) and ``between`` (two values), absent for the null checks, one value otherwise."""

    field: str
    op: str
    value: Any = None

    def __post_init__(self) -> None:
        _one_of("A filter's op", self.op, OPS)


@dataclass(frozen=True)
class Order:
    """``field``: one of the query's dimensions as written, ``<time_column>_<grain>``, or the metric's name."""

    field: str
    direction: str = "asc"

    def __post_init__(self) -> None:
        _one_of("An order's direction", self.direction, ("asc", "desc"))


@dataclass(frozen=True)
class MetricQuery:
    """One metric by ``dimensions`` (``"table.column"`` or a bare base column), filtered, optionally by time grain."""

    metric: str
    dimensions: Sequence[str] = ()
    filters: Sequence[Filter] = ()
    grain: str | None = None
    order: Order | None = None
    limit: int = 1000

    def __post_init__(self) -> None:
        if self.grain is not None:
            _one_of("A grain", self.grain, GRAINS)
        if not 1 <= self.limit <= MAX_LIMIT:
            raise ValueError(f"A limit is a number from 1 to {MAX_LIMIT}, got {self.limit}")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> MetricQuery:
        """Build one from plain JSON (filters and order as objects)."""
        order = d.get("order")
        try:
            return cls(**{
                **d,
                "dimensions": list(d.get("dimensions") or ()),
                "filters": [Filter(**f) for f in d.get("filters") or ()],
                "order": Order(**order) if order else None,
            })
        except TypeError as exc:
            raise ValueError(f"Not a metric query: {exc}") from None


# ------------------------------------------------------------------ what is known
def expression_sql(expression: Any) -> str:
    """The portable SQL of a metric's expression: the text itself, or an Ossie expression's ``sql``."""
    if isinstance(expression, Mapping):
        expression = expression.get("sql")
    return (expression or "").strip()


def _key(obj: Any) -> str:
    """A metric's or join's subject, whether it carries it as ``key`` (a view) or ``subject`` (a modelling object)."""
    return str(getattr(obj, "key", None) or getattr(obj, "subject", ""))


@dataclass(frozen=True)
class QueryModel:
    """What a query is compiled from. Anything with these three attributes will do; a ``ModelView`` is one.

    ``metrics`` carry ``name``, ``table``, ``expression``, ``filter``, ``time_column`` and ``dimensions``; ``joins``
    carry ``name``, ``from_table``, ``to_table``, ``from_columns`` and ``to_columns`` (``glosslayer.Metric`` and
    ``glosslayer.Relationship`` do). ``columns`` maps a dataset key to its column names; a table that is only joined
    to needs to be there too.
    """

    metrics: Sequence[Any]
    joins: Sequence[Any]
    columns: Mapping[str, Collection[str]]

    @classmethod
    def read(cls, gloss: Gloss, *, include_pending: bool = False) -> QueryModel:
        """The accepted model of a ledger (optionally with what waits for review), as this handle may read it."""
        return cls(
            gloss.metrics(include_pending=include_pending),
            gloss.relationships(include_pending=include_pending),
            {t.key: list(t.columns) for t in gloss.tables(include_pending=include_pending)},
        )

    @classmethod
    def of(cls, model: Any) -> QueryModel:
        """``model`` itself, or a copy of the three attributes of something that carries them."""
        return model if isinstance(model, cls) else cls(model.metrics, model.joins, model.columns)

    def with_metric(self, metric: Any) -> QueryModel:
        """This model with ``metric`` in place of any of its name on its table, for trying a proposal."""
        kept = [m for m in self.metrics if (m.table, m.name) != (metric.table, metric.name)]
        return replace(self, metrics=[*kept, metric])

    def without_join(self, key: str) -> QueryModel:
        """This model without the join whose subject is ``key``."""
        return replace(self, joins=[j for j in self.joins if _key(j) != key])


@dataclass(frozen=True)
class CompiledQuery:
    """The SQL, the dataset keys it reads (base first) and the metric it computes."""

    sql: str
    tables: list[str]
    metric: Any


@dataclass(frozen=True)
class ColumnReference:
    """A ``table.column`` or ``schema.table.column`` chain in free SQL: its span in the text and its parts."""

    start: int
    end: int
    parts: tuple[str, ...]
    text: str


@lru_cache
def _token(q: str) -> re.Pattern[str]:
    return re.compile(
        rf"""(?P<skip>\s+|--[^\n]*\n?|/\*.*?(?:\*/|\Z))
        |(?P<string>'(?:[^']|'')*'?)
        |{q}(?P<qident>(?:[^{q}]|{q}{q})*){q}?
        |(?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
        |(?P<ident>[^\W\d]\w*)
        |(?P<punct>.)""",
        re.VERBOSE | re.DOTALL,
    )


def column_references(sql: str, dialect: str = "ansi") -> list[ColumnReference]:
    """The two- and three-part identifier chains of ``sql``, in order; string literals and comments are never read.

    One followed by ``(`` is a function call and one starting with an unquoted keyword is SQL, so neither counts; a
    bare identifier is never a reference, and a chain of four or more parts is left alone.
    """
    q = quote_char(dialect)
    tokens = [m for m in _token(q).finditer(sql) if m.lastgroup != "skip"]

    def is_name(at: int) -> bool:
        return at < len(tokens) and tokens[at].lastgroup in ("ident", "qident")

    def is_punct(at: int, char: str) -> bool:
        return at < len(tokens) and tokens[at]["punct"] == char

    refs: list[ColumnReference] = []
    i = 0
    while i < len(tokens):
        first = i
        i += 1
        if not is_name(first):
            continue
        while is_punct(i, ".") and is_name(i + 1):
            i += 2
        names = tokens[first:i:2]
        is_keyword = names[0].lastgroup == "ident" and names[0].group().upper() in _KEYWORDS
        if 2 <= len(names) <= 3 and not is_keyword and not is_punct(i, "("):
            parts = tuple(t["ident"] or t["qident"].replace(q + q, q) for t in names)
            start, end = names[0].start(), names[-1].end()
            refs.append(ColumnReference(start, end, parts, sql[start:end]))
    return refs


# ------------------------------------------------------------------ the join graph
def _paths_from(model: QueryModel, start: str) -> dict[str, list[Any]]:
    """Shortest join path from ``start`` to each table it reaches, joins followed from ``from`` to ``to`` only."""
    paths: dict[str, list[Any]] = {start: []}
    queue = deque([start])
    joins = sorted(model.joins, key=lambda j: (j.name, _key(j)))
    while queue:
        cur = queue.popleft()
        for j in joins:
            if j.from_table == cur and j.to_table not in paths:
                paths[j.to_table] = [*paths[cur], j]
                queue.append(j.to_table)
    return paths


def reachable(model: Any, table: str) -> list[str]:
    """The dataset keys a metric on ``table`` can read, ``table`` first, then each table reached many → one."""
    return list(_paths_from(QueryModel.of(model), table))


def _one_row_of(name: str) -> str:
    """A naive singular for a message (``customers`` → ``customer``); a name that is not plain English stays."""
    if name.endswith("ies") and len(name) > 4:
        return name[:-3] + "y"
    if name.endswith("s") and not name.endswith("ss") and len(name) > 3:
        return name[:-1]
    return f"row of {name}"


class _Graph:
    """The tables a base table reaches by following joins from ``from`` to ``to``, nearest first."""

    def __init__(self, model: QueryModel, base: str):
        self.model = model
        self.base = base
        self.paths = _paths_from(model, base)
        self.columns = {key: set(names) for key, names in model.columns.items()}
        self.reach = ", ".join(short_name(k) for k in self.paths)

    def resolve(self, ref: str, what: str) -> tuple[str, str]:
        """``"table.column"`` or a bare base column as ``(dataset key, column)``; a ``ValueError`` otherwise."""
        if not ref.strip():
            raise ValueError(f"A {what} must name a column")
        if ref in self.columns.get(self.base, ()):
            return self.base, ref
        table_name, _, column = ref.rpartition(".")
        if not table_name:
            raise ValueError(
                f"Table {short_name(self.base)!r} has no column {ref!r}; write a column of a joined table as "
                f"'table.column' (reachable tables: {self.reach})"
            )
        key = next((k for k in self.paths if table_name in (k, short_name(k))), None)
        if key is None:
            if any(table_name in (k, short_name(k)) for k in self.columns):
                raise ValueError(
                    f"The {what} {ref!r} is on table {table_name!r}, which {short_name(self.base)!r} does not reach "
                    f"by following joins from their many side to their one side. Reachable tables: {self.reach}"
                )
            raise ValueError(f"Unknown table {table_name!r} in {what} {ref!r}. Reachable tables: {self.reach}")
        return self._checked(key, column, f"in {what} {ref!r}")

    def resolve_reference(self, ref: ColumnReference, where: str) -> tuple[str, str]:
        """A chain of a metric's free SQL as ``(dataset key, column)``; a ``ValueError`` naming ``where`` otherwise.

        The base table wins its own short name; any other table must be reachable many → one, and two reachable
        tables that answer the same name are an error that names both.
        """
        *table_parts, column = ref.parts
        name = ".".join(table_parts)
        at = f"{ref.text} in {where}"
        base = short_name(self.base)
        if name == base:
            return self._checked(self.base, column, at)

        def named(keys: Iterable[str]) -> list[str]:
            return [k for k in keys if k.split(".")[-len(table_parts):] == table_parts]

        matches = named(self.paths)
        if len(matches) > 1:
            both = " and ".join(".".join(k.split(".")[-2:]) for k in matches)
            raise ValueError(f"{name!r} could mean {both} ({at}); write schema.table.column to choose one")
        if matches:
            return self._checked(matches[0], column, at)
        known = named(self.columns)
        if not known:
            raise ValueError(f"Unknown table {name!r} ({at}). Tables this metric can use: {self.reach}")
        other = short_name(known[0])
        if self.base in _paths_from(self.model, known[0]):
            raise ValueError(
                f"{other} has many rows per {_one_row_of(base)}; a metric on {base} cannot use a column of "
                f"{other} ({at})"
            )
        raise ValueError(
            f"Table {other!r} is not reached from {base} by following joins from their many side to their one side "
            f"({at}). Tables this metric can use: {self.reach}"
        )

    def _checked(self, key: str, column: str, at: str) -> tuple[str, str]:
        if column not in self.columns.get(key, ()):
            raise ValueError(f"Table {short_name(key)!r} has no column {column!r} ({at})")
        return key, column


# ------------------------------------------------------------------ compiling
def _find_metric(model: QueryModel, name: str, tables: Collection[str] | None) -> Any:
    """The metric called ``name`` on one of ``tables`` (dataset keys; every table when None)."""
    metrics = [m for m in model.metrics if tables is None or m.table in tables]
    found = [m for m in metrics if m.name == name]
    if not found:
        known = sorted({m.name for m in metrics})
        hint = f" Known metrics: {', '.join(known)}" if known else " This model has no metrics yet."
        raise ValueError(f"No metric called {name!r} in this model.{hint}")
    if len(found) > 1:
        where = ", ".join(m.table or "no table" for m in found)
        raise ValueError(f"Metric {name!r} exists on several tables: {where}")
    if not found[0].table:
        raise ValueError(f"Metric {name!r} belongs to no table, so there are no rows to compute it over")
    return found[0]


def _norm(sql: str) -> str:
    return re.sub(r"\s+", "", sql).lower()


def compile_query(
    model: Any,
    query: MetricQuery,
    *,
    tables: Collection[str] | None = None,
    table_names: Mapping[str, str] | None = None,
    dialect: str = "ansi",
) -> CompiledQuery:
    """Compile ``query`` against ``model``; a ``ValueError`` with an actionable message when it cannot compile.

    ``tables`` limits which tables' metrics may be asked for (a joined table need not be among them). A table is
    written by its dataset key, as one quoted identifier, unless ``table_names`` maps its key to another name.
    """
    model = QueryModel.of(model)
    names_for = table_names or {}
    metric = _find_metric(model, query.metric, tables)
    expression = expression_sql(metric.expression)
    if not expression:
        raise ValueError(f"Metric {metric.name!r} has no expression")
    base = metric.table
    graph = _Graph(model, base)

    def q(name: str) -> str:
        return quote(name, dialect)

    qb = q(short_name(base))
    joined: dict[str, tuple[str, Any]] = {}
    projected: dict[tuple[str, str], str] = {}

    def table_alias(key: str) -> str:
        for j in graph.paths[key]:
            joined.setdefault(j.to_table, (q(f"j{len(joined) + 1}"), j))
        return joined[key][0] if key != base else qb

    def inner_ref(key: str, column: str) -> str:
        """The column as the inner query reads it."""
        return f"{table_alias(key)}.{q(column)}"

    def outer_ref(key: str, column: str) -> str:
        """The column as the outer query reads it: a base column by name, a joined one by its safe alias."""
        if key == base:
            return inner_ref(key, column)
        table_alias(key)
        return f"{qb}.{q(projected.setdefault((key, column), f'__gloss_d{len(projected) + 1}'))}"

    select: list[tuple[str, str]] = []
    if query.grain:
        if not metric.time_column:
            raise ValueError(f"Metric {metric.name!r} has no time column, so it cannot be broken down by {query.grain}")
        key, column = graph.resolve(metric.time_column, "time column")
        select.append((date_trunc(query.grain, outer_ref(key, column), dialect), f"{column}_{query.grain}"))
    for ref in dict.fromkeys(query.dimensions):
        key, column = graph.resolve(ref, "dimension")
        written = {column if key == base else None, f"{short_name(key)}.{column}", f"{key}.{column}"}
        if metric.dimensions and not written & set(metric.dimensions):
            raise ValueError(
                f"Metric {metric.name!r} may only be broken down by {', '.join(metric.dimensions)}; "
                f"{ref!r} is not one of them"
            )
        select.append((outer_ref(key, column), ref))

    where = [_render_filter(f, inner_ref(*graph.resolve(f.field, "filter field"))) for f in query.filters]

    def rewrite(text: str, where_name: str) -> str:
        """``text`` with each chain naming a joined table read from the inner query's alias for it."""
        for ref in reversed(column_references(text, dialect)):
            key, column = graph.resolve_reference(ref, where_name)
            if key != base or len(ref.parts) == 3:
                text = text[:ref.start] + outer_ref(key, column) + text[ref.end:]
        return text

    stored_filter = (metric.filter or "").strip()
    expression_text = rewrite(expression, "the expression")
    filter_text = ""
    if stored_filter and _norm(stored_filter) not in _norm(expression):
        filter_text = rewrite(stored_filter, "the filter")

    names = [name for _expr, name in select]
    if metric.name in names:
        raise ValueError(f"A dimension may not share the metric's name {metric.name!r}")
    order_sql = f"{q(names[0])} ASC" if names else None
    if query.order is not None:
        if query.order.field not in (*names, metric.name):
            raise ValueError(f"Order by a dimension or the metric; {query.order.field!r} is neither")
        order_sql = f"{q(query.order.field)} {query.order.direction.upper()}"

    inner_select = [f"{qb}.*"] + [f"{inner_ref(k, c)} AS {q(a)}" for (k, c), a in projected.items()]
    inner = [f"SELECT {', '.join(inner_select)}", f"FROM {q(names_for.get(base, base))} AS {qb}"]
    for table, (alias, j) in joined.items():
        on = " AND ".join(
            f"{table_alias(j.from_table)}.{q(fc)} = {alias}.{q(tc)}"
            for fc, tc in zip(j.from_columns, j.to_columns, strict=True)
        )
        inner.append(f"LEFT JOIN {q(names_for.get(table, table))} AS {alias} ON {on}")
    if where:
        inner.append("WHERE " + " AND ".join(where))

    outer_select = [f"{expr} AS {q(name)}" for expr, name in select]
    outer_select.append(f"{expression_text} AS {q(metric.name)}")
    sql = [f"SELECT {', '.join(outer_select)}", "FROM (" + " ".join(inner) + f") AS {qb}"]
    if filter_text:
        sql.append(f"WHERE ({filter_text})")
    if select:
        sql.append("GROUP BY " + ", ".join(expr for expr, _ in select))
    if order_sql:
        sql.append(f"ORDER BY {order_sql}")
    sql.append(f"LIMIT {query.limit}")
    return CompiledQuery(sql="\n".join(sql), tables=[base, *joined], metric=metric)


def _render_filter(f: Filter, ref: str) -> str:
    value, many = f.value, isinstance(f.value, list)
    if f.op in ("is null", "is not null"):
        return f"{ref} {f.op.upper()}"
    if f.op in ("in", "not in"):
        if not many or not value:
            raise ValueError(f"The {f.op!r} filter on {f.field!r} needs a non-empty list of values")
        return f"{ref} {f.op.upper()} {literal(value)}"
    if f.op == "between":
        if not many or len(value) != 2:
            raise ValueError(f"The 'between' filter on {f.field!r} needs a list of two values")
        return f"{ref} BETWEEN {literal(value[0])} AND {literal(value[1])}"
    if many or value is None:
        raise ValueError(f"The {f.op!r} filter on {f.field!r} needs one value; use 'is null' to match missing values")
    return f"{ref} {'<>' if f.op == '!=' else f.op} {literal(value)}"


# ------------------------------------------------------------------ trying definitions
def check_metric(
    model: Any,
    metric: Any,
    execute: Execute | None = None,
    *,
    table_names: Mapping[str, str] | None = None,
    dialect: str = "ansi",
) -> bool:
    """Refuse a proposed metric nobody could ask for: it must compile and, given ``execute``, run.

    Returns whether it was run. A compile failure or the engine's refusal is a ``ValueError`` for its author; an
    unavailable engine does not block authoring, so the metric then only has to compile.
    """
    try:
        trial = QueryModel.of(model).with_metric(metric)
        compiled = compile_query(trial, MetricQuery(metric.name), tables={metric.table}, table_names=table_names,
                                 dialect=dialect)
    except ValueError as exc:
        raise ValueError(f"Metric {metric.name!r} does not compile: {exc}") from None
    if execute is None:
        return False
    try:
        execute(compiled.sql)
    except EngineUnavailable as exc:
        log.warning("metric %r was not dry-run, compute unavailable: %s", metric.name, exc)
        return False
    except QueryFailed as exc:
        raise ValueError(f"Metric {metric.name!r} does not run: {exc}") from None
    return True


@dataclass(frozen=True)
class Dependent:
    """A metric that stops compiling without a join, and the error. ``uses`` is what of it needs the join:
    ``formula`` (its expression or stored filter), ``time column`` or ``dimension <name>``."""

    metric: Any
    uses: list[str]
    message: str


def dependents(model: Any, join: str) -> list[Dependent]:
    """The metrics that stop compiling without the join whose subject is ``join``; nothing is read from the data.

    Each is compiled with and without the join: plain (formula and filter), by its time grain, and by each allowed
    dimension. A metric that may be grouped by any column loses only the joined columns it could have been grouped
    by, so it is not listed.
    """
    model = QueryModel.of(model)
    without = model.without_join(join)
    out = []
    for m in model.metrics:
        cases: list[tuple[str, MetricQuery]] = [("formula", MetricQuery(m.name))]
        if m.time_column:
            cases.append(("time column", MetricQuery(m.name, grain="month")))
        cases += [(f"dimension {d}", MetricQuery(m.name, [d])) for d in m.dimensions]
        uses, message = [], ""
        for what, query in cases:
            try:
                compile_query(model, query, tables={m.table})
            except ValueError:
                continue
            try:
                compile_query(without, query, tables={m.table})
            except ValueError as exc:
                uses.append(what)
                message = message or str(exc)
        if uses:
            out.append(Dependent(m, uses, message))
    return out
