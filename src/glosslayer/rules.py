"""Rule-based suggestions for a model: column roles, joins and metrics, with no AI involved.

Three steps, so a host can show, limit or reword the candidates in between:

    stats, problem = rules.collect_stats(tables, execute)         # one aggregate statement per table
    found = rules.suggest(stats)                                   # pure: what the types and statistics support
    rules.propose(view, view.model_view("shop.sales", tables=tables, statuses=LIVE_STATUSES), found)

The rules read types and statistics, never what a name seems to mean, and each candidate carries its evidence as the
rationale a reviewer reads. Without an engine (see :mod:`glosslayer.dialect`) they propose only what a column's type
alone tells. :func:`propose` states them under an ``inferred:`` source, so they queue for review, and skips what the
model already holds.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from .dialect import EngineUnavailable, Execute, QueryFailed, quote
from .ids import SubjectId
from .ledger import Proposal
from .model import Status
from .modeling import Metric, Relationship
from .references import short_name
from .types import to_ossie_datatype

if TYPE_CHECKING:
    from .ledger import Gloss
    from .modelview import MetricView, ModelView, TableShape

SOURCE = "inferred:rules"
MAX_CATEGORIES = 20
WORDS_CONFIDENCE = 0.6
_TYPE_CLASSES = {
    "Integer": "integer",
    "Decimal": "numeric",
    "Float": "numeric",
    "Date": "time",
    "DateTime": "time",
    "DateTimeTz": "time",
    "String": "string",
    "Boolean": "boolean",
}


def type_class(dtype: str | None) -> str:
    """``integer``, ``numeric``, ``time``, ``string``, ``boolean`` or ``other`` for a catalog's type name."""
    return _TYPE_CLASSES.get(to_ossie_datatype(dtype) or "", "other")


def _numeric(tc: str) -> bool:
    return tc in ("integer", "numeric")


def _join_class(tc: str) -> str:
    return "number" if _numeric(tc) else tc


def singular(name: str) -> str:
    """A crude English singular, enough for ``orders`` → ``order`` and ``categories`` → ``category``."""
    n = name.lower()
    if n.endswith("ies") and len(n) > 4:
        return n[:-3] + "y"
    if n.endswith(("ses", "xes")) and len(n) > 4:
        return n[:-2]
    if n.endswith("s") and not n.endswith("ss") and len(n) > 2:
        return n[:-1]
    return n


def _words(name: str) -> str:
    return name.replace("_", " ").strip()


def _norm(expression: str | None) -> str:
    return (expression or "").replace(" ", "").lower()


# ------------------------------------------------------------------ statistics
@dataclass
class ColumnStats:
    """A column and, when its table was measured, its distinct and null counts and numeric or time bounds."""

    name: str
    dtype: str | None
    distinct: int | None = None
    nulls: int | None = None
    min: Any = None
    max: Any = None

    @property
    def tc(self) -> str:
        """The column's :func:`type_class`."""
        return type_class(self.dtype)


@dataclass
class TableStats:
    """A table as the rules see it. ``available`` says whether it was measured, or only its types are known."""

    table: TableShape
    row_count: int | None = None
    columns: list[ColumnStats] = field(default_factory=list)
    available: bool = False

    @property
    def key(self) -> str:
        return self.table.key

    @property
    def name(self) -> str:
        return getattr(self.table, "name", None) or short_name(self.table.key)

    def column(self, name: str) -> ColumnStats | None:
        """The column called ``name``, or None."""
        return next((c for c in self.columns if c.name == name), None)

    def unique(self, name: str) -> bool | None:
        """Whether the column is unique and non-null; ``None`` without statistics."""
        c = self.column(name)
        if not self.available or c is None or c.distinct is None or self.row_count is None:
            return None
        return self.row_count > 0 and c.distinct == self.row_count and not c.nulls


def stats_sql(table: TableShape, *, table_name: str | None = None, dialect: str = "ansi") -> str:
    """One aggregate statement: the row count and, per column, distinct and null counts and numeric or time bounds."""
    def q(name: str) -> str:
        return quote(name, dialect)

    parts = [f"COUNT(*) AS {q('__rows')}"]
    for i, c in enumerate(table.columns):
        col = q(c.name)
        parts.append(f"COUNT(DISTINCT {col}) AS {q(f'd{i}')}")
        parts.append(f"COUNT(*) - COUNT({col}) AS {q(f'n{i}')}")
        if type_class(c.dtype) in ("integer", "numeric", "time"):
            parts.append(f"MIN({col}) AS {q(f'lo{i}')}")
            parts.append(f"MAX({col}) AS {q(f'hi{i}')}")
    return f"SELECT {', '.join(parts)} FROM {q(table_name or table.key)}"


def collect_stats(
    tables: Iterable[TableShape],
    execute: Execute,
    *,
    table_names: Mapping[str, str] | None = None,
    dialect: str = "ansi",
) -> tuple[list[TableStats], str | None]:
    """Statistics per table via ``execute``; names and types only where that fails.

    Returns the stats and, when any table fell back, a sentence saying why. An unavailable engine stops the remaining
    queries too, since they would fail the same way.
    """
    out: list[TableStats] = []
    problem: str | None = None
    unreachable = False
    for t in tables:
        ts = TableStats(t, columns=[ColumnStats(c.name, c.dtype) for c in t.columns])
        out.append(ts)
        if unreachable or not ts.columns:
            continue
        try:
            rows = execute(stats_sql(t, table_name=(table_names or {}).get(t.key), dialect=dialect))
            if not rows:
                raise QueryFailed("no answer")
        except EngineUnavailable as exc:
            unreachable = True
            problem = f"Statistics were not available (compute unreachable: {exc}); rules used names and types only."
            continue
        except QueryFailed as exc:
            problem = problem or f"Statistics for {ts.name!r} failed ({exc}); its rules used names and types."
            continue
        values = iter(rows[0])
        ts.available, ts.row_count = True, next(values)
        for c in ts.columns:
            c.distinct, c.nulls = next(values), next(values)
            if c.tc in ("integer", "numeric", "time"):
                c.min, c.max = next(values), next(values)
    return out, problem


# ------------------------------------------------------------------ candidates
@dataclass
class RoleCandidate:
    table: str
    column: str
    role: str
    confidence: float
    rationale: str


@dataclass
class JoinCandidate:
    name: str
    from_table: str
    to_table: str
    from_columns: list[str]
    to_columns: list[str]
    description: str
    confidence: float
    rationale: str


@dataclass
class MetricCandidate:
    table: str
    name: str
    expression: str
    description: str
    aggregate: str
    time_column: str | None
    dimensions: list[str]
    confidence: float
    rationale: str


@dataclass
class Suggestions:
    """What the rules found, tables by dataset key, and the statistics (by key) they read."""

    roles: list[RoleCandidate] = field(default_factory=list)
    joins: list[JoinCandidate] = field(default_factory=list)
    metrics: list[MetricCandidate] = field(default_factory=list)
    stats: dict[str, TableStats] = field(default_factory=dict)

    @property
    def stats_available(self) -> bool:
        """Whether any table was measured."""
        return any(t.available for t in self.stats.values())

    def only(self, *, tables: Iterable[str] | None = None, kinds: Iterable[str] | None = None) -> Suggestions:
        """The candidates about ``tables`` (a join: touching one) and of ``kinds`` (``role``, ``join``, ``metric``)."""
        keep = None if tables is None else set(tables)
        want = None if kinds is None else set(kinds)

        def of(kind: str, items: list[Any], *ends: str) -> list[Any]:
            if want is not None and kind not in want:
                return []
            return [x for x in items if keep is None or any(getattr(x, e) in keep for e in ends)]

        return replace(
            self,
            roles=of("role", self.roles, "table"),
            joins=of("join", self.joins, "from_table", "to_table"),
            metrics=of("metric", self.metrics, "table"),
        )


def column_role(
    ts: TableStats, c: ColumnStats, *, foreign_key: bool = False, max_categories: int = MAX_CATEGORIES
) -> tuple[str, float, str] | None:
    """The role, a confidence and the evidence for one column, from its type and statistics; ``None`` when no rule
    fits. Without statistics only what the type alone tells is proposed (a date, a boolean)."""
    tc = c.tc
    rows = ts.row_count
    if tc == "time":
        return "time", 0.9, f"{c.name} is a {c.dtype} column"
    if tc == "boolean":
        return "category", 0.7, f"{c.name} is a boolean column"
    if foreign_key:
        return "key", 0.8, f"{c.name} points to a unique column of another table with the same name and type"
    if not ts.available or c.distinct is None:
        return None
    if ts.unique(c.name) and tc in ("integer", "string") and (rows or 0) >= 2:
        return "key", 0.7, f"{c.name} is unique and non-null ({c.distinct} of {rows} rows)"
    if _numeric(tc):
        return "amount", 0.7, f"{c.name} is numeric ({c.dtype}) and repeats ({c.distinct} distinct in {rows} rows)"
    if tc == "string" and (c.distinct <= max_categories or (rows and c.distinct <= rows / 2)):
        return "category", 0.8, f"{c.name} is text with {c.distinct} distinct values in {rows} rows"
    return None


def suggest(tables: list[TableStats], *, max_categories: int = MAX_CATEGORIES) -> Suggestions:
    """Roles, joins and metrics for ``tables`` (one model's tables, with or without statistics)."""
    out = Suggestions(stats={ts.key: ts for ts in tables})
    fks: dict[str, list[str]] = {ts.key: [] for ts in tables}
    for a, c, b in _foreign_keys(tables):
        target = b.column(c.name)
        why = (
            f"{a.name}.{c.name} matches {b.name}.{target.name} by name and type; "
            f"{target.name} is unique in {b.name} ({target.distinct} of {b.row_count} rows) and "
            f"repeats in {a.name} ({c.distinct} distinct in {a.row_count} rows)"
        )
        out.joins.append(JoinCandidate(
            f"{a.name}_{b.name}", a.key, b.key, [c.name], [target.name],
            f"Each {singular(a.name)} belongs to one {singular(b.name)}.", 0.85, f"Rule: {why}.",
        ))
        fks[a.key].append(c.name)

    roles: dict[str, dict[str, str]] = {}
    for ts in tables:
        roles[ts.key] = {}
        for c in ts.columns:
            found = column_role(ts, c, foreign_key=c.name in fks[ts.key], max_categories=max_categories)
            if found is None:
                continue
            role, confidence, why = found
            roles[ts.key][c.name] = role
            out.roles.append(RoleCandidate(ts.key, c.name, role, confidence, f"Rule: {why}."))

    for ts in tables:
        r = roles[ts.key]
        time_column = next((c.name for c in ts.columns if r.get(c.name) == "time"), None)
        dims = [c.name for c in ts.columns if r.get(c.name) == "category"]
        for j in out.joins:
            if j.from_table == ts.key:
                other = out.stats[j.to_table]
                dims += [f"{other.name}.{c.name}" for c in other.columns if roles[other.key].get(c.name) == "category"]
        lookup_only = ts.key in {j.to_table for j in out.joins} - {j.from_table for j in out.joins}
        for name, sql, description, aggregate, why, weight in _metric_rules(ts, r, fks[ts.key], lookup_only):
            confidence = (0.7 if ts.available else 0.5) * weight
            out.metrics.append(MetricCandidate(
                ts.key, name, sql, description, aggregate, time_column, dims, confidence, f"Rule: {why}."
            ))
    _dedupe_metric_names(out)
    return out


def _foreign_keys(tables: list[TableStats]) -> Iterator[tuple[TableStats, ColumnStats, TableStats]]:
    """Columns that repeat in one table and are unique, under the same name and kind of type, in another."""
    for a in tables:
        for c in a.columns:
            if not a.available or a.unique(c.name) is not False:
                continue
            for b in tables:
                target = b.column(c.name) if b is not a else None
                if target is not None and b.unique(target.name) and _join_class(target.tc) == _join_class(c.tc):
                    yield a, c, b
                    break


def _metric_rules(
    ts: TableStats, roles: Mapping[str, str], fks: list[str], lookup_only: bool
) -> Iterator[tuple[str, str, str, str, str, float]]:
    """Name, SQL, description, aggregate, evidence and weight of each metric one table is a candidate for."""
    noun, rows_of = singular(ts.name), _words(ts.name)
    evidence = f" ({ts.row_count} rows)" if ts.row_count is not None else ""
    why = f"every table gets a row count{evidence}"
    yield f"{noun}_count", "COUNT(*)", f"The number of {rows_of}.", "count", why, 1
    for name in (c.name for c in ts.columns if roles.get(c.name) == "amount"):
        col, words = quote(name), _words(name)
        # a per-row value of a lookup table (a product's price) has no meaningful sum
        if not lookup_only:
            why = f"{name} is an amount column, so its sum is a candidate metric"
            yield f"total_{name}", f"SUM({col})", f"The sum of {words} over {rows_of}.", "sum", why, 1
        why = f"{name} is an amount column, so its average is a candidate metric"
        yield f"average_{name}", f"AVG({col})", f"The average {words} per {noun}.", "avg", why, 0.9
    for fk in fks:
        description = f"The number of distinct {_words(fk)} values with at least one {noun}."
        why = f"{fk} is a foreign key, so its distinct count is a candidate metric"
        yield f"distinct_{fk}", f"COUNT(DISTINCT {quote(fk)})", description, "count_distinct", why, 1


def _dedupe_metric_names(found: Suggestions) -> None:
    """Metric names are unique per namespace: on a clash a table's own row count keeps the name and the others get
    their table's singular name as a prefix (``order_customer_count`` beside the customers' ``customer_count``)."""
    counts: dict[str, int] = {}
    for m in found.metrics:
        counts[m.name] = counts.get(m.name, 0) + 1
    for m in found.metrics:
        if counts[m.name] > 1 and m.aggregate != "count":
            m.name = f"{singular(found.stats[m.table].name)}_{m.name}"


# ------------------------------------------------------------------ proposing
@dataclass
class Words:
    """Better words for one table's candidates, written by an AI (or a person): a sentence for the table, one per
    column, and per metric candidate (by its rule name) a name and a meaning, either of them optional."""

    description: str | None = None
    columns: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, tuple[str | None, str | None]] = field(default_factory=dict)


@dataclass
class Counts:
    roles: int = 0
    descriptions: int = 0
    joins: int = 0
    metrics: int = 0

    def add(self, what: str, n: int = 1) -> None:
        setattr(self, what, getattr(self, what) + n)


@dataclass
class Stored:
    """``added`` are new proposals; ``skipped`` the candidates the model already holds (accepted, pending or once
    rejected). A join or a metric counts as one, whatever facts it has."""

    added: Counts = field(default_factory=Counts)
    skipped: Counts = field(default_factory=Counts)


def _common_namespace(a: str, b: str) -> str | None:
    """The longest dotted path both dataset keys lie under, or None."""
    shared = []
    for x, y in zip(a.split(".")[:-1], b.split(".")[:-1], strict=False):
        if x != y:
            break
        shared.append(x)
    return ".".join(shared) or None


class _Batch:
    """Proposals collected per candidate, stored in one call, counted per candidate."""

    def __init__(self, stored: Stored):
        self.stored = stored
        self.props: list[Proposal] = []
        self.parts: list[tuple[str, int]] = []

    def add(self, what: str, props: Iterable[Proposal]) -> None:
        props = list(props)
        self.props += props
        self.parts.append((what, len(props)))

    def store(self, gloss: Gloss, source: str) -> None:
        results = gloss.propose_many(self.props, source=source) if self.props else []
        starts = [0]
        for _what, n in self.parts:
            starts.append(starts[-1] + n)
        for (what, n), (start, end) in zip(self.parts, pairwise(starts), strict=True):
            made, total = sum(r.outcome == "created" for r in results[start:end]), n
            if what in ("joins", "metrics"):
                made, total = min(made, 1), min(n, 1)
            self.stored.added.add(what, made)
            self.stored.skipped.add(what, total - made)


def propose(
    gloss: Gloss,
    model: ModelView,
    found: Suggestions,
    *,
    source: str = SOURCE,
    owner: Callable[[str, str], str | None] = _common_namespace,
    words: Mapping[str, Words] | None = None,
    worded_by: tuple[Gloss, str] | None = None,
    by: str = "AI",
) -> Stored:
    """Store candidates as proposals through ``gloss`` (a view bound to whoever asked), skipping what ``model`` holds.

    ``model`` is the namespace's live view (``statuses=LIVE_STATUSES``). ``owner(from key, to key)`` names the
    namespace a join belongs to; by default the longest path both tables lie under. Each proposal carries its
    evidence as the comment and a confidence.

    ``words`` (per dataset key) are stated by ``worded_by``, a ``(view, source)`` pair such as an agent's: the
    descriptions, and the metrics it found a better name or meaning for; ``by`` names the author in the comments.
    A name that is taken falls back to the rule's. For a metric the model already holds the words rename nothing:
    the name becomes a synonym, and the meaning its description when it has none or only the rules' own.
    """
    if words and worded_by is None:
        raise ValueError("Say who states the words: worded_by=(view, source)")
    stored = Stored()
    by_rules, by_words = _Batch(stored), _Batch(stored)
    words = words or {}

    for r in found.roles:
        t = model.table(r.table)
        col = t.column(r.column) if t is not None else None
        if col is None or "role" in col.facts:
            stored.skipped.add("roles")
            continue
        sid = SubjectId.for_field(r.table, r.column)
        by_rules.add("roles", [Proposal(str(sid), "role", r.role, r.confidence, [], r.rationale)])

    held_joins = {(j.from_table, j.to_table, *j.from_columns, *j.to_columns) for j in model.joins}
    for j in found.joins:
        known = {j.from_table, j.to_table} <= model.columns.keys()
        if not known or (j.from_table, j.to_table, *j.from_columns, *j.to_columns) in held_joins:
            stored.skipped.add("joins")
            continue
        rel = Relationship(j.name, j.from_table, j.to_table, from_columns=list(j.from_columns),
                           to_columns=list(j.to_columns), description=j.description,
                           namespace=owner(j.from_table, j.to_table))
        by_rules.add("joins", _with(rel.proposals(), j.rationale, j.confidence))

    reworded: list[Proposal] = []
    names = set(gloss.metric_names(model.namespace))
    held = {(m.table, _norm(m.expression)): m for m in model.metrics}
    for m in found.metrics:
        if model.table(m.table) is None:
            stored.skipped.add("metrics")
            continue
        better = words[m.table].metrics.get(m.name) if m.table in words else None
        same = held.get((m.table, _norm(m.expression)))
        if same is not None:
            stored.skipped.add("metrics")
            if better:
                reworded += _reworded(same, *better, m.description, by)
            continue
        name, description = better or (None, None)
        name, description, batch = name or m.name, description or m.description, by_words if better else by_rules
        if name in names:
            if m.name in names:
                stored.skipped.add("metrics")
                continue
            name, description, batch = m.name, m.description, by_rules
        names.add(name)
        metric = Metric(name, m.expression, description=description, aggregate=m.aggregate, time_column=m.time_column,
                        dimensions=list(m.dimensions), table=m.table)
        note = f" Name and meaning written by {by}." if batch is by_words else ""
        batch.add("metrics", _with(metric.proposals(), m.rationale + note, m.confidence))

    described = _Batch(stored)
    if reworded:
        described.add("descriptions", reworded)
    for key, w in words.items():
        t = model.table(key)
        if t is None:
            continue
        lines = [(SubjectId.for_dataset(key), t, w.description, False)]
        lines += [(SubjectId.for_field(key, c), t.column(c), text, True) for c, text in w.columns.items()]
        for sid, entry, text, counted in lines:
            if entry is None or not text:
                continue
            if "description" in entry.facts:
                stored.skipped.add("descriptions", int(counted))
            else:
                described.add("descriptions", [_worded(str(sid), "description", text, f"Written by {by}.")])

    by_rules.store(gloss, source)
    if worded_by is not None:
        view, author = worded_by
        described.store(view, author)
        by_words.store(view, author)
    return stored


def _with(proposals: Iterable[Proposal], comment: str, confidence: float) -> list[Proposal]:
    return [replace(p, comment=comment, confidence=round(confidence, 2)) for p in proposals]


def _worded(subject: str, attribute: str, value: Any, comment: str, group: str | None = None) -> Proposal:
    return Proposal(subject, attribute, value, WORDS_CONFIDENCE, [], comment, group)


def _reworded(held: MetricView, name: str | None, meaning: str | None, templated: str, by: str) -> list[Proposal]:
    """A better name and meaning for a metric the model already holds, as proposals that rename nothing.

    The name becomes a **synonym** (so a question saying "revenue" finds ``total_amount``) and the sentence the
    description when the metric has none or only the rules' templated one. A pending metric's words join its review
    group so they are decided with the definition; an accepted metric's queue on their own.
    """
    group = None if held.status is Status.ACCEPTED else held.group
    out = []
    alias = (name or "").replace("_", " ")
    if alias and alias != held.name.replace("_", " ").lower() and alias not in held.synonyms:
        out.append(_worded(held.key, "synonyms", [*held.synonyms, alias], f"Name written by {by}.", group))
    if meaning and held.description in (None, templated) and meaning != held.description:
        out.append(_worded(held.key, "description", meaning, f"Written by {by}.", group))
    return out
