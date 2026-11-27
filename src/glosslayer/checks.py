"""Data quality checks attached to semantic types and fields.

    from glosslayer import checks

    checks.not_null()
    checks.unique()
    checks.matches(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$")
    checks.one_of(["open", "shipped", "cancelled"])
    checks.between(0, 10_000)
    checks.length(2, 2)
    checks.sql("{column} >= 0")            # escape hatch: any boolean SQL over the column

Checks are *declarations*: plain data stored as facts (``checks`` attribute), so they
are reviewed and logged like everything else. A field's effective checks are its
semantic type's checks plus its own.

Running them:

* ``run_polars(g, "shop.orders", df)`` evaluates on a Polars DataFrame (pip install polars);
* ``polars_plan(g, "shop.orders", lazy_frame)`` returns one ``LazyFrame`` that counts every
  failure in a single pass, for hosts that never collect in the calling process: ship
  ``plan.frame`` to wherever the data is, collect it there, and hand the one-row result to
  ``plan.results(row)``;
* ``run_plan(plan, collect)`` turns such a plan into results through the host's own ``collect``, one check at a
  time when the single pass fails, so a check that cannot be evaluated never hides the rest;
* ``plan(items, "shop.orders", lazy_frame)`` plans checks that are not in the ledger (a list being proposed);
* ``to_sql(g, "shop.orders", dialect="duckdb")`` produces one query per check to run anywhere.

Results can be logged with ``g.record_check_results(results, run_id=...)`` and read back with
``g.check_history(subject)``.

Custom checks: subclass :class:`Check`, implement ``polars`` and/or ``sql``, and
decorate with :func:`register`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .ledger import Gloss

_REGISTRY: dict[str, type[Check]] = {}


def register(cls: type[Check]) -> type[Check]:
    """Class decorator: make a check type known so stored checks can be rebuilt."""
    if not cls.kind:
        raise ValueError(f"{cls.__name__} needs a `kind`")
    _REGISTRY[cls.kind] = cls
    return cls


@dataclass(frozen=True)
class Check:
    """Base of the check declarations: a frozen dataclass whose fields are its parameters."""

    kind: ClassVar[str] = ""
    row_level: ClassVar[bool] = True  # False: evaluated over the whole column (e.g. unique)

    def to_dict(self) -> dict[str, Any]:
        """The stored form: ``{"check": kind, **parameters}``, leaving out parameters that are None."""
        return {"check": self.kind, **{k: v for k, v in asdict(self).items() if v is not None}}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Check:
        """Rebuild a check from its stored form; an unregistered kind becomes :class:`Unknown`."""
        d = dict(d)
        kind = d.pop("check")
        cls = _REGISTRY.get(kind)
        if cls is None:
            return Unknown(kind=kind, params=d)  # kept, not runnable until registered
        return cls(**d)

    def describe(self) -> str:
        """A short readable form, e.g. ``between(min=0, max=10)``."""
        args = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in fields(self) if getattr(self, f.name) is not None)
        return f"{self.kind}({args})"

    def verify(self) -> None:
        """Raise ``ValueError`` when the parameters cannot make a meaningful check; called when one is proposed."""

    # Implement one or both. Each returns a *pass* condition.
    def polars(self, col: str) -> Any:  # -> pl.Expr (True = row passes)
        """A Polars expression that is true for rows of ``col`` that pass.

        A null result counts as a failure, so a check that lets nulls pass must say so explicitly.
        """
        raise NotImplementedError(f"check {self.kind} has no Polars implementation")

    def sql(self, col: str, dialect: str = "ansi") -> str:  # boolean SQL, TRUE = row passes
        """A boolean SQL condition over ``col``, TRUE for rows that pass; a NULL result counts as a failure.

        Only row-level checks need one: ``to_sql`` writes its own query for checks with ``row_level`` False.
        """
        raise NotImplementedError(f"check {self.kind} has no SQL implementation")


def _q(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def _lit(v: Any) -> str:
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    return repr(v)


@register
@dataclass(frozen=True)
class NotNull(Check):
    """Passes when the value is not null."""

    kind: ClassVar[str] = "not_null"

    def polars(self, col: str) -> Any:
        import polars as pl

        return pl.col(col).is_not_null()

    def sql(self, col: str, dialect: str = "ansi") -> str:
        return f"{_q(col)} IS NOT NULL"


@register
@dataclass(frozen=True)
class Unique(Check):
    """Passes when no other row has the same value; nulls pass.

    Polars fails every copy of a duplicated value, while the SQL query from ``to_sql`` counts only
    the extra copies, so the two report different ``failed_rows`` for the same data.
    """

    kind: ClassVar[str] = "unique"
    row_level: ClassVar[bool] = False

    def polars(self, col: str) -> Any:
        import polars as pl

        return ~pl.col(col).is_duplicated() | pl.col(col).is_null()


@register
@dataclass(frozen=True)
class Matches(Check):
    """Passes when the value, as text, contains a match for the regular expression ``pattern``; nulls pass.

    The pattern is not anchored (add ``^``/``$``), except where the SQL dialect anchors it: Snowflake's
    ``REGEXP_LIKE`` and the ANSI fallback ``SIMILAR TO`` (which has its own syntax) match the whole value.
    """

    pattern: str = ""
    kind: ClassVar[str] = "matches"

    def __post_init__(self) -> None:
        re.compile(self.pattern)

    def verify(self) -> None:
        if not isinstance(self.pattern, str) or not self.pattern:
            raise ValueError("matches needs a pattern")

    def polars(self, col: str) -> Any:
        import polars as pl

        return pl.col(col).cast(pl.String).str.contains(self.pattern) | pl.col(col).is_null()

    def sql(self, col: str, dialect: str = "ansi") -> str:
        c, p = _q(col), _lit(self.pattern)
        if dialect == "postgres":
            cond = f"{c} ~ {p}"
        elif dialect == "duckdb":
            cond = f"REGEXP_MATCHES({c}, {p})"
        elif dialect in ("snowflake", "oracle", "mysql"):
            cond = f"REGEXP_LIKE({c}, {p})"
        elif dialect == "bigquery":
            cond = f"REGEXP_CONTAINS({c}, {p})"
        else:
            cond = f"{c} SIMILAR TO {p}"  # ANSI; patterns may need adapting
        return f"({c} IS NULL OR {cond})"


@register
@dataclass(frozen=True)
class OneOf(Check):
    """Passes when the value is one of ``values``; nulls pass."""

    values: tuple[Any, ...] = ()
    kind: ClassVar[str] = "one_of"

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.kind, "values": list(self.values)}

    def verify(self) -> None:
        if not self.values:
            raise ValueError("one_of needs at least one value")

    def polars(self, col: str) -> Any:
        import polars as pl

        return pl.col(col).is_in(list(self.values)) | pl.col(col).is_null()

    def sql(self, col: str, dialect: str = "ansi") -> str:
        return f"({_q(col)} IS NULL OR {_q(col)} IN ({', '.join(_lit(v) for v in self.values)}))"


@register
@dataclass(frozen=True)
class Between(Check):
    """Passes when ``min <= value <= max`` (either bound may be None); nulls pass."""

    min: Any = None
    max: Any = None
    kind: ClassVar[str] = "between"

    def verify(self) -> None:
        if self.min is None and self.max is None:
            raise ValueError("between needs a min or a max")

    def polars(self, col: str) -> Any:
        import polars as pl

        e = pl.col(col)
        cond = pl.lit(True)
        if self.min is not None:
            cond = cond & (e >= self.min)
        if self.max is not None:
            cond = cond & (e <= self.max)
        return cond | e.is_null()

    def sql(self, col: str, dialect: str = "ansi") -> str:
        parts = [f"{_q(col)} >= {_lit(self.min)}" if self.min is not None else "",
                 f"{_q(col)} <= {_lit(self.max)}" if self.max is not None else ""]
        return f"({_q(col)} IS NULL OR ({' AND '.join(p for p in parts if p) or 'TRUE'}))"


@register
@dataclass(frozen=True)
class Length(Check):
    """Passes when the value's length in characters is within ``min`` and ``max`` (inclusive); nulls pass."""

    min: int | None = None
    max: int | None = None
    kind: ClassVar[str] = "length"

    def verify(self) -> None:
        bounds = [b for b in (self.min, self.max) if b is not None]
        if not bounds:
            raise ValueError("length needs a min or a max")
        if any(isinstance(b, bool) or not isinstance(b, int) or b < 0 for b in bounds):
            raise ValueError("length bounds must be whole numbers of 0 or more")
        if len(bounds) == 2 and self.min > self.max:
            raise ValueError("length needs min to be at most max")

    def polars(self, col: str) -> Any:
        import polars as pl

        n = pl.col(col).cast(pl.String).str.len_chars()
        cond = pl.lit(True)
        if self.min is not None:
            cond = cond & (n >= self.min)
        if self.max is not None:
            cond = cond & (n <= self.max)
        return cond | pl.col(col).is_null()

    def sql(self, col: str, dialect: str = "ansi") -> str:
        n = f"CHAR_LENGTH({_q(col)})"
        parts = [f"{n} >= {self.min}" if self.min is not None else "", f"{n} <= {self.max}" if self.max is not None else ""]
        return f"({_q(col)} IS NULL OR ({' AND '.join(p for p in parts if p) or 'TRUE'}))"


@register
@dataclass(frozen=True)
class Sql(Check):
    """Any boolean SQL expression; ``{column}`` is replaced by the quoted column name."""

    expression: str = ""
    kind: ClassVar[str] = "sql"

    def polars(self, col: str) -> Any:
        import polars as pl

        return pl.sql_expr(self._filled(col))

    def sql(self, col: str, dialect: str = "ansi") -> str:
        return f"({self._filled(col)})"

    def _filled(self, col: str) -> str:
        # plain replacement: SQL is full of braces ("'^[A-Z]{2}$'") that str.format would trip over
        return self.expression.replace("{column}", _q(col))


@dataclass(frozen=True)
class Unknown(Check):
    """A stored check whose kind isn't registered in this process."""

    kind: str = ""  # type: ignore[misc]
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.kind, **self.params}

    def describe(self) -> str:
        return f"{self.kind}(…)"


# Friendly constructors
def not_null() -> Check:
    """A :class:`NotNull` check."""
    return NotNull()


def unique() -> Check:
    """A :class:`Unique` check."""
    return Unique()


def matches(pattern: str) -> Check:
    """A :class:`Matches` check; the pattern is compiled now, so a bad one fails here."""
    return Matches(pattern)


def one_of(values: Sequence[Any]) -> Check:
    """A :class:`OneOf` check."""
    return OneOf(tuple(values))


def between(min: Any = None, max: Any = None) -> Check:  # noqa: A002
    """A :class:`Between` check."""
    return Between(min, max)


def length(min: int | None = None, max: int | None = None) -> Check:  # noqa: A002
    """A :class:`Length` check."""
    return Length(min, max)


def sql(expression: str) -> Check:
    """A :class:`Sql` check: any boolean SQL, with ``{column}`` for the column; a NULL result fails."""
    return Sql(expression)


def normalise(items: Sequence[Check | dict[str, Any]]) -> list[dict[str, Any]]:
    """Checks as stored: a list of dicts with a ``check`` key."""
    out = []
    for c in items:
        d = c.to_dict() if isinstance(c, Check) else dict(c)
        if not isinstance(d.get("check"), str):
            raise ValueError(f"A check needs a 'check' kind: {d!r}")
        out.append(d)
    return out


def validate(items: Sequence[Check | dict[str, Any]]) -> list[dict[str, Any]]:
    """Checks as stored, refused when one cannot work: what a proposed ``checks`` value goes through.

    A check of a registered kind is built and asked to :meth:`Check.verify` itself, and a check listed twice is
    refused. A kind this process does not know is kept as it is: it may be registered where the checks run.
    """
    out = normalise(items)
    for i, d in enumerate(out):
        if d in out[:i]:
            raise ValueError(f"The same check is listed twice: {d!r}")
        if d["check"] not in _REGISTRY:
            continue
        try:
            Check.from_dict(d).verify()
        except (TypeError, re.error) as exc:
            raise ValueError(f"Check {d['check']!r} is not valid: {exc}") from None
    return out


# ---------------------------------------------------------------------- running
@dataclass
class CheckResult:
    """The outcome of one check on one field; ``error`` is set when it could not be evaluated."""

    subject: str  # field subject, e.g. field:shop.customers/email
    check: dict[str, Any]
    passed: bool
    failed_rows: int
    total_rows: int
    origin: str | None = None  # "type:email_address" when inherited from a semantic type
    sample: list[Any] = field(default_factory=list)  # a few failing values (omitted for confidential fields)
    error: str | None = None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:
        mark = "PASS" if self.passed else ("ERROR" if self.error else "FAIL")
        via = f"  (from {self.origin})" if self.origin else ""
        detail = self.error or f"{self.failed_rows}/{self.total_rows} rows fail"
        return f"{mark:<5} {self.subject:<40} {Check.from_dict(self.check).describe():<38} {detail}{via}"

    def to_dict(self) -> dict[str, Any]:
        """The wire form, for results computed on one machine and recorded on another.

        Failing sample values are data, so they are left out.
        """
        return {
            "subject": self.subject, "check": self.check, "passed": self.passed,
            "failed_rows": self.failed_rows, "total_rows": self.total_rows, "origin": self.origin,
            "error": self.error, "ts": self.ts.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CheckResult:
        """Rebuild a result from :meth:`to_dict`; without a timestamp it is stamped now."""
        ts = d.get("ts")
        return cls(
            d["subject"], d["check"], bool(d["passed"]), int(d["failed_rows"]), int(d["total_rows"]),
            d.get("origin"), [], d.get("error"),
            datetime.fromisoformat(ts) if isinstance(ts, str) else ts or datetime.now(timezone.utc),
        )


def effective_checks(gloss: Gloss, dataset: str) -> dict[str, list[tuple[dict[str, Any], str | None]]]:
    """{field: [(check, origin), ...]} for a dataset, from accepted facts (types included)."""
    from .model import Status

    state = gloss.snapshot([Status.ACCEPTED], datasets=[dataset], relationships=False)
    if dataset not in state.datasets:
        raise KeyError(f"No accepted facts for dataset {dataset!r}")
    out: dict[str, list[tuple[dict[str, Any], str | None]]] = {}
    for fname, f in state.datasets[dataset].fields.items():
        items = [(c, f.check_origins.get(i)) for i, c in enumerate(f.attrs.get("checks") or [])]
        if items:
            out[fname] = items
    return out


@dataclass
class CheckPlan:
    """Every effective check of a dataset as one lazy query.

    ``frame`` yields a single row: ``__total__`` and one failure count per runnable check. Nothing
    is collected here. Checks that could not be planned (column missing, no Polars form) are in
    ``errors`` already.
    """

    frame: Any  # pl.LazyFrame
    items: list[tuple[str, dict[str, Any], str | None, str]]  # (subject, check, origin, result column)
    errors: list[CheckResult]
    conditions: dict[str, Any] = field(default_factory=dict)  # result column -> pass condition
    source: Any = None  # the LazyFrame the plan reads

    def results(self, row: Any) -> list[CheckResult]:
        """Turn the collected one-row result (a mapping or a one-row DataFrame) into results."""
        if hasattr(row, "row"):
            row = row.row(0, named=True)
        total = int(row["__total__"])
        out = [CheckResult(subj, chk, int(row[col]) == 0, int(row[col]), total, origin)
               for subj, chk, origin, col in self.items]
        return out + [replace(e, total_rows=total) for e in self.errors]

    def failing(self, column: str) -> Any:
        """A LazyFrame of the rows failing one check, for an optional second pass that samples them."""
        return self.source.filter(~self.conditions[column].fill_null(False))


def plan(items: Mapping[str, Sequence[tuple[Check | dict[str, Any], str | None]]], dataset: str, frame: Any, *,
         schema: Iterable[str] | None = None) -> CheckPlan:
    """Plan ``{field: [(check, origin), ...]}`` over a Polars LazyFrame (or DataFrame) in one pass, without a ledger.

    ``schema`` is the frame's column names (or a mapping keyed by them) when the host knows them: the frame's own
    schema is then not resolved here, where its scan may not be readable. A check that cannot be planned (its
    column is missing, it has no Polars form, its parameters do not build) lands in ``errors``, never raises.
    """
    import polars as pl

    from .ids import SubjectId

    lf = frame.lazy()
    columns = set(schema) if schema is not None else set(lf.collect_schema().names())
    exprs = [pl.len().alias("__total__")]
    out = CheckPlan(None, [], [], source=lf)
    for fname, field_items in items.items():
        subj = str(SubjectId.for_field(dataset, fname))
        for chk, origin in field_items:
            chk = chk.to_dict() if isinstance(chk, Check) else chk
            if fname not in columns:
                out.errors.append(CheckResult(subj, chk, False, 0, 0, origin, error=f"column {fname!r} not in data"))
                continue
            try:
                passes = Check.from_dict(chk).polars(fname)
            except Exception as exc:  # noqa: BLE001
                out.errors.append(CheckResult(subj, chk, False, 0, 0, origin, error=_reason(exc)))
                continue
            col = f"c{len(out.items)}"
            out.conditions[col] = passes
            exprs.append(_failures(passes).alias(col))
            out.items.append((subj, chk, origin, col))
    out.frame = lf.select(exprs)
    return out


def polars_plan(gloss: Gloss, dataset: str, frame: Any, *, schema: Iterable[str] | None = None) -> CheckPlan:
    """Plan every effective check of ``dataset`` over a Polars LazyFrame (or DataFrame), in one pass.

    ``schema`` as in :func:`plan`.
    """
    return plan(effective_checks(gloss, dataset), dataset, frame, schema=schema)


def run_plan(plan: CheckPlan, collect: Callable[[Any], Any]) -> list[CheckResult]:
    """The results of ``plan``, each LazyFrame collected by ``collect`` wherever the data is.

    ``collect`` takes a LazyFrame that yields one row and returns that row (a mapping or a one-row DataFrame).
    When the single pass fails, every check is collected on its own, so a check that cannot be evaluated on this
    data reports its error and the rest still report. :class:`~glosslayer.dialect.EngineUnavailable` raised by
    ``collect`` is no check's failure and is raised.
    """
    import polars as pl

    from .dialect import EngineUnavailable

    try:
        return plan.results(collect(plan.frame))
    except EngineUnavailable:
        raise
    except Exception:  # noqa: BLE001
        pass
    try:
        total = int(_named(collect(plan.source.select(pl.len().alias("__total__"))))["__total__"])
    except EngineUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        failed = [CheckResult(subj, chk, False, 0, 0, origin, error=_reason(exc)) for subj, chk, origin, _c in plan.items]
        return failed + list(plan.errors)
    results = []
    for subj, chk, origin, col in plan.items:
        try:
            n = int(_named(collect(plan.source.select(_failures(plan.conditions[col]).alias(col))))[col])
            results.append(CheckResult(subj, chk, n == 0, n, total, origin))
        except EngineUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            results.append(CheckResult(subj, chk, False, 0, total, origin, error=_reason(exc)))
    return results + [replace(e, total_rows=total) for e in plan.errors]


def _failures(passes: Any) -> Any:
    return (~passes.fill_null(False)).sum()


def _named(row: Any) -> Any:
    return row.row(0, named=True) if hasattr(row, "row") else row


def _reason(exc: BaseException) -> str:
    """The first line of an error: engines append a plan dump that is no reason."""
    lines = str(exc).splitlines()
    return (lines[0].strip() if lines else "") or type(exc).__name__


def run_polars(gloss: Gloss, dataset: str, df: Any, *, sample_size: int = 5) -> list[CheckResult]:
    """Evaluate every effective check of ``dataset`` on a Polars DataFrame.

    One pass counts all failures; a few failing values are then sampled per failed check,
    except for fields above ``internal`` confidentiality.
    """
    from .model import Confidentiality
    from .visibility import levels

    plan = polars_plan(gloss, dataset, df)
    results = run_plan(plan, lambda frame: frame.collect())
    if sample_size:
        columns = {subj: col for subj, _chk, _origin, col in plan.items}
        hidden = {s for s, lv in levels(gloss.store, columns).items() if lv > Confidentiality.INTERNAL}
        by_item = {(subj, id(chk)): col for subj, chk, _origin, col in plan.items}
        for r in results:
            col = by_item.get((r.subject, id(r.check)))
            if col and not r.passed and not r.error and r.subject not in hidden:
                fname = r.subject.rsplit("/", 1)[-1]
                r.sample = plan.failing(col).select(fname).head(sample_size).collect().get_column(fname).to_list()
    order = {subj: i for i, subj in enumerate(dict.fromkeys(
        str(s) for s in [x[0] for x in plan.items] + [e.subject for e in plan.errors]))}
    return sorted(results, key=lambda r: order.get(r.subject, 0))


def to_sql(gloss: Gloss, dataset: str, *, dialect: str = "ansi", table: str | None = None) -> list[tuple[str, dict[str, Any], str]]:
    """[(field subject, check, query)]: each query returns ``failed_rows`` and ``total_rows``."""
    from .ids import SubjectId

    src = table or gloss.current(SubjectId.for_dataset(dataset)).get("source") or dataset
    out = []
    for fname, items in effective_checks(gloss, dataset).items():
        subj = str(SubjectId.for_field(dataset, fname))
        for chk, _origin in items:
            c = Check.from_dict(chk)
            if not c.row_level:
                q = f"SELECT COUNT({_q(fname)}) - COUNT(DISTINCT {_q(fname)}) AS failed_rows, COUNT(*) AS total_rows FROM {src}"
            else:
                try:
                    cond = c.sql(fname, dialect)
                except NotImplementedError:
                    continue
                q = (
                    f"SELECT SUM(CASE WHEN {cond} THEN 0 ELSE 1 END) AS failed_rows, "
                    f"COUNT(*) AS total_rows FROM {src}"
                )
            out.append((subj, chk, q))
    return out
