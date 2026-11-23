"""Data quality checks attached to semantic types and fields.

    from gloss import checks

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
* ``to_sql(g, "shop.orders", dialect="duckdb")`` produces one query per check to run anywhere.

Results can be logged with ``g.record_check_results(results, run_id=...)`` and read back with
``g.check_history(subject)``.

Custom checks: subclass :class:`Check`, implement ``polars`` and/or ``sql``, and
decorate with :func:`register`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
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
    kind: ClassVar[str] = ""
    row_level: ClassVar[bool] = True  # False: evaluated over the whole column (e.g. unique)

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.kind, **{k: v for k, v in asdict(self).items() if v is not None}}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Check:
        d = dict(d)
        kind = d.pop("check")
        cls = _REGISTRY.get(kind)
        if cls is None:
            return Unknown(kind=kind, params=d)  # kept, not runnable until registered
        return cls(**d)

    def describe(self) -> str:
        args = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in fields(self) if getattr(self, f.name) is not None)
        return f"{self.kind}({args})"

    # Implement one or both. Each returns a *pass* condition.
    def polars(self, col: str) -> Any:  # -> pl.Expr (True = row passes)
        raise NotImplementedError(f"check {self.kind} has no Polars implementation")

    def sql(self, col: str, dialect: str = "ansi") -> str:  # boolean SQL, TRUE = row passes
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
    kind: ClassVar[str] = "not_null"

    def polars(self, col: str) -> Any:
        import polars as pl

        return pl.col(col).is_not_null()

    def sql(self, col: str, dialect: str = "ansi") -> str:
        return f"{_q(col)} IS NOT NULL"


@register
@dataclass(frozen=True)
class Unique(Check):
    kind: ClassVar[str] = "unique"
    row_level: ClassVar[bool] = False

    def polars(self, col: str) -> Any:
        import polars as pl

        return ~pl.col(col).is_duplicated() | pl.col(col).is_null()


@register
@dataclass(frozen=True)
class Matches(Check):
    pattern: str = ""
    kind: ClassVar[str] = "matches"

    def __post_init__(self) -> None:
        re.compile(self.pattern)

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
    values: tuple[Any, ...] = ()
    kind: ClassVar[str] = "one_of"

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.kind, "values": list(self.values)}

    def polars(self, col: str) -> Any:
        import polars as pl

        return pl.col(col).is_in(list(self.values)) | pl.col(col).is_null()

    def sql(self, col: str, dialect: str = "ansi") -> str:
        return f"({_q(col)} IS NULL OR {_q(col)} IN ({', '.join(_lit(v) for v in self.values)}))"


@register
@dataclass(frozen=True)
class Between(Check):
    min: Any = None
    max: Any = None
    kind: ClassVar[str] = "between"

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
    min: int | None = None
    max: int | None = None
    kind: ClassVar[str] = "length"

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
    return NotNull()


def unique() -> Check:
    return Unique()


def matches(pattern: str) -> Check:
    return Matches(pattern)


def one_of(values: Sequence[Any]) -> Check:
    return OneOf(tuple(values))


def between(min: Any = None, max: Any = None) -> Check:  # noqa: A002
    return Between(min, max)


def length(min: int | None = None, max: int | None = None) -> Check:  # noqa: A002
    return Length(min, max)


def sql(expression: str) -> Check:
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


# ---------------------------------------------------------------------- running
@dataclass
class CheckResult:
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


def polars_plan(gloss: Gloss, dataset: str, frame: Any) -> CheckPlan:
    """Plan every effective check of ``dataset`` over a Polars LazyFrame (or DataFrame), in one pass."""
    import polars as pl

    from .ids import SubjectId

    lf = frame.lazy()
    columns = set(lf.collect_schema().names())
    exprs = [pl.len().alias("__total__")]
    plan = CheckPlan(None, [], [], source=lf)
    for fname, items in effective_checks(gloss, dataset).items():
        subj = str(SubjectId.for_field(dataset, fname))
        for chk, origin in items:
            if fname not in columns:
                plan.errors.append(CheckResult(subj, chk, False, 0, 0, origin, error=f"column {fname!r} not in data"))
                continue
            try:
                passes = Check.from_dict(chk).polars(fname)
            except (NotImplementedError, ValueError, TypeError, pl.exceptions.PolarsError) as exc:
                plan.errors.append(CheckResult(subj, chk, False, 0, 0, origin, error=str(exc)))
                continue
            col = f"c{len(plan.items)}"
            plan.conditions[col] = passes
            exprs.append((~passes.fill_null(False)).sum().alias(col))
            plan.items.append((subj, chk, origin, col))
    plan.frame = lf.select(exprs)
    return plan


def run_polars(gloss: Gloss, dataset: str, df: Any, *, sample_size: int = 5) -> list[CheckResult]:
    """Evaluate every effective check of ``dataset`` on a Polars DataFrame.

    One pass counts all failures; a few failing values are then sampled per failed check,
    except for fields above ``internal`` confidentiality.
    """
    import polars as pl

    from .model import Confidentiality
    from .visibility import levels

    plan = polars_plan(gloss, dataset, df)
    try:
        results = plan.results(plan.frame.collect())
    except pl.exceptions.PolarsError:
        # one check cannot be evaluated on this data: run them one by one so the rest still report
        total = df.lazy().select(pl.len()).collect().item()
        results = []
        for subj, chk, origin, col in plan.items:
            try:
                n = plan.failing(col).select(pl.len()).collect().item()
                results.append(CheckResult(subj, chk, n == 0, n, total, origin))
            except pl.exceptions.PolarsError as exc:
                results.append(CheckResult(subj, chk, False, 0, total, origin, error=str(exc)))
        results += [replace(e, total_rows=total) for e in plan.errors]
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
