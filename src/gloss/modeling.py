"""Object-style modelling API.

    import gloss
    from gloss import Table, Column, Relationship, Metric, Human

    g = gloss.open("sqlite:///gloss.db")

    orders = Table("shop.orders", primary_key=[Column("order_id", "Integer")])

    returns = Table("shop.returns", source="warehouse.shop.returns",
                    description="One row per returned order",
                    primary_key=[Column("return_id", "Integer")])
    returns.add_columns(
        Column("order_id", int, description="The order that was returned"),
        Column("reason", "String", synonyms=["return reason"]),
    )

    g.add(returns, orders,
          Relationship("returns_orders", on={returns["order_id"]: orders["order_id"]}),
          source=Human("edward"))

Objects are *descriptions*: saving them proposes one fact per attribute, so
everything still goes through review and the log. Saving the same object twice
is a no-op; changing one attribute proposes only that attribute.
``Table.load(g, key)`` / ``g.table(key)`` reads a table back from the ledger.
"""

from __future__ import annotations

import datetime as _dt
import decimal as _decimal
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

from .dtypes import DType
from .dtypes import parse as parse_dtype
from .ids import SubjectId
from .model import DATATYPES, Status, new_id
from .types import to_ossie_datatype

if TYPE_CHECKING:
    from .ledger import Gloss, Proposal

_PY_TYPES: dict[type, str] = {
    bool: "Boolean",  # before int: bool is a subclass of int
    int: "Integer",
    float: "Float",
    str: "String",
    _decimal.Decimal: "Decimal",
    _dt.datetime: "DateTime",  # before date: datetime is a subclass of date
    _dt.date: "Date",
    _dt.time: "Time",
}

# attributes shared by everything, in the order they are proposed
_COMMON = ("description", "ai_instructions", "synonyms", "examples", "confidentiality", "owner")


DatatypeLike = Union[str, type, DType, None]


def _datatype(value: DatatypeLike) -> tuple[str | None, str | None]:
    """Resolve to (Ossie datatype, physical type).

    Accepts Gloss types (``Int(32)``, ``Decimal(18, 2)``, or a bare ``Int``), Ossie names
    (``"Integer"``), catalog type names (``"Int64"``, ``"varchar(20)"``) and Python types (``int``).
    """
    if value is None:
        return None, None
    if isinstance(value, DType):
        return value.ossie, value.physical
    if isinstance(value, type):
        if issubclass(value, DType):
            inst = value()
            return inst.ossie, inst.physical
        for py, ossie in _PY_TYPES.items():
            if issubclass(value, py):
                return ossie, None
        raise ValueError(f"No Ossie datatype for Python type {value.__name__}")
    if value in DATATYPES:
        return value, None
    mapped = to_ossie_datatype(value)
    if mapped == "Opaque" and value.lower() != "opaque":
        raise ValueError(f"Unknown datatype {value!r}; use a Gloss type like Int(32), an Ossie name, or a catalog type")
    return mapped, value  # keep the catalog's own spelling as the physical type


def _facts(obj: Any, names: Sequence[str]) -> Iterator[tuple[str, Any]]:
    for name in names:
        value = getattr(obj, name)
        if value is not None and value != []:
            yield name, value


@dataclass
class SemanticType:
    """A reusable meaning that many columns share, e.g. an e-mail address.

        Email = SemanticType(
            "email_address",
            dtype=String(320),
            description="An e-mail address",
            confidentiality="restricted",
            classification=["pii.email"],
            synonyms=["e-mail", "mail address"],
            checks=[checks.matches(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$")],
            name_patterns=[r"e[-_]?mail"],          # used to suggest this type for matching columns
        )
        Column("customer_email", Email)

    A column using a type inherits everything it doesn't state itself; checks are combined.
    Changing an accepted type changes every column that uses it.
    """

    name: str
    dtype: DatatypeLike = None
    description: str | None = None
    label: str | None = None
    confidentiality: str | None = None
    classification: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    ai_instructions: str | None = None
    owner: str | None = None
    checks: list[Any] = field(default_factory=list)
    name_patterns: list[str] = field(default_factory=list)
    namespace: str | None = None  # the namespace that owns the type; None for a ledger-wide type

    def __post_init__(self) -> None:
        sid = SubjectId.for_type(self.name, self.namespace)  # also accepts "crm::email_address"
        self.name, self.namespace = sid.name or "", sid.namespace
        self.datatype, self.physical_type = _datatype(self.dtype)

    @property
    def subject(self) -> SubjectId:
        return SubjectId.for_type(self.name, self.namespace)

    @property
    def ref(self) -> str:
        """How columns refer to this type: ``crm::email_address``, or the bare name."""
        return self.subject.ref

    @property
    def resolved_dtype(self) -> DType | None:
        return parse_dtype(self.physical_type, self.datatype)

    def proposals(self) -> Iterator[Proposal]:
        from .checks import normalise
        from .ledger import Proposal

        subj = str(self.subject)
        for attr in ("datatype", "physical_type", "description", "label", "confidentiality", "classification",
                     "synonyms", "examples", "ai_instructions", "owner", "name_patterns"):
            value = getattr(self, attr)
            if value is not None and value != []:
                yield Proposal(subj, attr, value)
        if self.checks:
            yield Proposal(subj, "checks", normalise(self.checks))

    @classmethod
    def load(cls, gloss: Gloss, name: str, *, include_pending: bool = False) -> SemanticType:
        ref = SubjectId.for_type(name).ref
        state = gloss.snapshot(read_statuses(include_pending), datasets=[], relationships=False)
        if ref not in state.types:
            raise KeyError(f"No semantic type {name!r}")
        return cls._from_attrs(ref, state.types[ref].attrs)

    @classmethod
    def _from_attrs(cls, ref: str, attrs: dict[str, Any]) -> SemanticType:
        dtype = parse_dtype(attrs.get("physical_type"), attrs.get("datatype"))
        keep = {f for f in cls.__dataclass_fields__} - {"name", "dtype", "namespace"}
        return cls(ref, dtype=dtype, **{k: v for k, v in attrs.items() if k in keep})

    def __hash__(self) -> int:
        return hash(("semantic_type", self.ref))


@dataclass(eq=False)  # identity semantics: columns are hashable and can key `on={...}`
class Column:
    name: str
    datatype: DatatypeLike = None  # stored as the Ossie name after construction
    expression: str | dict[str, Any] | None = None  # defaults to the column name on export
    description: str | None = None
    label: str | None = None
    is_time: bool | None = None
    synonyms: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    ai_instructions: str | None = None
    confidentiality: str | None = None
    classification: list[str] = field(default_factory=list)
    owner: str | None = None
    role: str | None = None  # what the column is to a metric: "key", "time", "category" or "amount"
    physical_type: str | None = None  # e.g. "Int32"; filled in from a Gloss type like Int(32)
    semantic_type: SemanticType | str | None = None  # also accepted as the datatype: Column("email", Email)
    checks: list[Any] = field(default_factory=list)  # in addition to the semantic type's checks
    table: str | None = field(default=None, repr=False)  # set when the column is added to a Table

    def __post_init__(self) -> None:
        if isinstance(self.datatype, SemanticType):
            if self.semantic_type is not None and self.semantic_type is not self.datatype:
                raise ValueError("Give the semantic type once: as the datatype or as semantic_type=")
            self.semantic_type, self.datatype = self.datatype, None
        ossie, physical = _datatype(self.datatype)
        self.datatype = ossie
        if self.physical_type is None:
            self.physical_type = physical

    @property
    def semantic_type_name(self) -> str | None:
        st = self.semantic_type
        return st.ref if isinstance(st, SemanticType) else st

    @property
    def dtype(self) -> DType | None:
        """The precise type, e.g. ``Int(32)``; falls back to the semantic type's, then the Ossie type."""
        own = parse_dtype(self.physical_type, self.datatype)  # type: ignore[arg-type]
        if own is None and isinstance(self.semantic_type, SemanticType):
            return self.semantic_type.resolved_dtype
        return own

    def facts(self) -> Iterator[tuple[str, Any]]:
        from .checks import normalise

        yield from _facts(
            self, ("expression", "datatype", "physical_type", "is_time", "label", *_COMMON, "classification",
                   "role")
        )
        if self.semantic_type_name:
            yield "semantic_type", self.semantic_type_name
        if self.checks:
            yield "checks", normalise(self.checks)

    @property
    def qualified_name(self) -> str:
        return f"{self.table}.{self.name}" if self.table else self.name


@dataclass
class Metric:
    """A number with an agreed definition.

    ``expression`` is the SQL that computes it and is all Ossie needs. The structured parts are
    optional and let a host answer "this metric by these dimensions over this period" without
    parsing SQL: ``aggregate`` (``"sum(amount)"``), ``filter``, ``time_column``, the allowed
    ``dimensions`` and a number ``format``. Compiling a query from them is the host's job.
    """

    name: str
    expression: str | dict[str, Any]
    description: str | None = None
    datatype: DatatypeLike = None
    synonyms: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    ai_instructions: str | None = None
    confidentiality: str | None = None
    owner: str | None = None
    table: str | None = None  # set for a metric that belongs to one table
    namespace: str | None = None  # set for a metric owned by a namespace
    aggregate: str | None = None
    filter: str | None = None
    time_column: str | None = None
    dimensions: list[str] = field(default_factory=list)
    format: str | None = None
    version: int | None = field(default=None, compare=False, repr=False)  # set when read from a ledger

    def __post_init__(self) -> None:
        self.datatype = _datatype(self.datatype)[0]
        sid = SubjectId.for_metric(self.name, self.table, namespace=self.namespace)
        self.name, self.namespace = sid.name or "", sid.namespace

    @property
    def subject(self) -> SubjectId:
        return SubjectId.for_metric(self.name, self.table, namespace=self.namespace)

    def facts(self) -> Iterator[tuple[str, Any]]:
        yield from _facts(
            self, ("expression", "datatype", *_COMMON, "aggregate", "filter", "time_column", "dimensions", "format")
        )

    def proposals(self) -> Iterator[Proposal]:
        from .ledger import Proposal

        group = new_id()  # a metric's facts are reviewed as one definition
        for attr, value in self.facts():
            yield Proposal(str(self.subject), attr, value, group=group)


ColumnRef = Union["Column", str]
TableRef = Union["Table", str]


def _table_key(t: TableRef | None) -> str | None:
    return t.key if isinstance(t, Table) else t


@dataclass
class Relationship:
    """A join: many ``from_table`` rows to one ``to_table`` row.

    The tables can be left out when the columns already belong to tables:

        Relationship("returns_orders", on={returns["order_id"]: orders["order_id"]})
        Relationship("returns_orders", "shop.returns", "shop.orders", on={"order_id": "order_id"})
    """

    name: str
    from_table: TableRef | None = None
    to_table: TableRef | None = None
    on: Mapping[ColumnRef, ColumnRef] | None = None
    from_columns: list[ColumnRef] = field(default_factory=list)
    to_columns: list[ColumnRef] = field(default_factory=list)
    description: str | None = None
    ai_instructions: str | None = None
    namespace: str | None = None  # the namespace that owns the relationship
    version: int | None = field(default=None, compare=False, repr=False)  # set when read from a ledger

    @property
    def subject(self) -> SubjectId:
        return SubjectId.for_relationship(self.name, self.namespace)

    def __post_init__(self) -> None:
        sid = SubjectId.for_relationship(self.name, self.namespace)
        self.name, self.namespace = sid.name or "", sid.namespace
        if self.on:
            self.from_columns = list(self.on.keys())
            self.to_columns = list(self.on.values())
        if not self.from_columns or len(self.from_columns) != len(self.to_columns):
            raise ValueError(f"Relationship {self.name}: give `on` or matching from_columns/to_columns")
        self.from_table, self.from_columns = self._resolve("from", self.from_table, self.from_columns)
        self.to_table, self.to_columns = self._resolve("to", self.to_table, self.to_columns)

    def _resolve(self, side: str, table: TableRef | None, cols: list[ColumnRef]) -> tuple[str, list[str]]:
        key = _table_key(table)
        owners = {c.table for c in cols if isinstance(c, Column)}
        if None in owners:
            raise ValueError(f"Relationship {self.name}: {side}-columns must be added to a Table first")
        if len(owners) > 1:
            raise ValueError(f"Relationship {self.name}: {side}-columns come from different tables {sorted(owners)}")
        if owners:
            owner = owners.pop()
            if key is not None and key != owner:
                raise ValueError(f"Relationship {self.name}: {side}_table {key!r} but columns belong to {owner!r}")
            key = owner
        if key is None:
            raise ValueError(f"Relationship {self.name}: give {side}_table or use columns that belong to a Table")
        return key, [c.name if isinstance(c, Column) else c for c in cols]

    def proposals(self) -> Iterator[Proposal]:
        from .ledger import Proposal

        subj = str(self.subject)
        group = new_id()  # the four facts of a join only make sense together
        yield Proposal(subj, "from", self.from_table, group=group)
        yield Proposal(subj, "to", self.to_table, group=group)
        yield Proposal(subj, "from_columns", list(self.from_columns), group=group)
        yield Proposal(subj, "to_columns", list(self.to_columns), group=group)
        for attr, value in _facts(self, ("description", "ai_instructions")):
            yield Proposal(subj, attr, value, group=group)


@dataclass
class Table:
    key: str  # Gloss dataset key, e.g. "shop.returns"
    source: str | None = None  # physical table/view; defaults to the key on export
    description: str | None = None
    primary_key: list[ColumnRef] | ColumnRef = field(default_factory=list)  # Columns are added to the table
    unique_keys: list[list[ColumnRef]] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    ai_instructions: str | None = None
    confidentiality: str | None = None
    classification: list[str] = field(default_factory=list)
    owner: str | None = None
    columns: dict[str, Column] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cols, self.columns = self.columns, {}
        for c in cols.values() if isinstance(cols, dict) else cols:
            self.add_column(c)
        self.primary_key = self._key_names(self.primary_key)
        self.unique_keys = [self._key_names(k) for k in self.unique_keys]

    def _key_names(self, items: list[ColumnRef] | ColumnRef) -> list[str]:
        """Turn a key given as Columns and/or names into names, adding new Columns to the table."""
        if isinstance(items, (Column, str)):
            items = [items]
        names = []
        for item in items:
            if isinstance(item, Column):
                if self.columns.get(item.name) is not item:
                    self.add_column(item)
                names.append(item.name)
            else:
                names.append(item)
        return names

    # -- building -----------------------------------------------------------
    def add_column(self, column: Column | str, datatype: DatatypeLike = None, **attrs: Any) -> Table:
        """``add_column(Column("id", "Integer"))`` or the short form ``add_column("id", int)``."""
        if isinstance(column, str):
            column = Column(column, datatype, **attrs)
        elif datatype is not None or attrs:
            raise TypeError("Pass either a Column or a name with attributes, not both")
        if column.name in self.columns:
            raise ValueError(f"Table {self.key} already has a column {column.name!r}")
        if column.table not in (None, self.key):
            raise ValueError(
                f"Column {column.name!r} already belongs to {column.table!r}; "
                "use dataclasses.replace(column, table=None) to copy it"
            )
        column.table = self.key
        self.columns[column.name] = column
        return self

    def add_columns(self, *columns: Column) -> Table:
        for c in columns:
            self.add_column(c)
        return self

    def add_metric(self, metric: Metric) -> Table:
        metric.table = self.key
        self.metrics[metric.name] = metric
        return self

    def column(self, name: str) -> Column:
        return self.columns[name]

    def __getitem__(self, name: str) -> Column:
        return self.columns[name]

    def __contains__(self, name: object) -> bool:
        return name in self.columns

    def __iter__(self) -> Iterator[Column]:
        return iter(self.columns.values())

    def __len__(self) -> int:
        return len(self.columns)

    # -- saving -------------------------------------------------------------
    def proposals(self) -> Iterator[Proposal]:
        from .ledger import Proposal

        ds = str(SubjectId.for_dataset(self.key))
        for attr, value in _facts(
            self, ("source", "primary_key", "unique_keys", *_COMMON, "classification")
        ):
            yield Proposal(ds, attr, value)
        seen_types: set[str] = set()
        for col in self.columns.values():
            st = col.semantic_type
            if isinstance(st, SemanticType) and st.ref not in seen_types:
                seen_types.add(st.ref)
                yield from st.proposals()
        for col in self.columns.values():
            subj = str(SubjectId.for_field(self.key, col.name))
            facts = list(col.facts())
            if not any(a in ("expression", "datatype") for a, _ in facts):
                facts.insert(0, ("expression", {"sql": col.name}))  # a column must leave a trace
            for attr, value in facts:
                yield Proposal(subj, attr, value)
        for m in self.metrics.values():
            yield from m.proposals()

    # -- loading ------------------------------------------------------------
    @classmethod
    def load(cls, gloss: Gloss, key: str, *, include_pending: bool = False) -> Table:
        """Build a Table from the ledger (accepted facts; optionally open proposals too)."""
        statuses = read_statuses(include_pending)
        state = gloss.snapshot(statuses, datasets=[key], relationships=False)
        if key not in state.datasets:
            raise KeyError(
                f"No table {key!r} with facts in {[s.value for s in statuses]}"
                + ("" if include_pending else " (pass include_pending=True to see proposals)")
            )
        return cls._from_state(key, state, types_from_state(state))

    @classmethod
    def _from_state(cls, key: str, state: Any, types: dict[str, SemanticType]) -> Table:
        d = state.datasets[key]
        known = {f for f in cls.__dataclass_fields__ if f not in ("key", "columns", "metrics")}
        t = cls(key, **{k: v for k, v in d.attrs.items() if k in known})
        col_fields = set(Column.__dataclass_fields__) - {"name", "table", "semantic_type"}
        for name, f in d.fields.items():
            own = f.own_attrs
            col = Column(name, **{k: v for k, v in own.items() if k in col_fields})
            col.checks = [c for i, c in enumerate(f.attrs.get("checks") or []) if i not in f.check_origins]
            tname = own.get("semantic_type")
            col.semantic_type = types.get(tname, tname) if tname else None
            t.add_column(col)
        for name, m in d.metrics.items():
            met = metric_from_attrs(name, m.attrs, table=key)
            if met is not None:
                t.metrics[name] = met
        return t

    def __repr__(self) -> str:
        cols = ", ".join(f"{c.name}:{c.dtype or '?'}" for c in self.columns.values())
        return f"Table({self.key!r}, columns=[{cols}])"


# ---------------------------------------------------------------------- reading helpers
def read_statuses(include_pending: bool) -> list[Status]:
    return [Status.ACCEPTED] + ([Status.PROPOSED, Status.REVIEWED] if include_pending else [])


def types_from_state(state: Any) -> dict[str, SemanticType]:
    return {n: SemanticType._from_attrs(n, e.attrs) for n, e in state.types.items()}


def metric_from_attrs(name: str, attrs: dict[str, Any], *, table: str | None = None) -> Metric | None:
    """A Metric from snapshot attributes; None while it has no expression yet.

    ``name`` is the bare name for a table's metric, otherwise a reference like ``sales::revenue``.
    """
    if "expression" not in attrs:
        return None
    keep = set(Metric.__dataclass_fields__) - {"name", "table", "namespace", "version"}
    return Metric(name, table=table, **{k: v for k, v in attrs.items() if k in keep})


def relationship_from_attrs(name: str, attrs: dict[str, Any]) -> Relationship | None:
    """A Relationship from snapshot attributes; None while its facts do not form a valid join."""
    if not all(attrs.get(k) for k in ("from", "to", "from_columns", "to_columns")):
        return None
    try:
        return Relationship(
            name,
            attrs["from"],
            attrs["to"],
            from_columns=list(attrs["from_columns"]),
            to_columns=list(attrs["to_columns"]),
            description=attrs.get("description"),
            ai_instructions=attrs.get("ai_instructions"),
        )
    except ValueError:
        return None


@dataclass
class AddResult:
    created: int = 0
    corroborated: int = 0
    unchanged: int = 0
    previously_rejected: int = 0
    accepted: int = 0
    retracted: int = 0  # by Gloss.apply: stated before, no longer declared
    assertion_ids: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        parts = [f"{self.created} new", f"{self.unchanged} unchanged"]
        if self.corroborated:
            parts.append(f"{self.corroborated} corroborated")
        if self.previously_rejected:
            parts.append(f"{self.previously_rejected} previously rejected")
        if self.accepted:
            parts.append(f"{self.accepted} accepted")
        if self.retracted:
            parts.append(f"{self.retracted} retracted")
        return ", ".join(parts)
