"""One namespace's model with its review state beside it: what a screen that edits and reviews a model reads.

The other readers answer "what is the model" and merge statuses into one value per attribute (``Gloss.snapshot``,
``Gloss.tables``). A reviewer also needs what is waiting next to what is agreed, who stated it, why, and which
proposals are decided together. ``Gloss.model_view`` builds that for the tables a catalog lists:

    view = g.as_actor(Human("ann")).model_view("shop.sales", tables=catalog_tables, statuses=LIVE_STATUSES)
    for t in view.tables:
        t.value("description"), t.pending            # what is shown, and every open fact beside it
    view.metrics[0].status, view.metrics[0].pending_group, view.joins[0].version

What is *shown* is the accepted value, else the most recent open one. A metric or join shows one whole definition: the
accepted one when there is one (with ``pending_group`` naming an open edit of it), else the newest proposed group.
With accepted facts only (the default) the view is the published model, and it can be compiled from
(``glosslayer.query``). Reads are scoped and redacted like every other reader's, in a fixed number of queries.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from .actors import KNOWN_KINDS, actor_kind
from .ids import NS_SEP, Kind, SubjectId
from .model import ATOMIC_KINDS, DATASET_REFERENCES, OPEN_STATUSES, EventType, Status
from .references import short_name
from .seams import AssetSchema, Column

if TYPE_CHECKING:
    from .ledger import Gloss
    from .model import Assertion

_EVENT_CHUNK = 500
_METRIC_ATTRIBUTES = ("description", "aggregate", "filter", "time_column", "dimensions", "format", "synonyms")


class TableShape(Protocol):
    """What a catalog says about a table; ``AssetSchema`` fits, and so does any host type with these attributes.

    ``uid`` is the catalog's own id. A shape may also carry a ``name``; the key's last part is used otherwise.
    """

    key: str
    columns: Collection[Any]  # each with ``name`` and ``dtype``
    description: str | None
    uid: Any


@dataclass
class Fact:
    """One assertion with what its events say about it: ``via`` and ``rationale`` of the proposal (the channel it
    came through and its comment), and the last acceptance. ``redacted`` facts carry neither value nor rationale."""

    id: str
    subject: str
    attribute: str
    value: Any
    status: Status
    source: str
    created_by: str
    updated_at: datetime
    confidence: float | None = None
    group: str | None = None
    via: str | None = None
    rationale: str | None = None
    accepted_by: str | None = None
    accepted_at: datetime | None = None
    redacted: bool = False

    @property
    def source_kind(self) -> str:
        """The kind of actor that stated it (``human``, ``ai``, ``inferred``, …), or ``other``."""
        kind = actor_kind(self.source)
        return kind if kind in KNOWN_KINDS else "other"

    @property
    def is_open(self) -> bool:
        """Whether it still awaits a decision: proposed, reviewed or stale."""
        return self.status in OPEN_STATUSES


@dataclass
class ReviewEntry(Fact):
    """One decision to make. Its own fields are those of the fact, or of the first fact of a whole group, whose
    facts (this one included) are then in ``group_facts``."""

    subject_label: str = ""
    table_uid: Any = None
    group_facts: list[Fact] = field(default_factory=list)

    @property
    def subject_id(self) -> SubjectId:
        """The subject parsed into a :class:`~glosslayer.ids.SubjectId`."""
        return SubjectId.parse(self.subject)

    @property
    def subject_kind(self) -> str:
        """The kind of subject: ``field``, ``metric``, …"""
        return self.subject_id.kind.value

    @property
    def table(self) -> str | None:
        """The dataset key it is about, or None for a join, a model or a type."""
        return self.subject_id.dataset

    @property
    def column(self) -> str | None:
        """The column it is about, for a fact on a field."""
        sid = self.subject_id
        return sid.name if sid.kind is Kind.FIELD else None


@dataclass
class Described:
    """The shown fact per attribute, and every open fact on the subject (the shown ones included)."""

    facts: dict[str, Fact] = field(default_factory=dict)
    pending: list[Fact] = field(default_factory=list)

    def value(self, attribute: str) -> Any:
        """The shown value of ``attribute``, or None."""
        f = self.facts.get(attribute)
        return f.value if f is not None else None


@dataclass
class FieldView(Described):
    """A column: every column the catalog lists, then those only facts still name (``dtype`` None)."""

    name: str = ""
    dtype: str | None = None


@dataclass
class MetricView:
    """A metric's shown definition. ``accepted_by`` / ``accepted_at`` are of its expression, once accepted."""

    key: str
    name: str
    table: str | None
    status: Status
    table_uid: Any = None
    expression: str | None = None
    description: str | None = None
    aggregate: str | None = None
    filter: str | None = None
    time_column: str | None = None
    dimensions: list[str] = field(default_factory=list)
    format: str | None = None
    synonyms: list[str] = field(default_factory=list)
    version: int | None = None
    accepted_by: str | None = None
    accepted_at: datetime | None = None
    group: str | None = None
    pending_group: str | None = None


@dataclass
class JoinView:
    """A join's shown definition, ``from`` the many side ``to`` the one side; ``namespace`` is its owner."""

    key: str
    name: str
    namespace: str
    from_table: str
    to_table: str
    status: Status
    from_columns: list[str] = field(default_factory=list)
    to_columns: list[str] = field(default_factory=list)
    from_uid: Any = None
    to_uid: Any = None
    description: str | None = None
    version: int | None = None
    group: str | None = None


@dataclass
class TableView(Described):
    """A table of the model with its columns and its own metrics; ``description`` is the shown one, else the catalog's."""

    key: str = ""
    name: str = ""
    uid: Any = None
    description: str | None = None
    fields: list[FieldView] = field(default_factory=list)
    metrics: list[MetricView] = field(default_factory=list)

    def column(self, name: str) -> FieldView | None:
        """The column called ``name``, or None."""
        return next((c for c in self.fields if c.name == name), None)


@dataclass
class ModelView(Described):
    """The model of ``namespace``. ``facts`` and ``pending`` are the model's own (its name, its description);
    ``all_facts`` is everything read; ``columns`` the columns of every table known here, joined ones included."""

    namespace: str | None = None
    tables: list[TableView] = field(default_factory=list)
    joins: list[JoinView] = field(default_factory=list)
    own_metrics: list[MetricView] = field(default_factory=list)  # the namespace's, belonging to no table
    all_facts: list[Fact] = field(default_factory=list)
    columns: dict[str, set[str]] = field(default_factory=dict)

    def table(self, key_or_name: str) -> TableView | None:
        """The table with this dataset key or name, or None."""
        return next((t for t in self.tables if key_or_name in (t.key, t.name)), None)

    @property
    def metrics(self) -> list[MetricView]:
        """Every metric: the tables', then the namespace's own."""
        return [m for t in self.tables for m in t.metrics] + self.own_metrics

    def counts(self) -> dict[str, int]:
        """Live facts by what a person has to do: ``accepted``, ``pending`` (proposed, reviewed) and ``stale``.

        What the catalog itself accepted (column types and the like) is left out: nobody reviewed it.
        """
        return {
            "accepted": sum(f.status is Status.ACCEPTED and f.source_kind != "catalog" for f in self.all_facts),
            "pending": sum(f.status in (Status.PROPOSED, Status.REVIEWED) for f in self.all_facts),
            "stale": sum(f.status is Status.STALE for f in self.all_facts),
        }


# ---------------------------------------------------------------------- reading
def read_facts(gloss: Gloss, assertions: Iterable[Assertion]) -> list[Fact]:
    """``assertions`` as facts, in their order (see ``Gloss.facts``)."""
    facts = {
        a.id: Fact(a.id, a.subject, a.attribute, a.value, a.status, a.source, a.created_by, a.updated_at,
                   a.confidence, a.group, redacted=a.redacted)
        for a in assertions
    }
    ids = list(facts)
    for i in range(0, len(ids), _EVENT_CHUNK):
        for ev in gloss.store.events(assertion_ids=ids[i:i + _EVENT_CHUNK]):
            f = facts[ev.assertion_id]
            if ev.type is EventType.PROPOSED:
                f.via, f.rationale = ev.via, None if f.redacted else ev.data.get("comment")
            elif ev.type in (EventType.ACCEPTED, EventType.RESTORED):
                f.accepted_by, f.accepted_at = ev.actor, ev.ts
    return list(facts.values())


def _label(sid: SubjectId, table: str | None) -> str:
    """How an entry's subject is called: ``table.column``, the table, the namespace, else its own name."""
    if sid.kind is Kind.FIELD:
        return f"{table}.{sid.name}"
    if sid.kind is Kind.DATASET:
        return table or ""
    return (sid.namespace or "model") if sid.kind is Kind.MODEL else sid.name or ""


def review_queue(gloss: Gloss, namespace: str | None, tables: Iterable[TableShape]) -> list[ReviewEntry]:
    """The entries behind ``Gloss.review_queue``."""
    known = {t.key: t for t in tables}
    facts = read_facts(gloss, [a for a in gloss.pending(namespace=namespace) if not a.redacted])
    groups: dict[str, list[Fact]] = {}
    for f in sorted(facts, key=lambda f: f.updated_at):
        atomic = f.group and SubjectId.parse(f.subject).kind in ATOMIC_KINDS
        groups.setdefault(f.group if atomic else f.id, []).append(f)  # type: ignore[arg-type]
    out = []
    for key, rows in groups.items():
        head = rows[0]
        sid = SubjectId.parse(head.subject)
        t = known.get(sid.dataset or "")
        name = (getattr(t, "name", None) or short_name(t.key)) if t is not None else sid.dataset
        out.append(ReviewEntry(
            **vars(head), subject_label=_label(sid, name), table_uid=t.uid if t is not None else None,
            group_facts=rows if key != head.id else [],
        ))
    return out


def _best(rows: Iterable[Fact]) -> dict[str, Fact]:
    """One fact per attribute: accepted wins, then the most recently updated."""
    best: dict[str, Fact] = {}
    for f in rows:
        cur = best.get(f.attribute)
        if cur is None or (f.status is Status.ACCEPTED, f.updated_at) > (cur.status is Status.ACCEPTED, cur.updated_at):
            best[f.attribute] = f
    return best


def _open(rows: Iterable[Fact]) -> list[Fact]:
    return [f for f in rows if f.is_open]


def _definition(rows: list[Fact]) -> tuple[dict[str, Fact], Status, str | None, str | None]:
    """The shown definition's facts, its status, its group, and the group of an open edit beside it."""
    accepted = [f for f in rows if f.status is Status.ACCEPTED]
    open_rows = sorted(_open(rows), key=lambda f: f.updated_at, reverse=True)
    pending_group = open_rows[0].group if open_rows else None
    if accepted:
        group = next((f for f in accepted if f.attribute in ("expression", "from")), accepted[0]).group
        return _best(accepted), Status.ACCEPTED, group, pending_group if pending_group != group else None
    shown = [f for f in open_rows if f.group == pending_group] if pending_group else open_rows
    status = Status.STALE if any(f.status is Status.STALE for f in shown) else shown[0].status
    return _best(shown), status, pending_group, None


def _metric(sid: SubjectId, rows: list[Fact], uids: Mapping[str, Any], version: int | None) -> MetricView | None:
    facts, status, group, pending_group = _definition(rows)
    expr = facts.get("expression")
    if expr is None:
        return None
    accepted = status is Status.ACCEPTED
    return MetricView(
        str(sid), sid.name or "", sid.dataset, status, uids.get(sid.dataset or ""),
        expression=expr.value.get("sql") if isinstance(expr.value, dict) else expr.value,
        version=version,
        accepted_by=expr.accepted_by if accepted else None,
        accepted_at=expr.accepted_at if accepted else None,
        group=group,
        pending_group=pending_group,
        **{a: facts[a].value for a in _METRIC_ATTRIBUTES if a in facts and facts[a].value is not None},
    )


def _join(sid: SubjectId, rows: list[Fact], uids: Mapping[str, Any], version: int | None) -> JoinView | None:
    facts, status, group, _pending = _definition(rows)
    attrs = {a: f.value for a, f in facts.items()}
    if not all(attrs.get(k) for k in ("from", "to", "from_columns", "to_columns")):
        return None
    return JoinView(
        str(sid), sid.name or "", sid.namespace or "", attrs["from"], attrs["to"], status,
        attrs["from_columns"], attrs["to_columns"], uids.get(attrs["from"]), uids.get(attrs["to"]),
        attrs.get("description"), version, group,
    )


def _shapes_from(facts: Iterable[Fact]) -> list[AssetSchema]:
    """The tables and columns facts are about, for a ledger read without a catalog at hand."""
    found: dict[str, dict[str, str | None]] = {}
    for f in facts:
        sid = SubjectId.parse(f.subject)
        if sid.kind not in (Kind.DATASET, Kind.FIELD):
            continue
        columns = found.setdefault(sid.dataset or "", {})
        if sid.kind is Kind.FIELD:
            dtype = f.value if f.attribute in ("physical_type", "datatype") else None
            columns[sid.name or ""] = dtype or columns.get(sid.name or "")
    return [AssetSchema(k, [Column(n, d) for n, d in found[k].items()]) for k in sorted(found)]  # type: ignore[arg-type]


def build(
    gloss: Gloss,
    namespace: str | None = None,
    *,
    tables: Iterable[TableShape] | None = None,
    related: Iterable[TableShape] = (),
    joins_of: str | None = None,
    statuses: Iterable[Status] = (Status.ACCEPTED,),
) -> ModelView:
    """The view behind ``Gloss.model_view``."""
    wanted = [Status(s) for s in statuses]
    found = gloss.find(namespace=namespace, statuses=wanted)
    if joins_of:
        found += gloss.find(subject_prefix=f"{Kind.RELATIONSHIP.value}:{joins_of}{NS_SEP}", statuses=wanted)
    facts = read_facts(gloss, [a for a in found if not a.redacted])

    rows: dict[str, list[Fact]] = {}
    subjects: dict[str, SubjectId] = {}
    named: dict[str, set[str]] = {}  # columns that facts name, per dataset
    for f in facts:
        if f.subject not in subjects:
            sid = subjects[f.subject] = SubjectId.parse(f.subject)
            if sid.kind is Kind.FIELD:
                named.setdefault(sid.dataset or "", set()).add(sid.name or "")
        rows.setdefault(f.subject, []).append(f)

    shapes = list(tables) if tables is not None else _shapes_from(facts)
    keys = {t.key for t in shapes}
    known = {t.key: t for t in (*related, *shapes)}
    uids = {key: t.uid for key, t in known.items()}
    metrics = sorted(s for s, sid in subjects.items()
                     if sid.kind is Kind.METRIC and (sid.dataset is None or sid.dataset in keys))
    joins = sorted(s for s, sid in subjects.items() if sid.kind is Kind.RELATIONSHIP)
    versions = gloss.store.subject_versions(metrics + joins) if metrics or joins else {}

    own = rows.get(str(SubjectId.model(namespace)), [])
    view = ModelView(_best(own), _open(own), namespace, all_facts=facts)
    by_key: dict[str, TableView] = {}
    for t in shapes:
        own = rows.get(str(SubjectId.for_dataset(t.key)), [])
        shown = _best(own)
        table = by_key[t.key] = TableView(
            shown, _open(own), t.key, getattr(t, "name", None) or short_name(t.key), t.uid,
            shown["description"].value if "description" in shown else t.description,
        )
        dtypes = {c.name: c.dtype for c in t.columns}
        for name in [*dtypes, *sorted(named.get(t.key, set()) - dtypes.keys())]:
            own = rows.get(str(SubjectId.for_field(t.key, name)), [])
            table.fields.append(FieldView(_best(own), _open(own), name, dtypes.get(name)))
        view.tables.append(table)

    view.columns = {key: {c.name for c in t.columns} for key, t in known.items()}
    view.columns.update({t.key: {c.name for c in t.fields} for t in view.tables})
    for subject in metrics:
        m = _metric(subjects[subject], rows[subject], uids, versions.get(subject))
        if m is not None:
            (by_key[m.table].metrics if m.table else view.own_metrics).append(m)
    scope = gloss.scope
    for subject in joins:
        j = _join(subjects[subject], rows[subject], uids, versions.get(subject))
        if j is None or (joins_of and j.namespace == joins_of and not {j.from_table, j.to_table} & keys):
            continue
        # a join is only shown to a reader who may read both tables it joins
        if scope.all_paths or all(scope.covers(getattr(j, f"{side}_table")) for side in DATASET_REFERENCES):
            view.joins.append(j)
    return view
