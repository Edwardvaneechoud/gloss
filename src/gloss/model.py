"""Core data model: assertions (facts), events (the log) and the review workflow.

An *assertion* is one small piece of semantic knowledge: "the description of
field sales.orders/amount is 'Order value incl. VAT'", proposed by some
source with some confidence. Assertions are never edited in place. Every
change is an *event* appended to the log; the current state of an assertion
is a pure fold over its events (see :func:`apply_event`).
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .ids import NS_SEP, Kind, SubjectId


class Status(str, Enum):
    PROPOSED = "proposed"  # modelled, not yet reviewed
    REVIEWED = "reviewed"  # a reviewer looked at it and approved; awaiting acceptance
    ACCEPTED = "accepted"  # part of the published model
    REJECTED = "rejected"
    SUPERSEDED = "superseded"  # replaced by a newer accepted value
    STALE = "stale"  # the underlying data changed; needs re-review
    RETRACTED = "retracted"  # withdrawn by its source


OPEN_STATUSES = frozenset({Status.PROPOSED, Status.REVIEWED, Status.STALE})
LIVE_STATUSES = OPEN_STATUSES | {Status.ACCEPTED}


class InvalidTransition(ValueError):
    pass


class Confidentiality(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

    @property
    def rank(self) -> int:
        return _CONF_ORDER.index(self)

    @classmethod
    def from_rank(cls, rank: int) -> Confidentiality:
        return _CONF_ORDER[rank]

    def __ge__(self, other: Confidentiality) -> bool:  # type: ignore[override]
        return self.rank >= Confidentiality(other).rank

    def __gt__(self, other: Confidentiality) -> bool:  # type: ignore[override]
        return self.rank > Confidentiality(other).rank

    def __le__(self, other: Confidentiality) -> bool:  # type: ignore[override]
        return self.rank <= Confidentiality(other).rank

    def __lt__(self, other: Confidentiality) -> bool:  # type: ignore[override]
        return self.rank < Confidentiality(other).rank


_CONF_ORDER = [
    Confidentiality.PUBLIC,
    Confidentiality.INTERNAL,
    Confidentiality.CONFIDENTIAL,
    Confidentiality.RESTRICTED,
]

DATATYPES = frozenset(
    {"String", "Integer", "Decimal", "Float", "Boolean", "Date", "Time", "DateTime", "DateTimeTz", "Opaque"}
)


@dataclass(frozen=True)
class AttributeSpec:
    """One attribute a fact can state, and how the rest of Gloss treats it.

    ``inherited``   a field takes it from its semantic type when it does not state it itself;
    ``detail``      its value is hidden from readers below the subject's confidentiality level;
    ``searchable``  its words go into the search index (gated by level when ``detail``);
    ``ossie``       Ossie has a native place for it; otherwise it travels in ``nonportable_extensions``;
    ``structural``  it describes physical structure, which a catalog is the authority on.
    """

    name: str
    kinds: frozenset[Kind]
    validate: Callable[[Any], Any] | None = None
    inherited: bool = False
    detail: bool = False
    searchable: bool = False
    ossie: bool = False
    structural: bool = False


_ATTRIBUTES: dict[str, AttributeSpec] = {}


def register_attribute(
    name: str,
    kinds: Iterable[Kind | str],
    *,
    validate: Callable[[Any], Any] | None = None,
    inherited: bool = False,
    detail: bool = False,
    searchable: bool = False,
    ossie: bool = False,
    structural: bool = False,
) -> AttributeSpec:
    """Make an attribute known, so hosts can extend the vocabulary without changing Gloss.

        gloss.register_attribute("time_column", ["metric"], validate=str)

    Registering a name again replaces its definition. Facts stating an attribute this process
    does not know stay readable; they just cannot be proposed until it is registered.
    """
    if not name or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"Attribute name {name!r} must be lower snake case")
    spec = AttributeSpec(
        name, frozenset(Kind(k) for k in kinds), validate, inherited, detail, searchable, ossie, structural
    )
    if not spec.kinds:
        raise ValueError(f"Attribute {name!r} needs at least one subject kind")
    _ATTRIBUTES[name] = spec
    return spec


def attribute(name: str) -> AttributeSpec | None:
    return _ATTRIBUTES.get(name)


def attributes_for(kind: Kind) -> frozenset[str]:
    return frozenset(n for n, s in _ATTRIBUTES.items() if kind in s.kinds)


def inherited_attributes() -> tuple[str, ...]:
    """What a field inherits from its semantic type when it doesn't state the attribute itself."""
    return tuple(n for n, s in _ATTRIBUTES.items() if s.inherited)


def detail_attributes() -> frozenset[str]:
    return frozenset(n for n, s in _ATTRIBUTES.items() if s.detail)


def is_structural(subject: SubjectId, attribute_name: str) -> bool:
    """Physical structure of a dataset or field: the only facts a catalog may accept on its own."""
    spec = _ATTRIBUTES.get(attribute_name)
    return bool(spec and spec.structural and subject.kind in (Kind.DATASET, Kind.FIELD))


class _AttributesByKind(Mapping[Kind, frozenset[str]]):
    """Read-only ``{kind: attribute names}`` view of the registry."""

    def __getitem__(self, kind: Kind) -> frozenset[str]:
        return attributes_for(Kind(kind))

    def __iter__(self) -> Iterator[Kind]:
        return iter(Kind)

    def __len__(self) -> int:
        return len(Kind)


ATTRIBUTES: Mapping[Kind, frozenset[str]] = _AttributesByKind()


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("must be a non-empty string")
    return value


def _str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("must be a list of strings")
    return value


def _key_lists(value: Any) -> list[list[str]]:
    if not isinstance(value, list) or not all(isinstance(k, list) for k in value):
        raise ValueError("must be a list of lists of field names")
    return value


def _confidentiality(value: Any) -> str:
    return Confidentiality(value).value


def _datatype(value: Any) -> str:
    if value not in DATATYPES:
        raise ValueError(f"must be one of {sorted(DATATYPES)}, got {value!r}")
    return value


def _type_ref(value: Any) -> str:
    if not isinstance(value, str) or not value or "/" in value or ":" in value.replace(NS_SEP, ""):
        raise ValueError("must be a type name like 'email_address' or 'crm::email_address'")
    return SubjectId.for_type(value).ref


def _checks(value: Any) -> list[dict[str, Any]]:
    from .checks import normalise

    if not isinstance(value, list):
        raise ValueError("must be a list")
    return normalise(value)


def _patterns(value: Any) -> list[str]:
    value = _str_list(value)
    for p in value:
        re.compile(p)
    return value


def _bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("must be a boolean")
    return value


ROLES = frozenset({"key", "time", "category", "amount"})


def _role(value: Any) -> str:
    if value not in ROLES:
        raise ValueError(f"must be one of {sorted(ROLES)}, got {value!r}")
    return value


def normalise_expression(value: Any) -> dict[str, Any]:
    """Accept a bare SQL string or an Ossie Expression object."""
    if isinstance(value, str):
        return {"sql": value}
    if isinstance(value, dict) and ("sql" in value or "dialects" in value):
        return value
    raise ValueError("expression must be a SQL string or an Ossie expression object {sql|dialects}")


def _register_builtins() -> None:
    K = Kind
    everything = list(Kind)
    described = [K.DATASET, K.FIELD, K.RELATIONSHIP, K.METRIC, K.TYPE]
    reg = register_attribute
    # Registration order is the order fields inherit in and proposals are listed in.
    reg("name", [K.MODEL], validate=_text, ossie=True)
    reg("datatype", [K.FIELD, K.METRIC, K.TYPE], validate=_datatype, inherited=True, ossie=True, structural=True)
    reg("physical_type", [K.FIELD, K.TYPE], validate=_text, inherited=True, structural=True)
    reg("description", everything, inherited=True, detail=True, searchable=True, ossie=True)
    reg("label", [K.FIELD, K.TYPE], inherited=True, detail=True, searchable=True, ossie=True)
    reg("confidentiality", described, validate=_confidentiality, inherited=True)
    reg("classification", [K.DATASET, K.FIELD, K.TYPE], validate=_str_list, inherited=True, detail=True,
        searchable=True)
    reg("synonyms", everything, validate=_str_list, inherited=True, detail=True, searchable=True, ossie=True)
    reg("examples", everything, validate=_str_list, inherited=True, detail=True, searchable=True, ossie=True)
    reg("ai_instructions", everything, inherited=True, detail=True, searchable=True, ossie=True)
    reg("owner", everything, inherited=True)
    reg("source", [K.DATASET], searchable=True, ossie=True, structural=True)
    reg("primary_key", [K.DATASET], validate=_str_list, ossie=True, structural=True)
    reg("unique_keys", [K.DATASET], validate=_key_lists, ossie=True)
    reg("expression", [K.FIELD, K.METRIC], validate=normalise_expression, detail=True, ossie=True, structural=True)
    reg("is_time", [K.FIELD], validate=_bool, ossie=True)
    reg("semantic_type", [K.FIELD], validate=_type_ref)
    reg("checks", [K.FIELD, K.TYPE], validate=_checks)
    reg("name_patterns", [K.TYPE], validate=_patterns)
    reg("from", [K.RELATIONSHIP], validate=_text, searchable=True, ossie=True)
    reg("to", [K.RELATIONSHIP], validate=_text, searchable=True, ossie=True)
    reg("from_columns", [K.RELATIONSHIP], validate=_str_list, ossie=True)
    reg("to_columns", [K.RELATIONSHIP], validate=_str_list, ossie=True)
    # A metric's structured definition, beside its SQL expression. Ossie has no place for these yet.
    reg("aggregate", [K.METRIC], validate=_text, detail=True)
    reg("filter", [K.METRIC], validate=_text, detail=True)
    reg("time_column", [K.METRIC], validate=_text)
    reg("dimensions", [K.METRIC], validate=_str_list)
    reg("format", [K.METRIC], validate=_text)
    reg("role", [K.FIELD], validate=_role)


_register_builtins()

# Attributes whose value names another subject. A re-key rewrites them so references keep resolving.
DATASET_REFERENCES = ("from", "to")

# Subjects whose facts only make sense together: a group of proposals for one is decided as a unit.
ATOMIC_KINDS = frozenset({Kind.RELATIONSHIP, Kind.METRIC})


def validate_value(subject: SubjectId, attribute_name: str, value: Any) -> Any:
    """Check an attribute is allowed for the subject kind and normalise its value."""
    spec = _ATTRIBUTES.get(attribute_name)
    if spec is None or subject.kind not in spec.kinds:
        allowed = ", ".join(sorted(attributes_for(subject.kind)))
        raise ValueError(
            f"Attribute {attribute_name!r} is not valid for {subject.kind.value} subjects (allowed: {allowed})"
        )
    if spec.validate is None:
        return value
    try:
        return spec.validate(value)
    except ValueError as exc:
        raise ValueError(f"{attribute_name} {exc}" if not str(exc).startswith(attribute_name) else str(exc)) from None


def value_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class EventType(str, Enum):
    PROPOSED = "proposed"
    CORROBORATED = "corroborated"  # another source stated the same value
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REOPENED = "reopened"
    SUPERSEDED = "superseded"
    MARKED_STALE = "marked_stale"
    RETRACTED = "retracted"
    RESTORED = "restored"  # a superseded value was put back
    REKEYED = "rekeyed"  # the subject (or a subject its value names) was renamed or moved
    COMMENTED = "commented"


EVENT_TARGET_STATUS: dict[EventType, Status] = {
    EventType.REVIEWED: Status.REVIEWED,
    EventType.ACCEPTED: Status.ACCEPTED,
    EventType.REJECTED: Status.REJECTED,
    EventType.REOPENED: Status.PROPOSED,
    EventType.SUPERSEDED: Status.SUPERSEDED,
    EventType.MARKED_STALE: Status.STALE,
    EventType.RETRACTED: Status.RETRACTED,
    EventType.RESTORED: Status.ACCEPTED,
}

# The statuses each event may be applied from. Superseded values come back only through `restored`.
EVENT_FROM_STATUSES: dict[EventType, frozenset[Status]] = {
    EventType.REVIEWED: frozenset({Status.PROPOSED, Status.STALE}),
    EventType.ACCEPTED: frozenset({Status.PROPOSED, Status.REVIEWED, Status.STALE}),
    EventType.REJECTED: frozenset({Status.PROPOSED, Status.REVIEWED, Status.STALE}),
    EventType.REOPENED: frozenset({Status.REJECTED, Status.REVIEWED}),
    EventType.SUPERSEDED: frozenset({Status.ACCEPTED}),
    EventType.MARKED_STALE: frozenset({Status.PROPOSED, Status.REVIEWED, Status.ACCEPTED}),
    EventType.RETRACTED: frozenset({Status.PROPOSED, Status.REVIEWED, Status.ACCEPTED, Status.STALE}),
    EventType.RESTORED: frozenset({Status.SUPERSEDED}),
}


def _transitions() -> dict[Status, frozenset[Status]]:
    out: dict[Status, set[Status]] = {s: set() for s in Status}
    for etype, sources in EVENT_FROM_STATUSES.items():
        for s in sources:
            out[s].add(EVENT_TARGET_STATUS[etype])
    return {s: frozenset(t) for s, t in out.items()}


TRANSITIONS: dict[Status, frozenset[Status]] = _transitions()


def can_apply(event_type: EventType, status: Status) -> bool:
    return status in EVENT_FROM_STATUSES.get(event_type, frozenset())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class Event:
    type: EventType
    assertion_id: str
    actor: str
    data: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=new_id)
    ts: datetime = field(default_factory=utcnow)
    seq: int | None = None  # assigned by the store
    correlation: str | None = None  # groups the events of one request, sync run or bulk action
    via: str | None = None  # the channel the actor came through: an agent, a server

    def __repr__(self) -> str:
        note = self.data.get("comment") or self.data.get("reason")
        extra = f" {note!r}" if note else ""
        seq = f"#{self.seq} " if self.seq is not None else ""
        return f"<{seq}{self.type.value} {self.assertion_id[:8]} by {self.actor} {self.ts:%Y-%m-%d %H:%M:%S}{extra}>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "id": self.id,
            "ts": self.ts.isoformat(),
            "type": self.type.value,
            "assertion_id": self.assertion_id,
            "actor": self.actor,
            "via": self.via,
            "correlation": self.correlation,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Event:
        return cls(
            EventType(d["type"]), d["assertion_id"], d["actor"], d.get("data") or {}, d["id"],
            datetime.fromisoformat(d["ts"]), d.get("seq"), d.get("correlation"), d.get("via"),
        )


@dataclass(frozen=True)
class Assertion:
    id: str
    subject: str
    attribute: str
    value: Any
    source: str
    status: Status
    confidence: float | None
    created_at: datetime
    updated_at: datetime
    created_by: str
    sources: tuple[str, ...] = ()  # all sources that stated this value (corroboration)
    evidence: tuple[dict[str, Any], ...] = ()
    last_comment: str | None = None
    group: str | None = None  # proposals made together; decided as a unit for relationships and metrics
    redacted: bool = False  # view-only: the value was withheld from this reader

    @property
    def subject_id(self) -> SubjectId:
        return SubjectId.parse(self.subject)

    @property
    def dataset(self) -> str | None:
        return self.subject_id.owning_dataset

    @property
    def scope_path(self) -> str | None:
        return self.subject_id.scope_path

    def __repr__(self) -> str:
        v = self.value if isinstance(self.value, str) else json.dumps(self.value, ensure_ascii=False)
        if len(v) > 50:
            v = v[:49] + "…"
        conf = f" {self.confidence:.2f}" if self.confidence is not None else ""
        return f"<{self.status.value} {self.subject} {self.attribute}={v!r} by {self.source}{conf} id={self.id[:8]}>"

    @property
    def source_actor(self):  # -> gloss.actors.Actor
        from .actors import Actor

        return Actor.parse(self.source)

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES

    def to_dict(self) -> dict[str, Any]:
        out = {
            "id": self.id,
            "subject": self.subject,
            "attribute": self.attribute,
            "value": self.value,
            "source": self.source,
            "sources": list(self.sources),
            "status": self.status.value,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "last_comment": self.last_comment,
            "group": self.group,
        }
        if self.redacted:
            out["redacted"] = True
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Assertion:
        return cls(
            id=d["id"], subject=d["subject"], attribute=d["attribute"], value=d.get("value"),
            source=d["source"], status=Status(d["status"]), confidence=d.get("confidence"),
            created_at=datetime.fromisoformat(d["created_at"]), updated_at=datetime.fromisoformat(d["updated_at"]),
            created_by=d["created_by"], sources=tuple(d.get("sources") or ()),
            evidence=tuple(d.get("evidence") or ()), last_comment=d.get("last_comment"),
            group=d.get("group"), redacted=bool(d.get("redacted")),
        )


def apply_event(current: Assertion | None, event: Event) -> Assertion:
    """Fold one event into an assertion's state. Pure: the projection is rebuildable from the log."""
    d = event.data
    if event.type is EventType.PROPOSED:
        if current is not None:
            raise InvalidTransition(f"Assertion {event.assertion_id} already exists")
        return Assertion(
            id=event.assertion_id,
            subject=d["subject"],
            attribute=d["attribute"],
            value=d["value"],
            source=d["source"],
            status=Status.PROPOSED,
            confidence=d.get("confidence"),
            created_at=event.ts,
            updated_at=event.ts,
            created_by=event.actor,
            sources=(d["source"],),
            evidence=tuple(d.get("evidence") or ()),
            last_comment=d.get("comment"),
            group=d.get("group"),
        )

    if current is None:
        raise InvalidTransition(f"Event {event.type.value} for unknown assertion {event.assertion_id}")

    if event.type is EventType.CORROBORATED:
        sources = current.sources if d["source"] in current.sources else current.sources + (d["source"],)
        conf = current.confidence
        if d.get("confidence") is not None:
            conf = max(conf or 0.0, d["confidence"])
        return replace(
            current,
            sources=sources,
            confidence=conf,
            evidence=current.evidence + tuple(d.get("evidence") or ()),
            updated_at=event.ts,
        )

    if event.type is EventType.COMMENTED:
        return replace(current, last_comment=d.get("comment"), updated_at=event.ts)

    if event.type is EventType.REKEYED:
        return replace(
            current,
            subject=d.get("subject", current.subject),
            value=d["value"] if "value" in d else current.value,
            updated_at=event.ts,
        )

    if not can_apply(event.type, current.status):
        target = EVENT_TARGET_STATUS[event.type]
        raise InvalidTransition(
            f"Cannot move assertion {current.id} from {current.status.value} to {target.value}"
        )
    return replace(
        current,
        status=EVENT_TARGET_STATUS[event.type],
        updated_at=event.ts,
        last_comment=d.get("comment", current.last_comment),
    )
