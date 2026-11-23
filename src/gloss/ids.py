"""Stable, catalog-independent identifiers for the things semantic facts are about.

Every fact attaches to a *subject*. Subjects are plain strings so they can be
stored, logged and passed to AI tools without ceremony:

    model:                               the semantic model as a whole
    model:sales                          the model published from the ``sales`` namespace
    dataset:sales.orders                 a dataset (table, view, query)
    field:sales.orders/amount            a field of a dataset
    metric:sales.orders/avg_price        a metric that belongs to one dataset
    metric:sales::total_revenue          a metric owned by the ``sales`` namespace
    relationship:sales::orders_customers a relationship owned by the ``sales`` namespace
    type:crm::email_address              a semantic type owned by the ``crm`` namespace
    type:email_address                   the same kinds without a namespace are ledger-wide

Dataset keys are free-form dotted paths; ``/`` separates a dataset key from a
member name and ``::`` separates a namespace from a name. A *namespace* is a
dotted path in the same space as dataset keys (``main.sales`` owns
``main.sales.orders``), so one prefix rule covers access, stewardship and moves
for every kind of subject.

Subject strings are written into the event log, which is never edited. When a
table is renamed or moved the catalog of record reports it and Gloss appends
re-key events (see ``Gloss.rekey``); the strings themselves stay name-based so
two machines that know the same table agree on its subject.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

NS_SEP = "::"


class Kind(str, Enum):
    MODEL = "model"
    DATASET = "dataset"
    FIELD = "field"
    RELATIONSHIP = "relationship"
    METRIC = "metric"
    TYPE = "type"


def _check_part(what: str, value: str, original: str) -> None:
    if not value or "/" in value or NS_SEP in value:
        raise ValueError(f"Invalid {what} {value!r} in subject {original!r}: it cannot be empty or contain '/' or '::'")


@dataclass(frozen=True)
class SubjectId:
    kind: Kind
    dataset: str | None = None  # dataset key, for datasets, fields and dataset metrics
    name: str | None = None  # member name (field, relationship, metric, type)
    namespace: str | None = None  # owning namespace, for models, relationships, metrics and types

    # -- construction -------------------------------------------------------
    @classmethod
    def parse(cls, value: str | SubjectId) -> SubjectId:
        if isinstance(value, SubjectId):
            return value
        if not isinstance(value, str) or ":" not in value:
            raise ValueError(f"Invalid subject {value!r}: expected '<kind>:<path>'")
        kind_s, path = value.split(":", 1)
        try:
            kind = Kind(kind_s)
        except ValueError:
            raise ValueError(f"Unknown subject kind {kind_s!r} in {value!r}") from None

        if kind is Kind.MODEL:
            if not path:
                return cls(kind)
            _check_part("namespace", path, value)
            return cls(kind, namespace=path)
        if not path:
            raise ValueError(f"Subject {value!r} has an empty path")
        if kind is Kind.DATASET:
            if NS_SEP in path:
                raise ValueError(f"Dataset subject {value!r} cannot contain '::'")
            return cls(kind, dataset=path)
        if kind is Kind.FIELD or (kind is Kind.METRIC and "/" in path):
            ds, sep, name = path.rpartition("/")
            if not sep or not ds or not name or NS_SEP in path:
                raise ValueError(f"{kind.value} subject {value!r} must look like '{kind.value}:<dataset>/<name>'")
            return cls(kind, dataset=ds, name=name)
        if "/" in path:
            raise ValueError(f"{kind.value} subject {value!r} cannot contain '/'")
        ns, sep, name = path.rpartition(NS_SEP)
        if sep:
            _check_part("namespace", ns, value)
            _check_part("name", name, value)
            return cls(kind, name=name, namespace=ns)
        return cls(kind, name=path)

    @classmethod
    def model(cls, namespace: str | None = None) -> SubjectId:
        return cls(Kind.MODEL, namespace=namespace or None)

    @classmethod
    def for_dataset(cls, key: str) -> SubjectId:
        return cls(Kind.DATASET, dataset=key)

    @classmethod
    def for_field(cls, dataset: str, name: str) -> SubjectId:
        return cls(Kind.FIELD, dataset=dataset, name=name)

    @classmethod
    def for_relationship(cls, name: str, namespace: str | None = None) -> SubjectId:
        return cls._named(Kind.RELATIONSHIP, name, namespace)

    @classmethod
    def for_type(cls, name: str, namespace: str | None = None) -> SubjectId:
        return cls._named(Kind.TYPE, name, namespace)

    @classmethod
    def for_metric(cls, name: str, dataset: str | None = None, *, namespace: str | None = None) -> SubjectId:
        if dataset and namespace:
            raise ValueError("A metric belongs to a dataset or to a namespace, not both")
        if dataset:
            return cls(Kind.METRIC, dataset=dataset, name=name)
        return cls._named(Kind.METRIC, name, namespace)

    @classmethod
    def _named(cls, kind: Kind, name: str, namespace: str | None) -> SubjectId:
        """``name`` may itself be a reference like ``"crm::email_address"``."""
        if namespace is None and NS_SEP in name:
            namespace, _, name = name.rpartition(NS_SEP)
        return cls.parse(f"{kind.value}:{namespace}{NS_SEP}{name}" if namespace else f"{kind.value}:{name}")

    # -- rendering ------------------------------------------------------------
    def __str__(self) -> str:
        k = self.kind.value
        if self.kind is Kind.MODEL:
            return f"{k}:{self.namespace or ''}"
        if self.kind is Kind.DATASET:
            return f"{k}:{self.dataset}"
        if self.dataset:
            return f"{k}:{self.dataset}/{self.name}"
        return f"{k}:{self.ref}"

    @property
    def ref(self) -> str:
        """How other facts refer to this subject: ``crm::email_address``, or the bare name."""
        if self.dataset and self.name:
            return f"{self.dataset}/{self.name}"
        if self.namespace and self.name:
            return f"{self.namespace}{NS_SEP}{self.name}"
        return self.name or self.dataset or self.namespace or ""

    @property
    def owning_dataset(self) -> str | None:
        """The dataset this subject belongs to, used for progress roll-ups and drift."""
        return self.dataset

    @property
    def scope_path(self) -> str | None:
        """The dotted path that access rules and moves apply to: the dataset key or the namespace.

        ``None`` for ledger-wide subjects (``model:``, ``type:email_address``).
        """
        return self.dataset or self.namespace

    def moved(self, old: str, new: str) -> SubjectId | None:
        """This subject after the path ``old`` became ``new``, or None when it is not under ``old``."""
        path = self.scope_path
        if path is None or not path_covers(old, path):
            return None
        target = replace_prefix(path, old, new)
        return replace(self, dataset=target) if self.dataset else replace(self, namespace=target)


def subject(value: str | SubjectId) -> SubjectId:
    return SubjectId.parse(value)


def path_covers(prefix: str, path: str | None) -> bool:
    """Whether ``path`` is ``prefix`` or lies under it, compared per dotted segment.

    ``"finance"`` and ``"finance."`` both cover ``finance`` and ``finance.invoices`` and neither
    covers ``financial.x``. The empty prefix covers everything, including ledger-wide subjects.
    """
    p = prefix.rstrip(".")
    if not p:
        return True
    if path is None:
        return False
    return path == p or path.startswith(p + ".")


def replace_prefix(path: str, old: str, new: str) -> str:
    """``path`` with its leading ``old`` segments replaced by ``new``."""
    o, n = old.rstrip("."), new.rstrip(".")
    return n + path[len(o):] if o else path
