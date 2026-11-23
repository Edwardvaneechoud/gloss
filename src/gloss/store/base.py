"""Value types shared by the store and its callers.

The store itself is SQLAlchemy (:class:`gloss.store.sql.SQLStore`, on SQLite or Postgres).
There is no pluggable storage protocol: the ledger relies on transactions, indexes and a
search index that only a relational store provides, and one tested implementation is more
honest than a seam nobody exercises.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..model import Status

if TYPE_CHECKING:
    from ..policy import Scope


@dataclass(frozen=True)
class Query:
    """Filter for assertions. All criteria are AND-ed; ``None`` means 'any'."""

    subject: str | None = None
    subject_prefix: str | None = None  # e.g. "field:sales.orders/" or "dataset:sales."
    dataset: str | None = None  # dataset key the subject belongs to (dataset, fields, dataset metrics)
    dataset_prefix: str | None = None  # plain prefix of the dataset key, e.g. "sales."
    namespace: str | None = None  # dotted path: subjects at or under it (dataset keys and namespaces)
    kind: str | None = None
    attribute: str | None = None
    attributes: Sequence[str] | None = None
    statuses: Sequence[Status] | None = None
    source_prefix: str | None = None  # e.g. "ai:" or "import:ossie"
    value_hash: str | None = None
    group: str | None = None
    scope: Scope | None = None  # what the reader may see; None reads everything
    limit: int | None = None
    offset: int = 0


@dataclass(frozen=True)
class AssetRecord:
    """Last known physical shape of an asset, used for drift detection."""

    key: str
    catalog: str
    fingerprint: str
    schema: dict[str, Any]
    uid: str | None = None  # the catalog's own id for the asset, when it has one
