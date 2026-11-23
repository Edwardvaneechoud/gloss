"""Plugin seams. Gloss's core knows nothing about any specific catalog or tool.

Catalog   tells Gloss which data objects exist and what they physically look like.
Source    produces proposals (never accepted facts). Imports, profilers, lineage, AI.
Sink      publishes the accepted state somewhere. Ossie files, git, back into a catalog.

Implementations are plain classes; they do not need to inherit from anything.
They are registered in code or discovered through Python entry points (see :mod:`gloss.plugins`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .ledger import Gloss, Proposal


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str  # the catalog's own type name; mapped to an Ossie datatype by gloss.types
    nullable: bool | None = None
    description: str | None = None


@dataclass
class AssetSchema:
    """What a catalog knows about one asset (table, view, file)."""

    key: str  # stable Gloss dataset key, e.g. "sales.orders"
    columns: list[Column]
    description: str | None = None
    source: str | None = None  # physical reference, e.g. "warehouse.sales.orders"
    primary_key: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # catalog-specific; kept as evidence
    uid: str | None = None  # the catalog's own id for the asset, if it has one (helps spot renames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "columns": [asdict(c) for c in self.columns],
            "description": self.description,
            "source": self.source,
            "primary_key": self.primary_key,
        }


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:24]


def schema_fingerprint(columns: Iterable[Column]) -> str:
    """The *structure* of an asset: column names and types, order-insensitive."""
    return _digest(sorted((c.name, c.dtype) for c in columns))


def asset_fingerprint(schema: AssetSchema) -> str:
    """Everything about an asset Gloss should look at again when it changes.

    Structure plus what the catalog says in words: descriptions, the primary key, the source.
    Only a change of structure sends facts back to review; a changed description is proposed.
    """
    return _digest({
        "columns": sorted((c.name, c.dtype, c.description) for c in schema.columns),
        "description": schema.description,
        "primary_key": schema.primary_key,
        "source": schema.source,
    })


@dataclass(frozen=True)
class Coverage:
    """What one listing of a catalog covers, so "not listed" is not mistaken for "gone".

    ``complete``  the listing is everything under ``prefixes``; assets known before and missing now
                  were removed. False for a listing filtered by what one user may see.
    ``prefixes``  the key prefixes the listing speaks for; ``None`` means the whole catalog.
    """

    complete: bool = True
    prefixes: tuple[str, ...] | None = None

    def speaks_for(self, key: str) -> bool:
        from .ids import path_covers

        if not self.complete:
            return False
        return self.prefixes is None or any(path_covers(p, key) for p in self.prefixes)


@runtime_checkable
class Catalog(Protocol):
    """Read side of a catalog. ``coverage`` and ``write_back`` are optional.

    A catalog may define ``coverage() -> Coverage`` to say what its listing speaks for. Without it
    the listing is taken as complete. Removal is only ever inferred inside that coverage; a catalog
    that knows the moment something is deleted or renamed should report it instead
    (``gloss.sync.asset_removed`` / ``asset_moved``).
    """

    name: str

    def list_assets(self) -> Iterable[str]:
        """Dataset keys of every asset in scope."""

    def describe(self, key: str) -> AssetSchema: ...

    def fingerprint(self, key: str) -> str:
        """Changes whenever Gloss should look at the asset again (see :func:`asset_fingerprint`).

        Data-only changes should not change it.
        """


@runtime_checkable
class WritableCatalog(Catalog, Protocol):
    def write_back(self, key: str, facts: dict[str, Any]) -> None:
        """Receive accepted facts for a dataset: ``{"description": ..., "fields": {name: {...}}}``."""


@runtime_checkable
class Source(Protocol):
    """Produces proposals. ``source_id`` is recorded on every assertion it creates, e.g. ``import:dbt``."""

    source_id: str

    def proposals(self, gloss: Gloss) -> Iterable[Proposal]: ...


@runtime_checkable
class Sink(Protocol):
    name: str

    def publish(self, gloss: Gloss, **options: Any) -> Any: ...
