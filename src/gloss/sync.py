"""Synchronise Gloss with a catalog: discover assets and detect drift.

Two ways in, sharing one implementation:

* **Pull**: ``sync(g, catalog)`` compares everything the catalog lists with what Gloss saw last.
* **Push**: a catalog that knows the moment something changes reports it:
  ``sync_asset(g, catalog, key)``, ``asset_removed(g, catalog, key)``,
  ``asset_moved(g, catalog, old_key, new_key)``.

For each asset:

* **new**: structural facts (the field exists, its datatype, its expression, the dataset's
  source) are proposed by the catalog actor; the default policy accepts those at once, because
  the catalog is the authority on physical structure. Descriptions from the catalog are proposed
  only, and a person reviews them.
* **changed** (fingerprint differs):
    - removed columns: every live fact about that field is marked *stale*,
      and so are relationships and metrics that reference it;
    - columns whose *logical* type changed (e.g. Integer → String): the new
      datatype is recorded, and the field's semantic facts are marked stale;
    - added columns: handled like a new asset;
    - a changed description: proposed again, nothing goes stale.
* **moved** (renamed, or moved to another schema): its facts are re-keyed, so nothing is lost.
  A pull sync spots this only when the catalog gives assets a ``uid`` and the columns are unchanged.
* **gone**: every live fact about the dataset is marked stale. A pull sync infers this only inside
  what the catalog's listing covers (``Catalog.coverage``); a listing filtered by one user's
  access never marks anything gone.

Run a sync as the host's own principal, over everything, never per user.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .ids import Kind, SubjectId
from .ledger import Gloss, Proposal
from .model import LIVE_STATUSES, Event, EventType, Status
from .references import mentions_column, quote_ident
from .seams import AssetSchema, Catalog, Column, Coverage, schema_fingerprint
from .store import AssetRecord, Query
from .types import to_ossie_datatype

# Facts that depend on what kind of value a column holds: a logical type change sends them back to review.
SEMANTIC_FIELD_ATTRS = {
    "description", "label", "is_time", "classification", "ai_instructions", "synonyms",
    "semantic_type", "checks", "role",
}


@dataclass
class SyncReport:
    catalog: str
    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    moved: dict[str, str] = field(default_factory=dict)  # old key -> new key
    gone: list[str] = field(default_factory=list)
    unchanged: int = 0
    proposed: int = 0
    accepted: int = 0
    staled: int = 0
    notes: list[str] = field(default_factory=list)
    correlation: str | None = None  # every event of this run carries it

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _col_proposals(key: str, col: Column, catalog: str) -> list[Proposal]:
    subj = str(SubjectId.for_field(key, col.name))
    ev = [{"catalog": catalog, "column": col.name, "dtype": col.dtype}]
    out = [Proposal(subj, "expression", {"sql": quote_ident(col.name)}, 1.0, ev)]
    dt = to_ossie_datatype(col.dtype)
    if dt:
        out.append(Proposal(subj, "datatype", dt, 1.0, ev))
    if col.dtype:
        out.append(Proposal(subj, "physical_type", col.dtype, 1.0, ev))
    if col.description:
        out.append(Proposal(subj, "description", col.description, None, ev))
    return out


def _dataset_proposals(schema: AssetSchema, catalog: str) -> list[Proposal]:
    subj = str(SubjectId.for_dataset(schema.key))
    ev = [{"catalog": catalog, **{k: v for k, v in schema.extra.items() if _jsonable(v)}}]
    out = [Proposal(subj, "source", schema.source or schema.key, 1.0, ev)]
    if schema.primary_key:
        out.append(Proposal(subj, "primary_key", list(schema.primary_key), 1.0, ev))
    if schema.description:
        out.append(Proposal(subj, "description", schema.description, None, ev))
    return out


def _jsonable(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool, type(None), list, dict))


def _structure(schema: dict[str, Any]) -> str:
    return schema_fingerprint(Column(c["name"], c.get("dtype") or "") for c in schema.get("columns", []))


class CatalogSync:
    def __init__(self, gloss: Gloss, catalog: Catalog, *, auto_accept_structure: bool = True):
        self.g = gloss.correlated()
        self.catalog = catalog
        self.auto_accept = auto_accept_structure
        self.actor = f"catalog:{catalog.name}"
        self._refs: tuple[dict[str, dict[str, Any]], list[Any]] | None = None

    def _report(self) -> SyncReport:
        return SyncReport(self.catalog.name, correlation=self.g.correlation)

    def coverage(self) -> Coverage:
        fn = getattr(self.catalog, "coverage", None)
        return fn() if callable(fn) else Coverage()

    # -------------------------------------------------------------------------
    def run(self, *, detect_removed: bool = True, flush_every: int = 5000) -> SyncReport:
        rep = self._report()
        self._refs = None
        store = self.g.store
        known = {r.key: r for r in store.list_assets(self.catalog.name)}
        listed = list(self.catalog.list_assets())
        current = set(listed)
        coverage = self.coverage()
        missing = {
            k: r for k, r in known.items()
            if k not in current and detect_removed and coverage.speaks_for(k)
        }
        by_uid = {r.uid: r for r in missing.values() if r.uid}
        proposals: list[Proposal] = []
        records: list[AssetRecord] = []

        def flush() -> None:
            with store.transaction():
                self._submit(proposals, rep)
                store.put_assets(records)
            proposals.clear()
            records.clear()

        for key in listed:
            fp = self.catalog.fingerprint(key)
            prev = known.get(key)
            if prev is not None and prev.fingerprint == fp:
                rep.unchanged += 1
                continue
            schema = self.catalog.describe(key)
            if prev is None:
                old = by_uid.get(schema.uid) if schema.uid else None
                if old is not None and _structure(old.schema) == schema_fingerprint(schema.columns):
                    # same id, same columns, old name no longer listed: a rename, not a new table
                    flush()
                    self._move(old.key, schema.key, rep, old)
                    del missing[old.key], by_uid[old.uid]
                    prev = AssetRecord(key, old.catalog, old.fingerprint, old.schema, old.uid)
            if prev is None:
                proposals += self._on_new(schema)
                rep.new.append(key)
            else:
                proposals += self._on_changed(prev, schema, rep)
                if key not in rep.moved.values() or prev.fingerprint != fp:
                    rep.changed.append(key)
            records.append(AssetRecord(key, self.catalog.name, fp, schema.to_dict(), schema.uid))
            if len(proposals) >= flush_every:
                flush()
        flush()

        for rec in missing.values():
            self._gone(rec.key, rep)
        return rep

    def sync_asset(self, key: str, *, force: bool = False) -> SyncReport:
        """Look at one asset now: the catalog just created or changed it.

        ``force`` compares it with what was last seen even when its fingerprint is unchanged.
        """
        rep = self._report()
        self._refs = None
        store = self.g.store
        schema = self.catalog.describe(key)
        fp = self.catalog.fingerprint(key)
        prev = store.get_asset(self.catalog.name, key)
        if prev is not None and prev.fingerprint == fp and not force:
            rep.unchanged = 1
            return rep
        with store.transaction():
            if prev is None:
                self._submit(self._on_new(schema), rep)
                rep.new.append(key)
            else:
                self._submit(self._on_changed(prev, schema, rep), rep)
                if prev.fingerprint != fp:
                    rep.changed.append(key)
            store.put_assets([AssetRecord(key, self.catalog.name, fp, schema.to_dict(), schema.uid)])
        return rep

    def removed(self, key: str) -> SyncReport:
        """The catalog deleted an asset: everything known about it goes back to review."""
        rep = self._report()
        self._gone(key, rep)
        return rep

    def moved(self, old_key: str, new_key: str) -> SyncReport:
        """The catalog renamed an asset or moved it: its facts follow, then it is looked at again."""
        rep = self._report()
        self._move(old_key, new_key, rep, self.g.store.get_asset(self.catalog.name, old_key))
        try:
            again = self.sync_asset(new_key, force=True)  # facts that name the key itself, such as `source`
        except KeyError:
            return rep
        for name in ("new", "changed", "notes"):
            getattr(rep, name).extend(getattr(again, name))
        rep.proposed += again.proposed
        rep.accepted += again.accepted
        rep.staled += again.staled
        return rep

    # -------------------------------------------------------------------------
    def _submit(self, proposals: list[Proposal], rep: SyncReport) -> None:
        if not proposals:
            return
        results = self.g.propose_many(proposals, source=self.actor, auto_accept=self.auto_accept)
        rep.proposed += sum(r.outcome == "created" for r in results)
        rep.accepted += sum(r.accepted for r in results)

    def _move(self, old_key: str, new_key: str, rep: SyncReport, record: AssetRecord | None) -> None:
        store = self.g.store
        with store.transaction():
            try:
                res = self.g.rekey(SubjectId.for_dataset(old_key), SubjectId.for_dataset(new_key), actor=self.actor,
                                   comment=f"renamed in catalog {self.catalog.name}")
                rep.staled += res.staled
            except KeyError:
                pass  # no facts were ever recorded under the old name
            if record is not None:
                store.delete_asset(self.catalog.name, old_key)
                store.put_assets([AssetRecord(new_key, record.catalog, record.fingerprint,
                                              {**record.schema, "key": new_key}, record.uid)])
        rep.moved[old_key] = new_key
        rep.notes.append(f"{old_key} is now {new_key}")

    def _gone(self, key: str, rep: SyncReport) -> None:
        with self.g.store.transaction():
            rep.staled += self._stale(Query(dataset=key), "dataset no longer present in catalog")
            self.g.store.delete_asset(self.catalog.name, key)
        rep.gone.append(key)

    def _on_new(self, schema: AssetSchema) -> list[Proposal]:
        out = _dataset_proposals(schema, self.catalog.name)
        for col in schema.columns:
            out += _col_proposals(schema.key, col, self.catalog.name)
        return out

    def _on_changed(self, prev: AssetRecord, schema: AssetSchema, rep: SyncReport) -> list[Proposal]:
        old_cols = {c["name"]: c for c in prev.schema.get("columns", [])}
        new_cols = {c.name: c for c in schema.columns}
        key = schema.key

        for name in old_cols.keys() - new_cols.keys():
            reason = f"column {name!r} removed from {key}"
            rep.staled += self._stale(Query(subject=str(SubjectId.for_field(key, name))), reason)
            rep.staled += self._stale_references(key, name, reason)
            rep.notes.append(reason)

        out: list[Proposal] = []
        for name, col in new_cols.items():
            if name not in old_cols:
                out += _col_proposals(key, col, self.catalog.name)
                continue
            mine = _col_proposals(key, col, self.catalog.name)
            if old_cols[name].get("dtype") != col.dtype:
                # precise type changed (e.g. Int32 -> Int64): record it; no review needed by itself
                out += [p for p in mine if p.attribute == "physical_type"]
            if old_cols[name].get("description") != col.description:
                out += [p for p in mine if p.attribute == "description"]
            old_t = to_ossie_datatype(old_cols[name].get("dtype"))
            new_t = to_ossie_datatype(col.dtype)
            if old_t != new_t:
                reason = f"type of {key}.{name} changed from {old_t} to {new_t}"
                subj = str(SubjectId.for_field(key, name))
                rep.staled += self._stale(Query(subject=subj, attributes=sorted(SEMANTIC_FIELD_ATTRS)), reason)
                rep.staled += self._stale_references(key, name, reason)
                out += [p for p in mine if p.attribute == "datatype"]
                rep.notes.append(reason)
        return out + _dataset_proposals(schema, self.catalog.name)

    # -------------------------------------------------------------------------
    def _stale(self, query: Query, reason: str) -> int:
        statuses = [st for st in LIVE_STATUSES if st is not Status.STALE]
        q = Query(**{**query.__dict__, "statuses": statuses})
        ids = [a.id for a in self.g.store.find(q)]
        done, _ = self.g.transition_many("mark_stale", ids, actor=self.actor, comment=reason, reason=reason)
        return len(done)

    def _stale_references(self, key: str, column: str, reason: str) -> int:
        """Mark relationships and metrics that reference ``key.column`` as stale (heuristic for metrics)."""
        n = 0
        live = [st for st in LIVE_STATUSES if st is not Status.STALE]
        if self._refs is None:
            rels: dict[str, dict[str, Any]] = {}
            for a in self.g.store.find(Query(kind=Kind.RELATIONSHIP.value, statuses=live)):
                rels.setdefault(a.subject, {})[a.attribute] = a.value
            metrics = self.g.store.find(Query(kind=Kind.METRIC.value, attribute="expression", statuses=live))
            self._refs = (rels, metrics)
        rels, metrics = self._refs
        for subj, attrs in rels.items():
            hit = (attrs.get("from") == key and column in (attrs.get("from_columns") or [])) or (
                attrs.get("to") == key and column in (attrs.get("to_columns") or [])
            )
            if hit:
                n += self._stale(Query(subject=subj), reason)
        for a in metrics:
            if mentions_column(a.value, key, column, scoped=a.dataset == key):
                n += self._stale(Query(subject=a.subject), reason)
        return n


def sync(gloss: Gloss, catalog: Catalog, **kwargs: Any) -> SyncReport:
    """Compare everything the catalog lists with what Gloss last saw."""
    detect_removed = kwargs.pop("detect_removed", True)
    return CatalogSync(gloss, catalog, **kwargs).run(detect_removed=detect_removed)


def sync_asset(gloss: Gloss, catalog: Catalog, key: str, **kwargs: Any) -> SyncReport:
    """One asset was created or changed in the catalog."""
    return CatalogSync(gloss, catalog, **kwargs).sync_asset(key)


def asset_removed(gloss: Gloss, catalog: Catalog, key: str) -> SyncReport:
    """One asset was deleted from the catalog."""
    return CatalogSync(gloss, catalog).removed(key)


def asset_moved(gloss: Gloss, catalog: Catalog, old_key: str, new_key: str) -> SyncReport:
    """One asset was renamed or moved in the catalog."""
    return CatalogSync(gloss, catalog).moved(old_key, new_key)


def write_back(gloss: Gloss, catalog: Any) -> Callable[[], None]:
    """Make the catalog's own description fields a projection of accepted descriptions.

    Gloss owns meaning: whenever a description of a dataset or one of its fields is accepted
    (or restored), the catalog's ``write_back(key, facts)`` receives the accepted text. Returns
    a function that stops it.
    """

    def on_events(events: list[Event]) -> None:
        ids = {e.assertion_id for e in events if e.type in (EventType.ACCEPTED, EventType.RESTORED)}
        if not ids:
            return
        facts: dict[str, dict[str, Any]] = {}
        for a in gloss.store.get_many(ids).values():
            sid = a.subject_id
            if a.attribute != "description" or a.status is not Status.ACCEPTED or not sid.dataset:
                continue
            if sid.kind is Kind.DATASET:
                facts.setdefault(sid.dataset, {})["description"] = a.value
            elif sid.kind is Kind.FIELD:
                facts.setdefault(sid.dataset, {}).setdefault("fields", {})[sid.name] = {"description": a.value}
        for key, data in facts.items():
            catalog.write_back(key, data)

    return gloss.subscribe(on_events)
