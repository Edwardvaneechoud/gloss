"""SQLAlchemy store: SQLite for embedded use, Postgres for a shared service.

Tables (all prefixed ``gloss_``)
--------------------------------
gloss_events         append-only log (never updated or deleted)
gloss_assertions     projection: the current state of every assertion
gloss_search         projection: one search row per subject with accepted facts
gloss_assets         last seen physical schema per catalog asset (drift detection)
gloss_check_results  logged data quality runs
gloss_meta           schema version and index version

A store given a URL owns its engine: it sets SQLite pragmas and creates or upgrades the
schema. A store given a host's ``Engine`` touches neither: no connection listeners, no DDL.
The host runs :func:`glosslayer.schema.upgrade` from its own migrations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError

from ..model import Assertion, Event, EventType, Status, value_hash
from .base import AssetRecord, Query

if TYPE_CHECKING:
    from ..policy import Scope

metadata = sa.MetaData()

# 64-bit sequence numbers; SQLite only auto-increments a plain INTEGER primary key.
_BigSerial = sa.BigInteger().with_variant(sa.Integer, "sqlite")

events_t = sa.Table(
    "gloss_events",
    metadata,
    sa.Column("seq", _BigSerial, primary_key=True, autoincrement=True),
    sa.Column("id", sa.String(32), nullable=False, unique=True),
    sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
    sa.Column("type", sa.String(32), nullable=False),
    sa.Column("assertion_id", sa.String(32), nullable=False, index=True),
    sa.Column("subject", sa.String(512), nullable=False, index=True),  # the subject when the event happened
    sa.Column("actor", sa.String(256), nullable=False),
    sa.Column("via", sa.String(256), nullable=True),
    sa.Column("correlation", sa.String(64), nullable=True, index=True),
    sa.Column("data", sa.JSON, nullable=False),
)

assertions_t = sa.Table(
    "gloss_assertions",
    metadata,
    sa.Column("id", sa.String(32), primary_key=True),
    sa.Column("subject", sa.String(512), nullable=False, index=True),
    sa.Column("kind", sa.String(16), nullable=False, index=True),
    sa.Column("dataset", sa.String(512), nullable=True, index=True),
    sa.Column("namespace", sa.String(512), nullable=True),  # scope path: dataset key or owning namespace
    sa.Column("attribute", sa.String(64), nullable=False, index=True),
    sa.Column("value", sa.JSON, nullable=True),
    sa.Column("value_hash", sa.String(16), nullable=False),
    sa.Column("source", sa.String(256), nullable=False, index=True),
    sa.Column("sources", sa.JSON, nullable=False),
    sa.Column("status", sa.String(16), nullable=False, index=True),
    sa.Column("confidence", sa.Float, nullable=True),
    sa.Column("evidence", sa.JSON, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("created_by", sa.String(256), nullable=False),
    sa.Column("last_comment", sa.Text, nullable=True),
    sa.Column("group_id", sa.String(64), nullable=True, index=True),
    sa.Index("ix_gloss_assertions_subject_attr", "subject", "attribute"),
    # composite indexes so the planner never falls back to scanning every accepted fact
    sa.Index("ix_gloss_assertions_dataset_status", "dataset", "status"),
    sa.Index("ix_gloss_assertions_namespace_status", "namespace", "status"),
    sa.Index("ix_gloss_assertions_kind_status", "kind", "status"),
    sa.Index("ix_gloss_assertions_attr_hash", "attribute", "value_hash"),
    sa.Index("ix_gloss_assertions_subject_status", "subject", "status"),
    # at most one accepted value per subject and attribute, enforced by the database
    sa.Index(
        "uq_gloss_assertions_accepted",
        "subject",
        "attribute",
        unique=True,
        sqlite_where=sa.text("status = 'accepted'"),
        postgresql_where=sa.text("status = 'accepted'"),
    ),
)

assets_t = sa.Table(
    "gloss_assets",
    metadata,
    sa.Column("catalog", sa.String(128), primary_key=True),
    sa.Column("key", sa.String(512), primary_key=True),
    sa.Column("fingerprint", sa.String(128), nullable=False),
    sa.Column("schema", sa.JSON, nullable=False),
    sa.Column("uid", sa.String(256), nullable=True),
    sa.Column("seen_at", sa.DateTime(timezone=True), nullable=False),
)

search_t = sa.Table(
    "gloss_search",
    metadata,
    sa.Column("subject", sa.String(512), primary_key=True),
    sa.Column("kind", sa.String(16), nullable=False, index=True),
    sa.Column("dataset", sa.String(512), nullable=True, index=True),
    sa.Column("namespace", sa.String(512), nullable=True, index=True),
    sa.Column("name", sa.String(512), nullable=False),
    sa.Column("name_text", sa.Text, nullable=False),  # always searchable (names, keys, type)
    sa.Column("detail_text", sa.Text, nullable=False),  # searchable only up to the caller's level
    sa.Column("description", sa.Text, nullable=True),
    sa.Column("datatype", sa.String(32), nullable=True),
    sa.Column("semantic_type", sa.String(512), nullable=True),
    sa.Column("level", sa.Integer, nullable=False),  # confidentiality rank: 0 public … 3 restricted
)

meta_t = sa.Table(
    "gloss_meta",
    metadata,
    sa.Column("key", sa.String(64), primary_key=True),
    sa.Column("value", sa.Text, nullable=False),
)

check_results_t = sa.Table(
    "gloss_check_results",
    metadata,
    sa.Column("id", _BigSerial, primary_key=True, autoincrement=True),
    sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
    sa.Column("subject", sa.String(512), nullable=False, index=True),
    sa.Column("check", sa.JSON, nullable=False),
    sa.Column("passed", sa.Boolean, nullable=False),
    sa.Column("failed_rows", sa.BigInteger, nullable=False),
    sa.Column("total_rows", sa.BigInteger, nullable=False),
    sa.Column("origin", sa.String(512), nullable=True),
    sa.Column("error", sa.Text, nullable=True),
    sa.Column("runner", sa.String(256), nullable=False),
    sa.Column("run_id", sa.String(128), nullable=False, index=True),
)

FTS_TOKENIZER = "porter unicode61 remove_diacritics 2"  # stemming: emails → email


def pg_tsvector(column: str) -> str:
    """Postgres full-text expression for a search column, as indexed (see ``schema.create_fts``).

    Punctuation becomes spaces first, so ``sales.orders`` and ``order_id`` are words, not a host name.
    """
    return f"to_tsvector('english', regexp_replace({column}, '[^[:alnum:]]+', ' ', 'g'))"
_ADVISORY_LOCK = 0x676C6F7373  # "gloss": serialises appends on Postgres so seq order is commit order

# Events that can change which value of a subject is the accepted one.
VERSION_EVENTS = ("accepted", "superseded", "restored", "retracted", "marked_stale", "rekeyed")


class ConcurrentWrite(RuntimeError):
    """Another writer accepted a value for the same subject and attribute first. Read again and retry."""


def _aware(dt: datetime) -> datetime:
    # SQLite drops tzinfo; everything Gloss writes is UTC.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _row_to_assertion(r: Any) -> Assertion:
    return Assertion(
        id=r.id,
        subject=r.subject,
        attribute=r.attribute,
        value=r.value,
        source=r.source,
        status=Status(r.status),
        confidence=r.confidence,
        created_at=_aware(r.created_at),
        updated_at=_aware(r.updated_at),
        created_by=r.created_by,
        sources=tuple(r.sources or ()),
        evidence=tuple(r.evidence or ()),
        last_comment=r.last_comment,
        group=r.group_id,
    )


def _assertion_to_row(a: Assertion) -> dict[str, Any]:
    sid = a.subject_id
    return {
        "id": a.id,
        "subject": a.subject,
        "kind": sid.kind.value,
        "dataset": sid.owning_dataset,
        "namespace": sid.scope_path,
        "attribute": a.attribute,
        "value": a.value,
        "value_hash": value_hash(a.value),
        "source": a.source,
        "sources": list(a.sources),
        "status": a.status.value,
        "confidence": a.confidence,
        "evidence": list(a.evidence),
        "created_at": a.created_at,
        "updated_at": a.updated_at,
        "created_by": a.created_by,
        "last_comment": a.last_comment,
        "group_id": a.group,
    }


def _row_to_event(r: Any) -> Event:
    return Event(
        type=EventType(r.type),
        assertion_id=r.assertion_id,
        actor=r.actor,
        data=r.data or {},
        id=r.id,
        ts=_aware(r.ts),
        seq=r.seq,
        correlation=r.correlation,
        via=r.via,
    )


def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
    cur.execute("PRAGMA synchronous=NORMAL")  # durable at checkpoints; far fewer fsyncs
    cur.close()


@dataclass
class _Tx:
    conn: Connection
    after: list[Callable[[], None]] = field(default_factory=list)


class SQLStore:
    """Store backed by SQLAlchemy, on SQLite or Postgres."""

    def __init__(self, url_or_engine: str | Engine, *, create: bool | None = None, echo: bool = False):
        """Open a store on a URL (which it owns) or on a host's ``Engine`` (which it only uses).

        ``create`` defaults to whether the store owns its engine: True creates or upgrades the schema,
        False only checks that it is up to date (:func:`glosslayer.schema.require`).
        """
        from .. import schema

        self.owns_engine = not isinstance(url_or_engine, Engine)
        if isinstance(url_or_engine, Engine):
            self.engine = url_or_engine
        elif url_or_engine in ("sqlite://", "sqlite:///:memory:"):
            # One shared connection, otherwise every connection gets its own empty database.
            self.engine = sa.create_engine(
                url_or_engine,
                echo=echo,
                poolclass=sa.pool.StaticPool,
                connect_args={"check_same_thread": False},
            )
        else:
            self.engine = sa.create_engine(url_or_engine, echo=echo)
        self.dialect = self.engine.dialect.name
        if self.owns_engine and self.dialect == "sqlite":
            sa.event.listen(self.engine, "connect", _sqlite_pragmas)
        if self.owns_engine if create is None else create:
            schema.upgrade(self.engine)
        else:
            schema.require(self.engine)
        self._tx: ContextVar[_Tx | None] = ContextVar(f"gloss_tx_{id(self)}", default=None)
        # "fts5" (SQLite), "tsvector" (Postgres) or "like" (anything else, and SQLite without FTS5)
        self.search_engine = "like"
        if self.dialect == "postgresql":
            self.search_engine = "tsvector"
        elif self.dialect == "sqlite":
            with self._read() as conn:
                self.search_engine = "fts5" if schema.has_fts(conn) else "like"

    @property
    def fts(self) -> bool:
        """Whether the database's own full-text search is used (stemming, spelling vocabulary)."""
        return self.search_engine != "like"

    @fts.setter
    def fts(self, enabled: bool) -> None:
        if not enabled:
            self.search_engine = "like"  # plain matching, e.g. to test the portable path

    # -- prefixes and scopes ------------------------------------------------------
    def _starts(self, column: Any, prefix: str) -> Any:
        """``column`` starts with ``prefix``, written so the database can use an index for it.

        SQLite compares text bytewise, so a range is a prefix match and uses a plain index (its
        LIKE does not). Postgres uses LIKE, served by the ``text_pattern_ops`` indexes in the schema.
        """
        if not prefix:
            return column.isnot(None)
        if self.dialect == "sqlite" and ord(prefix[-1]) < 0x10FFFF:
            return sa.and_(column >= prefix, column < prefix[:-1] + chr(ord(prefix[-1]) + 1))
        return column.startswith(prefix, autoescape=True)

    def _under(self, column: Any, path: str) -> Any:
        """Rows whose dotted path is ``path`` or lies under it."""
        p = path.rstrip(".")
        if not p:
            return column.isnot(None)
        return sa.or_(column == p, self._starts(column, p + "."))

    def _readable(self, column: Any, scope: Scope | None) -> Any | None:
        """SQL for "this row's scope path is readable", or None when nothing needs filtering."""
        if scope is None or scope.all_paths:
            return None
        parts: list[Any] = []
        if scope.include_global:
            parts.append(column.is_(None))
        if scope.prefixes is None:
            parts.append(column.isnot(None))
        else:
            parts += [self._under(column, p) for p in scope.prefixes]
        return sa.or_(*parts) if parts else sa.false()

    # -- transactions -----------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        """One transaction for everything the store does inside the block. Re-entrant."""
        tx = self._tx.get()
        if tx is not None:
            yield tx.conn
            return
        with self.engine.begin() as conn:
            tx = _Tx(conn)
            token = self._tx.set(tx)
            try:
                yield conn
            finally:
                self._tx.reset(token)
        for fn in tx.after:
            fn()

    @contextmanager
    def using(self, connection: Connection) -> Iterator[Connection]:
        """Run inside the host's own transaction: Gloss neither commits nor rolls back.

        Post-commit callbacks (subscribers) run when the block ends, so end it after the host commits
        or accept that they fire slightly early.
        """
        tx = _Tx(connection)
        token = self._tx.set(tx)
        try:
            yield connection
        finally:
            self._tx.reset(token)
        for fn in tx.after:
            fn()

    @contextmanager
    def _read(self) -> Iterator[Connection]:
        tx = self._tx.get()
        if tx is not None:
            yield tx.conn
            return
        with self.engine.connect() as conn:
            yield conn

    def after_commit(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` once the current transaction has committed, or right away when none is open."""
        tx = self._tx.get()
        if tx is not None:
            tx.after.append(fn)
        else:
            fn()

    # -- writes ---------------------------------------------------------------
    def append(self, events: Sequence[Event], states: Sequence[Assertion], *, returning: bool = False) -> list[Event]:
        """Insert events and upsert the resulting states in one transaction.

        Events of type ``proposed`` create their assertion; all other states are updates. With
        ``returning`` the events come back carrying the ``seq`` the database gave them.
        """
        if not events and not states:
            return []
        subjects = {st.id: st.subject for st in states}
        new_ids = {ev.assertion_id for ev in events if ev.type is EventType.PROPOSED}
        out = list(events)
        try:
            with self.transaction() as conn:
                if self.dialect == "postgresql":
                    conn.execute(sa.text("SELECT pg_advisory_xact_lock(:k)"), {"k": _ADVISORY_LOCK})
                missing = {ev.assertion_id for ev in events} - subjects.keys() - new_ids
                if missing:
                    rows = conn.execute(
                        sa.select(assertions_t.c.id, assertions_t.c.subject).where(assertions_t.c.id.in_(missing))
                    )
                    subjects.update({r.id: r.subject for r in rows})
                # States leaving `accepted` are written before states entering it, so the
                # one-accepted-value index holds at every statement.
                leaving = [_assertion_to_row(st) for st in states if st.id not in new_ids
                           and st.status is not Status.ACCEPTED]
                entering = [_assertion_to_row(st) for st in states if st.id not in new_ids
                            and st.status is Status.ACCEPTED]
                inserts = [_assertion_to_row(st) for st in states if st.id in new_ids]
                if events:
                    rows_in = [
                        {
                            "id": ev.id,
                            "ts": ev.ts,
                            "type": ev.type.value,
                            "assertion_id": ev.assertion_id,
                            "subject": ev.data.get("subject") or subjects.get(ev.assertion_id, ""),
                            "actor": ev.actor,
                            "via": ev.via,
                            "correlation": ev.correlation,
                            "data": ev.data,
                        }
                        for ev in events
                    ]
                    if returning:
                        res = conn.execute(events_t.insert().returning(events_t.c.id, events_t.c.seq), rows_in)
                        seqs = {r.id: r.seq for r in res}
                        out = [replace(ev, seq=seqs.get(ev.id)) for ev in events]
                    else:
                        conn.execute(events_t.insert(), rows_in)
                for batch in (leaving, None, entering):
                    if batch is None:
                        if inserts:
                            conn.execute(assertions_t.insert(), inserts)
                        continue
                    if batch:
                        cols = [c for c in batch[0] if c != "id"]
                        stmt = (
                            assertions_t.update()
                            .where(assertions_t.c.id == sa.bindparam("_id"))
                            .values({c: sa.bindparam(c) for c in cols})
                        )
                        conn.execute(stmt, [{**{c: r[c] for c in cols}, "_id": r["id"]} for r in batch])
        except IntegrityError as exc:
            if "uq_gloss_assertions_accepted" in str(exc) or "gloss_assertions.subject" in str(exc):
                raise ConcurrentWrite(
                    "Another writer accepted a value for the same subject and attribute"
                ) from exc
            raise
        return out

    def reset_projection(self, states: Iterable[Assertion]) -> None:
        """Replace every row of ``gloss_assertions`` with ``states``, in one transaction."""
        with self.transaction() as conn:
            conn.execute(assertions_t.delete())
            rows = [_assertion_to_row(s) for s in states]
            for i in range(0, len(rows), 5000):
                conn.execute(assertions_t.insert(), rows[i : i + 5000])

    # -- reads ----------------------------------------------------------------
    def get(self, assertion_id: str) -> Assertion | None:
        """The assertion with this id, or None."""
        with self._read() as conn:
            r = conn.execute(sa.select(assertions_t).where(assertions_t.c.id == assertion_id)).first()
        return _row_to_assertion(r) if r else None

    def get_many(self, ids: Iterable[str]) -> dict[str, Assertion]:
        """Assertions by id, looked up in batches of 500; unknown ids are left out."""
        ids = list(ids)
        out: dict[str, Assertion] = {}
        with self._read() as conn:
            for i in range(0, len(ids), 500):
                rows = conn.execute(sa.select(assertions_t).where(assertions_t.c.id.in_(ids[i : i + 500])))
                out.update({r.id: _row_to_assertion(r) for r in rows})
        return out

    def by_id_prefix(self, prefix: str, limit: int = 2) -> list[Assertion]:
        """Assertions whose id starts with ``prefix``; two tell a unique prefix from an ambiguous one."""
        stmt = sa.select(assertions_t).where(self._starts(assertions_t.c.id, prefix)).limit(limit)
        with self._read() as conn:
            return [_row_to_assertion(r) for r in conn.execute(stmt)]

    def find_for_subjects(
        self,
        subjects: Iterable[str],
        statuses: Sequence[Status] | None = None,
        *,
        attributes: Sequence[str] | None = None,
        scope: Scope | None = None,
    ) -> list[Assertion]:
        """All assertions about any of the given subjects (batched lookups for bulk operations)."""
        subjects = sorted(set(subjects))
        out: list[Assertion] = []
        extra = []
        if statuses:
            extra.append(assertions_t.c.status.in_([Status(s).value for s in statuses]))
        if attributes:
            extra.append(assertions_t.c.attribute.in_(list(attributes)))
        clause = self._readable(assertions_t.c.namespace, scope)
        if clause is not None:
            extra.append(clause)
        with self._read() as conn:
            for i in range(0, len(subjects), 500):
                stmt = sa.select(assertions_t).where(assertions_t.c.subject.in_(subjects[i : i + 500]), *extra)
                out += [_row_to_assertion(r) for r in conn.execute(stmt.order_by(assertions_t.c.created_at))]
        return out

    def _where(self, q: Query) -> list[Any]:
        c = assertions_t.c
        w: list[Any] = []
        if q.subject is not None:
            w.append(c.subject == q.subject)
        if q.subject_prefix is not None:
            w.append(self._starts(c.subject, q.subject_prefix))
        if q.dataset is not None:
            w.append(c.dataset == q.dataset)
        if q.dataset_prefix is not None:
            w.append(self._starts(c.dataset, q.dataset_prefix))
        if q.namespace is not None:
            w.append(self._under(c.namespace, q.namespace))
        if q.kind is not None:
            w.append(c.kind == q.kind)
        if q.attribute is not None:
            w.append(c.attribute == q.attribute)
        if q.attributes:
            w.append(c.attribute.in_(list(q.attributes)))
        if q.statuses:
            w.append(c.status.in_([Status(s).value for s in q.statuses]))
        if q.source_prefix is not None:
            w.append(self._starts(c.source, q.source_prefix))
        if q.value_hash is not None:
            w.append(c.value_hash == q.value_hash)
        if q.group is not None:
            w.append(c.group_id == q.group)
        clause = self._readable(c.namespace, q.scope)
        if clause is not None:
            w.append(clause)
        return w

    def find(self, query: Query) -> list[Assertion]:
        """Matching assertions, ordered by subject, attribute and age.

        Without a limit the rows are sorted here, not in SQL: an ORDER BY tempts the planner into
        walking the whole subject index in order, when a filter's own index would find a handful of rows.
        """
        stmt = sa.select(assertions_t).where(*self._where(query))
        paged = query.limit is not None or query.offset
        if paged:
            stmt = stmt.order_by(
                assertions_t.c.subject, assertions_t.c.attribute, assertions_t.c.created_at
            ).offset(query.offset)
            if query.limit is not None:
                stmt = stmt.limit(query.limit)
        with self._read() as conn:
            rows = [_row_to_assertion(r) for r in conn.execute(stmt)]
        if not paged:
            rows.sort(key=lambda a: (a.subject, a.attribute, a.created_at))
        return rows

    def count(self, query: Query) -> int:
        """Number of assertions matching ``query`` (its ``limit`` and ``offset`` are ignored)."""
        stmt = sa.select(sa.func.count()).select_from(assertions_t).where(*self._where(query))
        with self._read() as conn:
            return conn.execute(stmt).scalar() or 0

    def count_by_status(self, query: Query) -> dict[tuple[str | None, Status], int]:
        """Counts grouped by (scope path, status)."""
        c = assertions_t.c
        stmt = (
            sa.select(c.namespace, c.status, sa.func.count())
            .where(*self._where(query))
            .group_by(c.namespace, c.status)
        )
        with self._read() as conn:
            return {(ns, Status(st)): n for ns, st, n in conn.execute(stmt)}

    def distinct_subjects(self, query: Query) -> list[str]:
        """Subjects with at least one assertion matching ``query``, in no particular order."""
        stmt = sa.select(assertions_t.c.subject).where(*self._where(query)).distinct()
        with self._read() as conn:
            return [r[0] for r in conn.execute(stmt)]

    def events(
        self,
        assertion_ids: Iterable[str] | None = None,
        after_seq: int = 0,
        limit: int | None = None,
        *,
        correlation: str | None = None,
        scope: Scope | None = None,
    ) -> list[Event]:
        """Events after ``after_seq`` in log order, optionally of some assertions or one correlation only.

        With ``scope``, only events of assertions whose current scope path the reader may see.
        """
        stmt = sa.select(events_t).where(events_t.c.seq > after_seq).order_by(events_t.c.seq)
        if assertion_ids is not None:
            stmt = stmt.where(events_t.c.assertion_id.in_(list(assertion_ids)))
        if correlation is not None:
            stmt = stmt.where(events_t.c.correlation == correlation)
        clause = self._readable(assertions_t.c.namespace, scope)
        if clause is not None:
            stmt = stmt.where(
                events_t.c.assertion_id.in_(sa.select(assertions_t.c.id).where(clause))
            )
        if limit is not None:
            stmt = stmt.limit(limit)
        with self._read() as conn:
            return [_row_to_event(r) for r in conn.execute(stmt)]

    def last_seq(self) -> int:
        """Sequence number of the newest event (0 when empty)."""
        with self._read() as conn:
            return conn.execute(sa.select(sa.func.max(events_t.c.seq))).scalar() or 0

    def events_for_subject(self, subject: str) -> list[Event]:
        """Every event of the assertions now at ``subject``, so history follows a renamed subject."""
        ids = sa.select(assertions_t.c.id).where(assertions_t.c.subject == subject)
        stmt = sa.select(events_t).where(events_t.c.assertion_id.in_(ids)).order_by(events_t.c.seq)
        with self._read() as conn:
            return [_row_to_event(r) for r in conn.execute(stmt)]

    def subject_versions(self, subjects: Iterable[str]) -> dict[str, int]:
        """Per subject, the seq of the last event that could have changed its accepted definition."""
        subjects = sorted(set(subjects))
        out: dict[str, int] = {}
        with self._read() as conn:
            for i in range(0, len(subjects), 500):
                stmt = (
                    sa.select(assertions_t.c.subject, sa.func.max(events_t.c.seq))
                    .select_from(events_t.join(assertions_t, assertions_t.c.id == events_t.c.assertion_id))
                    .where(assertions_t.c.subject.in_(subjects[i : i + 500]), events_t.c.type.in_(VERSION_EVENTS))
                    .group_by(assertions_t.c.subject)
                )
                out.update({s: int(v) for s, v in conn.execute(stmt)})
        return out

    # -- assets ---------------------------------------------------------------
    @staticmethod
    def _asset(r: Any) -> AssetRecord:
        return AssetRecord(r.key, r.catalog, r.fingerprint, r.schema, r.uid)

    def get_asset(self, catalog: str, key: str) -> AssetRecord | None:
        """The last recorded shape of a catalog's asset, or None when it was never seen."""
        with self._read() as conn:
            r = conn.execute(
                sa.select(assets_t).where(assets_t.c.catalog == catalog, assets_t.c.key == key)
            ).first()
        return self._asset(r) if r else None

    def put_asset(self, record: AssetRecord) -> None:
        """Record one asset's current shape (see :meth:`put_assets`)."""
        self.put_assets([record])

    def put_assets(self, records: Sequence[AssetRecord]) -> None:
        """Replace the stored shape of each asset in ``records``, stamped now, in one transaction."""
        if not records:
            return
        now = datetime.now(timezone.utc)
        catalogs = {r.catalog for r in records}
        with self.transaction() as conn:
            for cat in catalogs:
                keys = [r.key for r in records if r.catalog == cat]
                for i in range(0, len(keys), 500):
                    conn.execute(
                        assets_t.delete().where(assets_t.c.catalog == cat, assets_t.c.key.in_(keys[i : i + 500]))
                    )
            conn.execute(
                assets_t.insert(),
                [
                    {"catalog": r.catalog, "key": r.key, "fingerprint": r.fingerprint, "schema": r.schema,
                     "uid": r.uid, "seen_at": now}
                    for r in records
                ],
            )

    def list_assets(self, catalog: str) -> list[AssetRecord]:
        """Every recorded asset of one catalog."""
        with self._read() as conn:
            rows = conn.execute(sa.select(assets_t).where(assets_t.c.catalog == catalog))
            return [self._asset(r) for r in rows]

    def delete_asset(self, catalog: str, key: str) -> None:
        """Forget an asset's recorded shape."""
        with self.transaction() as conn:
            conn.execute(assets_t.delete().where(assets_t.c.catalog == catalog, assets_t.c.key == key))

    # -- search index -----------------------------------------------------------
    def recreate_search_tables(self) -> None:
        """Drop and recreate the full-text tables (needed when the tokenizer changes)."""
        from .. import schema

        if self.search_engine != "fts5":
            return
        with self.transaction() as conn:
            conn.exec_driver_sql("DROP TABLE IF EXISTS gloss_search_vocab")
            conn.exec_driver_sql("DROP TABLE IF EXISTS gloss_search_fts")
            schema.create_fts(conn)

    def _search_filters(
        self, kinds: Sequence[str] | None, dataset_prefix: str | None, scope: Scope | None
    ) -> list[Any]:
        filters: list[Any] = []
        if kinds:
            filters.append(search_t.c.kind.in_(list(kinds)))
        if dataset_prefix:
            filters.append(self._starts(search_t.c.dataset, dataset_prefix))
        clause = self._readable(search_t.c.namespace, scope)
        if clause is not None:
            filters.append(clause)
        return filters

    def search_names_containing(
        self,
        words: Sequence[str],
        *,
        kinds: Sequence[str] | None = None,
        dataset_prefix: str | None = None,
        scope: Scope | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Rows whose *names* contain every word anywhere ("mail" in "useremail"). A slow-path fallback."""
        if not words:
            return []
        where = [search_t.c.name_text.ilike(f"%{w}%") for w in words]
        stmt = (
            sa.select(search_t, sa.literal(0).label("score"))
            .where(*where, *self._search_filters(kinds, dataset_prefix, scope))
            .order_by(sa.func.length(search_t.c.name), search_t.c.subject)
            .limit(limit)
        )
        with self._read() as conn:
            return [dict(r._mapping) for r in conn.execute(stmt)]

    def replace_search_rows(self, subjects: Iterable[str], rows: Sequence[dict[str, Any]]) -> None:
        """Delete the index rows of ``subjects`` and insert ``rows`` (a subset of them) in one transaction."""
        subjects = list(set(subjects) | {r["subject"] for r in rows})
        with self.transaction() as conn:
            for i in range(0, len(subjects), 500):
                chunk = subjects[i : i + 500]
                conn.execute(search_t.delete().where(search_t.c.subject.in_(chunk)))
                if self.search_engine == "fts5":
                    marks = ",".join("?" * len(chunk))
                    conn.exec_driver_sql(f"DELETE FROM gloss_search_fts WHERE subject IN ({marks})", tuple(chunk))
            if rows:
                conn.execute(search_t.insert(), list(rows))
                if self.search_engine == "fts5":
                    conn.exec_driver_sql(
                        "INSERT INTO gloss_search_fts (subject, name_text, detail_text) VALUES (?, ?, ?)",
                        [(r["subject"], r["name_text"], r["detail_text"]) for r in rows],
                    )

    def clear_search(self) -> None:
        """Delete every row of the search index, including the full-text table on SQLite."""
        with self.transaction() as conn:
            conn.execute(search_t.delete())
            if self.search_engine == "fts5":
                conn.exec_driver_sql("DELETE FROM gloss_search_fts")

    def search_is_empty(self) -> bool:
        """Whether the search index has no rows (e.g. it was never built)."""
        with self._read() as conn:
            return conn.execute(sa.select(search_t.c.subject).limit(1)).first() is None

    def accepted_subjects(self) -> list[str]:
        """Every subject with at least one accepted fact."""
        return self.distinct_subjects(Query(statuses=[Status.ACCEPTED]))

    def search(
        self,
        tokens: Sequence[str],
        *,
        max_level: int,
        kinds: Sequence[str] | None = None,
        dataset_prefix: str | None = None,
        scope: Scope | None = None,
        limit: int = 20,
        match_any: bool = False,
    ) -> list[dict[str, Any]]:
        """Rows matching every token (or, with ``match_any``, at least one), ranked.

        Details only count for rows at or below ``max_level``; above it, only names match.
        """
        if not tokens:
            return []
        exact = "_".join(tokens)
        filters = self._search_filters(kinds, dataset_prefix, scope)
        lname = sa.func.lower(search_t.c.name)
        rank = sa.case(
            (sa.or_(lname == exact, lname.like(f"%.{exact}")), 0), (lname.like(f"{exact}%"), 1), else_=2
        )
        with self._read() as conn:
            if self.search_engine == "fts5":
                joiner = " OR " if match_any else " AND "
                q_all = joiner.join(f'"{t}"*' for t in tokens)
                q_name = joiner.join(f'name_text : "{t}"*' for t in tokens)
                fts = sa.table("gloss_search_fts", sa.column("subject"))
                match = sa.text("gloss_search_fts MATCH :q_all")
                by_name = sa.text("SELECT subject FROM gloss_search_fts WHERE gloss_search_fts MATCH :q_name")
                score = sa.literal_column("bm25(gloss_search_fts, 0.0, 10.0, 1.0)").label("score")
                stmt = (
                    sa.select(search_t, score)
                    .select_from(fts.join(search_t, search_t.c.subject == fts.c.subject))
                    .where(
                        match,
                        sa.or_(search_t.c.level <= max_level, search_t.c.subject.in_(by_name)),
                        *filters,
                    )
                    .order_by(*(["score"] if match_any else [rank, "score"]))
                    .limit(limit)
                )
                rows = conn.execute(stmt, {"q_all": q_all, "q_name": q_name})
                return [dict(r._mapping) for r in rows]
            # every token in the names, or in the details when visible: stemmed prefix matches on
            # Postgres, plain substring matches elsewhere
            if self.search_engine == "tsvector":
                names = sa.literal_column(pg_tsvector("gloss_search.name_text"))
                details = sa.literal_column(pg_tsvector("gloss_search.detail_text"))

                def in_name(t: str) -> Any:
                    return names.op("@@")(sa.func.to_tsquery(sa.literal_column("'english'"), f"{t}:*"))

                def in_detail(t: str) -> Any:
                    return details.op("@@")(sa.func.to_tsquery(sa.literal_column("'english'"), f"{t}:*"))
            else:
                def in_name(t: str) -> Any:
                    return search_t.c.name_text.ilike(f"%{t}%")

                def in_detail(t: str) -> Any:
                    return search_t.c.detail_text.ilike(f"%{t}%")

            cond = [
                sa.or_(in_name(t), sa.and_(in_detail(t), search_t.c.level <= max_level)) for t in tokens
            ]
            matched = cond[0] if len(cond) == 1 else (sa.or_(*cond) if match_any else sa.and_(*cond))
            hits_per_row = sum(sa.case((c, 1), else_=0) for c in cond)
            name_hits = sum(sa.case((in_name(t), 1), else_=0) for t in tokens)
            score = -(hits_per_row * 10 + name_hits)
            stmt = (
                sa.select(search_t, score.label("score"))
                .where(matched, *filters)
                .order_by(*(["score"] if match_any else [rank, "score"]), search_t.c.subject)
                .limit(limit)
            )
            return [dict(r._mapping) for r in conn.execute(stmt)]

    def search_vocabulary(self, *, names_only: bool = False) -> list[str]:
        """Words (stems) in the search index; ``names_only`` limits them to always-visible names.

        Needs the database's full-text search; empty for the plain-matching fallback.
        """
        if self.search_engine == "like":
            return []
        if self.search_engine == "tsvector":
            text = "name_text" if names_only else "name_text || ' ' || detail_text"
            inner = f"SELECT {pg_tsvector(text)} FROM gloss_search".replace("'", "''")
            with self._read() as conn:
                return [r[0] for r in conn.exec_driver_sql(f"SELECT word FROM ts_stat('{inner}')")]
        sql = "SELECT DISTINCT term FROM gloss_search_vocab"
        if names_only:
            sql += " WHERE col = 'name_text'"
        with self._read() as conn:
            return [r[0] for r in conn.exec_driver_sql(sql)]

    def get_meta(self, key: str) -> str | None:
        """A value from ``gloss_meta`` (schema and index versions), or None."""
        with self._read() as conn:
            return conn.execute(sa.select(meta_t.c.value).where(meta_t.c.key == key)).scalar()

    def set_meta(self, key: str, value: str) -> None:
        """Insert or overwrite a value in ``gloss_meta``."""
        with self.transaction() as conn:
            if not conn.execute(meta_t.update().where(meta_t.c.key == key).values(value=value)).rowcount:
                conn.execute(meta_t.insert().values(key=key, value=value))

    # -- check results --------------------------------------------------------
    def append_check_results(self, rows: Sequence[dict[str, Any]]) -> None:
        """Insert logged check results (rows shaped like ``gloss_check_results``) in one transaction."""
        if rows:
            with self.transaction() as conn:
                conn.execute(check_results_t.insert(), list(rows))

    def check_results(
        self,
        subject: str | None = None,
        subject_prefix: str | None = None,
        limit: int | None = 200,
        *,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Logged check results, newest first; a falsy ``limit`` returns all of them."""
        stmt = sa.select(check_results_t).order_by(check_results_t.c.id.desc())
        if subject:
            stmt = stmt.where(check_results_t.c.subject == subject)
        if subject_prefix:
            stmt = stmt.where(self._starts(check_results_t.c.subject, subject_prefix))
        if run_id:
            stmt = stmt.where(check_results_t.c.run_id == run_id)
        if limit:
            stmt = stmt.limit(limit)
        with self._read() as conn:
            return [{**r._asdict(), "ts": _aware(r.ts)} for r in conn.execute(stmt)]

    def rename_check_subjects(self, renames: dict[str, str]) -> None:
        """Keep logged check results with their field when its subject is re-keyed."""
        if not renames:
            return
        stmt = (
            check_results_t.update()
            .where(check_results_t.c.subject == sa.bindparam("_old"))
            .values(subject=sa.bindparam("_new"))
        )
        with self.transaction() as conn:
            conn.execute(stmt, [{"_old": o, "_new": n} for o, n in renames.items()])

    def close(self) -> None:
        """Dispose of an owned engine, refreshing SQLite's statistics first; a host's is left alone."""
        if not self.owns_engine:
            return  # the host's engine is the host's to dispose
        if self.dialect == "sqlite":
            try:
                with self.engine.begin() as conn:
                    conn.exec_driver_sql("PRAGMA optimize")  # refresh planner statistics cheaply
            except Exception:
                pass
        self.engine.dispose()
