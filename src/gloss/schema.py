"""Gloss's database schema, for hosts that own their database.

Gloss keeps its tables (all prefixed ``gloss_``) in whatever database it is given and
tracks their version itself, in ``gloss_meta``. It never needs a migration tool of its
own, and it never touches a table that is not its own.

A host with migrations calls :func:`upgrade` from one of them, on the connection the
migration already holds, and again whenever it upgrades Gloss:

    # an Alembic revision in the host's chain
    def upgrade():
        gloss.schema.upgrade(op.get_bind())

and keeps Gloss's tables out of its own autogenerate:

    context.configure(..., include_object=lambda obj, name, type_, *_:
                      not (type_ == "table" and gloss.schema.is_gloss_table(name)))

``upgrade`` is idempotent: it creates the tables in an empty database and otherwise walks
the version ladder from wherever the database is to :data:`LATEST`.
"""

from __future__ import annotations

from collections.abc import Callable

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

LATEST = 1
TABLE_PREFIX = "gloss_"
_VERSION_KEY = "schema_version"

# version -> the step that takes a database from version - 1 to version.
# Add a step (plain DDL on the connection) whenever a table changes after a release, and bump LATEST.
_STEPS: dict[int, Callable[[Connection], None]] = {}


class SchemaNotReady(RuntimeError):
    """The database has no Gloss tables, or they are at another version than this Gloss expects."""


def is_gloss_table(name: str) -> bool:
    """Whether a table is one of Gloss's (``gloss_`` prefix), e.g. to keep it out of a host's autogenerate."""
    return name.startswith(TABLE_PREFIX)


def metadata() -> sa.MetaData:
    """The SQLAlchemy metadata of Gloss's tables (never mixed into a host's own metadata)."""
    from .store.sql import metadata as md

    return md


def current_version(bind: Engine | Connection) -> int | None:
    """The schema version in this database: None when Gloss has no tables there."""
    if isinstance(bind, Engine):
        with bind.connect() as conn:
            return current_version(conn)
    from .store.sql import meta_t

    insp = sa.inspect(bind)
    if not insp.has_table(meta_t.name):
        if insp.has_table("gloss_events"):
            return 0
        return None
    value = bind.execute(sa.select(meta_t.c.value).where(meta_t.c.key == _VERSION_KEY)).scalar()
    return int(value) if value is not None else 0


def has_fts(conn: Connection) -> bool:
    """Whether the SQLite FTS5 search tables exist here (Postgres full-text search uses indexes instead)."""
    return sa.inspect(conn).has_table("gloss_search_fts")


def create_fts(conn: Connection) -> None:
    """Full-text search over the search index, in the database's own way.

    SQLite: FTS5 tables (raises when SQLite was built without FTS5). Postgres: GIN indexes on the
    ``tsvector`` expressions the search query uses. Other databases: nothing; search matches plainly.
    """
    from .store.sql import FTS_TOKENIZER, pg_tsvector

    if conn.dialect.name == "postgresql":
        for column in ("name_text", "detail_text"):
            conn.exec_driver_sql(
                f"CREATE INDEX IF NOT EXISTS ix_gloss_search_{column}_fts ON gloss_search "
                f"USING gin (({pg_tsvector(column)}))"
            )
        # prefix filters (paths, subject prefixes, short ids) are LIKE 'x%': these serve them in any collation
        for table, column in (("gloss_assertions", "namespace"), ("gloss_assertions", "dataset"),
                              ("gloss_assertions", "subject"), ("gloss_assertions", "id"),
                              ("gloss_search", "namespace"), ("gloss_search", "dataset")):
            conn.exec_driver_sql(
                f"CREATE INDEX IF NOT EXISTS ix_{table}_{column}_prefix ON {table} ({column} text_pattern_ops)"
            )
        return
    if conn.dialect.name != "sqlite":
        return
    conn.exec_driver_sql(
        "CREATE VIRTUAL TABLE IF NOT EXISTS gloss_search_fts USING fts5("
        f"subject UNINDEXED, name_text, detail_text, tokenize='{FTS_TOKENIZER}')"
    )
    # every indexed word with its document count, per column, so suggestions can be limited to names
    conn.exec_driver_sql(
        "CREATE VIRTUAL TABLE IF NOT EXISTS gloss_search_vocab USING fts5vocab(gloss_search_fts, 'col')"
    )


def _set_version(conn: Connection, version: int) -> None:
    from .store.sql import meta_t

    where = meta_t.c.key == _VERSION_KEY
    if not conn.execute(meta_t.update().where(where).values(value=str(version))).rowcount:
        conn.execute(meta_t.insert().values(key=_VERSION_KEY, value=str(version)))


def upgrade(bind: Engine | Connection) -> int:
    """Create Gloss's tables or bring them to :data:`LATEST`. Returns the version reached.

    Given a connection, it works inside the caller's transaction and leaves committing to the caller.
    """
    if isinstance(bind, Engine):
        with bind.begin() as conn:
            return upgrade(conn)
    conn = bind
    version = current_version(conn)
    if version == LATEST:
        return version
    if version is None:
        metadata().create_all(conn)
        try:
            with conn.begin_nested():
                create_fts(conn)
        except sa.exc.DBAPIError:
            if conn.dialect.name != "sqlite":
                raise
            # SQLite built without FTS5: search falls back to plain matching
        _set_version(conn, LATEST)
        return LATEST
    if version == 0:
        raise SchemaNotReady(
            "This database holds a Gloss ledger from a pre-release without a schema version. "
            "Its event format is not supported; start a new ledger."
        )
    if version > LATEST:
        raise SchemaNotReady(f"The ledger is at schema version {version}; this Gloss supports up to {LATEST}")
    for step in range(version + 1, LATEST + 1):
        _STEPS[step](conn)
        _set_version(conn, step)
    return LATEST


def require(bind: Engine | Connection) -> None:
    """Raise :class:`SchemaNotReady` unless the database is at :data:`LATEST`."""
    version = current_version(bind)
    if version != LATEST:
        state = "has no Gloss tables" if version is None else f"is at schema version {version}"
        raise SchemaNotReady(
            f"The database {state}; this Gloss needs version {LATEST}. "
            "Run gloss.schema.upgrade(connection) from the host's migrations (or pass create=True)."
        )
