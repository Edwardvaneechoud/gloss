"""Embedding in a host that owns its database: Gloss changes nothing it was not asked to."""

import pytest
import sqlalchemy as sa

import glosslayer
from glosslayer import ConcurrentWrite, Gloss, SQLStore, Status, schema
from glosslayer.model import Event, EventType, apply_event, new_id

F = "field:sales.orders/amount"


def _host_engine(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'host.db'}")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE host_things (id INTEGER PRIMARY KEY)")
    return engine


def _pragma(engine, name):
    with engine.connect() as conn:
        return conn.exec_driver_sql(f"PRAGMA {name}").scalar()


def test_a_host_engine_is_left_alone(tmp_path):
    engine = _host_engine(tmp_path)
    before = (_pragma(engine, "foreign_keys"), _pragma(engine, "synchronous"))
    with pytest.raises(schema.SchemaNotReady):
        SQLStore(engine)  # no tables yet, and Gloss does not create them uninvited
    assert sa.inspect(engine).get_table_names() == ["host_things"]

    with engine.begin() as conn:  # what a host migration does
        assert schema.upgrade(conn) == schema.LATEST
        assert schema.upgrade(conn) == schema.LATEST  # idempotent
    store = SQLStore(engine)
    engine.dispose()  # new connections: a listener would show up here
    assert (_pragma(engine, "foreign_keys"), _pragma(engine, "synchronous")) == before
    assert all(schema.is_gloss_table(t) for t in sa.inspect(engine).get_table_names() if t != "host_things")
    assert schema.current_version(engine) == schema.LATEST

    g = Gloss(store)
    a = g.propose(F, "description", "x", source="human:a")
    g.accept(a.id, actor="human:a")
    assert g.search("amount", for_ai=False)[0]["subject"] == F
    g.close()  # must not dispose the host's engine
    assert _pragma(engine, "foreign_keys") == before[0]


def test_writes_can_join_the_hosts_transaction(tmp_path):
    engine = _host_engine(tmp_path)
    schema.upgrade(engine)
    g = Gloss(SQLStore(engine))
    with engine.connect() as conn:
        tx = conn.begin()
        with g.store.using(conn):
            g.propose(F, "description", "x", source="human:a")
        tx.rollback()  # the host's call
    assert g.find() == []
    with engine.begin() as conn, g.store.using(conn):
        g.propose(F, "description", "x", source="human:a")
    assert len(g.find()) == 1


def test_a_ledger_without_a_schema_version_is_refused(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE gloss_events (seq INTEGER PRIMARY KEY)")
    with pytest.raises(schema.SchemaNotReady, match="pre-release"):
        schema.upgrade(engine)


def test_owned_database_is_created_and_versioned(tmp_path):
    url = f"sqlite:///{tmp_path / 'g.db'}"
    with glosslayer.open(url) as g:
        assert schema.current_version(g.store.engine) == schema.LATEST
        assert g.store.fts
    with glosslayer.open(url) as g:  # reopening does not recreate anything
        assert g.find() == []


def test_the_database_refuses_two_accepted_values(g):
    """Two writers racing to accept different values: the second one is told, not silently applied."""
    a = g.propose(F, "description", "one", source="human:a")
    b = g.propose(F, "description", "two", source="human:b")
    g.accept(a.id, actor="human:a")
    stale_view = g.get(b.id)  # what a second writer read before the first one committed
    ev = Event(EventType.ACCEPTED, b.id, "human:b", {"from_status": "proposed"}, new_id())
    with pytest.raises(ConcurrentWrite):
        g.store.append([ev], [apply_event(stale_view, ev)])
    assert g.current(F) == {"description": "one"}
    assert len(g.log()) == 3  # the failed append left no event behind
    assert g.accept(b.id, actor="human:b").status is Status.ACCEPTED  # the normal path supersedes
    assert g.get(a.id).status is Status.SUPERSEDED


def test_search_index_commits_with_the_events(g, monkeypatch):
    """A failure while indexing undoes the accept: the log and the index cannot drift apart."""
    a = g.propose(F, "description", "x", source="human:a")

    def boom(*args, **kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr("glosslayer.search_index.build_rows", boom)
    with pytest.raises(RuntimeError):
        g.accept(a.id, actor="human:a")
    assert g.get(a.id).status is Status.PROPOSED
    assert [e.type for e in g.log()] == [EventType.PROPOSED]
