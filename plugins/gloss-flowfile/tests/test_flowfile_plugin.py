"""Tests against a fake of Flowfile's CatalogService (same shape as CatalogTableOut).

The fake mirrors what matters for embedding: ``access`` is None for an unrestricted service,
and a restricted one lists only what its user may see.
"""

import pytest

import gloss
from gloss import AI, PermissionDenied, Status
from gloss.sync import sync
from gloss_flowfile import Directory, Embedded, FlowfileCatalog, LineageSource, actor_for, policy_for


def _table(id, name, cols, *, produced_by=None, read_by=(), description=None, source="local"):
    return {
        "id": id,
        "name": name.rsplit(".", 1)[-1],
        "qualified_name": name,
        "description": description,
        "schema_columns": [{"name": c, "dtype": t} for c, t in cols],
        "table_type": "physical",
        "source_registration_id": produced_by,
        "source_registration_name": f"flow_{produced_by}" if produced_by else None,
        "read_by_flows": [{"id": f, "name": f"flow_{f}"} for f in read_by],
        "source": source,
    }


class FakeService:
    def __init__(self, tables, *, visible_ids=None):
        self.tables = tables
        self.access = None if visible_ids is None else object()  # a resolver means a user's view
        self._visible = visible_ids
        self.updated = {}

    def list_tables(self, namespace_id=None, user_id=None):
        return [t for t in self.tables if self._visible is None or t["id"] in self._visible]

    def update_table(self, table_id, name=None, description=None, **_):
        self.updated[table_id] = description
        for t in self.tables:
            if t["id"] == table_id and description is not None:
                t["description"] = description


CUSTOMER_COLS = [("customer_id", "Int64"), ("email", "Utf8"), ("country", "Utf8")]


def make_tables():
    return [
        _table(1, "main.raw.customers", CUSTOMER_COLS, read_by=[7]),
        _table(2, "main.mart.customers_clean", CUSTOMER_COLS, produced_by=7),
    ]


def test_catalog_seam_embedded():
    svc = FakeService(make_tables())
    cat = FlowfileCatalog(service=svc)
    assert cat.list_assets() == ["main.raw.customers", "main.mart.customers_clean"]
    assert cat.coverage().complete
    schema = cat.describe("main.mart.customers_clean")
    assert schema.extra["produced_by_flow"] == "flow_7" and schema.uid == "2"
    with gloss.open("sqlite://") as g:
        rep = sync(g, cat)
        assert len(rep.new) == 2
        assert g.current("field:main.raw.customers/customer_id")["datatype"] == "Integer"
    cat.write_back("main.raw.customers", {"description": "Raw CRM export"})
    assert svc.updated == {1: "Raw CRM export"}


def test_a_users_listing_never_marks_other_users_tables_gone():
    tables = make_tables()
    with gloss.open("sqlite://") as g:
        sync(g, FlowfileCatalog(service=FakeService(tables)))  # the backend's own, unrestricted sync
        bobs = g.propose("dataset:main.mart.customers_clean", "description", "Bob's mart", source="human:2")
        g.accept(bobs.id, actor="human:2")

        alice = FlowfileCatalog(service=FakeService(tables, visible_ids={1}))  # what Alice may see
        assert not alice.coverage().complete
        rep = sync(g, alice)
        assert rep.gone == [] and rep.staled == 0
        assert g.get(bobs.id).status is Status.ACCEPTED

        remote = FlowfileCatalog(fetch_tables=lambda: tables[:1])  # a user's token over REST: same caution
        assert sync(g, remote).gone == []


def test_hub_rows_are_not_copied_into_the_local_ledger():
    tables = make_tables() + [_table(-5, "org.finance.invoices", [("total", "Float64")], source="hub")]
    cat = FlowfileCatalog(service=FakeService(tables))
    assert "org.finance.invoices" not in cat.list_assets()
    with gloss.open("sqlite://") as g:
        sync(g, cat)
        assert g.find(dataset="org.finance.invoices") == []


def test_backend_events_keep_meaning_with_the_table():
    tables = make_tables()
    svc = FakeService(tables)
    with gloss.open("sqlite://", policy=policy_for(sharing=False)) as g:
        semantics = Embedded(g, FlowfileCatalog(service=svc))
        semantics.sync()
        me = semantics.view(user_id=1)
        # desktop mode: one user, own edits accepted at once, and Flowfile's description follows
        a = me.propose("dataset:main.raw.customers", "description", "Raw CRM export")
        assert a.status is Status.ACCEPTED and svc.updated == {1: "Raw CRM export"}
        me.propose("field:main.raw.customers/email", "confidentiality", "restricted")

        tables[0].update(name="clients", qualified_name="main.raw.clients")  # Flowfile's update_table(name=…)
        rep = semantics.table_renamed("main.raw.customers", "main.raw.clients")
        assert rep.moved == {"main.raw.customers": "main.raw.clients"} and rep.staled == 0
        assert g.current("dataset:main.raw.clients")["description"] == "Raw CRM export"
        assert g.current("field:main.raw.clients/email")["confidentiality"] == "restricted"
        assert semantics.sync().gone == []

        tables[0]["schema_columns"].append({"name": "phone", "dtype": "Utf8"})
        assert semantics.table_written("main.raw.clients").changed == ["main.raw.clients"]
        assert g.current("field:main.raw.clients/phone")["datatype"] == "String"

        del tables[0]
        assert semantics.table_deleted("main.raw.clients").gone == ["main.raw.clients"]
        assert g.find(dataset="main.raw.clients", statuses=[Status.ACCEPTED]) == []
        semantics.close()


def test_policy_from_request_context():
    with gloss.open("sqlite://", policy=policy_for(sharing=True)) as g:
        semantics = Embedded(g, FlowfileCatalog(service=FakeService(make_tables())), own_descriptions=False)
        semantics.sync()
        alice = semantics.view(user_id="u-alice", manage=["main.raw"], read=["main.raw"])
        bob = semantics.view(user_id="u-bob", manage=["main.mart"], read=["main.mart", "main.raw"])
        admin = semantics.view(user_id="u-root", admin=True)

        mine = alice.propose("dataset:main.raw.customers", "description", "Raw CRM export")
        assert mine.status is Status.PROPOSED  # shared mode: queued
        with pytest.raises(PermissionDenied):
            bob.accept(mine.id)  # he reads main.raw but does not manage it
        assert alice.accept(mine.id).status is Status.ACCEPTED
        assert [t.key for t in alice.tables()] == ["main.raw.customers"]
        assert len(bob.tables()) == 2 and len(admin.tables()) == 2
        with pytest.raises(PermissionDenied):
            alice.rekey("dataset:main.raw.customers", "dataset:main.mart.customers")  # needs both ends
        admin.rekey("dataset:main.raw.customers", "dataset:main.mart.customers")
        agent = semantics.view(user_id="u-alice", manage=["main.raw"], via="ai:assistant")
        idea = agent.propose("dataset:main.mart.customers", "synonyms", ["clients"], source=AI("assistant"))
        with pytest.raises(PermissionDenied):
            agent.accept(idea.id)


def test_actors_are_ids_and_names_come_from_the_host():
    assert actor_for(42) == "human:42"
    names = Directory({"42": "Ann"}.get)
    assert names.display_name("human:42") == "Ann"
    assert names.display_name("human:7") is None  # someone who left: the log names no one
    assert names.display_name("ai:assistant") == "assistant"


def test_lineage_propagates_confidentiality_and_descriptions():
    cat = FlowfileCatalog(service=FakeService(make_tables()))
    with gloss.open("sqlite://") as g:
        sync(g, cat)
        for attr, val in (("confidentiality", "restricted"), ("classification", ["pii.email"]),
                          ("description", "Login e-mail")):
            a = g.propose("field:main.raw.customers/email", attr, val, source="human:dpo")
            g.accept(a.id, actor="human:dpo")

        src = LineageSource(cat)
        assert src.upstream() == {"main.mart.customers_clean": [("main.raw.customers", "flow_7")]}
        res = g.propose_many(src.proposals(g), source=src.source_id)
        got = {(r.assertion.subject, r.assertion.attribute): r.assertion for r in res}

        conf = got[("field:main.mart.customers_clean/email", "confidentiality")]
        assert conf.value == "restricted" and conf.confidence == 0.6 and conf.status is Status.PROPOSED
        assert conf.evidence[0]["upstream"] == "main.raw.customers"
        assert got[("field:main.mart.customers_clean/email", "classification")].value == ["pii.email"]
        assert got[("dataset:main.mart.customers_clean", "confidentiality")].value == "restricted"
        assert ("field:main.mart.customers_clean/country", "confidentiality") not in got

        # running again does not duplicate proposals
        again = g.propose_many(src.proposals(g), source=src.source_id)
        assert {r.outcome for r in again} == {"unchanged"}


def test_lineage_reads_in_batches(monkeypatch):
    wide = [(f"c{i}", "Utf8") for i in range(40)]
    tables = [_table(1, "main.raw.wide", wide, read_by=[7]), _table(2, "main.mart.wide", wide, produced_by=7)]
    cat = FlowfileCatalog(service=FakeService(tables))
    with gloss.open("sqlite://") as g:
        sync(g, cat)
        calls = []
        for name in ("find", "find_for_subjects", "get", "get_many"):
            original = getattr(g.store, name)
            monkeypatch.setattr(g.store, name, lambda *a, _o=original, _n=name, **k: (calls.append(_n), _o(*a, **k))[1])
        list(LineageSource(cat).proposals(g))
        assert len(calls) <= 3  # not several lookups per column
