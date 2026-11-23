"""The REST contract from the other side: a client, and two ledgers used as one."""

import pytest

import gloss
from gloss import Column, Human, PermissionDenied, Proposal, Relationship, RolePolicy, Status, Table
from gloss.client import Client, Federation

pytest.importorskip("fastapi")


def _client(ledger, actor, catalogs=None):
    from fastapi.testclient import TestClient

    from gloss.server import create_app

    http = TestClient(create_app(ledger, catalogs=catalogs))

    def send(method, path, params, body):
        r = http.request(method, path, params=params, json=body, headers={"X-Gloss-Actor": actor})
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, r.text

    return Client(send=send)


@pytest.fixture
def hub():
    policy = RolePolicy(admins={"human:root"}, stewards={"org.": {"human:ann"}}, readers={"org.": {"human:me"}})
    with gloss.open("sqlite://", policy=policy) as g:
        orders = Table("org.sales.orders", description="One row per order", primary_key=[Column("order_id", int)])
        orders.add_columns(Column("customer_id", int), Column("amount", float, description="Order value"))
        customers = Table("org.sales.customers", primary_key=[Column("customer_id", int)])
        g.add(orders, customers, source=Human("root"), accept=True)
        yield g


def test_client_reads_and_proposes(hub):
    me, ann = _client(hub, "human:me"), _client(hub, "human:ann")
    assert me.current("field:org.sales.orders/amount")["description"] == "Order value"
    assert [t["key"] for t in me.tables()] == ["org.sales.customers", "org.sales.orders"]
    assert me.table("org.sales.orders")["columns"][2]["name"] == "amount"
    assert me.search("order value")[0]["subject"] == "field:org.sales.orders/amount"

    a = me.propose("field:org.sales.orders/amount", "label", "Amount", confidence=0.9)
    assert a.status is Status.PROPOSED and a.created_by == "human:me"
    assert [p.id for p in me.pending()] == [a.id] and me.count(statuses=["proposed"]) == 1
    assert me.get(a.id) == hub.get(a.id)  # the same object on both sides of the wire
    with pytest.raises(PermissionDenied):
        me.accept(a.id)
    assert ann.accept(a.id, comment="ok").status is Status.ACCEPTED
    assert [e.type.value for e in me.history("field:org.sales.orders/amount")][-2:] == ["proposed", "accepted"]
    with pytest.raises(KeyError):
        me.get("0" * 32)
    with pytest.raises(ValueError):
        me.propose("field:org.sales.orders/amount", "datatype", "Money")
    with pytest.raises(gloss.InvalidTransition):
        ann.accept(a.id)


def test_client_groups_bulk_and_versions(hub):
    me, ann = _client(hub, "human:me"), _client(hub, "human:ann")
    rel = Relationship("orders_customers", "org.sales.orders", "org.sales.customers",
                       on={"customer_id": "customer_id"}, namespace="org.sales")
    results = me.propose_many(list(rel.proposals()))
    assert {r.outcome for r in results} == {"created"}
    assert me.version("relationship:org.sales::orders_customers") is None
    done = ann.accept_group(results[0].assertion.group)
    assert len(done) == 4
    assert me.relationships()[0]["from_table"] == "org.sales.orders"
    assert me.version("relationship:org.sales::orders_customers") == me.relationships()[0]["version"]

    more = me.propose_many([Proposal("field:org.sales.orders/amount", "synonyms", ["revenue"]),
                            Proposal("field:org.sales.orders/customer_id", "description", "Who ordered")])
    changed, errors = ann.transition_many("accept", [r.assertion.id for r in more] + ["f" * 32])
    assert len(changed) == 2 and errors[0][0] == "f" * 32
    assert "orders_customers" in me.to_yaml()


def test_check_results_and_sync_over_rest(hub):
    from gloss.catalogs.file import FileCatalog
    from gloss.checks import CheckResult, not_null

    worker = _client(hub, "system:worker-1")
    run = worker.record_check_results(
        [CheckResult("field:org.sales.orders/amount", not_null().to_dict(), False, 2, 10)], run_id="run-77"
    )
    assert run == "run-77" and hub.check_history(run_id="run-77")[0]["failed_rows"] == 2
    assert hub.check_history(run_id="run-77")[0]["runner"] == "system:worker-1"

    from pathlib import Path

    catalog = FileCatalog(Path(__file__).resolve().parents[1] / "examples" / "catalog.yaml")
    root = _client(hub, "human:root", catalogs={"shop": catalog})
    rep = root._call("POST", "/sync/shop")
    assert sorted(rep["new"]) == ["shop.customers", "shop.orders"] and rep["accepted"] > 0
    with pytest.raises(PermissionDenied):
        _client(hub, "human:me", catalogs={"shop": catalog})._call("POST", "/sync/shop")
    with pytest.raises(KeyError):
        root._call("POST", "/sync/unknown")


def test_two_ledgers_as_one(hub):
    with gloss.open("sqlite://", policy=gloss.DefaultPolicy(mode="solo")) as local:
        mine = Table("scratch.notes", description="My notes").add_column("note", str)
        local.add(mine, source=Human("me"))
        both = Federation({"org.": _client(hub, "human:me")}, default=local.as_actor(Human("me")))

        assert both.current("dataset:org.sales.orders")["description"] == "One row per order"
        assert both.current("dataset:scratch.notes")["description"] == "My notes"
        assert {str(getattr(t, "key", None) or t["key"]) for t in both.tables()} == {
            "org.sales.customers", "org.sales.orders", "scratch.notes"}

        shared = both.propose("field:org.sales.orders/amount", "label", "Amount")
        own = both.propose("field:scratch.notes/note", "description", "Free text")
        assert shared.status is Status.PROPOSED  # the organisation reviews
        assert own.status is Status.ACCEPTED  # my own ledger is solo
        assert hub.get(shared.id).created_by == "human:me" and local.get(own.id)
        assert [a.id for a in both.pending()] == [shared.id]
        assert both.get(shared.id).attribute == "label" and both.get(own.id).attribute == "description"
        with pytest.raises(PermissionDenied):
            both.accept(shared.id)

        results = both.propose_many([Proposal("dataset:scratch.notes", "synonyms", ["memo"]),
                                     Proposal("dataset:org.sales.orders", "synonyms", ["sales"]),
                                     Proposal("field:scratch.notes/note", "label", "Note")])
        assert [r.assertion.subject for r in results] == [
            "dataset:scratch.notes", "dataset:org.sales.orders", "field:scratch.notes/note"]
        assert [r.accepted for r in results] == [True, False, True]
        hits = {h["subject"] for h in both.search("notes order", for_ai=False)}
        assert "dataset:scratch.notes" in hits and "dataset:org.sales.orders" in hits
        # nothing of the organisation's model was copied into the personal ledger
        assert local.find(namespace="org") == []
