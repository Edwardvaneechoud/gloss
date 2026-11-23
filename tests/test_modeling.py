import datetime as dt
from decimal import Decimal

import pytest

import glosslayer
from glosslayer import Column, Metric, PermissionDenied, Relationship, Status, Table
from glosslayer.ossie import to_ossie, validate


def make_returns() -> Table:
    t = Table("shop.returns", source="warehouse.shop.returns", description="Returns", primary_key=["return_id"])
    t.add_columns(
        Column("return_id", "Integer"),
        Column("order_id", int, description="The order that was returned"),
        Column("reason", "String", synonyms=["return reason"]),
    )
    t.add_column("refund", "Decimal", expression="refund_cents / 100.0")
    return t


def test_datatype_forms():
    assert Column("a", "Integer").datatype == "Integer"
    assert Column("a", "Int64").datatype == "Integer"
    assert Column("a", "varchar(20)").datatype == "String"
    assert Column("a", bool).datatype == "Boolean"
    assert Column("a", Decimal).datatype == "Decimal"
    assert Column("a", dt.datetime).datatype == "DateTime"
    assert Column("a", dt.date).datatype == "Date"
    assert Column("a").datatype is None
    with pytest.raises(ValueError):
        Column("a", "Money")


def test_table_building():
    t = make_returns()
    assert len(t) == 4 and "reason" in t and t["refund"].expression == "refund_cents / 100.0"
    assert [c.name for c in t] == ["return_id", "order_id", "reason", "refund"]
    with pytest.raises(ValueError):
        t.add_column(Column("reason"))
    with pytest.raises(TypeError):
        t.add_column(Column("x"), "Integer")


def test_add_proposes_and_is_idempotent(g):
    t = make_returns()
    r = g.add(t, source="human:a")
    assert r.created == len(r.assertion_ids) and r.created > 0
    assert all(a.status is Status.PROPOSED for a in g.find(dataset="shop.returns"))
    assert g.add(t, source="human:a").created == 0

    t["reason"].description = "Customer-selected reason"
    assert g.add(t, source="human:a").created == 1


def test_add_with_accept_and_load(g):
    t = make_returns()
    t.add_metric(Metric("refund_total", "SUM(returns.refund)", datatype="Decimal"))
    r = g.add(t, actor="human:a", accept=True)
    assert r.accepted == r.created
    back = g.table("shop.returns")
    assert [c.name for c in back] == ["return_id", "order_id", "reason", "refund"]
    assert back["order_id"].description == "The order that was returned"
    assert back["refund"].expression == {"sql": "refund_cents / 100.0"}
    assert back.primary_key == ["return_id"] and back.source == "warehouse.shop.returns"
    assert back.metrics["refund_total"].expression == {"sql": "SUM(returns.refund)"}
    assert g.current("metric:shop.returns/refund_total")["datatype"] == "Decimal"


def test_column_without_type_still_exists(g):
    g.add(Table("t").add_columns(Column("bare")), actor="human:a", accept=True)
    assert "bare" in g.table("t")


def test_load_pending_and_missing(g):
    g.add(make_returns(), source="human:a")
    with pytest.raises(KeyError):
        g.table("shop.returns")
    assert len(g.table("shop.returns", include_pending=True)) == 4


def test_relationship_and_export(g):
    orders = Table("shop.orders").add_columns(Column("order_id", "Integer"))
    rel = Relationship("returns_orders", "shop.returns", "shop.orders", on={"order_id": "order_id"})
    g.add(make_returns(), orders, rel, actor="human:a", accept=True)
    doc = to_ossie(g).document
    assert validate(doc) == []
    assert doc["relationships"] == [
        {"name": "returns_orders", "from": "returns", "to": "orders", "from_columns": ["order_id"], "to_columns": ["order_id"]}
    ]
    with pytest.raises(ValueError):
        Relationship("bad", "a", "b")


def test_ai_source_cannot_self_accept(g):
    with pytest.raises(PermissionDenied):
        g.add(make_returns(), source="ai:claude", accept=True)
    assert all(a.status is Status.PROPOSED for a in g.find(dataset="shop.returns"))


def test_confidentiality_on_columns(g):
    t = Table("crm.people").add_columns(Column("email", str, confidentiality="restricted", classification=["pii.email"]))
    g.add(t, actor="human:a", accept=True)
    assert g.current("field:crm.people/email")["confidentiality"] == "restricted"
    from glosslayer import ai

    assert ai.model_context(g)["datasets"][0]["fields"][0]["redacted"] is True


def test_requires_source(g):
    with pytest.raises(ValueError):
        g.add(make_returns())


def test_top_level_exports():
    assert glosslayer.Table is Table and glosslayer.Column is Column
