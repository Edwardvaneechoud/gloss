"""Column objects in keys/relationships, typed actors, precise dtypes."""

import copy
import pickle

import pytest

import gloss
from gloss import (
    AI,
    Actor,
    Column,
    DateTime,
    Decimal,
    Human,
    Import,
    Int,
    PermissionDenied,
    Relationship,
    String,
    Table,
    UInt,
)
from gloss.dtypes import Float, Opaque, parse
from gloss.ossie import OssieSource, to_ossie, validate


# ----------------------------------------------------------------- actors
def test_actors_are_strings():
    h = Human("edward")
    assert h == "human:edward" and isinstance(h, str) and h.name == "edward" and h.kind == "human"
    assert repr(h) == "Human('edward')"
    assert Human("human:edward") == h
    assert Actor.parse("ai:claude") == AI("claude") and isinstance(Actor.parse("ai:claude"), AI)
    assert Actor.parse("weird:thing") == "weird:thing"
    assert copy.copy(h) == h and pickle.loads(pickle.dumps(h)) == h
    with pytest.raises(ValueError):
        Human("")
    with pytest.raises(TypeError):
        Actor("x")


def test_actors_in_the_ledger(g):
    a = g.propose("field:t/x", "description", "d", source=AI("claude"))
    with pytest.raises(PermissionDenied):
        g.accept(a.id, actor=AI("claude"))
    g.accept(a.id, actor=Human("alice"))
    got = g.get(a.id)
    assert got.source == "ai:claude" and isinstance(got.source_actor, AI)
    assert g.history("field:t/x")[-1].actor == Human("alice")


# ----------------------------------------------------------------- columns as objects
def test_primary_key_columns_are_added():
    rid = Column("return_id", Int(32))
    t = Table("shop.returns", primary_key=[rid])
    assert t.primary_key == ["return_id"] and t["return_id"] is rid and rid.table == "shop.returns"
    t2 = Table("x", primary_key=Column("id"))  # single column, no list
    assert t2.primary_key == ["id"]
    t3 = Table("y", primary_key=["a", Column("b")], unique_keys=[[Column("c"), "b"]])
    assert t3.primary_key == ["a", "b"] and t3.unique_keys == [["c", "b"]] and "c" in t3


def test_column_can_only_belong_to_one_table():
    c = Column("id")
    Table("a").add_column(c)
    with pytest.raises(ValueError, match="already belongs"):
        Table("b").add_column(c)


def test_relationship_infers_tables_from_columns():
    orders = Table("shop.orders", primary_key=[Column("order_id", Int(64))])
    returns = Table("shop.returns").add_columns(Column("order_id", Int(64)))
    rel = Relationship("returns_orders", on={returns["order_id"]: orders["order_id"]})
    assert (rel.from_table, rel.to_table) == ("shop.returns", "shop.orders")
    assert rel.from_columns == ["order_id"] and rel.to_columns == ["order_id"]
    rel2 = Relationship("r2", returns, orders, on={"order_id": "order_id"})  # Table objects
    assert rel2.from_table == "shop.returns"
    with pytest.raises(ValueError, match="belong to"):
        Relationship("bad", "shop.other", on={returns["order_id"]: orders["order_id"]})
    with pytest.raises(ValueError, match="added to a Table"):
        Relationship("bad", on={Column("loose"): orders["order_id"]})


# ----------------------------------------------------------------- dtypes
@pytest.mark.parametrize(
    "dtype,ossie,physical",
    [
        (Int(32), "Integer", "Int32"),
        (UInt(8), "Integer", "UInt8"),
        (Int, "Integer", None),
        (Float(32), "Float", "Float32"),
        (Decimal(18, 2), "Decimal", "Decimal(18,2)"),
        (Decimal(10), "Decimal", "Decimal(10)"),
        (String(255), "String", "String(255)"),
        (String, "String", None),
        (DateTime(), "DateTime", None),
        (DateTime(tz="UTC"), "DateTimeTz", "DateTime(tz=UTC)"),
        (DateTime(tz="Europe/Amsterdam", unit="us"), "DateTimeTz", "DateTime(us, tz=Europe/Amsterdam)"),
        (Opaque("geometry"), "Opaque", "geometry"),
        ("Int16", "Integer", "Int16"),
        ("Integer", "Integer", None),
        (int, "Integer", None),
    ],
)
def test_dtype_mapping(dtype, ossie, physical):
    c = Column("c", dtype)
    assert (c.datatype, c.physical_type) == (ossie, physical)


def test_dtype_round_trip_through_parse():
    for d in (Int(32), UInt(16), Float(64), Decimal(18, 2), Decimal(9), String(80),
              DateTime(tz="UTC"), DateTime(unit="ns"), Opaque("geometry")):
        assert parse(d.physical, d.ossie) == d
    assert parse(None, "Integer") == Int()
    assert Column("x", Int(32)).dtype == Int(32)


@pytest.mark.parametrize("bad", [lambda: Int(12), lambda: Decimal(scale=2), lambda: Decimal(4, 6),
                                 lambda: String(0), lambda: DateTime(unit="h")])
def test_dtype_validation(bad):
    with pytest.raises(ValueError):
        bad()


# ----------------------------------------------------------------- all together
def test_full_example(g):
    orders = Table("shop.orders", primary_key=[Column("order_id", Int(64))])
    returns = Table(
        "shop.returns",
        source="warehouse.shop.returns",
        description="One row per returned order",
        primary_key=[Column("return_id", Int(32))],
    )
    returns.add_columns(
        Column("order_id", Int(64), description="The order that was returned"),
        Column("reason", String(255), synonyms=["return reason"]),
        Column("refund", Decimal(18, 2), expression="refund_cents / 100.0"),
        Column("returned_at", DateTime(tz="UTC")),
    )
    rel = Relationship("returns_orders", on={returns["order_id"]: orders["order_id"]})
    r = g.add(returns, orders, rel, source=Human("edward"))
    assert r.created > 0 and all(a.source == "human:edward" for a in g.find(dataset="shop.returns"))
    g.bulk("accept", gloss.Query(), actor=Human("edward"))

    back = g.table("shop.returns")
    assert back["return_id"].dtype == Int(32)
    assert back["refund"].dtype == Decimal(18, 2)
    assert back["returned_at"].dtype == DateTime(tz="UTC")
    assert repr(back).startswith("Table('shop.returns', columns=[return_id:Int32, order_id:Int64")

    doc = to_ossie(g).document
    assert validate(doc) == []
    ret = next(d for d in doc["datasets"] if d["name"] == "returns")
    refund = next(f for f in ret["fields"] if f["name"] == "refund")
    assert refund["datatype"] == "Decimal"
    assert refund["nonportable_extensions"][0]["data"]["physical_type"] == "Decimal(18,2)"

    # and it survives a round trip through Ossie into a fresh ledger
    with gloss.open("sqlite://") as g2:
        g2.propose_many(OssieSource(doc).proposals(), source=Import("ossie"))
        g2.bulk("accept", gloss.Query(), actor=Human("edward"))
        assert g2.table("returns")["refund"].dtype == Decimal(18, 2)


def test_apply_converges_on_what_the_code_declares(g):
    from gloss import Column, Human, Metric, Status, Table

    me = Human("me")
    v1 = Table("shop.orders", description="Orders")
    v1.add_columns(Column("amount", float, description="Order value"), Column("legacy_flag", bool))
    v1.add_metric(Metric("revenue", "SUM(amount)"))
    g.apply(v1, source=me, accept=True)
    g.propose("field:shop.orders/legacy_flag", "description", "Do not use", source="human:other", accept=True)

    v2 = Table("shop.orders", description="All orders")  # description changed, column and metric dropped
    v2.add_columns(Column("amount", float, description="Order value"))
    res = g.apply(v2, source=me, accept=True)
    assert (res.created, res.accepted, res.retracted) == (1, 1, 2)  # the column's datatype, the metric
    assert "retracted" in str(res)
    assert g.current("dataset:shop.orders")["description"] == "All orders"
    table = g.table("shop.orders")
    assert table.metrics == {} and "amount" in table
    # what someone else said about the dropped column is theirs, and stays
    assert g.current("field:shop.orders/legacy_flag") == {"description": "Do not use"}
    assert g.apply(v2, source=me).retracted == 0
    # add() alone never retracts
    g.add(Table("shop.orders"), source=me)
    assert g.current("field:shop.orders/amount")["description"] == "Order value"
    assert {a.status for a in g.find(subject="metric:shop.orders/revenue")} == {Status.RETRACTED}
