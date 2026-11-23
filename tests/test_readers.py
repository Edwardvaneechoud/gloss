"""Reading the model back as objects, and as Ossie YAML."""

import pytest
import yaml

import glosslayer
from glosslayer import Column, Decimal, Human, Int, Metric, Relationship, SemanticType, String, Table
from glosslayer.ossie import validate

ED = Human("edward")


@pytest.fixture
def proposed(g):
    Email = SemanticType("email_address", dtype=String(320), confidentiality="restricted")
    customers = Table("crm.customers", primary_key=[Column("customer_id", Int(64))])
    customers.add_columns(Column("email", Email), Column("country", String(2)))
    orders = Table("crm.orders", primary_key=[Column("order_id", Int(64))])
    orders.add_columns(Column("customer_id", Int(64)), Column("amount", Decimal(18, 2)))
    orders.add_metric(Metric("revenue", "SUM(orders.amount)", datatype=Decimal))
    g.add(customers, orders, Relationship("orders_customers", on={orders["customer_id"]: customers["customer_id"]}),
          Metric("order_total", "SUM(orders.amount)"), source=ED)
    return g


def test_nothing_is_accepted_yet(proposed):
    g = proposed
    assert g.tables() == [] and g.semantic_types() == [] and g.relationships() == [] and g.metrics() == []
    with pytest.raises(KeyError, match="include_pending=True"):
        g.table("crm.customers")
    with pytest.raises(KeyError, match="include_pending=True"):
        g.relationship("orders_customers")
    with pytest.raises(ValueError, match="include_pending=True"):
        g.to_yaml()


def test_readers_with_pending(proposed):
    g = proposed
    assert [t.key for t in g.tables(include_pending=True)] == ["crm.customers", "crm.orders"]
    assert [t.key for t in g.tables("crm.cust", include_pending=True)] == ["crm.customers"]
    [st] = g.semantic_types(include_pending=True)
    assert st.name == "email_address" and st.confidentiality == "restricted"
    assert g.semantic_type("email_address", include_pending=True).resolved_dtype == String(320)
    [rel] = g.relationships(include_pending=True)
    assert (rel.from_table, rel.to_table, rel.from_columns) == ("crm.orders", "crm.customers", ["customer_id"])
    assert [r.name for r in g.relationships("crm.customers", include_pending=True)] == ["orders_customers"]
    assert g.relationships("crm.other", include_pending=True) == []
    names = [(m.table, m.name) for m in g.metrics(include_pending=True)]
    assert names == [(None, "order_total"), ("crm.orders", "revenue")]
    assert [m.name for m in g.metrics("crm.orders", include_pending=True)] == ["revenue"]
    assert g.metric("revenue", table="crm.orders", include_pending=True).datatype == "Decimal"
    assert g.metric("order_total", include_pending=True).table is None


def test_readers_after_accepting(proposed):
    g = proposed
    g.bulk("accept", glosslayer.Query(), actor=Human("alice"))
    assert len(g.tables()) == 2 and g.relationship("orders_customers").to_table == "crm.customers"
    assert g.semantic_types()[0].name == "email_address"
    t = g.tables()[0]
    assert t["email"].semantic_type.name == "email_address"  # types resolved to objects


def test_yaml(proposed, tmp_path):
    g = proposed
    text = g.to_yaml(include_pending=True)
    assert "&id" not in text and "*id" not in text  # inherited values written out, not aliased
    doc = yaml.safe_load(text)
    assert validate(doc) == [] and {d["name"] for d in doc["datasets"]} == {"customers", "orders"}
    g.to_yaml(tmp_path / "crm.yaml", include_pending=True, tables="crm.orders", name="crm")
    only = yaml.safe_load((tmp_path / "crm.yaml").read_text())
    assert only["name"] == "crm" and [d["name"] for d in only["datasets"]] == ["orders"]
    g.bulk("accept", glosslayer.Query(), actor=Human("alice"))
    assert g.to_ossie().warnings == [] and "datasets:" in g.to_yaml()


def test_subscript_access(proposed):
    g = proposed
    with pytest.raises(KeyError, match="only proposed facts"):
        g["crm.customers"]
    with pytest.raises(KeyError, match="Nothing known"):
        g["crm.nope"]
    assert "crm.customers" not in g
    g.bulk("accept", glosslayer.Query(), actor=Human("alice"))
    assert g["crm.customers"].key == "crm.customers" and "crm.customers" in g
    assert g["dataset:crm.customers"].key == "crm.customers"
    assert g["field:crm.customers/email"].semantic_type.name == "email_address"
    assert g["type:email_address"].confidentiality == "restricted"
    assert g["relationship:orders_customers"].to_table == "crm.customers"
    assert g["metric:crm.orders/revenue"].expression == {"sql": "SUM(orders.amount)"}
    assert g["metric:order_total"].table is None
    with pytest.raises(KeyError, match="no accepted column"):
        g["field:crm.customers/nope"]
    with pytest.raises(TypeError):
        g[1]
