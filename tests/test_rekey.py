"""Renames and moves: knowledge follows the table, and the log records how."""

import pytest

import glosslayer
from glosslayer import Column, Human, Metric, PermissionDenied, Relationship, RolePolicy, SemanticType, Status, Table
from glosslayer import String, checks
from glosslayer.model import EventType

ME = Human("me")


@pytest.fixture
def shop(g):
    Email = SemanticType("email_address", dtype=String(320), confidentiality="restricted", namespace="crm")
    customers = Table("crm.customers", description="One row per customer",
                      primary_key=[Column("customer_id", int)])
    customers.add_columns(Column("email", Email, description="Login e-mail"))
    orders = Table("crm.orders", description="One row per order", primary_key=[Column("order_id", int)],
                   unique_keys=[["order_id", "customer_id"]])
    orders.add_columns(Column("customer_id", int, description="Who ordered"), Column("amount", float))
    orders.add_metric(Metric("avg_amount", "AVG(amount)"))
    g.add(
        customers, orders,
        Relationship("orders_customers", on={orders["customer_id"]: customers["customer_id"]}, namespace="crm"),
        Metric("revenue", "SUM(orders.amount)", namespace="crm", description="Total order value"),
        Metric("customer_count", "COUNT(DISTINCT customers.customer_id)", namespace="crm"),
        source=ME, accept=True,
    )
    return g


def test_renamed_table_keeps_what_was_known(shop):
    g = shop
    rejected = g.propose("field:crm.orders/amount", "description", "wrong", source="ai:claude")
    g.reject(rejected.id, actor=ME)
    before = len(g.history("dataset:crm.orders"))

    res = g.rekey("dataset:crm.orders", "dataset:crm.order_lines", actor="catalog:wh", comment="renamed upstream")
    assert res.subjects["field:crm.orders/amount"] == "field:crm.order_lines/amount"

    assert g.current("dataset:crm.order_lines")["description"] == "One row per order"
    assert g.current("field:crm.order_lines/customer_id")["description"] == "Who ordered"
    assert g.find(dataset="crm.orders") == []
    table = g.table("crm.order_lines")
    assert [m.name for m in table.metrics.values()] == ["avg_amount"]
    # the join was rewritten, not broken, and nobody has to review it again
    rel = g.relationship("crm::orders_customers")
    assert (rel.from_table, rel.to_table) == ("crm.order_lines", "crm.customers")
    assert g.pending(kind="relationship") == []
    # history follows the subject, and says what happened
    history = g.history("dataset:crm.order_lines")
    assert len(history) > before and history[-1].type is EventType.REKEYED
    assert history[-1].data["from_subject"] == "dataset:crm.orders" and history[-1].actor == "catalog:wh"
    # a rejected suggestion stays rejected under the new name
    again = g.propose_many([glosslayer.Proposal("field:crm.order_lines/amount", "description", "wrong")],
                           source="ai:claude")
    assert again[0].outcome == "previously_rejected"
    # SQL that names the old table goes back to review; SQL that does not is left alone
    assert {a.status for a in g.find(subject="metric:crm::revenue", attribute="expression")} == {Status.STALE}
    assert g.metric("customer_count", namespace="crm").expression == {"sql": "COUNT(DISTINCT customers.customer_id)"}
    assert g.search("order_lines", for_ai=False)[0]["subject"] == "dataset:crm.order_lines"
    assert "dataset:crm.orders" not in {h["subject"] for h in g.search("orders", for_ai=False)}


def test_rekey_is_rebuilt_from_the_log(shop):
    g = shop
    g.rekey("dataset:crm.orders", "dataset:crm.order_lines", actor=ME)
    g.rekey("field:crm.order_lines/customer_id", "field:crm.order_lines/buyer_id", actor=ME)
    before = sorted((a.to_dict() for a in g.find()), key=lambda d: d["id"])
    hits = g.search("buyer", for_ai=False)
    g.rebuild_projection()
    assert sorted((a.to_dict() for a in g.find()), key=lambda d: d["id"]) == before
    assert g.search("buyer", for_ai=False) == hits and hits


def test_renamed_column(shop):
    g = shop
    res = g.rekey("field:crm.orders/customer_id", "field:crm.orders/buyer_id", actor=ME)
    assert res.moved and res.rewritten
    assert g.current("field:crm.orders/buyer_id")["description"] == "Who ordered"
    assert g.relationship("crm::orders_customers").from_columns == ["buyer_id"]
    assert g.relationship("crm::orders_customers").to_columns == ["customer_id"]  # the other table's column
    assert g.current("dataset:crm.orders")["unique_keys"] == [["order_id", "buyer_id"]]
    assert g.current("dataset:crm.customers")["primary_key"] == ["customer_id"]
    with pytest.raises(ValueError):
        g.rekey("field:crm.orders/amount", "field:crm.customers/amount", actor=ME)


def test_moving_a_whole_schema(shop):
    g = shop
    res = g.rekey_namespace("crm", "main.sales", actor=ME)
    assert res.subjects["type:crm::email_address"] == "type:main.sales::email_address"
    assert [t.key for t in g.tables()] == ["main.sales.customers", "main.sales.orders"]
    rel = g.relationships()[0]
    assert str(rel.subject) == "relationship:main.sales::orders_customers"
    assert (rel.from_table, rel.to_table) == ("main.sales.orders", "main.sales.customers")
    assert g.metric("revenue", namespace="main.sales").description == "Total order value"
    # the column still points at its type, under the type's new name, and still inherits from it
    email = g.table("main.sales.customers")["email"]
    assert email.semantic_type.ref == "main.sales::email_address"
    assert g.effective("field:main.sales.customers/email")["confidentiality"] == "restricted"
    assert g.find(namespace="crm") == []
    with pytest.raises(ValueError):
        g.rekey_namespace("main.sales", "main.sales.eu", actor=ME)
    with pytest.raises(KeyError):
        g.rekey_namespace("nowhere", "elsewhere", actor=ME)


def test_renaming_a_type_updates_its_users(shop):
    g = shop
    g.rekey("type:crm::email_address", "type:email", actor=ME)
    assert g.current("field:crm.customers/email")["semantic_type"] == "email"
    assert g.type_usages("email") == ["field:crm.customers/email"]
    assert g.semantic_type("email").confidentiality == "restricted"


def test_a_value_already_accepted_at_the_target_wins(shop):
    g = shop
    newer = Table("crm.order_lines", description="Lines of an order", source="warehouse.order_lines")
    g.add(newer.add_column("amount", float), source=ME, accept=True)
    res = g.rekey("dataset:crm.orders", "dataset:crm.order_lines", actor=ME)
    assert res.superseded >= 1
    assert g.current("dataset:crm.order_lines")["description"] == "Lines of an order"
    moved = [a for a in g.find(subject="dataset:crm.order_lines", attribute="description")
             if a.value == "One row per order"]
    assert moved[0].status is Status.SUPERSEDED  # kept, and can be restored
    assert g.restore(moved[0].id, actor=ME).status is Status.ACCEPTED


def test_check_results_follow_a_renamed_column(shop):
    g = shop
    from glosslayer.checks import CheckResult

    g.record_check_results([CheckResult("field:crm.orders/amount", checks.not_null().to_dict(), True, 0, 3)])
    g.rekey("dataset:crm.orders", "dataset:crm.sales", actor=ME)
    assert g.check_history("field:crm.sales/amount")[0]["passed"] is True
    assert g.check_history(dataset="crm.orders") == []


def test_a_move_needs_a_say_at_both_ends():
    policy = RolePolicy(stewards={"sales.": {"human:alice"}, "finance.": {"human:bob"}}, admins={"human:root"})
    with glosslayer.open("sqlite://", policy=policy) as g:
        g.add(Table("sales.orders", description="x").add_column("amount", float), source="human:root", accept=True)
        with pytest.raises(PermissionDenied):
            g.rekey("dataset:sales.orders", "dataset:finance.orders", actor="human:alice")
        with pytest.raises(PermissionDenied):
            g.rekey("dataset:sales.orders", "dataset:sales.order_lines", actor="ai:claude")
        assert g.table("sales.orders")  # nothing moved
        g.rekey("dataset:sales.orders", "dataset:sales.order_lines", actor="human:alice")
        g.rekey("dataset:sales.order_lines", "dataset:finance.orders", actor="human:root")
        assert [t.key for t in g.tables()] == ["finance.orders"]


def test_bad_rekeys(g):
    with pytest.raises(ValueError):
        g.rekey("dataset:a", "field:a/b", actor=ME)
    with pytest.raises(ValueError):
        g.rekey("dataset:a", "dataset:a", actor=ME)
    with pytest.raises(KeyError):
        g.rekey("dataset:a", "dataset:b", actor=ME)
