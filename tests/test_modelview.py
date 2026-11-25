"""A namespace's model with its review state beside it."""

from dataclasses import dataclass

import pytest

import glosslayer
from glosslayer import AI, Column, Human, Metric, Proposal, Relationship, Status, Table
from glosslayer.model import LIVE_STATUSES
from glosslayer.query import MetricQuery, compile_query
from glosslayer.seams import AssetSchema
from glosslayer.seams import Column as SeamColumn

ANN, BOB = Human("ann"), Human("bob")
ORDERS, CUSTOMERS, PRODUCTS = "shop.sales.orders", "shop.sales.customers", "shop.stock.products"


@dataclass
class HostTable:
    """A host's own table type: the attributes are what counts."""

    key: str
    columns: list
    uid: int
    name: str | None = None
    description: str | None = None


def shape(key, *columns, uid=None, **kw):
    return HostTable(key, [SeamColumn(c, "Int64") for c in columns], uid, **kw)


TABLES = [
    shape(ORDERS, "order_id", "customer_id", "product_id", "amount", uid=1, description="From the catalog"),
    shape(CUSTOMERS, "customer_id", "country", uid=2, name="Clients"),
]
RELATED = [*TABLES, shape(PRODUCTS, "product_id", "price", uid=3)]


def join(name, a, b, col, namespace):
    return Relationship(name, a, b, from_columns=[col], to_columns=[col], namespace=namespace)


@pytest.fixture
def ledger(g):
    g.add(
        Metric("revenue", "SUM(amount)", table=ORDERS, description="Booked", dimensions=["status"]),
        join("oc", ORDERS, CUSTOMERS, "customer_id", "shop.sales"),
        join("op", ORDERS, PRODUCTS, "product_id", "shop"),
        join("elsewhere", PRODUCTS, "shop.stock.x", "product_id", "shop"),
        source=ANN, accept=True,
    )
    g.propose("model:shop.sales", "name", "Sales", source=ANN, accept=True)
    g.propose(f"dataset:{ORDERS}", "description", "One row per order", source=ANN, accept=True)
    g.propose(f"field:{ORDERS}/amount", "label", "Amount", source=ANN, accept=True)
    g.propose(f"field:{ORDERS}/amount", "label", "Order value", source=AI("claude"), comment="clearer", confidence=0.6)
    g.propose(f"field:{ORDERS}/dropped", "label", "Gone", source=ANN)
    return g


def live(g, **kw):
    return g.model_view("shop.sales", tables=TABLES, related=RELATED, joins_of="shop", statuses=LIVE_STATUSES, **kw)


def test_tables_columns_and_what_waits_beside_them(ledger):
    v = live(ledger)
    assert [(t.key, t.name, t.uid, t.description) for t in v.tables] == [
        (ORDERS, "orders", 1, "One row per order"), (CUSTOMERS, "Clients", 2, None)
    ]
    assert v.value("name") == "Sales" and v.namespace == "shop.sales"
    orders = v.table("orders")
    assert orders is v.table(ORDERS) and v.table("Clients").uid == 2 and v.table("zz") is None
    # the catalog's columns in its order, then one that only a fact still names
    assert [(c.name, c.dtype) for c in orders.fields] == [
        ("order_id", "Int64"), ("customer_id", "Int64"), ("product_id", "Int64"), ("amount", "Int64"), ("dropped", None)
    ]
    amount = orders.column("amount")
    assert amount.value("label") == "Amount" and amount.value("role") is None and orders.column("zz") is None
    waiting = amount.pending[0]
    assert (waiting.value, waiting.status, waiting.source_kind, waiting.rationale, waiting.confidence) == (
        "Order value", Status.PROPOSED, "ai", "clearer", 0.6
    )
    assert waiting.is_open and amount.facts["label"].accepted_by == "human:ann"
    assert orders.column("dropped").value("label") == "Gone"  # nothing accepted yet: the newest open one is shown
    assert v.counts()["pending"] == 2 and v.counts()["stale"] == 0


def test_accepted_only_is_the_published_model_and_compiles(ledger):
    v = ledger.model_view("shop.sales", tables=TABLES, related=RELATED, joins_of="shop")
    assert v.table("orders").column("amount").pending == [] and v.table("orders").column("dropped") is None
    assert v.columns[PRODUCTS] == {"product_id", "price"}
    v2 = live(ledger)
    assert "dropped" in v2.columns[ORDERS]
    q = compile_query(v, MetricQuery("revenue"), tables={ORDERS})
    assert (q.metric.accepted_by, q.metric.table_uid, q.metric.version) == (
        "human:ann", 1, ledger.version(f"metric:{ORDERS}/revenue")
    )
    ledger.add(Metric("value", "SUM(products.price)", table=ORDERS), source=ANN, accept=True)
    v = ledger.model_view("shop.sales", tables=TABLES, related=RELATED, joins_of="shop")
    assert compile_query(v, MetricQuery("value")).tables == [ORDERS, PRODUCTS]


def test_joins_of_a_schema_and_of_its_catalog(ledger):
    v = live(ledger)
    assert [(j.name, j.namespace, j.from_uid, j.to_uid) for j in v.joins] == [
        ("oc", "shop.sales", 1, 2), ("op", "shop", 1, 3)  # the catalog's join that touches no table here is left out
    ]
    assert v.joins[1].version and v.joins[1].from_columns == ["product_id"]
    assert [j.name for j in ledger.model_view("shop.sales", tables=TABLES).joins] == ["oc"]
    assert {j.name for j in ledger.model_view("shop", tables=RELATED).joins} == {"oc", "op", "elsewhere"}
    # a join half proposed is not a join yet
    ledger.propose("relationship:shop.sales::half", "from", ORDERS, source=ANN)
    assert "half" not in {j.name for j in live(ledger).joins}


def test_a_metric_shows_one_definition_and_names_the_edit_beside_it(ledger):
    key = f"metric:{ORDERS}/revenue"
    edit = Metric("revenue", "SUM(amount) / 2", table=ORDERS, filter="amount > 0")
    ledger.add(edit, source=BOB)
    (m,) = live(ledger).metrics
    assert (m.expression, m.filter, m.status, m.dimensions) == ("SUM(amount)", None, Status.ACCEPTED, ["status"])
    assert m.pending_group and m.pending_group != m.group
    ledger.accept_group(m.pending_group, actor=ANN)
    (m,) = live(ledger).metrics
    assert (m.expression, m.filter, m.pending_group, m.key) == ("SUM(amount) / 2", "amount > 0", None, key)

    ledger.add(Metric("fresh", "COUNT(*)", table=CUSTOMERS, description="How many"), source=BOB)
    fresh = next(m for m in live(ledger).metrics if m.name == "fresh")
    assert (fresh.status, fresh.description, fresh.accepted_by, fresh.group is not None) == (
        Status.PROPOSED, "How many", None, True
    )
    ledger.mark_stale(ledger.find(subject=fresh.key, attribute="expression")[0].id, actor="system:x", reason="r")
    assert next(m for m in live(ledger).metrics if m.name == "fresh").status is Status.STALE
    ledger.propose(f"metric:{ORDERS}/wordy", "description", "no expression", source=BOB)
    assert "wordy" not in {m.name for m in live(ledger).metrics}


def test_metrics_of_the_namespace_itself(ledger):
    ledger.add(Metric("margin", "SUM(a) / SUM(b)", namespace="shop.sales"), source=ANN, accept=True)
    v = live(ledger)
    assert [(m.name, m.table) for m in v.own_metrics] == [("margin", None)] and v.metrics[-1].name == "margin"


def test_without_a_catalog_the_tables_are_those_facts_are_about(g):
    g.add(Table("s.t", primary_key=[Column("id", int)]).add_columns(Column("email", str)), source=ANN, accept=True)
    v = g.model_view()
    assert [(t.key, [(c.name, c.dtype) for c in t.fields]) for t in v.tables] == [
        ("s.t", [("email", "String"), ("id", "Integer")])
    ]
    assert g.model_view("s", tables=[AssetSchema("s.t", [])]).tables[0].column("id").dtype is None
    assert g.model_view("other").tables == []


def test_facts_keep_their_order_and_say_where_they_came_from(ledger):
    view = ledger.as_actor(ANN, via="server:x")
    first = view.propose("dataset:shop.sales.orders", "owner", "team", comment="they run it")
    other = ledger.find(subject=f"field:{ORDERS}/amount", attribute="label")
    facts = ledger.facts([*other, first])
    assert [f.id for f in facts] == [*(a.id for a in other), first.id]
    assert (facts[-1].via, facts[-1].rationale, facts[-1].accepted_by) == ("server:x", "they run it", None)
    assert ledger.facts([]) == []
    ledger.propose("dataset:x", "owner", "me", source="dbt:run", actor=ANN)
    assert ledger.facts(ledger.find(subject="dataset:x"))[0].source_kind == "other"


def test_details_above_the_readers_level_are_not_in_the_view(ledger):
    ledger.propose(f"field:{ORDERS}/amount", "confidentiality", "restricted", source=ANN, accept=True)
    agent = ledger.as_actor(AI("claude"))
    amount = agent.model_view("shop.sales", tables=TABLES, statuses=LIVE_STATUSES).table("orders").column("amount")
    assert amount.value("confidentiality") == "restricted" and amount.value("label") is None and not amount.pending
    hidden = agent.facts(agent.find(subject=f"field:{ORDERS}/amount", attribute="label"))
    assert len(hidden) == 2 and all(f.redacted and f.value is None and f.rationale is None for f in hidden)


def test_the_review_queue_has_one_entry_per_decision(ledger):
    ledger.add(Metric("fresh", "COUNT(*)", table=CUSTOMERS, description="How many"),
               join("cc", CUSTOMERS, PRODUCTS, "customer_id", "shop.sales"), source=BOB)
    ledger.propose("model:shop.sales", "description", "All sales", source=BOB)
    ledger.propose("dataset:shop.sales.unlisted", "description", "x", source=BOB)
    queue = ledger.review_queue("shop.sales", tables=TABLES)
    assert [(e.subject_kind, e.subject_label, e.attribute, len(e.group_facts)) for e in queue] == [
        ("field", "orders.amount", "label", 0),
        ("field", "orders.dropped", "label", 0),
        ("metric", "fresh", "expression", 2),
        ("relationship", "cc", "from", 4),
        ("model", "shop.sales", "description", 0),
        ("dataset", "shop.sales.unlisted", "description", 0),  # a table the catalog does not list goes by its key
    ]
    amount, _dropped, fresh, cc, *_ = queue
    assert (amount.table, amount.column, amount.table_uid, amount.rationale) == (ORDERS, "amount", 1, "clearer")
    assert (fresh.table, fresh.column, fresh.table_uid) == (CUSTOMERS, None, 2) and fresh.group_facts[0].id == fresh.id
    assert (cc.table, cc.table_uid) == (None, None) and {f.group for f in cc.group_facts} == {cc.group}
    assert next(e for e in queue if e.table == CUSTOMERS).group  # facts of other kinds may carry one, decided alone
    ledger.accept_group(fresh.group, actor=ANN)
    assert len(ledger.review_queue("shop.sales", tables=TABLES)) == 5
    assert ledger.as_actor(ANN).review_queue("elsewhere") == []
    assert next(e for e in ledger.review_queue() if e.column == "amount").subject_label == f"{ORDERS}.amount"


def test_counts_leave_out_what_the_catalog_accepted(g):
    g.propose(f"field:{ORDERS}/amount", "datatype", "Integer", source="catalog:files")
    g.propose(f"field:{ORDERS}/amount", "label", "Amount", source=ANN, accept=True)
    stale = g.propose(f"field:{ORDERS}/amount", "description", "x", source=ANN)
    g.mark_stale(stale.id, actor="system:x", reason="r")
    assert g.model_view("shop", statuses=LIVE_STATUSES).counts() == {"accepted": 1, "pending": 0, "stale": 1}


def test_a_big_view_is_read_in_a_fixed_number_of_queries(g):
    import sqlalchemy as sa

    def fill(n):
        g.propose_many([Proposal(f"field:s.t{i}/c{j}", "label", f"{i}.{j}") for i in range(n) for j in range(5)],
                       source=ANN)

    def queries():
        seen = []
        listen = (g.store.engine, "before_cursor_execute", lambda *a: seen.append(a[2]))
        sa.event.listen(*listen)
        g.model_view("s", statuses=LIVE_STATUSES)
        sa.event.remove(*listen)
        return len(seen)

    fill(2)
    few = queries()
    fill(60)
    assert queries() == few


def test_an_empty_ledger():
    with glosslayer.open("sqlite://") as g:
        assert (g.model_view().tables, g.model_view().columns, g.model_view().counts()["pending"]) == ([], {}, 0)
