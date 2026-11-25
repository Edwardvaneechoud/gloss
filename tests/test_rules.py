"""Rule-based suggestions: statistics through a host's ``execute`` (SQLite here), the rules, and storing them."""

import sqlite3

import pytest

import glosslayer
from glosslayer import AI, Human, Metric, Relationship, Status
from glosslayer.dialect import EngineUnavailable, QueryFailed
from glosslayer.model import LIVE_STATUSES
from glosslayer.rules import (
    Counts,
    Words,
    collect_stats,
    column_role,
    propose,
    singular,
    stats_sql,
    suggest,
    type_class,
)
from glosslayer.seams import AssetSchema, Column

ANN = Human("ann")
ORDERS, CUSTOMERS, PRODUCTS = "shop.sales.orders", "shop.sales.customers", "shop.stock.products"


def asset(key, **columns):
    return AssetSchema(key, [Column(n, t) for n, t in columns.items()])


TABLES = [
    asset(CUSTOMERS, customer_id="Int64", country="String", name="String"),
    asset(ORDERS, order_id="Int64", customer_id="Int64", product_id="Int32", day="Date", amount="Float64",
          paid="Boolean", extra="Struct({'a': Int64})"),
    asset(PRODUCTS, product_id="Int64", price="Decimal(18, 2)"),
]


@pytest.fixture
def execute():
    con = sqlite3.connect(":memory:")
    con.executescript(f"""
        CREATE TABLE "{CUSTOMERS}" (customer_id, country, name);
        INSERT INTO "{CUSTOMERS}" VALUES (1, 'NL', 'a'), (2, 'DE', 'b'), (3, 'NL', 'c');
        CREATE TABLE "{ORDERS}" (order_id, customer_id, product_id, day, amount, paid, extra);
        INSERT INTO "{ORDERS}" VALUES (1, 1, 1, '2026-01-01', 5.0, 1, 'x'), (2, 1, 2, '2026-01-02', 7.5, 0, 'x'),
                                      (3, 2, 2, '2026-01-03', 5.0, 1, 'x'), (4, 2, 1, NULL, 5.0, 1, 'x');
        CREATE TABLE "{PRODUCTS}" (product_id, price);
        INSERT INTO "{PRODUCTS}" VALUES (1, 9.5), (2, 9.5);
    """)

    def run(sql):
        try:
            return con.execute(sql).fetchall()
        except sqlite3.Error as exc:
            raise QueryFailed(str(exc)) from None

    return run


@pytest.fixture
def found(execute):
    stats, problem = collect_stats(TABLES, execute)
    assert problem is None
    return suggest(stats)


def live(g, namespace="shop"):
    return g.model_view(namespace, tables=TABLES, statuses=LIVE_STATUSES)


# ------------------------------------------------------------------ statistics
def test_type_classes_come_from_the_ossie_datatype():
    assert [type_class(t) for t in ("Int64", "UInt8", "Decimal(18, 2)", "Float32", "Date", "Datetime('us')", "String",
                                    "Categorical", "Boolean", "Time", "Struct({'a': Int64})", "List(String)", None)] == [
        "integer", "integer", "numeric", "numeric", "time", "time", "string", "string", "boolean",
        "other", "other", "other", "other"
    ]
    assert [singular(n) for n in ("orders", "Categories", "boxes", "address", "bus", "as")] == [
        "order", "category", "box", "address", "bu", "as"
    ]


def test_one_statement_per_table(execute):
    seen = []
    stats, problem = collect_stats(TABLES[:1], lambda sql: seen.append(sql) or execute(sql))
    assert seen == [
        'SELECT COUNT(*) AS "__rows", COUNT(DISTINCT "customer_id") AS "d0", '
        'COUNT(*) - COUNT("customer_id") AS "n0", MIN("customer_id") AS "lo0", MAX("customer_id") AS "hi0", '
        'COUNT(DISTINCT "country") AS "d1", COUNT(*) - COUNT("country") AS "n1", COUNT(DISTINCT "name") AS "d2", '
        'COUNT(*) - COUNT("name") AS "n2" FROM "shop.sales.customers"'
    ]
    (t,) = stats
    assert (t.key, t.name, t.row_count, t.available) == (CUSTOMERS, "customers", 3, True)
    assert (t.column("customer_id").min, t.column("customer_id").max, t.column("country").distinct) == (1, 3, 2)
    assert t.unique("customer_id") is True and t.unique("country") is False and t.unique("zz") is None
    assert stats_sql(TABLES[2], table_name="p", dialect="bigquery").endswith("FROM `p`")


def test_a_table_that_cannot_be_read_falls_back(execute):
    missing = asset("shop.gone", id="Int64")
    stats, problem = collect_stats([missing, TABLES[0]], execute)
    assert problem.startswith("Statistics for 'gone' failed (no such table") and [t.available for t in stats] == [
        False, True
    ]
    assert collect_stats([TABLES[0]], lambda sql: [])[1].startswith("Statistics for 'customers' failed (no answer)")


def test_an_unavailable_engine_is_asked_once():
    asked = []

    def down(sql):
        asked.append(sql)
        raise EngineUnavailable("no worker")

    stats, problem = collect_stats(TABLES, down)
    assert len(asked) == 1 and problem == (
        "Statistics were not available (compute unreachable: no worker); rules used names and types only."
    )
    sugg = suggest(stats)
    assert not sugg.stats_available and sugg.joins == []
    assert {(r.column, r.role) for r in sugg.roles} == {("day", "time"), ("paid", "category")}  # what types alone tell
    assert {m.name for m in sugg.metrics} == {"customer_count", "order_count", "product_count"}
    assert {m.confidence for m in sugg.metrics} == {0.5}


# ------------------------------------------------------------------ the rules
def test_joins_follow_a_repeating_column_to_where_it_is_unique(found):
    assert [(j.name, j.from_table, j.to_table, j.from_columns) for j in found.joins] == [
        ("orders_customers", ORDERS, CUSTOMERS, ["customer_id"]),
        ("orders_products", ORDERS, PRODUCTS, ["product_id"]),  # Int32 and Int64 are one kind of key
    ]
    assert found.joins[0].description == "Each order belongs to one customer."
    assert found.joins[0].rationale == (
        "Rule: orders.customer_id matches customers.customer_id by name and type; customer_id is unique in customers "
        "(3 of 3 rows) and repeats in orders (2 distinct in 4 rows)."
    )


def test_roles_come_from_types_and_statistics(found):
    roles = {(r.table, r.column): r.role for r in found.roles}
    assert roles == {
        (CUSTOMERS, "customer_id"): "key", (CUSTOMERS, "country"): "category", (CUSTOMERS, "name"): "key",
        (ORDERS, "order_id"): "key", (ORDERS, "customer_id"): "key", (ORDERS, "product_id"): "key",
        (ORDERS, "day"): "time", (ORDERS, "amount"): "amount", (ORDERS, "paid"): "category",
        (PRODUCTS, "product_id"): "key", (PRODUCTS, "price"): "amount",
    }
    wide = found.stats[CUSTOMERS]
    wide.row_count, wide.column("country").distinct = 100, 60
    assert column_role(wide, wide.column("country")) is None  # free text: too many values to group by
    assert column_role(wide, wide.column("country"), max_categories=60)[0] == "category"


def test_metrics_per_table(found):
    by_table = {t: [m.name for m in found.metrics if m.table == t] for t in (CUSTOMERS, ORDERS, PRODUCTS)}
    assert by_table == {
        CUSTOMERS: ["customer_count"],
        ORDERS: ["order_count", "total_amount", "average_amount", "distinct_customer_id", "distinct_product_id"],
        PRODUCTS: ["product_count", "average_price"],  # a lookup table's price is not summed
    }
    total = next(m for m in found.metrics if m.name == "total_amount")
    assert (total.expression, total.aggregate, total.time_column, total.dimensions) == (
        'SUM("amount")', "sum", "day", ["paid", "customers.country"]
    )
    assert total.rationale == "Rule: amount is an amount column, so its sum is a candidate metric."


def test_clashing_metric_names_get_their_tables_name():
    tables = [asset("s.orders", day="Date"), asset("s.visits", order="Int64"), asset("t.orders", day="Date")]
    names = [m.name for m in suggest(collect_stats(tables, lambda sql: [[3, 2, 0, 1, 2]])[0]).metrics]
    assert names.count("order_count") == 2  # two tables' own row counts: the host resolves it on storing
    assert "total_order" in names


def test_only_some_candidates(found):
    assert [m.table for m in found.only(tables=[PRODUCTS]).metrics] == [PRODUCTS, PRODUCTS]
    assert [j.name for j in found.only(tables=[PRODUCTS]).joins] == ["orders_products"]  # either end
    limited = found.only(kinds=["role"])
    assert limited.roles and not limited.joins and not limited.metrics and limited.stats is found.stats


# ------------------------------------------------------------------ storing
def test_candidates_are_queued_with_their_evidence_and_a_second_run_adds_nothing(g, found):
    first = propose(g, live(g), found)
    assert first.added == Counts(roles=11, joins=2, metrics=8) and first.skipped == Counts()
    v = live(g)
    amount = v.table(ORDERS).column("amount").facts["role"]
    assert (amount.status, amount.source, amount.confidence) == (Status.PROPOSED, "inferred:rules", 0.7)
    assert amount.rationale.startswith("Rule: amount is numeric")
    # a join belongs to the longest path both of its tables lie under
    assert {j.name: j.namespace for j in v.joins} == {"orders_customers": "shop.sales", "orders_products": "shop"}
    assert {m.group is not None for m in v.metrics} == {True}
    again = propose(g, live(g), found)
    assert again.added == Counts() and again.skipped == Counts(roles=11, joins=2, metrics=8)


def test_what_the_model_holds_is_skipped(g, found):
    g.propose(f"field:{ORDERS}/amount", "role", "key", source=ANN, accept=True)
    g.add(Relationship("other_name", ORDERS, CUSTOMERS, from_columns=["customer_id"], to_columns=["customer_id"],
                       namespace="shop.sales"),
          Metric("order_count", "COUNT(order_id)", table=ORDERS),  # takes the name
          Metric("mine", "sum( \"amount\" )", table=ORDERS), source=ANN)  # holds the formula
    out = propose(g, live(g), found)
    assert out.skipped == Counts(roles=1, joins=1, metrics=2) and out.added == Counts(roles=10, joins=1, metrics=6)
    rejected = g.find(subject=f"field:{ORDERS}/paid", attribute="role")[0]
    g.reject(rejected.id, actor=ANN)
    assert propose(g, live(g), found.only(kinds=["role"])).added.roles == 0  # once rejected is not proposed again


def test_tables_that_are_not_in_the_model_are_left_out(g, found):
    view = g.model_view("shop.sales", tables=TABLES[:2], statuses=LIVE_STATUSES)
    out = propose(g, view, found, owner=lambda a, b: "shop")
    assert out.skipped == Counts(roles=2, joins=1, metrics=2)
    assert {j.namespace for j in live(g).joins} == {"shop"}


def test_words_are_stated_by_whoever_wrote_them(g, found):
    agent = g.as_actor(ANN, via="ai:claude")
    words = {
        ORDERS: Words("Every order placed.", {"amount": "Order value", "day": "", "zz": "x"},
                      {"total_amount": ("revenue", "All the money."), "order_count": (None, None),
                       "average_amount": ("customer_count", "taken name")}),
        "shop.gone": Words("x"),
    }
    with pytest.raises(ValueError, match="worded_by"):
        propose(g, live(g), found, words=words)
    out = propose(g, live(g), found, words=words, worded_by=(agent, "ai:claude"))
    assert out.added == Counts(roles=11, joins=2, metrics=8, descriptions=2)
    v = live(g)
    by_name = {m.name: m for m in v.metrics}
    revenue = {f.attribute: f for f in g.facts(g.find(subject=by_name["revenue"].key))}
    assert (revenue["description"].value, revenue["description"].source, revenue["description"].via) == (
        "All the money.", "ai:claude", "ai:claude"
    )
    assert revenue["expression"].rationale.endswith("Name and meaning written by AI.")
    assert "average_amount" in by_name and g.facts(g.find(subject=by_name["average_amount"].key))[0].source == (
        "inferred:rules"  # the better name was taken, so the rule's own stands
    )
    described = v.table(ORDERS).column("amount").facts["description"]
    assert (described.value, described.rationale, described.confidence) == ("Order value", "Written by AI.", 0.6)

    # the model holds these now: new words become a synonym and, where there was only the rules' text, a description
    words = {ORDERS: Words(None, {"amount": "Again"}, {"total_amount": ("turn over", "New."),
                                                       "distinct_product_id": ("products_sold", "How many products.")})}
    out = propose(g, live(g), found, words=words, worded_by=(agent, "ai:claude"), by="Claude")
    assert out.skipped.metrics == 8 and out.skipped.descriptions == 1 and out.added.descriptions == 3
    held = {m.name: m for m in live(g).metrics}
    waiting = {a.attribute: a for a in g.find(subject=held["distinct_product_id"].key, source_prefix="ai:")}
    assert waiting["synonyms"].value == ["products sold"] and waiting["synonyms"].group == held["distinct_product_id"].group
    assert waiting["description"].value == "How many products."
    assert g.find(subject=held["revenue"].key, attribute="synonyms")[0].value == ["turn over"]
    assert "revenue" in held  # nothing was renamed


def test_metric_names_of_a_path(g):
    g.add(Metric("a", "1", table="s.t"), Metric("b", "1", namespace="s.sub"), Metric("c", "1"), source=ANN)
    assert {n: str(sid) for n, sid in g.metric_names("s").items()} == {"a": "metric:s.t/a", "b": "metric:s.sub::b"}
    assert set(g.metric_names()) == {"a", "b", "c"}
    g.reject_group(g.find(subject="metric:s.t/a")[0].group, actor=ANN)
    assert set(g.metric_names("s")) == {"b"}


def test_an_agent_cannot_state_rules_as_someone_else():
    with glosslayer.open("sqlite://") as g:
        view = g.as_actor(AI("claude"))
        out = propose(view, view.model_view("shop", tables=TABLES, statuses=LIVE_STATUSES), suggest([]))
        assert out.added == Counts()
