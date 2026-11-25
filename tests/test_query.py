"""Compiling metric queries: what is emitted, what is refused and why, and that it runs."""

import sqlite3

import pytest

from glosslayer import Column, Human, Metric, Relationship, Table
from glosslayer.dialect import EngineUnavailable, QueryFailed
from glosslayer.query import (
    Filter,
    MetricQuery,
    Order,
    QueryModel,
    check_metric,
    column_references,
    compile_query,
    dependents,
    expression_sql,
    reachable,
)

ORDERS, CUSTOMERS, PRODUCTS, REGIONS = "shop.sales.orders", "shop.sales.customers", "shop.stock.products", "shop.geo.regions"
COLUMNS = {
    ORDERS: ["order_id", "customer_id", "product_id", "amount", "status", "day"],
    CUSTOMERS: ["customer_id", "country", "region_id"],
    PRODUCTS: ["product_id", "price", "label"],
    REGIONS: ["region_id", "name"],
}


def join(name, a, b, col, to=None):
    return Relationship(name, a, b, from_columns=[col], to_columns=[to or col], namespace="shop")


JOINS = [
    join("orders_customers", ORDERS, CUSTOMERS, "customer_id"),
    join("orders_products", ORDERS, PRODUCTS, "product_id"),
    join("customers_regions", CUSTOMERS, REGIONS, "region_id"),
]


def model(*metrics, joins=JOINS, columns=COLUMNS):
    return QueryModel(list(metrics), joins, columns)


def metric(name="revenue", expression="SUM(amount)", table=ORDERS, **kw):
    return Metric(name, expression, table=table, **kw)


REVENUE = metric(time_column="day")


def sql(m, query, **kw):
    return compile_query(model(m), query, **kw).sql


def refused(m, query, **kw):
    with pytest.raises(ValueError) as err:
        compile_query(m if isinstance(m, QueryModel) else model(m), query, **kw)
    return str(err.value)


# ------------------------------------------------------------------ what is emitted
def test_a_plain_metric():
    q = compile_query(model(REVENUE), MetricQuery("revenue"))
    assert q.sql == (
        'SELECT SUM(amount) AS "revenue"\n'
        'FROM (SELECT "orders".* FROM "shop.sales.orders" AS "orders") AS "orders"\n'
        "LIMIT 1000"
    )
    assert q.tables == [ORDERS] and q.metric is REVENUE


def test_dimensions_grain_filters_and_order():
    q = compile_query(
        model(REVENUE),
        MetricQuery(
            "revenue",
            ["status", "customers.country", "status"],
            [Filter("regions.name", "in", ["North", "o'x"]), Filter("amount", "between", [1, 2.5]),
             Filter("status", "!=", "open"), Filter("day", "is not null")],
            grain="month",
            order=Order("revenue", "desc"),
            limit=5,
        ),
    )
    assert q.tables == [ORDERS, CUSTOMERS, REGIONS]
    assert q.sql.split("\n") == [
        'SELECT DATE_TRUNC(\'month\', "orders"."day") AS "day_month", "orders"."status" AS "status", '
        '"orders"."__gloss_d1" AS "customers.country", SUM(amount) AS "revenue"',
        'FROM (SELECT "orders".*, "j1"."country" AS "__gloss_d1" FROM "shop.sales.orders" AS "orders" '
        'LEFT JOIN "shop.sales.customers" AS "j1" ON "orders"."customer_id" = "j1"."customer_id" '
        'LEFT JOIN "shop.geo.regions" AS "j2" ON "j1"."region_id" = "j2"."region_id" '
        'WHERE "j2"."name" IN (\'North\', \'o\'\'x\') AND "orders"."amount" BETWEEN 1 AND 2.5 '
        'AND "orders"."status" <> \'open\' AND "orders"."day" IS NOT NULL) AS "orders"',
        'GROUP BY DATE_TRUNC(\'month\', "orders"."day"), "orders"."status", "orders"."__gloss_d1"',
        'ORDER BY "revenue" DESC',
        "LIMIT 5",
    ]


def test_a_joined_column_in_the_expression_is_read_under_a_safe_alias():
    value = metric("value", "SUM(products.price * orders.amount) / 'products.price'", filter='"products"."label" = 1')
    text = sql(value, MetricQuery("value", ["products.label"]))
    assert 'SUM("orders"."__gloss_d2" * orders.amount) / \'products.price\' AS "value"' in text
    assert 'WHERE ("orders"."__gloss_d1" = 1)' in text
    assert '"j1"."label" AS "__gloss_d1", "j1"."price" AS "__gloss_d2"' in text


def test_a_filter_the_expression_already_holds_is_not_applied_twice():
    paid = metric("paid", "SUM(amount) FILTER (WHERE status = 'paid')", filter="status='paid'")
    assert "WHERE (" not in sql(paid, MetricQuery("paid")).replace("FILTER (WHERE", "")


def test_other_table_names_and_dialects():
    text = sql(REVENUE, MetricQuery("revenue", grain="week"), table_names={ORDERS: "input_1"}, dialect="bigquery")
    assert text.startswith("SELECT DATE_TRUNC(`orders`.`day`, WEEK) AS `day_week`") and "FROM `input_1`" in text
    assert [r.parts for r in column_references("`a`.`b` \"c\".d", "bigquery")] == [("a", "b")]


# ------------------------------------------------------------------ what is refused
def test_names_that_do_not_resolve():
    assert refused(REVENUE, MetricQuery("nope")) == "No metric called 'nope' in this model. Known metrics: revenue"
    assert refused(model(), MetricQuery("x")).endswith("This model has no metrics yet.")
    assert refused(REVENUE, MetricQuery("revenue"), tables={CUSTOMERS}).startswith("No metric called")
    assert "has no column 'zz'" in refused(REVENUE, MetricQuery("revenue", ["zz"]))
    assert "has no column 'zz' (in dimension" in refused(REVENUE, MetricQuery("revenue", ["customers.zz"]))
    assert refused(REVENUE, MetricQuery("revenue", ["nope.x"])).startswith("Unknown table 'nope' in dimension")
    assert refused(REVENUE, MetricQuery("revenue", [" "])) == "A dimension must name a column"
    assert "has no expression" in refused(metric(expression={"dialects": []}), MetricQuery("revenue"))
    two = model(REVENUE, metric(table=CUSTOMERS))
    assert refused(two, MetricQuery("revenue")) == f"Metric 'revenue' exists on several tables: {ORDERS}, {CUSTOMERS}"
    assert "belongs to no table" in refused(Metric("r", "1", namespace="shop"), MetricQuery("r"))


def test_joins_are_only_followed_many_to_one():
    count = metric("n", "COUNT(*)", CUSTOMERS)
    assert "does not reach by following joins" in refused(count, MetricQuery("n", ["orders.status"]))
    assert refused(metric("x", "SUM(orders.amount)", CUSTOMERS), MetricQuery("x")) == (
        "orders has many rows per customer; a metric on customers cannot use a column of orders "
        "(orders.amount in the expression)"
    )
    assert "is not reached from customers" in refused(metric("x", "SUM(products.price)", CUSTOMERS), MetricQuery("x"))
    assert "Unknown table 'zz'" in refused(metric("x", "SUM(zz.a)"), MetricQuery("x"))
    assert reachable(model(), ORDERS) == [ORDERS, CUSTOMERS, PRODUCTS, REGIONS] and reachable(model(), REGIONS) == [REGIONS]


def test_two_reachable_tables_of_one_name_need_the_schema():
    other = "shop.crm.customers"
    m = model(metric("x", "MAX(customers.country)"), joins=[*JOINS, join("oc2", ORDERS, other, "customer_id")],
              columns={**COLUMNS, other: ["customer_id"]})
    assert "could mean crm.customers and sales.customers" in refused(m, MetricQuery("x"))
    m = m.with_metric(metric("x", "MAX(sales.customers.country)"))
    assert 'MAX("orders"."__gloss_d1")' in compile_query(m, MetricQuery("x")).sql


def test_what_a_metric_does_not_allow():
    limited = metric(dimensions=["status", "customers.country"])
    assert sql(limited, MetricQuery("revenue", [f"{CUSTOMERS}.country", "orders.status"]))
    assert "may only be broken down by status, customers.country" in refused(limited, MetricQuery("revenue", ["day"]))
    assert "has no time column" in refused(limited, MetricQuery("revenue", grain="day"))
    assert "is neither" in refused(REVENUE, MetricQuery("revenue", order=Order("status")))
    assert "may not share the metric's name" in refused(metric("status", "COUNT(*)"), MetricQuery("status", ["status"]))


@pytest.mark.parametrize("op,value,message", [
    ("in", [], "non-empty list"), ("in", "a", "non-empty list"), ("between", [1], "list of two values"),
    ("=", None, "needs one value"), ("=", [1], "needs one value"), ("=", {"a": 1}, "must be text, a number"),
])
def test_filters_that_cannot_be_rendered(op, value, message):
    assert message in refused(REVENUE, MetricQuery("revenue", filters=[Filter("amount", op, value)]))


@pytest.mark.parametrize("build", [
    lambda: Filter("a", "~", 1), lambda: Order("a", "up"), lambda: MetricQuery("m", grain="eon"),
    lambda: MetricQuery("m", limit=0), lambda: MetricQuery.from_dict({"metric": "m", "nope": 1}),
])
def test_queries_check_themselves(build):
    with pytest.raises(ValueError):
        build()


def test_a_query_from_json():
    q = MetricQuery.from_dict({
        "metric": "m", "dimensions": ["a"], "filters": [{"field": "a", "op": "=", "value": 1}],
        "grain": "day", "order": {"field": "a", "direction": "desc"}, "limit": 3,
    })
    assert q == MetricQuery("m", ["a"], [Filter("a", "=", 1)], "day", Order("a", "desc"), 3)
    assert MetricQuery.from_dict({"metric": "m", "order": None, "filters": None}).order is None


def test_references_in_free_sql():
    text = "SUM(a.b) + s.t.\"c\"\"d\" -- x.y\n + 'e.f' /* g.h */ + fn.call(1) + CASE.x + 1.5 + a.b.c.d + bare"
    assert [(r.parts, r.text) for r in column_references(text)] == [
        (("a", "b"), "a.b"), (("s", "t", 'c"d'), 's.t."c""d"')
    ]
    assert [expression_sql(x) for x in ({"sql": " a "}, "b", None, {"dialects": []})] == ["a", "b", "", ""]


# ------------------------------------------------------------------ trying and removing definitions
def test_a_proposed_metric_must_compile_and_run():
    seen = []

    def engine(text):
        seen.append(text)
        if "bad" in text:
            raise QueryFailed("no such function: bad")
        return [[1]]

    m = model(metric(expression="SUM(order_id)"))
    assert check_metric(m, metric(expression="SUM(products.price)"), engine) is True  # replaces the one of its name
    assert "__gloss_d1" in seen[0]
    assert check_metric(m, REVENUE) is False
    with pytest.raises(ValueError, match=r"^Metric 'revenue' does not compile: Unknown table 'zz'"):
        check_metric(m, metric(expression="SUM(zz.a)"), engine)
    with pytest.raises(ValueError, match="^Metric 'revenue' does not run: no such function: bad$"):
        check_metric(m, metric(expression="bad(amount)"), engine)

    def down(text):
        raise EngineUnavailable("no worker")

    assert check_metric(m, REVENUE, down) is False  # authoring does not wait for compute


def test_what_depends_on_a_join():
    key = "relationship:shop::orders_customers"
    m = model(
        metric("by_country", dimensions=["status", "customers.country"]),
        metric("formula", "SUM(customers.region_id)"),
        metric("far", "COUNT(*)", dimensions=["regions.name"]),
        metric("plain"),
        metric("broken", "SUM(zz.a)"),
    )
    found = {d.metric.name: d.uses for d in dependents(m, key)}
    assert found == {"by_country": ["dimension customers.country"], "formula": ["formula"],
                     "far": ["dimension regions.name"]}
    assert "does not reach" in dependents(m, key)[0].message
    assert dependents(m, "relationship:shop::other") == []


# ------------------------------------------------------------------ from a ledger, on an engine
def test_a_model_read_from_a_ledger_compiles_and_runs(g):
    ann = Human("ann")
    orders = Table(ORDERS, primary_key=[Column("order_id", int)]).add_columns(
        Column("customer_id", int), Column("amount", float), Column("status", str))
    customers = Table(CUSTOMERS, primary_key=[Column("customer_id", int)]).add_columns(Column("country", str))
    g.add(orders, customers,
          Relationship("oc", on={orders["customer_id"]: customers["customer_id"]}, namespace="shop"),
          metric(filter="status = 'paid'"), source=ann, accept=True)
    g.add(metric("pending", "AVG(amount)"), source=ann)
    m = QueryModel.read(g)
    assert [x.name for x in m.metrics] == ["revenue"] and set(m.columns) == {ORDERS, CUSTOMERS}
    assert {x.name for x in QueryModel.read(g, include_pending=True).metrics} == {"revenue", "pending"}

    con = sqlite3.connect(":memory:")
    con.executescript(f"""
        CREATE TABLE "{ORDERS}" (order_id, customer_id, amount, status);
        INSERT INTO "{ORDERS}" VALUES (1, 1, 10, 'paid'), (2, 1, 5, 'open'), (3, 2, 7, 'paid'), (4, 9, 1, 'paid');
        CREATE TABLE "{CUSTOMERS}" (customer_id, country);
        INSERT INTO "{CUSTOMERS}" VALUES (1, 'NL'), (2, 'DE');
    """)
    q = compile_query(m, MetricQuery("revenue", ["customers.country"]))
    assert con.execute(q.sql).fetchall() == [(None, 1), ("DE", 7), ("NL", 10)]
    assert q.metric.version is not None


def test_a_compiled_grain_runs_on_duckdb():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute(f'CREATE TABLE "{ORDERS}" AS SELECT * FROM (VALUES '
                "(1, DATE '2026-01-03', 10), (2, DATE '2026-01-30', 5), (3, DATE '2026-02-01', 7)) t(order_id, day, amount)")
    q = compile_query(model(REVENUE), MetricQuery("revenue", grain="month"), dialect="duckdb")
    assert [r[1] for r in con.execute(q.sql).fetchall()] == [15, 7]
