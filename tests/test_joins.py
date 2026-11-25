"""Measuring a join against data, through a host's ``execute`` (here: SQLite), and the SQL text helpers."""

import sqlite3

import pytest

from glosslayer import Relationship, joins
from glosslayer.dialect import EngineUnavailable, QueryFailed, date_trunc, literal, quote


@pytest.fixture
def db():
    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        CREATE TABLE "shop.customers" (customer_id INT, email TEXT);
        INSERT INTO "shop.customers" VALUES (1, 'a'), (2, 'b'), (3, 'b'), (NULL, 'c');
        CREATE TABLE "shop.orders" (order_id INT, customer_id INT, email TEXT, line INT);
        INSERT INTO "shop.orders" VALUES (1, 1, 'a', 1), (2, 1, 'a', 2), (3, 2, 'b', 1), (4, NULL, NULL, NULL);
        CREATE TABLE "shop.empty" (customer_id INT);
        """
    )
    return con


@pytest.fixture
def execute(db):
    def run(sql):
        run.seen.append(sql)
        try:
            return db.execute(sql).fetchall()
        except sqlite3.Error as exc:
            raise QueryFailed(str(exc)) from None

    run.seen = []
    return run


ORDERS, CUSTOMERS = "shop.orders", "shop.customers"


def test_a_join_to_a_unique_side_is_ok_and_says_what_was_measured(execute):
    found = joins.check(ORDERS, ["customer_id"], CUSTOMERS, ["customer_id"], execute)
    assert found.verdict == "ok"
    assert found.message == (
        "customer_id is unique in customers (3 of 3 rows) and repeats in orders (2 distinct in 3 rows)."
    )
    assert execute.seen[0] == (  # the to side first; one statement per side, rows without a key left out
        'SELECT COUNT(*) AS "keys", COALESCE(SUM("n"), 0) AS "rows" FROM (SELECT "customer_id", COUNT(*) AS "n" '
        'FROM "shop.customers" WHERE "customer_id" IS NOT NULL GROUP BY "customer_id") AS "g"'
    )
    assert len(execute.seen) == 2


def test_one_to_one_and_keys_of_several_columns(execute):
    assert joins.check(ORDERS, ["order_id"], CUSTOMERS, ["customer_id"], execute).message == (
        "customer_id is unique in customers (3 of 3 rows) and order_id in orders (4 of 4 rows): one to one."
    )
    # neither email nor line identifies an order, together with customer_id line does
    both = joins.check("shop.orders", ["customer_id", "line"], "shop.orders", ["customer_id", "line"], execute)
    assert both.verdict == "ok" and "customer_id, line is unique in orders (3 of 3 rows)" in both.message


def test_a_reversed_join_is_told_to_swap(execute):
    found = joins.check(CUSTOMERS, ["customer_id"], ORDERS, ["customer_id"], execute)
    assert found.verdict == "reversed"
    assert found.message.endswith("swap the two sides so the join goes from orders to customers.")


def test_a_join_that_multiplies_rows(execute):
    found = joins.check(ORDERS, ["email"], CUSTOMERS, ["email"], execute)
    assert found.verdict == "many_to_many"
    assert found.message.endswith("The to side needs columns that identify one row of customers.")


def test_what_cannot_be_measured_is_unchecked(execute):
    empty = joins.check(ORDERS, ["customer_id"], "shop.empty", ["customer_id"], execute)
    assert empty == joins.unchecked("empty has no rows with a key (an empty table)")
    missing = joins.check(ORDERS, ["customer_id"], "shop.nope", ["id"], execute, labels={"shop.nope": "Nope"})
    assert missing.verdict == "unchecked" and missing.message.startswith("Direction not checked: Nope could not be")
    assert joins.check(ORDERS, ["a"], CUSTOMERS, ["b"], lambda sql: []).message.endswith("(no answer).")

    def down(sql):
        raise EngineUnavailable("gone")

    assert joins.check(ORDERS, ["a"], CUSTOMERS, ["b"], down) == joins.unchecked("compute was not available")


def test_tables_can_go_by_another_name_and_dialect(db, execute):
    db.execute('CREATE VIEW "stg_customers" AS SELECT * FROM "shop.customers"')
    found = joins.check(ORDERS, ["customer_id"], CUSTOMERS, ["customer_id"], execute,
                        table_names={CUSTOMERS: "stg_customers"}, labels={CUSTOMERS: "Clients"})
    assert found.verdict == "ok" and "unique in Clients" in found.message
    assert "`shop.x`" in joins.measure_sql("shop.x", ["a"], dialect="bigquery")


def test_every_join_of_a_model_is_checked(execute):
    columns = {ORDERS: ["order_id", "customer_id"], CUSTOMERS: ["customer_id"]}

    def rel(name, to, column):
        return Relationship(name, ORDERS, to, from_columns=["customer_id"], to_columns=[column])

    given = [rel("ok", CUSTOMERS, "customer_id"), rel("no_table", "shop.gone", "id"), rel("no_col", CUSTOMERS, "id")]
    found = joins.check_all(given, columns, execute)
    assert [(j.name, c.verdict) for j, c in found] == [("ok", "ok"), ("no_table", "unchecked"), ("no_col", "unchecked")]
    assert len(execute.seen) == 2  # what is missing is not asked of the engine

    def broken(sql):
        raise RuntimeError("boom")

    assert joins.check_all(given[:1], columns, broken)[0][1].message == "Direction not checked: boom"


def test_sql_text_helpers():
    assert quote('a"b') == '"a""b"' and quote("a`b", "bigquery") == "`a``b`"
    assert date_trunc("month", "x") == "DATE_TRUNC('month', x)"
    assert date_trunc("month", "x", "bigquery") == "DATE_TRUNC(x, MONTH)"
    assert [literal(v) for v in (None, True, 1, 2.5, "o'x", [1, "a"])] == [
        "NULL", "TRUE", "1", "2.5", "'o''x'", "(1, 'a')"
    ]
    with pytest.raises(ValueError):
        literal(float("nan"))
    with pytest.raises(ValueError):
        literal({"a": 1})
    with pytest.raises(ValueError):
        quote("a", "klingon")


def test_joins_that_cannot_be_one_are_refused_before_the_data_is_asked():
    columns = {ORDERS: ["order_id", "customer_id", "parent_id"], CUSTOMERS: ["customer_id"]}

    def refused(f, fc, t, tc):
        with pytest.raises(ValueError) as err:
            joins.validate(f, fc, t, tc, columns, labels={CUSTOMERS: "Clients"})
        return str(err.value)

    assert joins.validate(ORDERS, ["customer_id"], CUSTOMERS, ["customer_id"], columns) is None
    assert joins.validate(ORDERS, ["parent_id"], ORDERS, ["order_id"], columns) is None
    for a, b in (([], []), (["customer_id"], [])):
        assert refused(ORDERS, a, CUSTOMERS, b).startswith("A join needs the same number of columns")
    assert refused(ORDERS, ["customer_id"], CUSTOMERS, ["id"]) == "Table 'Clients' has no column 'id'"
    assert refused("shop.gone", ["a"], CUSTOMERS, ["customer_id"]) == "Table 'gone' has no column 'a'"
    assert "orders to itself needs different columns" in refused(ORDERS, ["order_id"], ORDERS, ["order_id"])
