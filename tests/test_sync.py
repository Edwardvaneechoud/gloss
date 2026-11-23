from gloss import Status
from gloss.catalogs.file import FileCatalog
from gloss.ossie import OssieSource
from gloss.sync import sync
from gloss.types import to_ossie_datatype


def test_type_mapping():
    assert to_ossie_datatype("Int64") == "Integer"
    assert to_ossie_datatype("UInt8") == "Integer"
    assert to_ossie_datatype("Float32") == "Float"
    assert to_ossie_datatype("Decimal(18,2)") == "Decimal"
    assert to_ossie_datatype("Utf8") == "String"
    assert to_ossie_datatype("Datetime(time_unit='us', time_zone='UTC')") == "DateTimeTz"
    assert to_ossie_datatype("Datetime(time_unit='us', time_zone=None)") == "DateTime"
    assert to_ossie_datatype("timestamp with time zone") == "DateTimeTz"
    assert to_ossie_datatype("List(Int64)") == "Opaque"


def test_initial_sync_accepts_structure_and_proposes_descriptions(g, examples):
    rep = sync(g, FileCatalog(examples / "catalog.yaml"))
    assert sorted(rep.new) == ["shop.customers", "shop.orders"]
    assert g.current("field:shop.orders/amount") == {
        "expression": {"sql": "amount"},
        "datatype": "Decimal",
        "physical_type": "Decimal(18,2)",
    }
    assert g.current("dataset:shop.orders")["primary_key"] == ["order_id"]
    desc = g.find(subject="dataset:shop.orders", attribute="description")
    assert desc[0].status is Status.PROPOSED  # descriptions need a human


def test_resync_unchanged_is_noop(g, examples):
    cat = FileCatalog(examples / "catalog.yaml")
    sync(g, cat)
    n = len(g.log())
    rep = sync(g, cat)
    assert rep.unchanged == 2 and len(g.log()) == n


def test_drift_marks_facts_stale(g, examples):
    sync(g, FileCatalog(examples / "catalog.yaml"))
    g.propose_many(OssieSource(examples / "semantic_model.yaml", key_by="source").proposals(), source="import:ossie")
    g.bulk("accept", __import__("gloss").Query(source_prefix="import:"), actor="human:a")

    rep = sync(g, FileCatalog(examples / "catalog_v2.yaml"))
    assert rep.changed == ["shop.orders"]

    # removed column: all its facts stale, and the metric that uses it
    status_facts = g.find(subject="field:shop.orders/status")
    assert status_facts and all(a.status is Status.STALE for a in status_facts)
    assert {a.status for a in g.find(subject="metric:total_revenue")} == {Status.STALE}
    assert {a.status for a in g.find(subject="metric:order_count")} == {Status.ACCEPTED}

    # logical type change: new datatype accepted, the join that depends on it goes back to review
    assert g.current("field:shop.orders/customer_id")["datatype"] == "String"
    assert {a.status for a in g.find(subject="relationship:orders_customers")} == {Status.STALE}

    # added column
    assert g.current("field:shop.orders/channel")["datatype"] == "String"


def test_removed_asset(g, examples, tmp_path):
    p = tmp_path / "cat.yaml"
    p.write_text((examples / "catalog.yaml").read_text())
    sync(g, FileCatalog(p))
    text = p.read_text()
    p.write_text(text[: text.index("  shop.customers:")])
    rep = sync(g, FileCatalog(p))
    assert rep.gone == ["shop.customers"]
    assert all(a.status is Status.STALE for a in g.find(dataset="shop.customers"))


def test_write_back(g, examples, tmp_path):
    p = tmp_path / "cat.yaml"
    p.write_text((examples / "catalog.yaml").read_text())
    cat = FileCatalog(p)
    cat.write_back("shop.orders", {"description": "accepted text", "fields": {"amount": {"description": "EUR"}}})
    again = FileCatalog(p).describe("shop.orders")
    assert again.description == "accepted text"
    assert [c.description for c in again.columns if c.name == "amount"] == ["EUR"]


def test_precise_type_change_is_recorded_without_review(g, tmp_path, examples):
    p = tmp_path / "cat.yaml"
    p.write_text((examples / "catalog.yaml").read_text())
    sync(g, FileCatalog(p))
    p.write_text(p.read_text().replace("{name: order_id, dtype: Int64}", "{name: order_id, dtype: Int32}"))
    rep = sync(g, FileCatalog(p))
    assert rep.changed == ["shop.orders"] and rep.staled == 0
    assert g.current("field:shop.orders/order_id")["physical_type"] == "Int32"
    assert g.current("field:shop.orders/order_id")["datatype"] == "Integer"


def test_logical_type_change_sends_semantic_type_and_checks_back(g, tmp_path, examples):
    from gloss import Column, Human, SemanticType, String, Table, checks

    p = tmp_path / "cat.yaml"
    p.write_text((examples / "catalog.yaml").read_text())
    sync(g, FileCatalog(p))
    Country = SemanticType("country_code", dtype=String(2), checks=[checks.length(2, 2)])
    g.add(Table("shop.customers").add_columns(Column("country", Country, checks=[checks.not_null()])),
          actor=Human("a"), accept=True)
    p.write_text(p.read_text().replace("{name: country, dtype: Utf8}", "{name: country, dtype: Int32}"))
    sync(g, FileCatalog(p))
    stale = {a.attribute for a in g.find(subject="field:shop.customers/country", statuses=[Status.STALE])}
    assert {"semantic_type", "checks"} <= stale
    assert g.current("field:shop.customers/country")["datatype"] == "Integer"
