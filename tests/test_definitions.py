"""Namespaces, proposal groups and versions: definitions that belong to someone and change as a unit."""

import pytest

import glosslayer
from glosslayer import AI, Column, Human, Metric, PermissionDenied, Proposal, Relationship, SemanticType, Status, Table
from glosslayer.ids import SubjectId, path_covers

ANN, BOB = Human("ann"), Human("bob")


@pytest.mark.parametrize(
    "text,kind,dataset,name,namespace",
    [
        ("model:sales", "model", None, None, "sales"),
        ("relationship:sales::orders_customers", "relationship", None, "orders_customers", "sales"),
        ("metric:main.sales::revenue", "metric", None, "revenue", "main.sales"),
        ("metric:main.sales.orders/avg", "metric", "main.sales.orders", "avg", None),
        ("type:crm::email_address", "type", None, "email_address", "crm"),
        ("type:email_address", "type", None, "email_address", None),
    ],
)
def test_namespaced_subjects_round_trip(text, kind, dataset, name, namespace):
    sid = SubjectId.parse(text)
    assert (sid.kind.value, sid.dataset, sid.name, sid.namespace) == (kind, dataset, name, namespace)
    assert str(sid) == text
    assert sid.scope_path == (dataset or namespace)


@pytest.mark.parametrize("bad", ["relationship:a/b", "type:::x", "metric:ns::", "dataset:a::b", "field:a::b/c",
                                 "relationship:a::b::c/d"])
def test_malformed_namespaced_subjects(bad):
    with pytest.raises(ValueError):
        SubjectId.parse(bad)


def test_paths_are_compared_by_segment():
    assert path_covers("finance", "finance") and path_covers("finance.", "finance.invoices")
    assert not path_covers("fin", "finance.invoices") and not path_covers("finance", None)
    assert path_covers("", None) and path_covers("", "anything")
    assert SubjectId.parse("field:a.b.t/c").moved("a.b", "x") == SubjectId.parse("field:x.t/c")
    assert SubjectId.parse("type:a.b::t").moved("a", "z.y") == SubjectId.parse("type:z.y.b::t")
    assert SubjectId.parse("type:t").moved("a", "b") is None


def test_two_teams_can_each_define_revenue(g):
    g.add(Metric("revenue", "SUM(orders.amount)", namespace="sales", description="Booked order value"),
          Metric("revenue", "SUM(invoices.paid)", namespace="finance", description="Cash received"),
          source=ANN, accept=True)
    assert g.metric("revenue", namespace="sales").description == "Booked order value"
    assert g.metric("revenue", namespace="finance").description == "Cash received"
    assert {a.status for a in g.find(kind="metric")} == {Status.ACCEPTED}  # neither superseded the other
    assert [m.namespace for m in g.metrics()] == ["finance", "sales"]
    assert g["metric:sales::revenue"].expression == {"sql": "SUM(orders.amount)"}
    assert set(g.progress()) == {"finance", "sales"}


def test_a_relationship_is_accepted_as_a_unit(g):
    orders = Table("s.orders").add_columns(Column("a", int), Column("b", int))
    custs = Table("s.customers").add_columns(Column("a", int), Column("b", int))
    g.add(orders, custs, source=ANN, accept=True)
    g.add(Relationship("oc", on={orders["a"]: custs["a"]}, namespace="s"), source=ANN)
    first = g.find(subject="relationship:s::oc", attribute="from_columns")[0]
    assert g.relationships(include_pending=False) == []

    accepted = g.accept(first.id, actor=BOB)  # one fact of the group: the whole definition follows
    assert accepted.status is Status.ACCEPTED
    assert {a.status for a in g.find(subject="relationship:s::oc")} == {Status.ACCEPTED}
    assert g.relationship("s::oc").from_columns == ["a"]
    v1 = g.relationship("s::oc").version

    # change both sides of the join: two new facts, one group
    g.add(Relationship("oc", on={orders["a"]: custs["a"], orders["b"]: custs["b"]}, namespace="s"), source=ANN)
    pending = g.pending(subject="relationship:s::oc")
    assert sorted(a.attribute for a in pending) == ["from_columns", "to_columns"]
    assert len({a.group for a in pending}) == 1
    assert g.relationship("s::oc").to_columns == ["a"]  # the accepted join is untouched meanwhile
    done = g.accept_group(pending[0].group, actor=BOB)
    assert len(done) == 2
    rel = g.relationship("s::oc")
    assert (rel.from_columns, rel.to_columns) == (["a", "b"], ["a", "b"])
    assert rel.version > v1 and g.version("relationship:s::oc") == rel.version


def test_a_group_is_all_or_nothing():
    from glosslayer import DefaultPolicy

    with glosslayer.open("sqlite://", policy=DefaultPolicy(mode="four_eyes")) as g:
        g.propose_many(
            [Proposal("relationship:s::oc", "from", "s.orders", group="g1"),
             Proposal("relationship:s::oc", "to", "s.customers", group="g1"),
             Proposal("relationship:s::oc", "from_columns", ["a"], group="g1")],
            source=ANN,
        )
        # bob states the last part himself, so he cannot be the one to accept that part
        g.propose_many([Proposal("relationship:s::oc", "to_columns", ["a"], group="g1")], source=BOB)
        ids = [a.id for a in g.pending()]
        done, errors = g.transition_many("accept", ids[:1], actor=BOB)
        assert done == [] and "decided as a group" in str(errors[0][1])
        assert {a.status for a in g.find()} == {Status.PROPOSED}
        with pytest.raises(PermissionDenied):
            g.accept_group("g1", actor=BOB)
        assert len(g.accept_group("g1", actor=Human("cyd"))) == 4

        rejected = g.propose_many([Proposal("metric:s::m", "expression", "SUM(x)", group="g2"),
                                   Proposal("metric:s::m", "description", "x", group="g2")], source=AI("claude"))
        g.reject(rejected[0].assertion.id, actor=ANN)
        assert {a.status for a in g.find(subject="metric:s::m")} == {Status.REJECTED}


def test_groups_do_not_bind_descriptive_facts(g):
    """Only relationships and metrics are decided as a unit; a table's columns are reviewed one by one."""
    res = g.propose_many([Proposal("field:t/a", "description", "x", group="g"),
                          Proposal("field:t/b", "description", "y", group="g")], source=ANN)
    g.accept(res[0].assertion.id, actor=BOB)
    assert [a.status for a in g.find(attribute="description")] == [Status.ACCEPTED, Status.PROPOSED]


def test_an_incomplete_relationship_does_not_break_the_listing(g):
    for attr, value in (("from", "s.orders"), ("to", "s.customers"), ("from_columns", ["a", "b"]),
                        ("to_columns", ["a"])):
        g.propose("relationship:broken", attr, value, source=ANN, accept=True)
    g.add(Relationship("fine", "s.orders", "s.customers", on={"a": "a"}), source=ANN, accept=True)
    assert [r.name for r in g.relationships()] == ["fine"]
    with pytest.raises(KeyError):
        g.relationship("broken")


def test_versions(g):
    assert g.version("metric:s::m") is None
    g.add(Metric("m", "SUM(x)", namespace="s"), source=ANN)
    assert g.version("metric:s::m") is None  # proposed only: nothing is defined yet
    g.accept(g.pending()[0].id, actor=BOB)
    v1 = g.metric("m", namespace="s").version
    assert v1 is not None
    g.comment(g.find(subject="metric:s::m")[0].id, actor=BOB, comment="nice")
    assert g.version("metric:s::m") == v1  # a comment changes nothing
    g.add(Metric("m", "SUM(y)", namespace="s"), source=ANN, accept=True)
    assert g.metric("m", namespace="s").version > v1


def test_structured_metrics_and_column_roles(g):
    t = Table("s.orders").add_columns(Column("amount", float, role="amount"), Column("day", "Date", role="time"),
                                      Column("country", str, role="category"))
    t.add_metric(Metric("revenue", "SUM(amount) FILTER (WHERE status = 'paid')", aggregate="sum(amount)",
                        filter="status = 'paid'", time_column="day", dimensions=["country"], format="#,##0.00"))
    g.add(t, source=ANN, accept=True)
    m = g.metric("revenue", table="s.orders")
    assert (m.aggregate, m.filter, m.time_column, m.dimensions, m.format) == (
        "sum(amount)", "status = 'paid'", "day", ["country"], "#,##0.00")
    assert g.table("s.orders")["day"].role == "time"
    with pytest.raises(ValueError):
        g.propose("field:s.orders/amount", "role", "measure", source=ANN)

    from glosslayer.ossie import OssieSource, validate

    export = g.to_ossie()
    assert validate(export.document) == []
    with glosslayer.open("sqlite://") as other:
        other.propose_many(OssieSource(export.document).proposals(), source="import:ossie", actor=ANN, accept=True)
        assert other.metric("revenue", table="orders").dimensions == ["country"]
        assert other.table("orders")["amount"].role == "amount"


def test_host_attributes_round_trip_through_ossie(g):
    glosslayer.register_attribute("trusted", ["dataset", "metric"], validate=bool)
    try:
        g.add(Table("s.orders", description="x").add_column("a", int), source=ANN, accept=True)
        g.propose("dataset:s.orders", "trusted", True, source=ANN, accept=True)
        with pytest.raises(ValueError):
            g.propose("field:s.orders/a", "trusted", True, source=ANN)
        export = g.to_ossie()
        from glosslayer.ossie import OssieSource, validate

        assert validate(export.document) == []
        ext = export.document["datasets"][0]["nonportable_extensions"][0]["data"]
        assert ext["trusted"] is True
        with glosslayer.open("sqlite://") as other:
            other.propose_many(OssieSource(export.document).proposals(), source="import:ossie", actor=ANN,
                               accept=True)
            assert other.current("dataset:orders")["trusted"] is True
    finally:
        from glosslayer import model

        model._ATTRIBUTES.pop("trusted")


def test_namespaces_survive_an_ossie_round_trip(g):
    Email = SemanticType("email", dtype="String", namespace="crm", synonyms=["e-mail"])
    custs = Table("crm.customers").add_columns(Column("id", int), Column("email", Email))
    orders = Table("crm.orders").add_columns(Column("customer_id", int))
    g.add(custs, orders, Relationship("oc", on={orders["customer_id"]: custs["id"]}, namespace="crm"),
          Metric("customers", "COUNT(*)", namespace="crm"), source=ANN, accept=True)
    g.propose("model:crm", "name", "crm_model", source=ANN, accept=True)
    from glosslayer.ossie import OssieSource, validate

    export = g.to_ossie(namespace="crm")
    assert validate(export.document) == [] and export.document["name"] == "crm_model"
    assert [r["name"] for r in export.document["relationships"]] == ["oc"]
    with glosslayer.open("sqlite://") as other:
        src = OssieSource(export.document, key_by="source")
        other.propose_many(src.proposals(), source="import:ossie", actor=ANN, accept=True)
        assert str(other.relationships()[0].subject) == "relationship:crm::oc"
        assert other.metric("customers", namespace="crm").expression == {"sql": "COUNT(*)"}
        assert other.table("crm.customers")["email"].semantic_type.ref == "crm::email"
        again = other.propose_many(src.proposals(), source="import:ossie", actor=ANN)
        assert {r.outcome for r in again} == {"unchanged"}  # re-importing duplicates nothing
