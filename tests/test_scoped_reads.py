"""Reads are made as an actor: the policy's scope decides what comes back."""

import pytest

import glosslayer
from glosslayer import AI, Column, Confidentiality, DefaultPolicy, Human, Relationship, RolePolicy, Table
from glosslayer import PermissionDenied, Status


@pytest.fixture
def ledger():
    policy = RolePolicy(
        admins={"human:root"},
        stewards={"sales.": {"human:alice"}, "hr.": {"human:bob"}},
        readers={"sales.": {"human:carol"}},
    )
    with glosslayer.open("sqlite://", policy=policy) as g:
        orders = Table("sales.orders", description="One row per order",
                       primary_key=[Column("order_id", int)])
        orders.add_columns(Column("employee_id", int), Column("amount", float, description="Order value"))
        people = Table("hr.employees", description="Everyone on payroll",
                       primary_key=[Column("employee_id", int)])
        people.add_columns(Column("salary", float, description="Gross yearly salary", confidentiality="restricted"))
        g.add(orders, people,
              Relationship("orders_employees", on={orders["employee_id"]: people["employee_id"]}, namespace="sales"),
              source=Human("root"), accept=True)
        g.propose("dataset:hr.employees", "synonyms", ["staff"], source=AI("claude"))
        yield g


def test_unbound_ledger_reads_everything(ledger):
    assert {t.key for t in ledger.tables()} == {"sales.orders", "hr.employees"}
    assert ledger.scope.unrestricted


def test_a_view_only_reads_its_scope(ledger):
    carol = ledger.as_actor(Human("carol"))
    assert [t.key for t in carol.tables()] == ["sales.orders"]
    assert {a.dataset for a in carol.find()} == {"sales.orders"} | {None}
    assert carol.pending() == []  # the proposal about hr.employees is not hers to see
    assert carol.current("dataset:hr.employees") == {}
    assert carol.history("dataset:hr.employees") == []
    assert all("hr." not in e.data.get("subject", "") for e in carol.log())
    assert set(carol.progress()) == {"sales", "sales.orders"}
    assert "hr.employees" not in carol
    with pytest.raises(KeyError):
        carol.table("hr.employees")
    assert [h["subject"] for h in carol.search("employees")] == ["field:sales.orders/employee_id"]
    assert carol.search("payroll") == []
    assert "hr.employees" not in carol.to_yaml()
    # a join into a table she cannot read is not shown either
    assert carol.relationships() == [] and len(ledger.relationships()) == 1

    hidden = ledger.find(subject="dataset:hr.employees", attribute="description")[0]
    with pytest.raises(KeyError):
        carol.get(hidden.id)
    with pytest.raises(KeyError):
        carol.accept(hidden.id)  # not "forbidden": she cannot know it exists
    with pytest.raises(PermissionDenied):
        carol.propose("dataset:hr.employees", "description", "mine now")


def test_a_model_view_is_scoped_and_redacted(ledger):
    everything = ledger.model_view()
    assert {t.key for t in everything.tables} == {"sales.orders", "hr.employees"} and len(everything.joins) == 1
    view = ledger.as_actor(Human("carol")).model_view()
    assert [t.key for t in view.tables] == ["sales.orders"]
    assert view.joins == []  # a join into a table she cannot read is not shown
    assert ledger.as_actor(Human("carol")).model_view("hr").tables == []


def test_stewards_and_admins_read_their_own(ledger):
    assert [t.key for t in ledger.as_actor(Human("bob")).tables()] == ["hr.employees"]
    assert len(ledger.as_actor(Human("root")).tables()) == 2
    assert ledger.as_actor(Human("nobody")).tables() == []


def test_confidentiality_applies_to_every_reader_below_the_level():
    policy = DefaultPolicy(human_max=Confidentiality.INTERNAL)
    with glosslayer.open("sqlite://", policy=policy) as g:
        t = Table("hr.employees", primary_key=[Column("employee_id", int)])
        t.add_columns(Column("salary", float, description="Gross yearly salary", confidentiality="restricted"),
                      Column("team", str, description="Team name"))
        g.add(t, source=Human("root"), accept=True)
        view = g.as_actor(Human("intern"))

        row = view.find(subject="field:hr.employees/salary", attribute="description")[0]
        assert row.redacted and row.value is None and row.to_dict()["redacted"] is True
        assert "description" not in view.current("field:hr.employees/salary")
        assert view.current("field:hr.employees/team")["description"] == "Team name"
        assert view.table("hr.employees")["salary"].description is None
        assert view.table("hr.employees")["salary"].datatype == "Float"  # structure stays
        proposed = [e for e in view.history("field:hr.employees/salary") if e.data.get("attribute") == "description"]
        assert proposed[0].data.get("redacted") is True and "value" not in proposed[0].data
        hit = view.search("salary")[0]
        assert hit["redacted"] and "description" not in hit
        assert view.search("gross yearly") == []  # not findable by words it may not read

        # the unbound ledger passes for_ai=False to read as the host itself
        assert g.search("gross yearly", for_ai=False)[0]["description"] == "Gross yearly salary"
        assert g.find(subject="field:hr.employees/salary", attribute="description")[0].value


def test_ai_scope_and_a_person_working_through_an_agent(ledger):
    agent = ledger.as_actor(AI("claude"))
    assert agent.scope.max_confidentiality is Confidentiality.INTERNAL
    assert "description" not in agent.current("field:hr.employees/salary")

    carol_via_agent = ledger.as_actor(Human("carol"), via="ai:claude")
    assert carol_via_agent.scope.prefixes == ("sales.",)
    assert carol_via_agent.scope.max_confidentiality is Confidentiality.INTERNAL
    a = carol_via_agent.propose("field:sales.orders/amount", "label", "Amount", source=AI("claude"))
    assert a.status is Status.PROPOSED and a.created_by == "human:carol"
    alice_via_agent = ledger.as_actor(Human("alice"), via="ai:claude")
    with pytest.raises(PermissionDenied):
        alice_via_agent.accept(a.id)  # she may; the agent may not, and it is the agent asking
    assert ledger.as_actor(Human("alice")).accept(a.id).status is Status.ACCEPTED


def test_ai_views_report_what_they_handed_out(ledger):
    from glosslayer import ai

    seen = []
    stop = ledger.on_read(seen.append)
    view = ledger.as_actor(Human("carol"), via="ai:claude")
    ai.model_context(view)
    ai.search(view, "order")
    stop()
    ai.search(view, "order")
    assert [r.tool for r in seen] == ["model_context", "search"]
    assert seen[0].actor == "human:carol" and seen[0].via == "ai:claude"
    assert "field:sales.orders/amount" in seen[0].subjects
    assert not any("hr." in s for r in seen for s in r.subjects)


def test_rest_reads_are_scoped(ledger):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from glosslayer.server import create_app

    c = TestClient(create_app(ledger))
    carol, root = {"X-Gloss-Actor": "human:carol"}, {"X-Gloss-Actor": "human:root"}
    assert c.get("/tables").status_code == 401
    assert [t["key"] for t in c.get("/tables", headers=carol).json()] == ["sales.orders"]
    assert len(c.get("/tables", headers=root).json()) == 2
    assert c.get("/tables/hr.employees", headers=carol).status_code == 404
    assert c.get("/assertions", params={"dataset": "hr.employees"}, headers=carol).json()["total"] == 0
    assert c.get("/subjects/dataset:hr.employees", headers=carol).json()["assertions"] == []
    assert c.get("/search", params={"q": "payroll"}, headers=carol).json() == []
    assert c.get("/search", params={"q": "payroll"}, headers=root).json()[0]["subject"] == "dataset:hr.employees"
    assert "hr.employees" not in c.get("/export/ossie", headers=carol).text
    assert c.get("/tables", headers={"X-Gloss-Actor": "carol"}).status_code == 401  # no kind, no rights


def test_router_takes_a_bound_view_from_the_host(ledger):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI, Header
    from fastapi.testclient import TestClient

    from glosslayer.server import create_router

    def host_ledger(x_user: str = Header()) -> glosslayer.Gloss:  # the host's own auth, with its own context
        return ledger.as_actor(Human(x_user), via="server:test", context={"sign_in": "direct"})

    app = FastAPI()
    app.include_router(create_router(host_ledger), prefix="/semantic")
    c = TestClient(app)
    r = c.post("/semantic/assertions", headers={"X-User": "alice"},
               json={"subject": "field:sales.orders/amount", "attribute": "label", "value": "Amount"})
    assert r.status_code == 201 and r.json()["assertion"]["created_by"] == "human:alice"
    assert ledger.log()[-1].via == "server:test"
    schema = c.get("/openapi.json").json()
    out = schema["paths"]["/semantic/tables"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert out["items"]["$ref"].endswith("/TableOut")  # a typed contract, not dict[str, Any]
