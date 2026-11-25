"""Taking facts away: rejecting or retracting by who stated them, all or nothing."""

import pytest

import glosslayer
from glosslayer import AI, DefaultPolicy, Human, Metric, PermissionDenied, Proposal, RolePolicy, Status

ANN, BOB = Human("ann"), Human("bob")
F = "field:sales.orders/amount"
M = "metric:sales.orders/revenue"


def state(g, subject, **kw):
    return sorted((a.attribute, a.status.value) for a in g.find(subject=subject, **kw))


def test_each_fact_is_taken_away_the_way_that_fits_it(g):
    g.propose(F, "label", "Amount", source=ANN, accept=True)
    g.propose(F, "role", "amount", source=AI("claude"))
    g.propose(F, "owner", "me", source=BOB)
    changed = g.withdraw(g.find(subject=F), actor=BOB, comment="not any more")
    assert {a.attribute: a.status for a in changed} == {
        "label": Status.RETRACTED,  # accepted, so it was held
        "owner": Status.RETRACTED,  # Bob's own proposal
        "role": Status.REJECTED,  # a decision on someone else's proposal
    }
    assert {e.data["comment"] for e in g.history(F)[-3:]} == {"not any more"}
    again = g.propose_many([Proposal(F, "role", "amount")], source=AI("claude"))[0]
    assert again.outcome == "previously_rejected"
    assert g.propose(F, "label", "Amount", source=ANN).status is Status.PROPOSED  # a retracted value may come back
    assert g.withdraw([], actor=BOB) == []


def test_the_first_refusal_undoes_them_all():
    with glosslayer.open("sqlite://", policy=RolePolicy(stewards={"sales.": {"human:ann"}})) as g:
        g.propose(F, "label", "Amount", source=ANN, accept=True)
        g.propose("field:hr.people/pay", "label", "Pay", source=BOB)
        before = state(g, F) + state(g, "field:hr.people/pay")
        with pytest.raises(PermissionDenied):
            g.withdraw(g.find(), actor=ANN)  # she decides in sales, not in hr
        assert state(g, F) + state(g, "field:hr.people/pay") == before
        assert g.can_withdraw([F, "field:hr.people/pay", "field:x/y"], actor=ANN) == {
            F: True, "field:hr.people/pay": False
        }
        assert g.as_actor(BOB).can_withdraw([F, "field:hr.people/pay"]) == {F: False, "field:hr.people/pay": True}


def test_a_definition_is_removed_whole_and_its_name_is_free_again(g):
    g.add(Metric("revenue", "SUM(amount)", table="sales.orders", description="d"), source=ANN, accept=True)
    g.add(Metric("revenue", "SUM(amount) / 2", table="sales.orders"), source=BOB)
    assert g.can_withdraw([M], actor=ANN) == {M: True}
    changed = g.withdraw_subject(M, actor=ANN, comment="removed")
    assert {a.status for a in changed} == {Status.RETRACTED, Status.REJECTED}
    assert g.metric_names("sales") == {} and g.model_view("sales").metrics == []
    with pytest.raises(KeyError, match="holds nothing to remove"):
        g.withdraw_subject(M, actor=ANN)
    assert g.can_withdraw([M], actor=ANN) == {}
    g.add(Metric("revenue", "COUNT(*)", table="sales.orders"), source=ANN, accept=True)
    assert g.metric("revenue", table="sales.orders").expression == {"sql": "COUNT(*)"}


def test_clearing_values(g):
    g.propose(F, "label", "Amount", source=ANN, accept=True)
    g.propose(F, "label", "Order value", source=AI("claude"))
    g.propose(F, "role", "amount", source=ANN, accept=True)
    g.propose("dataset:sales.orders", "description", "Orders", source=ANN, accept=True)
    cleared = g.clear([(F, "label"), ("dataset:sales.orders", "owner")], actor=ANN, comment="cleared")
    assert sorted(a.status.value for a in cleared) == ["rejected", "retracted"]
    assert g.current(F) == {"role": "amount"} and g.current("dataset:sales.orders") == {"description": "Orders"}
    assert g.clear([], actor=ANN) == []


def define(g, actor, **parts):
    metric = Metric("revenue", "SUM(amount)", table="sales.orders", **parts)
    return g.as_actor(actor).propose_many(metric.proposals())


def test_an_accepted_edit_takes_away_what_it_no_longer_states():
    optional = ("description", "filter", "format")
    with glosslayer.open("sqlite://", policy=DefaultPolicy(mode="solo")) as g:
        define(g, ANN, description="d", filter="amount > 0", format="0.0", owner="sales team")
        results = define(g, ANN, filter="amount > 1")
        gone = g.as_actor(ANN).retract_left_out(results, optional, comment="cleared")
        assert sorted(a.attribute for a in gone) == ["description", "format"]
        assert g.current(M) == {"expression": {"sql": "SUM(amount)"}, "filter": "amount > 1", "owner": "sales team"}
        assert g.as_actor(ANN).retract_left_out([], optional) == []


def test_a_queued_edit_changes_nothing_until_it_is_decided(g):
    g.add(Metric("revenue", "SUM(amount)", table="sales.orders", description="d"), source=ANN, accept=True)
    results = define(g, BOB, filter="amount > 1")
    assert g.as_actor(BOB).retract_left_out(results, ("description",)) == []
    assert g.current(M)["description"] == "d"


def test_only_what_the_reader_may_read_is_withdrawn():
    policy = RolePolicy(stewards={"sales.": {"human:ann"}}, readers={"sales.": set()})
    with glosslayer.open("sqlite://", policy=policy) as g:
        g.propose("field:hr.people/pay", "label", "Pay", source=BOB)
        view = g.as_actor(ANN)
        with pytest.raises(KeyError):  # out of scope is "not there", never "forbidden"
            view.withdraw_subject("field:hr.people/pay")
        assert view.can_withdraw(["field:hr.people/pay"]) == {} and view.clear([("field:hr.people/pay", "label")]) == []
        g.propose(F, "label", "Amount", source=ANN)
        assert [a.status for a in view.withdraw_subject(F)] == [Status.RETRACTED]


def test_withdrawing_is_rebuilt_from_the_log(g):
    g.add(Metric("revenue", "SUM(amount)", table="sales.orders"), source=ANN, accept=True)
    g.withdraw_subject(M, actor=BOB)
    before = state(g, M)
    g.rebuild_projection()
    assert state(g, M) == before
