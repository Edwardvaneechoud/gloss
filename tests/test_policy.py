"""Review modes, who may do what, and what happens to a proposal."""

import pytest

import glosslayer
from glosslayer import AI, Confidentiality, DefaultPolicy, Human, PermissionDenied, Proposal, RolePolicy, Status
from glosslayer.model import EventType

F = "field:sales.orders/amount"


def _open(**policy):
    return glosslayer.open("sqlite://", policy=DefaultPolicy(**policy))


def test_solo_accepts_your_own_edit_in_one_append():
    with _open(mode="solo") as g:
        a = g.propose(F, "description", "Order value", source=Human("me"))
        assert a.status is Status.ACCEPTED and g.current(F) == {"description": "Order value"}
        proposed, accepted = g.log()
        assert (proposed.type, accepted.type) == (EventType.PROPOSED, EventType.ACCEPTED)
        assert accepted.data["auto"] is True and accepted.data["rule"] == "solo"
        assert proposed.correlation == accepted.correlation  # one write

        b = g.propose(F, "description", "Order value incl. VAT", source=Human("me"))
        assert g.get(a.id).status is Status.SUPERSEDED and b.status is Status.ACCEPTED


def test_solo_still_queues_the_machine():
    with _open(mode="solo") as g:
        assert g.propose(F, "description", "x", source=AI("claude")).status is Status.PROPOSED
        assert g.propose(F, "label", "x", source="import:dbt").status is Status.PROPOSED
        # a person passing on an AI's suggestion is not stating it themselves
        relayed = g.propose(F, "synonyms", ["amt"], source=AI("claude"), actor=Human("me"))
        assert relayed.status is Status.PROPOSED
        # nor is a person working through an agent
        via_agent = g.as_actor(Human("me"), via="ai:claude").propose(F, "is_time", False)
        assert via_agent.status is Status.PROPOSED
        assert g.accept(relayed.id, actor=Human("me")).status is Status.ACCEPTED


def test_solo_restating_an_ai_suggestion_accepts_it():
    with _open(mode="solo") as g:
        idea = g.propose(F, "description", "x", source=AI("claude"))
        mine = g.propose(F, "description", "x", source=Human("me"))
        assert mine.id == idea.id and mine.status is Status.ACCEPTED
        assert mine.sources == ("ai:claude", "human:me")


def test_mode_is_per_path():
    with _open(mode="solo", modes={"prod.": "four_eyes"}) as g:
        assert g.propose("dataset:sandbox.t", "description", "x", source=Human("a")).status is Status.ACCEPTED
        a = g.propose("dataset:prod.t", "description", "x", source=Human("a"))
        assert a.status is Status.PROPOSED
        with pytest.raises(PermissionDenied):
            g.accept(a.id, actor=Human("a"))
        assert g.accept(a.id, actor=Human("b")).status is Status.ACCEPTED
        # "prod." does not swallow a path that merely starts with the same letters
        assert g.propose("dataset:production.t", "description", "x", source=Human("a")).status is Status.ACCEPTED


def test_four_eyes():
    with _open(mode="four_eyes") as g:
        mine = g.propose(F, "description", "x", source=Human("ann"))
        with pytest.raises(PermissionDenied):
            g.accept(mine.id, actor=Human("ann"))
        with pytest.raises(PermissionDenied):
            g.add(glosslayer.Table("t").add_column("c", int), actor=Human("ann"), accept=True)
        assert g.find(dataset="t") == []  # refused before anything was written

        # a second person stating the same value is the second pair of eyes
        g.propose(F, "description", "x", source=Human("bob"))
        assert g.accept(mine.id, actor=Human("ann")).status is Status.ACCEPTED

        # an AI's work reviewed by one person is fine
        idea = g.propose(F, "label", "Amount", source=AI("claude"))
        assert g.accept(idea.id, actor=Human("ann")).status is Status.ACCEPTED


def test_restore_puts_a_superseded_value_back():
    with _open(mode="four_eyes") as g:
        v1 = g.propose(F, "description", "v1", source=Human("ann"))
        g.accept(v1.id, actor=Human("bob"))
        v2 = g.propose(F, "description", "v2", source=Human("bob"))
        g.accept(v2.id, actor=Human("ann"))
        with pytest.raises(glosslayer.InvalidTransition):
            g.accept(v1.id, actor=Human("bob"))  # superseded values come back only through restore
        # its author may put it back without waiting for a second person: it was reviewed before
        assert g.restore(v1.id, actor=Human("ann")).status is Status.ACCEPTED
        assert g.get(v2.id).status is Status.SUPERSEDED and g.current(F) == {"description": "v1"}
        assert g.history(F)[-1].type is EventType.RESTORED
        with pytest.raises(PermissionDenied):
            g.restore(v2.id, actor=AI("claude"))


def test_a_non_steward_cannot_retract_or_stale_a_stewarded_fact():
    pol = RolePolicy(stewards={"finance.": {"human:alice"}})
    with glosslayer.open("sqlite://", policy=pol) as g:
        a = g.propose("field:finance.invoices/total", "description", "x", source="ai:claude")
        g.accept(a.id, actor="human:alice")
        with pytest.raises(PermissionDenied):
            g.retract(a.id, actor="human:mallory")
        with pytest.raises(PermissionDenied):
            g.mark_stale(a.id, actor="human:mallory", reason="no")
        assert g.retract(a.id, actor="ai:claude").status is Status.RETRACTED  # its own statement


def test_machines_accept_structure_only(g):
    structural = g.propose(F, "datatype", "Decimal", source="import:x")
    meaning = g.propose(F, "description", "x", source="import:x")
    assert g.accept(structural.id, actor="catalog:wh").status is Status.ACCEPTED
    for machine in ("catalog:wh", "system:job"):
        with pytest.raises(PermissionDenied):
            g.accept(meaning.id, actor=machine)
    # what a catalog states about structure is accepted as it is stated; meaning is queued
    assert g.propose(F, "physical_type", "Decimal(18,2)", source="catalog:wh").status is Status.ACCEPTED
    assert g.propose(F, "label", "Amount", source="catalog:wh").status is Status.PROPOSED
    # a metric's expression is meaning, not structure
    assert g.propose("metric:revenue", "expression", "SUM(x)", source="catalog:wh").status is Status.PROPOSED


@pytest.mark.parametrize("actor", ["alice", "", "root:alice", ":alice", "human:"])
def test_actors_of_unknown_shape_have_no_rights(g, actor):
    a = g.propose(F, "description", "x", source="human:a")
    with pytest.raises((PermissionDenied, ValueError)):
        g.accept(a.id, actor=actor)
    with pytest.raises((PermissionDenied, ValueError)):
        g.propose(F, "label", "x", source=actor)
    assert g.get(a.id).status is Status.PROPOSED and len(g.find()) == 1
    assert not DefaultPolicy().can(actor, "accept", a)


def test_a_host_can_refuse_attributes_that_hold_data():
    with _open(refuse_attributes=frozenset({"examples"})) as g:
        with pytest.raises(PermissionDenied, match="examples"):
            g.propose(F, "examples", ["4111 1111"], source=Human("a"))
        assert g.propose(F, "description", "x", source=Human("a")).status is Status.PROPOSED


def test_host_context_reaches_the_policy_and_via_is_recorded():
    class DirectSignInOnly(DefaultPolicy):
        def may_decide(self, actor, assertion, context):
            return bool(context) and context.attrs.get("sign_in") == "direct"

    with glosslayer.open("sqlite://", policy=DirectSignInOnly()) as g:
        a = g.propose(F, "description", "x", source=Human("a"))
        delegated = g.as_actor(Human("b"), via="server:eu-1", context={"sign_in": "delegated"})
        with pytest.raises(PermissionDenied):
            delegated.accept(a.id)
        direct = g.as_actor(Human("b"), context={"sign_in": "direct"})
        assert direct.accept(a.id).status is Status.ACCEPTED
        with pytest.raises(PermissionDenied):
            direct.accept(a.id, actor=Human("someone-else"))  # a view acts as its own actor only
        b = delegated.propose(F, "label", "x")
        assert g.history(F)[-1].via == "server:eu-1" and b.created_by == "human:b"


def test_a_custom_policy_decides_proposals(g):
    class AcceptImports(DefaultPolicy):
        def on_propose(self, actor, source, subject, attribute, context=None):
            return "accept" if source.startswith("import:trusted") else "queue"

    with glosslayer.open("sqlite://", policy=AcceptImports()) as h:
        res = h.propose_many([Proposal(F, "description", "x")], source="import:trusted-model")
        assert res[0].accepted and h.log()[-1].data == {"from_status": "proposed", "auto": True, "rule": "policy"}
        queued = h.propose_many([Proposal(F, "label", "x")], source="import:trusted-model", auto_accept=False)
        assert not queued[0].accepted and queued[0].assertion.status is Status.PROPOSED


def test_scope_intersection():
    from glosslayer import Scope

    a = Scope(prefixes=("sales", "finance.eu"), max_confidentiality=Confidentiality.CONFIDENTIAL)
    b = Scope(prefixes=("sales.emea", "finance"), max_confidentiality=Confidentiality.INTERNAL)
    both = a.intersect(b)
    assert set(both.prefixes) == {"sales.emea", "finance.eu"}
    assert both.max_confidentiality is Confidentiality.INTERNAL
    assert both.covers("sales.emea.orders") and not both.covers("sales.apac.orders")
    assert Scope.nothing().covers("x") is False and Scope.everything().unrestricted
