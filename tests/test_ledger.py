import pytest

import gloss
from gloss import Confidentiality, InvalidTransition, PermissionDenied, Proposal, Status
from gloss.model import EventType

F = "field:sales.orders/amount"


def test_propose_accept_current(g):
    a = g.propose(F, "description", "Order value incl. VAT", source="ai:claude", confidence=0.8)
    assert a.status is Status.PROPOSED
    assert g.current(F) == {}
    g.accept(a.id, actor="human:alice")
    assert g.current(F) == {"description": "Order value incl. VAT"}


def test_review_then_accept_and_history(g):
    a = g.propose(F, "description", "x", source="import:dbt")
    g.review(a.id, actor="human:bob", comment="looks right")
    a = g.accept(a.id, actor="human:alice")
    types = [e.type for e in g.history(F)]
    assert types == [EventType.PROPOSED, EventType.REVIEWED, EventType.ACCEPTED]
    assert g.history(F)[1].data["comment"] == "looks right"


def test_accept_supersedes_previous_value(g):
    a1 = g.propose(F, "description", "old", source="human:alice")
    g.accept(a1.id, actor="human:alice")
    a2 = g.propose(F, "description", "new", source="human:bob")
    g.accept(a2.id, actor="human:alice")
    assert g.get(a1.id).status is Status.SUPERSEDED
    assert g.current(F)["description"] == "new"


def test_same_value_from_other_source_corroborates(g):
    a = g.propose(F, "datatype", "Decimal", source="inferred:x", confidence=0.7)
    res = g.propose_many([Proposal(F, "datatype", "Decimal", 0.9)], source="import:ossie")
    assert res[0].outcome == "corroborated"
    b = g.get(a.id)
    assert b.sources == ("inferred:x", "import:ossie")
    assert b.confidence == 0.9
    assert len(g.find(subject=F)) == 1


def test_repeat_from_same_source_is_unchanged(g):
    g.propose(F, "datatype", "Decimal", source="import:x")
    res = g.propose_many([Proposal(F, "datatype", "Decimal")], source="import:x")
    assert res[0].outcome == "unchanged"
    assert len(g.log()) == 1


def test_rejected_values_are_not_reproposed(g):
    a = g.propose(F, "description", "wrong", source="ai:claude")
    g.reject(a.id, actor="human:alice", comment="not VAT")
    res = g.propose_many([Proposal(F, "description", "wrong")], source="ai:claude")
    assert res[0].outcome == "previously_rejected"
    forced = g.propose(F, "description", "wrong", source="ai:claude", force=True)
    assert forced.id != a.id and forced.status is Status.PROPOSED


def test_ai_cannot_accept(g):
    a = g.propose(F, "description", "x", source="ai:claude")
    with pytest.raises(PermissionDenied):
        g.accept(a.id, actor="ai:claude")


def test_invalid_transition(g):
    a = g.propose(F, "description", "x", source="human:a")
    g.reject(a.id, actor="human:a")
    with pytest.raises(InvalidTransition):
        g.accept(a.id, actor="human:a")
    g.reopen(a.id, actor="human:a")
    assert g.accept(a.id, actor="human:a").status is Status.ACCEPTED


def test_attribute_validation(g):
    with pytest.raises(ValueError):
        g.propose(F, "primary_key", ["x"], source="human:a")  # not a field attribute
    with pytest.raises(ValueError):
        g.propose(F, "datatype", "Money", source="human:a")
    with pytest.raises(ValueError):
        g.propose(F, "confidentiality", "top-secret", source="human:a")
    a = g.propose(F, "expression", "amount / 100", source="human:a")
    assert a.value == {"sql": "amount / 100"}


def test_short_ids(g):
    a = g.propose(F, "description", "x", source="human:a")
    assert g.accept(a.id[:8], actor="human:a").id == a.id


def test_rebuild_projection_matches(g):
    a = g.propose(F, "description", "v1", source="human:a")
    g.accept(a.id, actor="human:a")
    b = g.propose(F, "description", "v2", source="ai:x")
    g.propose(F, "datatype", "Decimal", source="catalog:c")
    g.propose_many([Proposal(F, "datatype", "Decimal", 0.5)], source="import:o")
    g.accept(b.id, actor="human:a")
    before = sorted((x.to_dict() for x in g.find()), key=lambda d: d["id"])
    assert g.rebuild_projection() == 3
    after = sorted((x.to_dict() for x in g.find()), key=lambda d: d["id"])
    assert before == after


def test_progress_and_pending(g):
    g.propose("dataset:sales.orders", "description", "d", source="human:a")
    a = g.propose(F, "description", "x", source="human:a")
    g.accept(a.id, actor="human:a")
    g.propose("metric:revenue", "expression", "SUM(x)", source="human:a")
    prog = g.progress()
    assert prog["sales.orders"]["accepted"] == 1
    assert prog["sales.orders"]["proposed"] == 1
    assert prog["(model-level)"]["proposed"] == 1
    assert len(g.pending(dataset="sales.orders")) == 1


def test_bulk_accept(g):
    for i in range(5):
        g.propose(f"field:sales.orders/c{i}", "datatype", "Integer", source="import:x")
    done, errors = g.bulk("accept", gloss.Query(source_prefix="import:"), actor="human:a")
    assert len(done) == 5 and not errors


def test_role_policy():
    pol = gloss.RolePolicy(stewards={"finance.": {"human:alice"}}, reviewers={"human:carol"})
    with gloss.open("sqlite://", policy=pol) as g:
        a = g.propose("field:finance.invoices/total", "description", "x", source="ai:claude")
        b = g.propose("field:sales.orders/total", "description", "x", source="ai:claude")
        g.review(a.id, actor="human:carol")
        with pytest.raises(PermissionDenied):
            g.accept(a.id, actor="human:carol")
        with pytest.raises(PermissionDenied):
            g.accept(b.id, actor="human:alice")  # not her domain
        assert g.accept(a.id, actor="human:alice").status is Status.ACCEPTED


def test_confidentiality_ordering():
    assert Confidentiality.RESTRICTED > Confidentiality.CONFIDENTIAL > Confidentiality.INTERNAL
    assert Confidentiality.PUBLIC <= Confidentiality.INTERNAL


def test_persistence(tmp_path):
    url = f"sqlite:///{tmp_path / 'g.db'}"
    with gloss.open(url) as g:
        a = g.propose(F, "description", "x", source="human:a")
        g.accept(a.id, actor="human:a")
    with gloss.open(url) as g:
        assert g.current(F) == {"description": "x"}
        assert len(g.log()) == 2
