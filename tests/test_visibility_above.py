"""What lies above a level by the strongest live claim: for withholding, not for publishing."""

import pytest

from glosslayer import AI, Confidentiality, Human, Status
from glosslayer.visibility import above, levels

ANN, BOB = Human("ann"), Human("bob")
INTERNAL = Confidentiality.INTERNAL


def say(g, subject, value, *, accept=False, attribute="confidentiality", source=ANN):
    return g.propose(subject, attribute, value, source=source, accept=accept)


def test_nothing_is_above_an_empty_ledger(g):
    found = above(g.store, INTERNAL)
    assert (found.datasets, found.columns) == (frozenset(), {})


def test_own_facts_types_and_datasets(g):
    say(g, "field:crm.people/salary", "restricted", accept=True)
    say(g, "field:crm.people/team", "internal", accept=True)
    say(g, "dataset:hr.reviews", "confidential", accept=True)
    say(g, "type:crm::email", "restricted", accept=True)
    say(g, "type:plain", "public", accept=True)
    say(g, "field:crm.people/mail", "crm::email", attribute="semantic_type", accept=True)
    say(g, "field:crm.people/other", "plain", attribute="semantic_type", accept=True)
    found = above(g.store, INTERNAL)
    assert found.datasets == {"hr.reviews"}
    assert found.columns == {"crm.people": {"salary", "mail"}}
    assert above(g.store, Confidentiality.RESTRICTED).columns == {}


def test_a_proposal_protects_at_once_and_the_strongest_wins(g):
    say(g, "field:crm.people/salary", "public", accept=True)
    say(g, "field:crm.people/salary", "confidential", source=AI("claude"))
    # a column's own lower level does not undo its type's
    say(g, "type:email", "restricted", accept=True)
    say(g, "field:crm.people/mail", "email", attribute="semantic_type")
    say(g, "field:crm.people/mail", "public", accept=True)
    assert above(g.store, INTERNAL).columns == {"crm.people": {"salary", "mail"}}
    assert above(g.store, INTERNAL, statuses=[Status.ACCEPTED]).columns == {}
    # the published level says otherwise: there, what was agreed is what counts
    published = levels(g.store, ["field:crm.people/salary", "field:crm.people/mail"])
    assert set(published.values()) == {Confidentiality.PUBLIC}


@pytest.mark.parametrize("end", ["reject", "retract"])
def test_a_claim_that_was_withdrawn_no_longer_counts(g, end):
    fact = say(g, "field:crm.people/salary", "restricted")
    stale = say(g, "dataset:hr.reviews", "restricted", accept=True)
    g.mark_stale(stale.id, actor="system:test", reason="table changed")
    getattr(g, end)(fact.id, actor=ANN)
    found = above(g.store, INTERNAL)
    assert found.columns == {} and found.datasets == {"hr.reviews"}  # stale still protects
