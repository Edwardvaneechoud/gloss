"""Several writers at once: the database keeps one accepted value, and nothing is lost."""

import os
import threading

import pytest

import gloss
from gloss import Human, Status

F = "field:sales.orders/amount"


@pytest.fixture
def shared(tmp_path):
    """A ledger with real connections per thread (in-memory SQLite shares one connection)."""
    url = "sqlite://" if os.environ.get("GLOSS_TEST_DATABASE_URL") else f"sqlite:///{tmp_path / 'c.db'}"
    with gloss.open(url) as g:
        yield g


def test_racing_accepts_leave_exactly_one_accepted_value(shared):
    g = shared
    ids = [g.propose(F, "description", f"version {i}", source=Human(f"u{i}")).id for i in range(8)]
    start = threading.Barrier(len(ids))
    failures: list[BaseException] = []

    def accept(i: int) -> None:
        try:
            start.wait()
            g.accept(ids[i], actor=Human("reviewer"))
        except BaseException as exc:  # noqa: BLE001 - collected and asserted on below
            failures.append(exc)

    threads = [threading.Thread(target=accept, args=(i,)) for i in range(len(ids))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    statuses = [a.status for a in g.find(subject=F)]
    assert statuses.count(Status.ACCEPTED) == 1
    # every writer either got through (and was later superseded) or was told to retry: none vanished
    assert statuses.count(Status.SUPERSEDED) + statuses.count(Status.PROPOSED) == len(ids) - 1
    assert all(isinstance(f, gloss.ConcurrentWrite) for f in failures)
    assert statuses.count(Status.PROPOSED) == len(failures)

    seqs = [e.seq for e in g.log()]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    before = sorted((a.id, a.status.value) for a in g.find())
    g.rebuild_projection()  # the log alone reproduces the state the writers left
    assert sorted((a.id, a.status.value) for a in g.find()) == before


def test_parallel_proposals_of_the_same_solo_fact(shared):
    """Two sessions of one solo user saving different values: one wins, the other is superseded or retried."""
    g = gloss.Gloss(shared.store, gloss.DefaultPolicy(mode="solo"))
    start = threading.Barrier(4)
    errors: list[BaseException] = []

    def save(i: int) -> None:
        try:
            start.wait()
            g.propose(F, "label", f"Label {i}", source=Human("me"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=save, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(isinstance(e, gloss.ConcurrentWrite) for e in errors)
    accepted = g.find(subject=F, attribute="label", statuses=[Status.ACCEPTED])
    assert len(accepted) == 1
