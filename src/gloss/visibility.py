"""Confidentiality levels of subjects, shared by search, AI views and scoped reads.

A subject's effective level is its own accepted ``confidentiality``, else its semantic type's
(fields), else its dataset's, else ``internal``. A relationship is as confidential as the
more confidential of the datasets it joins. A semantic type's *definition* is never
confidential; the columns that use it are.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .ids import Kind, SubjectId
from .model import Confidentiality, Status

if TYPE_CHECKING:
    from .store import SQLStore

DEFAULT = Confidentiality.INTERNAL
_NEEDED = ("confidentiality", "semantic_type", "from", "to")


def _rank(value: Any) -> int | None:
    return Confidentiality(value).rank if value else None


def effective_rank(
    sid: SubjectId,
    own: Any = None,
    from_type: Any = None,
    from_dataset: Any = None,
    ends: Iterable[Any] = (),
) -> int:
    """The level rank for one subject, given the confidentiality values it can inherit from."""
    if sid.kind is Kind.TYPE:
        return DEFAULT.rank
    level = _rank(own)
    if level is None and sid.kind is Kind.FIELD:
        level = _rank(from_type)
    if level is None and sid.dataset and sid.kind is not Kind.DATASET:
        level = _rank(from_dataset)
    if sid.kind is Kind.RELATIONSHIP:
        found = [r for r in (_rank(e) for e in ends) if r is not None]
        if level is not None:
            found.append(level)
        level = max(found, default=None)
    return DEFAULT.rank if level is None else level


def levels(store: SQLStore, subjects: Iterable[str]) -> dict[str, Confidentiality]:
    """Effective confidentiality of many subjects in a fixed number of queries."""
    sids = {s: SubjectId.parse(s) for s in set(subjects)}
    accepted = [Status.ACCEPTED]
    own: dict[str, dict[str, Any]] = defaultdict(dict)
    for a in store.find_for_subjects(sids, accepted, attributes=_NEEDED):
        own[a.subject][a.attribute] = a.value

    parents: set[str] = set()
    for s, sid in sids.items():
        if sid.dataset and sid.kind is not Kind.DATASET:
            parents.add(str(SubjectId.for_dataset(sid.dataset)))
        if own[s].get("semantic_type"):
            parents.add(str(SubjectId.for_type(own[s]["semantic_type"])))
        parents.update(str(SubjectId.for_dataset(x)) for x in (own[s].get("from"), own[s].get("to"))
                       if isinstance(x, str))
    inherited: dict[str, Any] = {s: own[s].get("confidentiality") for s in parents & sids.keys()}
    for a in store.find_for_subjects(parents - sids.keys(), accepted, attributes=("confidentiality",)):
        inherited[a.subject] = a.value

    out: dict[str, Confidentiality] = {}
    for s, sid in sids.items():
        mine = own[s]
        tname = mine.get("semantic_type")
        rank = effective_rank(
            sid,
            mine.get("confidentiality"),
            inherited.get(str(SubjectId.for_type(tname))) if tname else None,
            inherited.get(str(SubjectId.for_dataset(sid.dataset))) if sid.dataset else None,
            [inherited.get(str(SubjectId.for_dataset(x))) for x in (mine.get("from"), mine.get("to"))
             if isinstance(x, str)],
        )
        out[s] = Confidentiality.from_rank(rank)
    return out
