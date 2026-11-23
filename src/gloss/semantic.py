"""Applying semantic types at scale.

``TypeSuggester`` is a Source: for every accepted field without a semantic type, it
proposes the type whose ``name_patterns`` match the column name (and whose datatype
is compatible). With 10,000 tables this is how ``email_address`` reaches every
``*_email`` column: as proposals a person accepts in bulk.

    g.propose_many(TypeSuggester().proposals(g), source=TypeSuggester.source_id)
    gloss import semantic-types            # same, from the CLI
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from .ids import SubjectId
from .ledger import Proposal
from .model import Status

if TYPE_CHECKING:
    from .ledger import Gloss


class TypeSuggester:
    source_id = "inferred:type-names"

    def __init__(self, confidence: float = 0.7, dataset_prefix: str | None = None):
        self.confidence = confidence
        self.dataset_prefix = dataset_prefix

    @classmethod
    def from_spec(cls, arg: str, **kwargs: Any) -> TypeSuggester:
        return cls(dataset_prefix=arg or None, **kwargs)

    def proposals(self, gloss: Gloss) -> Iterator[Proposal]:
        state = gloss.snapshot([Status.ACCEPTED])
        rules = [
            (name, re.compile(p, re.IGNORECASE), t.attrs.get("datatype"))
            for name, t in sorted(state.types.items())
            for p in t.attrs.get("name_patterns") or []
        ]
        if not rules:
            return
        for dk, d in state.datasets.items():
            if self.dataset_prefix and not dk.startswith(self.dataset_prefix):
                continue
            for fk, f in d.fields.items():
                if "semantic_type" in f.attrs:
                    continue
                for tname, rx, tdt in rules:
                    if not rx.search(fk):
                        continue
                    fdt = f.attrs.get("datatype")
                    if tdt and fdt and tdt != fdt:
                        continue  # e.g. an Integer column called email_count is not an e-mail address
                    yield Proposal(
                        str(SubjectId.for_field(dk, fk)),
                        "semantic_type",
                        tname,
                        self.confidence,
                        [{"rule": "column name matches type pattern", "pattern": rx.pattern, "column": fk}],
                    )
                    break
