"""Lineage-based proposals from Flowfile.

Flowfile knows which flow produced each table and which tables each flow
reads. That gives table-level lineage: the *upstream* tables of a table are
the tables read by the flow that produced it.

What this source proposes today
-------------------------------
* **Confidentiality and classification follow the data.** When an upstream
  column is accepted as e.g. ``confidential`` / ``pii.email``, a downstream
  column with the same name is *proposed* with the same level (confidence 0.6,
  as it is a name match). If a downstream table has upstream confidential data
  at all, the table itself is proposed at that level (confidence 0.5).
* **Descriptions carry over** for same-named columns that have no accepted
  description downstream (confidence 0.5).

Every proposal records the upstream table, the flow and the rule as evidence.

TODO (needs Flowfile's flow graph, available in embedded mode):
* column-level lineage through select/rename/formula nodes (raises confidence to ~0.9);
* relationships from join nodes (join keys → from_columns/to_columns);
* metric candidates from group-by + aggregate nodes.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from gloss.ids import SubjectId
from gloss.ledger import Proposal
from gloss.model import Confidentiality, Status

from .catalog import FlowfileCatalog, _get

if TYPE_CHECKING:
    from gloss.ledger import Gloss

_ATTRS = ("confidentiality", "classification", "description")


class LineageSource:
    """Source plugin: proposes confidentiality, classification and descriptions along table lineage."""

    source_id = "inferred:flowfile-lineage"

    def __init__(self, catalog: FlowfileCatalog):
        self.catalog = catalog

    @classmethod
    def from_spec(cls, arg: str, **kwargs: Any) -> LineageSource:
        """``arg`` is a Flowfile base URL, as for ``FlowfileCatalog.from_spec``."""
        return cls(FlowfileCatalog.from_spec(arg, **kwargs))

    def upstream(self) -> dict[str, list[tuple[str, str | None]]]:
        """{table_key: [(upstream_key, flow_name), ...]}"""
        tables = self.catalog.tables(refresh=True)
        readers: dict[int, list[str]] = {}
        for key, t in tables.items():
            for flow in _get(t, "read_by_flows") or []:
                readers.setdefault(_get(flow, "id"), []).append(key)
        out: dict[str, list[tuple[str, str | None]]] = {}
        for key, t in tables.items():
            producer = _get(t, "source_registration_id") or _get(t, "producer_registration_id")
            if producer is None:
                continue
            flow_name = _get(t, "source_registration_name") or _get(t, "producer_registration_name")
            ups = [(u, flow_name) for u in readers.get(producer, []) if u != key]
            if ups:
                out[key] = ups
        return out

    def proposals(self, gloss: Gloss) -> Iterator[Proposal]:
        """Accepted upstream facts proposed for same-named downstream columns (and tables) lacking one."""
        lineage = self.upstream()
        columns = {key: [c.name for c in self.catalog.describe(key).columns] for key in lineage}

        # everything this run needs to know, in two lookups instead of several per column
        wanted: set[str] = set()
        for key, ups in lineage.items():
            for table in (key, *(u for u, _ in ups)):
                wanted.add(str(SubjectId.for_dataset(table)))
                wanted.update(str(SubjectId.for_field(table, col)) for col in columns[key])
        known = gloss.current_many(wanted, attributes=_ATTRS)
        open_levels = {
            a.subject
            for a in gloss.find(kind="dataset", attribute="confidentiality",
                                statuses=[Status.PROPOSED, Status.REVIEWED])
        }

        def facts(subject: SubjectId) -> dict[str, Any]:
            return known.get(str(subject), {})

        for key, ups in lineage.items():
            strongest: Confidentiality | None = None
            for up_key, flow in ups:
                ev_base = {"upstream": up_key, "flow": flow}
                ds_level = facts(SubjectId.for_dataset(up_key)).get("confidentiality")
                for col in columns[key]:
                    up_facts = facts(SubjectId.for_field(up_key, col))
                    if not up_facts:
                        continue
                    down_subj = SubjectId.for_field(key, col)
                    down_facts = facts(down_subj)
                    level = up_facts.get("confidentiality") or ds_level
                    if level and Confidentiality(level) > Confidentiality.INTERNAL:
                        if "confidentiality" not in down_facts:
                            yield Proposal(
                                str(down_subj), "confidentiality", level, 0.6,
                                [{**ev_base, "rule": "same-name column downstream of confidential column"}],
                            )
                        lvl = Confidentiality(level)
                        strongest = lvl if strongest is None or lvl > strongest else strongest
                    if up_facts.get("classification") and "classification" not in down_facts:
                        yield Proposal(
                            str(down_subj), "classification", up_facts["classification"], 0.6,
                            [{**ev_base, "rule": "classification propagated by column name"}],
                        )
                    if up_facts.get("description") and "description" not in down_facts:
                        yield Proposal(
                            str(down_subj), "description", up_facts["description"], 0.5,
                            [{**ev_base, "rule": "description carried over from same-named upstream column"}],
                        )
            if strongest is not None:
                ds_subj = SubjectId.for_dataset(key)
                if "confidentiality" not in facts(ds_subj) and str(ds_subj) not in open_levels:
                    yield Proposal(
                        str(ds_subj), "confidentiality", strongest.value, 0.5,
                        [{"upstream": [u for u, _ in ups], "rule": "contains data derived from confidential columns"}],
                    )
