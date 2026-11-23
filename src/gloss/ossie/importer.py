"""Import Apache Ossie semantic model documents as proposals.

Imported facts are *proposed*, never accepted directly: a file is a claim,
and claims get reviewed. (The CLI's ``--accept`` flag accepts them in one go
on behalf of a named person, for bootstrapping from an already-reviewed model.)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ..ids import SubjectId
from ..ledger import Proposal
from ..model import attribute, new_id
from . import GLOSS_VENDOR

if TYPE_CHECKING:
    from ..ledger import Gloss


def load_document(path_or_doc: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_doc, dict):
        return path_or_doc
    text = Path(path_or_doc).read_text()
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or "datasets" not in doc:
        raise ValueError(f"{path_or_doc} does not look like an Ossie semantic model (no 'datasets')")
    return doc


def _ai_context(subj: str, ctx: Any, ev: list[dict[str, Any]]) -> Iterator[Proposal]:
    if ctx is None:
        return
    if isinstance(ctx, str):
        yield Proposal(subj, "ai_instructions", ctx, None, ev)
        return
    if ctx.get("instructions"):
        yield Proposal(subj, "ai_instructions", ctx["instructions"], None, ev)
    if ctx.get("synonyms"):
        yield Proposal(subj, "synonyms", list(ctx["synonyms"]), None, ev)
    if ctx.get("examples"):
        yield Proposal(subj, "examples", list(ctx["examples"]), None, ev)


def _gloss_data(exts: list[dict[str, Any]] | None) -> dict[str, Any]:
    for ext in exts or []:
        if ext.get("vendor_name") == GLOSS_VENDOR:
            return ext.get("data") or {}
    return {}


_EXT_NOTES = frozenset({"inherited", "pending", "namespace"})  # about the export, not facts


def _gloss_ext(subj: str, exts: list[dict[str, Any]] | None, ev: list[dict[str, Any]],
               group: str | None = None) -> Iterator[Proposal]:
    """Gloss-only attributes from the extension: whatever the attribute registry knows for this kind."""
    data = _gloss_data(exts)
    inherited = set(data.get("inherited") or [])
    kind = SubjectId.parse(subj).kind
    for attr, value in data.items():
        spec = attribute(attr)
        if attr in _EXT_NOTES or attr in inherited or value is None or spec is None or kind not in spec.kinds:
            continue
        yield Proposal(subj, attr, value, None, ev, group=group)


class OssieSource:
    """Source plugin: ``OssieSource("model.yaml").proposals(g)``.

    ``namespace`` is where the document's relationships, model-level metrics and model facts
    land when the document does not say (a Gloss export records each one's own namespace).
    """

    def __init__(self, path_or_doc: str | Path | dict[str, Any], *, key_by: str = "name",
                 namespace: str | None = None):
        if key_by not in ("name", "source"):
            raise ValueError("key_by must be 'name' or 'source'")
        self.doc = load_document(path_or_doc)
        self.key_by = key_by
        self.namespace = namespace
        self.origin = str(path_or_doc) if not isinstance(path_or_doc, dict) else "<document>"
        self.source_id = "import:ossie"

    @classmethod
    def from_spec(cls, arg: str, **kwargs: Any) -> OssieSource:
        return cls(arg, **kwargs)

    def dataset_key(self, ds: dict[str, Any]) -> str:
        return ds["source"] if self.key_by == "source" and ds.get("source") else ds["name"]

    def proposals(self, gloss: Gloss | None = None) -> Iterator[Proposal]:
        doc = self.doc
        base_ev = {"file": self.origin, "ossie_version": doc.get("version")}
        keys = {ds["name"]: self.dataset_key(ds) for ds in doc.get("datasets", [])}

        model_ext = _gloss_data(doc.get("nonportable_extensions"))
        m = str(SubjectId.model(model_ext.get("namespace") or self.namespace))
        ev = [{**base_ev, "path": "$"}]
        for tname, tattrs in (_gloss_data(doc.get("nonportable_extensions")).get("types") or {}).items():
            tsubj = str(SubjectId.for_type(tname))
            tev = [{**base_ev, "path": f"$.nonportable_extensions[gloss].types[{tname}]"}]
            for attr, value in tattrs.items():
                yield Proposal(tsubj, attr, value, None, tev)
        if doc.get("name"):
            yield Proposal(m, "name", doc["name"], None, ev)
        if doc.get("description"):
            yield Proposal(m, "description", doc["description"], None, ev)
        yield from _ai_context(m, doc.get("ai_context"), ev)

        for ds in doc.get("datasets", []):
            key = keys[ds["name"]]
            subj = str(SubjectId.for_dataset(key))
            ev = [{**base_ev, "path": f"$.datasets[{ds['name']}]"}]
            for attr in ("source", "primary_key", "unique_keys", "description"):
                if ds.get(attr) is not None:
                    yield Proposal(subj, attr, ds[attr], None, ev)
            yield from _ai_context(subj, ds.get("ai_context"), ev)
            yield from _gloss_ext(subj, ds.get("nonportable_extensions"), ev)

            for f in ds.get("fields", []) or []:
                fsubj = str(SubjectId.for_field(key, f["name"]))
                fev = [{**base_ev, "path": f"$.datasets[{ds['name']}].fields[{f['name']}]"}]
                inherited = set(_gloss_data(f.get("nonportable_extensions")).get("inherited") or [])
                for attr in ("expression", "datatype", "label", "description"):
                    if f.get(attr) is not None and attr not in inherited:
                        yield Proposal(fsubj, attr, f[attr], None, fev)
                dim = f.get("dimension") or {}
                if "is_time" in dim:
                    yield Proposal(fsubj, "is_time", bool(dim["is_time"]), None, fev)
                for p in _ai_context(fsubj, f.get("ai_context"), fev):
                    if p.attribute not in inherited:
                        yield p
                yield from _gloss_ext(fsubj, f.get("nonportable_extensions"), fev)

            for met in ds.get("metrics", []) or []:
                yield from self._metric(met, key, base_ev)

        for rel in doc.get("relationships", []) or []:
            ns = _gloss_data(rel.get("nonportable_extensions")).get("namespace") or self.namespace
            subj = str(SubjectId.for_relationship(rel["name"], ns))
            ev = [{**base_ev, "path": f"$.relationships[{rel['name']}]"}]
            group = new_id()  # a join is reviewed as one definition
            yield Proposal(subj, "from", keys.get(rel["from"], rel["from"]), None, ev, group=group)
            yield Proposal(subj, "to", keys.get(rel["to"], rel["to"]), None, ev, group=group)
            yield Proposal(subj, "from_columns", list(rel["from_columns"]), None, ev, group=group)
            yield Proposal(subj, "to_columns", list(rel["to_columns"]), None, ev, group=group)
            for p in _ai_context(subj, rel.get("ai_context"), ev):
                p.group = group
                yield p
            yield from _gloss_ext(subj, rel.get("nonportable_extensions"), ev, group)

        for met in doc.get("metrics", []) or []:
            yield from self._metric(met, None, base_ev)

    def _metric(self, met: dict[str, Any], dataset: str | None, base_ev: dict[str, Any]) -> Iterator[Proposal]:
        ns = None if dataset else _gloss_data(met.get("nonportable_extensions")).get("namespace") or self.namespace
        subj = str(SubjectId.for_metric(met["name"], dataset, namespace=ns))
        ev = [{**base_ev, "path": f"$.metrics[{met['name']}]"}]
        group = new_id()  # a metric is reviewed as one definition
        for attr in ("expression", "description", "datatype"):
            if met.get(attr) is not None:
                yield Proposal(subj, attr, met[attr], None, ev, group=group)
        for p in _ai_context(subj, met.get("ai_context"), ev):
            p.group = group
            yield p
        yield from _gloss_ext(subj, met.get("nonportable_extensions"), ev, group)
