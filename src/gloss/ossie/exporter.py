"""Export Gloss state as a valid Apache Ossie semantic model document.

Ossie carries what the data *means*. What Gloss knows *about* those
definitions (confidentiality, classification, owner, review state) travels
in ``nonportable_extensions`` under ``vendor_name: gloss``, which other tools
ignore and Ossie validation accepts.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ..ids import path_covers
from ..model import Status, attribute
from . import GLOSS_VENDOR, OSSIE_VERSION

if TYPE_CHECKING:
    from ..ledger import Entry, Gloss, ModelState


class _PlainDumper(yaml.SafeDumper):
    """Write every value out in full: no &anchors / *aliases for values shared by several fields."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


@dataclass
class ExportResult:
    document: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_yaml(self) -> str:
        return yaml.dump(self.document, Dumper=_PlainDumper, sort_keys=False, allow_unicode=True, width=100)


def _ai_context(attrs: dict[str, Any]) -> Any:
    instr, syn, ex = attrs.get("ai_instructions"), attrs.get("synonyms"), attrs.get("examples")
    if not (syn or ex):
        return instr
    ctx: dict[str, Any] = {}
    if instr:
        ctx["instructions"] = instr
    if syn:
        ctx["synonyms"] = syn
    if ex:
        ctx["examples"] = ex
    return ctx


# Listed first in the extension, for stable and readable output; other Gloss-only attributes follow.
_EXT_FIRST = ("semantic_type", "confidentiality", "classification", "owner", "physical_type")
_EXT_SPECIAL = frozenset({"checks", "name_patterns"})


def _extension(entry: Entry, annotate: bool, namespace: str | None = None) -> list[dict[str, Any]] | None:
    """Everything Ossie has no place for: attributes not marked ``ossie`` in the attribute registry."""
    data: dict[str, Any] = {}
    if namespace:
        data["namespace"] = namespace
    rest = [a for a in entry.attrs if a not in _EXT_FIRST and a not in _EXT_SPECIAL]
    for attr in (*_EXT_FIRST, *rest):
        spec = attribute(attr)
        if attr in entry.attrs and (spec is None or not spec.ossie):
            data[attr] = entry.attrs[attr]
    own_checks = [c for i, c in enumerate(entry.attrs.get("checks") or []) if i not in entry.check_origins]
    if own_checks:
        data["checks"] = own_checks
    if entry.inherited:
        data["inherited"] = sorted(entry.inherited)  # values that come from the semantic type
    if annotate:
        pending = {
            attr: a.status.value for attr, a in entry.assertions.items() if a.status is not Status.ACCEPTED
        }
        if pending:
            data["pending"] = pending
    return [{"vendor_name": GLOSS_VENDOR, "data": data}] if data else None


def _put(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None and value != [] and value != {}:
        target[key] = value


def _first_seen(entry: Entry) -> float:
    return min((a.created_at.timestamp() for a in entry.assertions.values()), default=0.0)


def _ossie_names(keys: Iterable[str]) -> dict[str, str]:
    """Gloss dataset keys may contain dots; Ossie names are referenced in SQL, so keep them simple."""
    keys = list(keys)
    short = {k: k.rsplit(".", 1)[-1] for k in keys}
    counts: dict[str, int] = {}
    for s in short.values():
        counts[s] = counts.get(s, 0) + 1
    return {k: (short[k] if counts[short[k]] == 1 else k.replace(".", "_")) for k in keys}


def _ref_names(refs: Iterable[str]) -> dict[str, str]:
    """Ossie names for namespaced references: the bare name, or ``ns_name`` when two share one."""
    refs = list(refs)
    short = {r: r.rpartition("::")[2] for r in refs}
    counts: dict[str, int] = {}
    for s in short.values():
        counts[s] = counts.get(s, 0) + 1
    return {r: (short[r] if counts[short[r]] == 1 else r.replace("::", "_").replace(".", "_")) for r in refs}


def _namespace_of(ref: str) -> str | None:
    return ref.rpartition("::")[0] or None


def build_document(
    state: ModelState,
    *,
    name: str | None = None,
    annotate: bool = True,
    dataset_prefix: str | None = None,
    namespace: str | None = None,
    log_seq: int | None = None,
) -> ExportResult:
    """``namespace`` exports one namespace's model: its datasets, relationships, metrics and model facts."""
    warnings: list[str] = []
    datasets = {
        k: d for k, d in state.datasets.items()
        if (not dataset_prefix or k.startswith(dataset_prefix)) and (not namespace or path_covers(namespace, k))
    }
    names = _ossie_names(datasets)

    def owned(ref: str) -> bool:
        return not namespace or path_covers(namespace, _namespace_of(ref))

    model = state.models.get(namespace, state.model) if namespace else state.model
    doc: dict[str, Any] = {
        "version": OSSIE_VERSION,
        "name": name or model.attrs.get("name") or "gloss_model",
    }
    _put(doc, "description", model.attrs.get("description"))
    _put(doc, "ai_context", _ai_context(model.attrs))

    out_datasets = []
    for key in sorted(datasets):
        d = datasets[key]
        ds: dict[str, Any] = {"name": names[key], "source": d.attrs.get("source") or key}
        for attr in ("primary_key", "unique_keys", "description"):
            _put(ds, attr, d.attrs.get(attr))
        _put(ds, "ai_context", _ai_context(d.attrs))

        fields = []
        for fname, f in sorted(d.fields.items(), key=lambda kv: _first_seen(kv[1])):
            fo: dict[str, Any] = {"name": fname, "expression": f.attrs.get("expression") or {"sql": fname}}
            if "is_time" in f.attrs:
                fo["dimension"] = {"is_time": f.attrs["is_time"]}
            for attr in ("label", "description", "datatype"):
                _put(fo, attr, f.attrs.get(attr))
            _put(fo, "ai_context", _ai_context(f.attrs))
            _put(fo, "nonportable_extensions", _extension(f, annotate))
            fields.append(fo)
        _put(ds, "fields", fields)

        pk = ds.get("primary_key") or []
        missing = [c for c in pk if c not in d.fields]
        if missing:
            warnings.append(f"dataset {key}: primary key references unknown fields {missing}")

        metrics = [m for m in (_metric(mn, me, annotate, warnings, key) for mn, me in d.metrics.items()) if m]
        _put(ds, "metrics", metrics)
        _put(ds, "nonportable_extensions", _extension(d, annotate))
        out_datasets.append(ds)

    if not out_datasets:
        raise ValueError(
            "Nothing to export: no tables have accepted facts yet. "
            "Accept some, or export proposals too with include_pending=True (CLI: --include-pending)."
        )
    doc["datasets"] = out_datasets

    rels = []
    rel_names = _ref_names(r for r in state.relationships if owned(r))
    for rname, r in sorted(state.relationships.items()):
        if not owned(rname):
            continue
        a = r.attrs
        missing = [k for k in ("from", "to", "from_columns", "to_columns") if not a.get(k)]
        if missing:
            warnings.append(f"relationship {rname}: skipped, missing {', '.join(missing)}")
            continue
        if a["from"] not in names or a["to"] not in names:
            warnings.append(f"relationship {rname}: skipped, references a dataset outside the export")
            continue
        ro: dict[str, Any] = {
            "name": rel_names[rname],
            "from": names[a["from"]],
            "to": names[a["to"]],
            "from_columns": a["from_columns"],
            "to_columns": a["to_columns"],
        }
        _put(ro, "ai_context", _ai_context(a))
        _put(ro, "nonportable_extensions", _extension(r, annotate, _namespace_of(rname)))
        rels.append(ro)
    _put(doc, "relationships", rels)

    metric_names = _ref_names(m for m in state.metrics if owned(m))
    metrics = [
        m
        for m in (
            _metric(metric_names[ref], me, annotate, warnings, namespace=_namespace_of(ref))
            for ref, me in sorted(state.metrics.items())
            if owned(ref)
        )
        if m
    ]
    _put(doc, "metrics", metrics)

    model_ext: dict[str, Any] = {"exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if log_seq is not None:
        model_ext["log_seq"] = log_seq
    if namespace:
        model_ext["namespace"] = namespace
    for attr in ("owner",):
        _put(model_ext, attr, model.attrs.get(attr))
    if state.types:
        model_ext["types"] = {ref: dict(t.attrs) for ref, t in sorted(state.types.items())}
    doc["nonportable_extensions"] = [{"vendor_name": GLOSS_VENDOR, "data": model_ext}]
    return ExportResult(doc, warnings)


def _metric(
    mname: str, m: Entry, annotate: bool, warnings: list[str], dataset: str | None = None,
    namespace: str | None = None,
) -> dict[str, Any] | None:
    where = f"{dataset}/{mname}" if dataset else mname
    if not m.attrs.get("expression"):
        warnings.append(f"metric {where}: skipped, no expression")
        return None
    mo: dict[str, Any] = {"name": mname, "expression": m.attrs["expression"]}
    for attr in ("description", "datatype"):
        _put(mo, attr, m.attrs.get(attr))
    _put(mo, "ai_context", _ai_context(m.attrs))
    _put(mo, "nonportable_extensions", _extension(m, annotate, namespace))
    return mo


def to_ossie(
    gloss: Gloss,
    *,
    name: str | None = None,
    include_pending: bool = False,
    annotate: bool = True,
    dataset_prefix: str | None = None,
    namespace: str | None = None,
) -> ExportResult:
    """Build an Ossie document from accepted facts (optionally also open proposals)."""
    statuses = [Status.ACCEPTED]
    if include_pending:
        statuses += [Status.PROPOSED, Status.REVIEWED]
    return build_document(
        gloss.snapshot(statuses),
        name=name,
        annotate=annotate,
        dataset_prefix=dataset_prefix,
        namespace=namespace,
        log_seq=gloss.store.last_seq(),
    )


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    return json.loads(resources.files("gloss.ossie").joinpath("ossie-schema.json").read_text())


def validate(document: dict[str, Any]) -> list[str]:
    """Validate against the vendored Ossie JSON Schema. Returns error messages (empty = valid)."""
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Validation needs the 'validate' extra: pip install gloss-semantics[validate]") from exc
    validator = jsonschema.Draft202012Validator(_schema())
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '$'}: {err.message}"
        for err in sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
    ]


class OssieSink:
    """Sink plugin: write the accepted model to an Ossie YAML file."""

    name = "ossie"

    def __init__(self, path: str | Path | None = None, **defaults: Any):
        self.path = Path(path) if path else None
        self.defaults = defaults

    @classmethod
    def from_spec(cls, arg: str, **kwargs: Any) -> OssieSink:
        return cls(arg, **kwargs)

    def publish(self, gloss: Gloss, **options: Any) -> ExportResult:
        opts = {**self.defaults, **options}
        path = opts.pop("path", None) or self.path
        result = to_ossie(gloss, **opts)
        if path:
            Path(path).write_text(result.to_yaml())
        return result
