"""AI-facing views of the model, with confidentiality enforced.

Everything AI tools read goes through here. Subjects whose confidentiality is
above the reader's level are shown by name only (so the AI knows they exist and
can say so) with descriptions, expressions and examples removed. Confidentiality
set on a dataset applies to all its fields unless a field states its own level.

Pass an actor-bound view (``g.as_actor(AI("claude"))``, or a person's view with
``via="ai:claude"``) and the policy's scope for that actor applies: which paths it
may read and up to which level. Pass an unbound ledger and the policy's default AI
level applies to everything in it. Each call reports what it handed out to the
ledger's read hook (``Gloss.on_read``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .model import Confidentiality, Status, attribute, detail_attributes

if TYPE_CHECKING:
    from .ledger import Entry, Gloss


def ai_limit(gloss: Gloss) -> Confidentiality:
    """The confidentiality ceiling for what this handle may show an AI."""
    if gloss.actor is not None:
        return gloss.scope.max_confidentiality
    return gloss.policy.scope("ai:anonymous", None).max_confidentiality


def _view(name: str, entry: Entry, extra: tuple[str, ...] = ()) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name}
    if "datatype" in entry.attrs:
        out["datatype"] = entry.attrs["datatype"]
    if entry.redacted:
        out["confidentiality"] = entry.redacted
        out["redacted"] = True
        return out
    shown = detail_attributes() | set(extra)
    for attr, value in entry.attrs.items():
        if attr in shown and attr != "datatype":
            out[attr] = value
    level = entry.attrs.get("confidentiality")
    if level and level != Confidentiality.INTERNAL.value:
        out["confidentiality"] = level
    return out


def model_context(
    gloss: Gloss,
    *,
    datasets: list[str] | None = None,
    include_pending: bool = False,
) -> dict[str, Any]:
    """Accepted model as compact JSON for an LLM prompt or an MCP tool result."""
    limit = ai_limit(gloss)
    statuses = [Status.ACCEPTED] + ([Status.PROPOSED, Status.REVIEWED] if include_pending else [])
    state = gloss.snapshot(statuses, datasets=datasets)
    state.redact(limit)
    wanted = set(datasets) if datasets else None
    metric_extra = tuple(n for n in ("time_column", "dimensions", "format") if attribute(n))

    out: dict[str, Any] = {"visibility_limit": limit.value, "datasets": [], "relationships": [], "metrics": []}
    if state.model.attrs.get("description"):
        out["model"] = {"description": state.model.attrs["description"]}
    shown: list[str] = []
    for key, d in sorted(state.datasets.items()):
        if wanted is not None and key not in wanted:
            continue
        dv = _view(key, d, extra=("primary_key",))
        shown.append(f"dataset:{key}")
        if not d.redacted:
            dv["fields"] = [_view(fn, f) for fn, f in d.fields.items()]
            dv["metrics"] = [_view(mn, m, metric_extra) for mn, m in d.metrics.items()]
            shown += [f"field:{key}/{fn}" for fn, f in d.fields.items() if not f.redacted]
        out["datasets"].append(dv)
    for ref, r in sorted(state.relationships.items()):
        a = r.attrs
        if wanted is not None and not ({a.get("from"), a.get("to")} & wanted):
            continue
        if r.redacted:
            continue
        out["relationships"].append({k: a[k] for k in ("from", "to", "from_columns", "to_columns") if k in a}
                                    | {"name": ref})
        shown.append(f"relationship:{ref}")
    if wanted is None:
        for ref, m in sorted(state.metrics.items()):
            out["metrics"].append(_view(ref, m, metric_extra))
            shown.append(f"metric:{ref}")
    gloss.report_read("model_context", shown)
    return out


def subject_level(gloss: Gloss, subject: str) -> Confidentiality:
    """Effective confidentiality of a subject (own accepted level, else its type's or dataset's, else internal)."""
    from .ids import SubjectId
    from .visibility import levels

    s = str(SubjectId.parse(subject))
    return levels(gloss.store, [s])[s]


def review_status(gloss: Gloss, subject: str) -> list[dict[str, Any]]:
    """Facts about one subject and their status, with values hidden above the reader's level."""
    from .ids import SubjectId

    sid = SubjectId.parse(subject)
    visible = subject_level(gloss, str(sid)) <= ai_limit(gloss)
    detail = detail_attributes()
    out = []
    for a in gloss.find(subject=str(sid)):
        d: dict[str, Any] = {"attribute": a.attribute, "status": a.status.value, "source": a.source}
        if not a.redacted and (visible or a.attribute not in detail):
            d["value"] = a.value
        out.append(d)
    if out:
        gloss.report_read("review_status", [str(sid)])
    return out


def search(gloss: Gloss, text: str, *, limit: int = 20, kinds: list[str] | None = None) -> list[dict[str, Any]]:
    """Indexed search over accepted knowledge, with the AI visibility limit applied (see Gloss.search)."""
    hits = gloss.search(text, limit=limit, kinds=kinds, for_ai=True)
    gloss.report_read("search", [h["subject"] for h in hits if not h.get("redacted")])
    return hits
