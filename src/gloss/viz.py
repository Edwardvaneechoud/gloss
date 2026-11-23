"""Visualisations: the model as a Mermaid ER diagram, and review progress as text or HTML."""

from __future__ import annotations

import html
import json
import re
from typing import TYPE_CHECKING

from .model import Status

if TYPE_CHECKING:
    from .ledger import Gloss

_SAFE = re.compile(r"[^A-Za-z0-9_]")


def _ent(key: str) -> str:
    return _SAFE.sub("_", key)


def mermaid_er(gloss: Gloss, *, include_pending: bool = True, dataset_prefix: str | None = None) -> str:
    """Datasets, fields and relationships as a Mermaid ``erDiagram``.

    Fields whose facts are not all accepted are marked with a trailing ``"pending"`` comment.
    """
    statuses = [Status.ACCEPTED] + ([Status.PROPOSED, Status.REVIEWED, Status.STALE] if include_pending else [])
    state = gloss.snapshot(statuses)
    lines = ["erDiagram"]
    keys = [k for k in sorted(state.datasets) if not dataset_prefix or k.startswith(dataset_prefix)]
    for key in keys:
        d = state.datasets[key]
        pk = set(d.attrs.get("primary_key") or [])
        lines.append(f"    {_ent(key)} {{")
        for fname, f in d.fields.items():
            dtype = f.attrs.get("datatype", "Opaque")
            marks = ["PK"] if fname in pk else []
            note = []
            if f.attrs.get("semantic_type"):
                note.append(f.attrs["semantic_type"])
            if any(a.status is Status.STALE for a in f.assertions.values()):
                note.append("stale")
            elif not f.all_accepted:
                note.append("pending")
            conf = f.attrs.get("confidentiality")
            if conf in ("confidential", "restricted"):
                note.append(conf)
            line = f"        {dtype} {_SAFE.sub('_', fname)}"
            if marks:
                line += " " + ",".join(marks)
            if note:
                line += f' "{", ".join(note)}"'
            lines.append(line)
        lines.append("    }")
    for rname, r in sorted(state.relationships.items()):
        a = r.attrs
        if a.get("from") in keys and a.get("to") in keys:
            label = rname + ("" if r.all_accepted else " (pending)")
            # from (many) -> to (one): the usual foreign-key shape
            lines.append(f'    {_ent(a["from"])} }}o--|| {_ent(a["to"])} : "{label}"')
    return "\n".join(lines)


def progress_table(gloss: Gloss, dataset_prefix: str | None = None) -> str:
    prog = gloss.progress(dataset_prefix)
    cols = ["accepted", "reviewed", "proposed", "stale", "rejected"]
    width = max([len(k) for k in prog] + [7])
    out = [f"{'dataset':<{width}}  " + "  ".join(f"{c:>8}" for c in cols) + "  progress"]
    for ds, counts in prog.items():
        live = sum(counts[c] for c in ("accepted", "reviewed", "proposed", "stale"))
        pct = (counts["accepted"] / live * 100) if live else 0.0
        bar = "#" * round(pct / 10) + "." * (10 - round(pct / 10))
        out.append(f"{ds:<{width}}  " + "  ".join(f"{counts[c]:>8}" for c in cols) + f"  {bar} {pct:5.1f}%")
    return "\n".join(out)


def html_report(gloss: Gloss, *, title: str = "Gloss model") -> str:
    """A self-contained HTML page: ER diagram plus review progress per dataset."""
    diagram = mermaid_er(gloss)
    prog = gloss.progress()
    rows = []
    for ds, c in prog.items():
        live = sum(c[s] for s in ("accepted", "reviewed", "proposed", "stale"))
        pct = round(c["accepted"] / live * 100) if live else 0
        rows.append(
            f"<tr><td>{html.escape(ds)}</td><td>{c['accepted']}</td><td>{c['reviewed']}</td>"
            f"<td>{c['proposed']}</td><td>{c['stale']}</td><td>{c['rejected']}</td>"
            f'<td><div class="bar"><span style="width:{pct}%"></span></div> {pct}%</td></tr>'
        )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;color:#1d2125}}
 table{{border-collapse:collapse;margin:1rem 0}} td,th{{padding:.3rem .8rem;border-bottom:1px solid #e4e6e8;text-align:right}}
 td:first-child,th:first-child{{text-align:left}} .bar{{display:inline-block;width:120px;height:8px;background:#eceef0;border-radius:4px;vertical-align:middle}}
 .bar span{{display:block;height:100%;background:#2f7d5b;border-radius:4px}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<h2>Review progress</h2>
<table><tr><th>dataset</th><th>accepted</th><th>reviewed</th><th>proposed</th><th>stale</th><th>rejected</th><th>progress</th></tr>
{''.join(rows)}</table>
<h2>Model</h2>
<pre class="mermaid">{html.escape(diagram)}</pre>
<script type="module">
 import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
 mermaid.initialize({{startOnLoad:true}});
</script>
<script type="application/json" id="gloss-progress">{html.escape(json.dumps(prog))}</script>
</body></html>"""
