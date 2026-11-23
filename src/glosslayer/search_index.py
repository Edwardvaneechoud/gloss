"""The search index: one compact row per subject, kept current as facts are accepted.

Only *accepted* knowledge is indexed. Each row has

* ``name_text``: names, keys, kind and semantic type; always searchable, so confidential
  things can still be *found* (and reported as redacted);
* ``detail_text``: description, label, synonyms, examples, AI instructions,
  classification; searchable only by callers allowed to see that confidentiality level;
* ``level``: the effective confidentiality (own → semantic type → table → internal).

The index is a projection like ``gloss_assertions``: ``Gloss.rebuild_search_index()``
recreates it from scratch, and the ledger updates the affected rows whenever a fact is
accepted, superseded, marked stale or retracted, including every field that inherits
from a changed semantic type or table.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .ids import Kind, SubjectId
from .model import Status, attribute, inherited_attributes, value_hash
from .store import Query
from .visibility import effective_rank

if TYPE_CHECKING:
    from .ledger import Gloss

# Bump when what goes into index rows changes; ledgers with an older index rebuild it on first search.
INDEX_VERSION = "5"

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)  # words; underscores and punctuation split


def tokens(text: str) -> list[str]:
    """The lower-cased words of ``text`` (underscores and punctuation split them), at most 16."""
    return [t.lower() for t in _TOKEN.findall(text)][:16]


# Words that carry no meaning in a question about data ("revenue per country by month").
STOPWORDS = frozenset(
    "a an the of per by for in on at to and or with from into is are was were be been what which who how "
    "many much show give get list find me my our all each every this that these those there do does".split()
)


def content_tokens(text: str) -> list[str]:
    """The words of a question without stopwords, or all of them when nothing else is left."""
    words = tokens(text)
    kept = [w for w in words if w not in STOPWORDS]
    return kept or words


def correct_spelling(words: list[str], vocabulary: list[str], cutoff: float = 0.7) -> list[str]:
    """Replace words that match nothing in the index with their closest indexed word."""
    import difflib

    if not vocabulary:
        return words
    vocab = sorted(set(vocabulary))
    out = []
    for w in words:
        if any(v.startswith(w) for v in vocab):
            out.append(w)
            continue
        close = difflib.get_close_matches(w, vocab, n=1, cutoff=cutoff)
        out.append(close[0] if close else w)
    return out


def _words(*values: Any) -> str:
    out: list[str] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            out.extend(str(x) for x in v)
        else:
            out.append(str(v))
    text = " ".join(out)
    # also split camelCase so "orderDate" is found by "date"
    return text + " " + re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text) if re.search(r"[a-z][A-Z]", text) else text


def _searchable(eff: dict[str, Any], *, detail: bool) -> list[Any]:
    """Values of the registered attributes whose words are indexed, names or details."""
    out = []
    for name, value in eff.items():
        spec = attribute(name)
        if spec is not None and spec.searchable and spec.detail is detail:
            out.append(value)
    return out


def expand(gloss: Gloss, subjects: Iterable[str]) -> set[str]:
    """Subjects whose index rows may change when these subjects change."""
    out = set(subjects)
    accepted = [Status.ACCEPTED]
    for s in list(out):
        sid = SubjectId.parse(s)
        if sid.kind is Kind.TYPE:  # every field using the type
            for a in gloss.store.find(
                Query(attribute="semantic_type", value_hash=value_hash(sid.ref), statuses=accepted)
            ):
                out.add(a.subject)
        elif sid.kind is Kind.DATASET:  # fields, scoped metrics and relationships inherit its level
            for a in gloss.store.find(Query(dataset=sid.dataset, statuses=accepted)):
                out.add(a.subject)
            for attr in ("from", "to"):
                for a in gloss.store.find(
                    Query(attribute=attr, value_hash=value_hash(sid.dataset), statuses=accepted)
                ):
                    out.add(a.subject)
    return out


def build_rows(gloss: Gloss, subjects: Iterable[str]) -> list[dict[str, Any]]:
    """Index rows for ``subjects`` from their accepted facts; a subject without any gets no row.

    Fields take inherited attributes from their semantic type, and levels are inherited as in
    :mod:`glosslayer.visibility`.
    """
    subjects = set(subjects)
    attrs: dict[str, dict[str, Any]] = defaultdict(dict)
    for a in gloss.store.find_for_subjects(subjects, [Status.ACCEPTED]):
        attrs[a.subject][a.attribute] = a.value

    # what the rows inherit from: semantic types and tables
    type_names = {v["semantic_type"] for v in attrs.values() if v.get("semantic_type")}
    datasets = {SubjectId.parse(s).dataset for s in attrs if SubjectId.parse(s).dataset}
    for v in attrs.values():
        datasets.update(x for x in (v.get("from"), v.get("to")) if isinstance(x, str))
    lookup = {str(SubjectId.for_type(t)) for t in type_names} | {str(SubjectId.for_dataset(d)) for d in datasets}
    inherited: dict[str, dict[str, Any]] = defaultdict(dict)
    for a in gloss.store.find_for_subjects(lookup - subjects, [Status.ACCEPTED]):
        inherited[a.subject][a.attribute] = a.value
    for s in lookup & subjects:
        inherited[s] = attrs.get(s, {})

    inherit = inherited_attributes()

    def conf(subject: SubjectId) -> Any:
        return inherited.get(str(subject), {}).get("confidentiality")

    rows = []
    for subj, own in attrs.items():
        sid = SubjectId.parse(subj)
        eff = dict(own)
        tname = own.get("semantic_type")
        if sid.kind is Kind.FIELD and tname:
            t = inherited.get(str(SubjectId.for_type(tname)), {})
            for attr in inherit:
                if attr not in eff and attr in t:
                    eff[attr] = t[attr]
        level = effective_rank(
            sid,
            eff.get("confidentiality"),
            None,  # the type's level is already in `eff`
            conf(SubjectId.for_dataset(sid.dataset)) if sid.dataset else None,
            [conf(SubjectId.for_dataset(x)) for x in (eff.get("from"), eff.get("to")) if isinstance(x, str)],
        )
        name = sid.name or sid.dataset or sid.namespace or "model"
        # A semantic type's own words (its synonyms) are definition, not data: always searchable,
        # on the type itself and on every column using it ("mail" finds email columns via "e-mail").
        type_words = None
        if sid.kind is Kind.TYPE:
            type_words = eff.get("synonyms")
        elif tname:
            type_words = inherited.get(str(SubjectId.for_type(tname)), {}).get("synonyms")
        rows.append(
            {
                "subject": subj,
                "kind": sid.kind.value,
                "dataset": sid.dataset,
                "namespace": sid.scope_path,
                "name": name,
                "name_text": _words(sid.kind.value, sid.dataset, sid.namespace, sid.name, tname, type_words,
                                    *_searchable(eff, detail=False)),
                "detail_text": _words(*_searchable(eff, detail=True)),
                "description": eff.get("description"),
                "datatype": eff.get("datatype"),
                "semantic_type": tname,
                "level": level,
            }
        )
    return rows


def reindex(gloss: Gloss, subjects: Iterable[str], *, cascade: Iterable[str] | None = None,
            chunk: int = 2000) -> int:
    """Rebuild the rows of ``subjects``, and of everything that inherits from ``cascade`` (default: all)."""
    subjects = set(subjects)
    targets = sorted(subjects | expand(gloss, subjects if cascade is None else cascade))
    n = 0
    for i in range(0, len(targets), chunk):
        part = targets[i : i + chunk]
        rows = build_rows(gloss, part)
        gloss.store.replace_search_rows(part, rows)
        n += len(rows)
    return n


def rebuild(gloss: Gloss, *, chunk: int = 2000) -> int:
    """Recreate the whole index from accepted facts, stamped :data:`INDEX_VERSION`. Returns the row count."""
    gloss.store.set_meta("search_index_version", INDEX_VERSION)
    gloss.store.clear_search()
    gloss.store.recreate_search_tables()
    subjects = sorted(gloss.store.accepted_subjects())
    n = 0
    for i in range(0, len(subjects), chunk):
        part = subjects[i : i + chunk]
        rows = build_rows(gloss, part)
        gloss.store.replace_search_rows(part, rows)
        n += len(rows)
    return n
