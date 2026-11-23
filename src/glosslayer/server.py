"""REST API. Run standalone (``gloss serve``) or mount the router inside a host app.

Every route, reads included, runs as an actor: the host says who is asking, the policy
decides what they see and may do. Mounted in a host with its own logins:

    from glosslayer.server import create_router

    def ledger(user=Depends(get_current_user)) -> glosslayer.Gloss:      # one bound view per request
        return g.as_actor(Human(user.id), context={"sign_in": user.sign_in})

    app.include_router(create_router(ledger), prefix="/semantic")

Or give the ledger and a dependency that returns the actor string:

    app.include_router(create_router(g, actor=current_actor), prefix="/semantic")

Standalone, the actor comes from the ``X-Gloss-Actor`` header. That is fine
behind an authenticating proxy and not for direct exposure.

Responses are typed, so the OpenAPI schema is a contract a client can be generated from
(:mod:`glosslayer.client` is one).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from . import policy as P
from .ledger import Gloss, Proposal
from .model import InvalidTransition, Status
from .policy import PermissionDenied
from .store import ConcurrentWrite, Query

try:
    from fastapi import APIRouter, Body, Depends, FastAPI, Header, HTTPException
    from fastapi import Query as Param
    from fastapi.responses import HTMLResponse, PlainTextResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("The server needs the 'server' extra: pip install glosslayer[server]") from exc

ACTIONS = ("review", "accept", "reject", "retract", "reopen", "restore")


class AssertionOut(BaseModel):
    """One fact about a subject with its review status; ``redacted`` means the value is withheld from you."""

    id: str
    subject: str
    attribute: str
    value: Any = None
    source: str
    sources: list[str] = []
    status: Status
    confidence: float | None = None
    evidence: list[dict[str, Any]] = []
    created_at: str
    updated_at: str
    created_by: str
    last_comment: str | None = None
    group: str | None = None
    redacted: bool = False


class AssertionPage(BaseModel):
    """A page of assertions; ``total`` counts every match, for paging with ``limit`` and ``offset``."""

    items: list[AssertionOut]
    total: int
    limit: int
    offset: int


class EventOut(BaseModel):
    """One entry of the append-only event log; ``seq`` is its position in commit order."""

    seq: int | None = None
    id: str
    ts: str
    type: str
    assertion_id: str
    actor: str
    via: str | None = None
    correlation: str | None = None
    data: dict[str, Any] = {}


class ProposeIn(BaseModel):
    """One fact to propose; ``accept`` asks to accept it at once, which the policy may refuse."""

    subject: str
    attribute: str
    value: Any
    source: str | None = None
    confidence: float | None = None
    evidence: list[dict[str, Any]] = []
    comment: str | None = None
    group: str | None = None
    accept: bool = False


class ProposalIn(BaseModel):
    """One fact in a ``POST /proposals`` batch."""

    subject: str
    attribute: str
    value: Any
    confidence: float | None = None
    evidence: list[dict[str, Any]] = []
    comment: str | None = None
    group: str | None = None


class ProposeManyIn(BaseModel):
    """A batch of facts to propose, with one ``source``, ``accept`` flag and ``comment`` for all."""

    proposals: list[ProposalIn]
    source: str | None = None
    accept: bool = False
    comment: str | None = None


class ProposeOut(BaseModel):
    """What became of a proposal: created, corroborated, unchanged or previously_rejected."""

    outcome: str
    accepted: bool = False
    assertion: AssertionOut


class ActionIn(BaseModel):
    """An optional comment recorded with a review action."""

    comment: str | None = None


class BulkIn(BaseModel):
    """Act on ``ids``, or on everything matching the filters (at least one filter is required)."""

    ids: list[str] | None = None
    subject: str | None = None
    subject_prefix: str | None = None
    dataset: str | None = None
    namespace: str | None = None
    kind: str | None = None
    attribute: str | None = None
    source_prefix: str | None = None
    group: str | None = None
    status: list[Status] | None = None
    comment: str | None = None


class BulkError(BaseModel):
    """An assertion a bulk action could not change, and why."""

    id: str
    error: str


class BulkOut(BaseModel):
    """The assertions a bulk action changed, and the ones it could not."""

    changed: list[AssertionOut]
    errors: list[BulkError]


class SubjectOut(BaseModel):
    """One subject: its accepted value per attribute, its version and the assertions about it you may read."""

    subject: str
    version: int | None = None
    current: dict[str, Any]
    assertions: list[AssertionOut]


class RekeyIn(BaseModel):
    """Rename or move one subject (``old``/``new`` are subjects), or a whole path with ``namespace=true``."""

    old: str
    new: str
    namespace: bool = False
    comment: str | None = None


class RekeyOut(BaseModel):
    """What a rename changed: subjects re-keyed (old to new), facts moved, rewritten and sent back."""

    subjects: dict[str, str]
    moved: int
    rewritten: int
    superseded: int
    staled: int


class SearchHit(BaseModel):
    """One search result; ``matched`` lists the words it covers when no hit has them all."""

    subject: str
    kind: str
    name: str
    dataset: str | None = None
    datatype: str | None = None
    semantic_type: str | None = None
    description: str | None = None
    confidentiality: str | None = None
    redacted: bool = False
    matched: list[str] | None = None
    did_you_mean: str | None = None


class CheckOut(BaseModel):
    """A data quality check in effect on a field; ``origin`` is the semantic type it was inherited from."""

    check: dict[str, Any]
    origin: str | None = None


class ColumnOut(BaseModel):
    """A column with its agreed meaning: datatype, description, semantic type, confidentiality and checks."""

    name: str
    datatype: str | None = None
    physical_type: str | None = None
    description: str | None = None
    label: str | None = None
    is_time: bool | None = None
    role: str | None = None
    semantic_type: str | None = None
    confidentiality: str | None = None
    classification: list[str] = []
    synonyms: list[str] = []
    checks: list[dict[str, Any]] = []


class MetricOut(BaseModel):
    """A metric: its expression and, when structured, its aggregate, filter, time column and dimensions."""

    name: str
    table: str | None = None
    namespace: str | None = None
    expression: Any = None
    description: str | None = None
    datatype: str | None = None
    aggregate: str | None = None
    filter: str | None = None
    time_column: str | None = None
    dimensions: list[str] = []
    format: str | None = None
    version: int | None = None


class TableOut(BaseModel):
    """A table with its primary key, columns and the metrics defined on it."""

    key: str
    source: str | None = None
    description: str | None = None
    primary_key: list[str] = []
    confidentiality: str | None = None
    owner: str | None = None
    columns: list[ColumnOut] = []
    metrics: list[MetricOut] = []


class RelationshipOut(BaseModel):
    """A join: ``from_columns`` of ``from_table`` reference ``to_columns`` of ``to_table``."""

    name: str
    namespace: str | None = None
    from_table: str
    to_table: str
    from_columns: list[str]
    to_columns: list[str]
    description: str | None = None
    version: int | None = None


class TypeOut(BaseModel):
    """A semantic type with its checks and name patterns; ``used_by`` counts the fields that use it."""

    name: str
    namespace: str | None = None
    datatype: str | None = None
    physical_type: str | None = None
    description: str | None = None
    confidentiality: str | None = None
    classification: list[str] = []
    synonyms: list[str] = []
    checks: list[dict[str, Any]] = []
    name_patterns: list[str] = []
    used_by: int = 0


class CheckResultIO(BaseModel):
    """One check result for a field: how many rows failed, and the run that produced it."""

    subject: str
    check: dict[str, Any]
    passed: bool
    failed_rows: int
    total_rows: int
    origin: str | None = None
    error: str | None = None
    ts: str | None = None
    runner: str | None = None
    run_id: str | None = None


class CheckRunIn(BaseModel):
    """A run of check results to record."""

    results: list[CheckResultIO]
    run_id: str | None = Field(default=None, description="The host's id for the run that produced these results")


class CheckRunOut(BaseModel):
    """The run id the results were recorded under, and how many were recorded."""

    run_id: str
    recorded: int


class SyncOut(BaseModel):
    """What a catalog sync found and did to tables and facts; every event it wrote carries ``correlation``."""

    catalog: str
    new: list[str]
    changed: list[str]
    moved: dict[str, str]
    gone: list[str]
    unchanged: int
    proposed: int
    accepted: int
    staled: int
    notes: list[str]
    correlation: str | None = None


class HealthOut(BaseModel):
    """The server is up; ``schema_version`` is the ledger schema version this Gloss release uses."""

    ok: bool
    schema_version: int | None = None


def header_actor(x_gloss_actor: str | None = Header(default=None)) -> str:
    """The actor named in the ``X-Gloss-Actor`` header (401 without it); trust it only behind a proxy."""
    if not x_gloss_actor:
        raise HTTPException(401, "Set the X-Gloss-Actor header, e.g. 'human:alice'")
    return x_gloss_actor


def _column(c: Any) -> ColumnOut:
    return ColumnOut(
        name=c.name, datatype=c.datatype, physical_type=c.physical_type, description=c.description,
        label=c.label, is_time=c.is_time, role=c.role, semantic_type=c.semantic_type_name,
        confidentiality=c.confidentiality, classification=list(c.classification), synonyms=list(c.synonyms),
        checks=[x if isinstance(x, dict) else x.to_dict() for x in c.checks],
    )


def _metric(m: Any) -> MetricOut:
    return MetricOut(
        name=m.name, table=m.table, namespace=m.namespace, expression=m.expression, description=m.description,
        datatype=m.datatype, aggregate=m.aggregate, filter=m.filter, time_column=m.time_column,
        dimensions=list(m.dimensions), format=m.format, version=m.version,
    )


def _table(t: Any) -> TableOut:
    return TableOut(
        key=t.key, source=t.source, description=t.description, primary_key=list(t.primary_key),
        confidentiality=t.confidentiality, owner=t.owner,
        columns=[_column(c) for c in t.columns.values()], metrics=[_metric(m) for m in t.metrics.values()],
    )


def create_router(
    gloss: Gloss | Callable[..., Gloss],
    *,
    actor: Callable[..., str] = header_actor,
    catalogs: Mapping[str, Any] | None = None,
) -> APIRouter:
    """The REST routes.

    ``gloss`` is a ledger (bound per request to whoever ``actor`` returns), or a FastAPI
    dependency that returns an already bound view (``g.as_actor(...)``), which lets the host add
    ``via`` and context. ``catalogs`` names the catalogs ``POST /sync/{name}`` may run.
    """
    r = APIRouter()

    if isinstance(gloss, Gloss):
        root = gloss

        def ledger(who: str = Depends(actor)) -> Gloss:
            try:
                return root.as_actor(who)
            except ValueError as e:
                raise HTTPException(401, str(e)) from e
    else:
        ledger = gloss

    def guard(fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except PermissionDenied as e:
            raise HTTPException(403, str(e)) from e
        except KeyError as e:
            raise HTTPException(404, str(e.args[0]) if e.args else "not found") from e
        except ConcurrentWrite as e:
            raise HTTPException(409, str(e)) from e
        except (InvalidTransition, ValueError) as e:
            raise HTTPException(422, str(e)) from e

    def out(a: Any) -> AssertionOut:
        return AssertionOut(**a.to_dict())

    @r.get("/health", response_model=HealthOut)
    def health() -> Any:
        """Liveness check."""
        return {"ok": True, "schema_version": __import__("glosslayer").schema.LATEST}

    @r.get("/assertions", response_model=AssertionPage)
    def list_assertions(
        subject: str | None = None,
        subject_prefix: str | None = None,
        dataset: str | None = None,
        namespace: str | None = None,
        kind: str | None = None,
        attribute: str | None = None,
        status: list[Status] | None = Param(default=None),
        source_prefix: str | None = None,
        group: str | None = None,
        limit: int = 200,
        offset: int = 0,
        g: Gloss = Depends(ledger),
    ) -> Any:
        """Assertions matching the filters, a page at a time (``limit`` at most 5000)."""
        criteria = dict(
            subject=subject, subject_prefix=subject_prefix, dataset=dataset, namespace=namespace, kind=kind,
            attribute=attribute, statuses=status, source_prefix=source_prefix, group=group,
        )
        limit = max(1, min(limit, 5000))
        rows = guard(lambda: g.find(**criteria, limit=limit, offset=offset))
        return {"items": [out(a) for a in rows], "total": g.count(**criteria), "limit": limit, "offset": offset}

    @r.post("/assertions", status_code=201, response_model=ProposeOut)
    def propose(body: ProposeIn, g: Gloss = Depends(ledger)) -> Any:
        """Propose one fact as the caller."""
        def go() -> Any:
            res = g.propose_many(
                [Proposal(body.subject, body.attribute, body.value, body.confidence, body.evidence, body.comment,
                          body.group)],
                source=body.source,
                accept=body.accept,
            )[0]
            return {"outcome": res.outcome, "accepted": res.accepted, "assertion": out(res.assertion)}

        return guard(go)

    @r.post("/proposals", status_code=201, response_model=list[ProposeOut])
    def propose_many(body: ProposeManyIn, g: Gloss = Depends(ledger)) -> Any:
        """Propose many facts at once, de-duplicated against what is already known."""
        def go() -> Any:
            props = [Proposal(p.subject, p.attribute, p.value, p.confidence, p.evidence, p.comment, p.group)
                     for p in body.proposals]
            results = g.propose_many(props, source=body.source, accept=body.accept, comment=body.comment)
            return [{"outcome": x.outcome, "accepted": x.accepted, "assertion": out(x.assertion)} for x in results]

        return guard(go)

    @r.get("/assertions/{assertion_id}", response_model=AssertionOut)
    def get_assertion(assertion_id: str, g: Gloss = Depends(ledger)) -> Any:
        """One assertion by id or unambiguous id prefix."""
        return guard(lambda: out(g.get(assertion_id)))

    @r.post("/assertions/{assertion_id}/{action}", response_model=AssertionOut)
    def act(
        assertion_id: str,
        action: str,
        body: ActionIn = Body(default_factory=ActionIn),
        g: Gloss = Depends(ledger),
    ) -> Any:
        """Apply a review action (review, accept, reject, retract, reopen, restore) to one assertion."""
        if action not in ACTIONS:
            raise HTTPException(404, f"Unknown action {action!r}")
        return guard(lambda: out(getattr(g, action)(assertion_id, comment=body.comment)))

    @r.post("/bulk/{action}", response_model=BulkOut)
    def bulk(action: str, body: BulkIn, g: Gloss = Depends(ledger)) -> Any:
        """Apply a review action to the given ids, or to every matching assertion that can take it."""
        if action not in ACTIONS:
            raise HTTPException(404, f"Unknown action {action!r}")

        def go() -> Any:
            if body.ids is not None:
                done, errors = g.transition_many(action, body.ids, comment=body.comment)
                errs = [{"id": i, "error": str(e.args[0] if e.args else e)} for i, e in errors]
            else:
                q = Query(
                    subject=body.subject, subject_prefix=body.subject_prefix, dataset=body.dataset,
                    namespace=body.namespace, kind=body.kind, attribute=body.attribute,
                    source_prefix=body.source_prefix, group=body.group, statuses=body.status,
                )
                if not any([q.subject, q.subject_prefix, q.dataset, q.namespace, q.kind, q.attribute,
                            q.source_prefix, q.group]):
                    raise ValueError("Give ids or at least one filter")
                done, pairs = g.bulk(action, q, comment=body.comment)
                errs = [{"id": i, "error": e} for i, e in pairs]
            return {"changed": [out(a) for a in done], "errors": errs}

        return guard(go)

    @r.post("/groups/{group}/{action}", response_model=list[AssertionOut])
    def decide_group(group: str, action: str, body: ActionIn = Body(default_factory=ActionIn),
                     g: Gloss = Depends(ledger)) -> Any:
        """Accept or reject every open fact of a proposal group, all or nothing."""
        if action not in ("accept", "reject"):
            raise HTTPException(404, f"Unknown action {action!r}")
        fn = g.accept_group if action == "accept" else g.reject_group
        return guard(lambda: [out(a) for a in fn(group, comment=body.comment)])

    @r.get("/subjects/{subject:path}/history", response_model=list[EventOut])
    def subject_history(subject: str, g: Gloss = Depends(ledger)) -> Any:
        """Every event of the facts now at a subject, including those from before it was re-keyed."""
        return guard(lambda: [e.to_dict() for e in g.history(subject)])

    @r.get("/subjects/{subject:path}", response_model=SubjectOut)
    def subject_view(subject: str, g: Gloss = Depends(ledger)) -> Any:
        """A subject's accepted values, version and assertions."""
        return guard(
            lambda: {
                "subject": subject,
                "version": g.version(subject),
                "current": g.current(subject),
                "assertions": [out(a) for a in g.find(subject=subject)],
            }
        )

    @r.post("/rekey", response_model=RekeyOut)
    def rekey(body: RekeyIn, g: Gloss = Depends(ledger)) -> Any:
        """Record that a subject, or a whole path with ``namespace``, was renamed or moved."""
        fn = g.rekey_namespace if body.namespace else g.rekey
        return guard(lambda: fn(body.old, body.new, comment=body.comment).__dict__)

    @r.get("/log", response_model=list[EventOut])
    def log(after_seq: int = 0, limit: int = 500, correlation: str | None = None,
            g: Gloss = Depends(ledger)) -> Any:
        """The event log after ``after_seq``, in commit order (at most 5000 events per call)."""
        return [e.to_dict() for e in g.log(after_seq=after_seq, limit=min(limit, 5000), correlation=correlation)]

    @r.get("/progress", response_model=dict[str, dict[str, int]])
    def progress(dataset_prefix: str | None = None, g: Gloss = Depends(ledger)) -> Any:
        """Review progress per dataset or namespace: ``{path: {status: count}}``."""
        return g.progress(dataset_prefix)

    @r.get("/search", response_model=list[SearchHit])
    def search(q: str, kind: list[str] | None = Param(default=None), dataset_prefix: str | None = None,
               limit: int = 20, g: Gloss = Depends(ledger)) -> Any:
        """Search as the caller: what they may read, redacted above their level."""
        return g.search(q, kinds=kind, dataset_prefix=dataset_prefix, limit=min(limit, 200))

    @r.get("/tables", response_model=list[TableOut])
    def tables(prefix: str | None = None, include_pending: bool = False, g: Gloss = Depends(ledger)) -> Any:
        """All tables, or those whose key starts with ``prefix``, sorted by key."""
        return [_table(t) for t in g.tables(prefix, include_pending=include_pending)]

    @r.get("/tables/{key:path}", response_model=TableOut)
    def table(key: str, include_pending: bool = False, g: Gloss = Depends(ledger)) -> Any:
        """One table by key."""
        return guard(lambda: _table(g.table(key, include_pending=include_pending)))

    @r.get("/relationships", response_model=list[RelationshipOut])
    def relationships(table: str | None = None, include_pending: bool = False, g: Gloss = Depends(ledger)) -> Any:
        """Complete relationships, optionally only those touching ``table``."""
        return [
            RelationshipOut(
                name=x.name, namespace=x.namespace, from_table=x.from_table, to_table=x.to_table,
                from_columns=list(x.from_columns), to_columns=list(x.to_columns), description=x.description,
                version=x.version,
            )
            for x in g.relationships(table, include_pending=include_pending)
        ]

    @r.get("/metrics", response_model=list[MetricOut])
    def metrics(table: str | None = None, include_pending: bool = False, g: Gloss = Depends(ledger)) -> Any:
        """Ledger-wide, namespace and table metrics; with ``table``, only that table's."""
        return [_metric(m) for m in g.metrics(table, include_pending=include_pending)]

    @r.get("/types", response_model=list[TypeOut])
    def types(include_pending: bool = False, g: Gloss = Depends(ledger)) -> Any:
        """Semantic types, each with the number of fields that use it."""
        return [
            TypeOut(
                name=t.name, namespace=t.namespace, datatype=t.datatype, physical_type=t.physical_type,
                description=t.description, confidentiality=t.confidentiality,
                classification=list(t.classification), synonyms=list(t.synonyms),
                checks=[x if isinstance(x, dict) else x.to_dict() for x in t.checks],
                name_patterns=list(t.name_patterns), used_by=len(g.type_usages(t.ref)),
            )
            for t in g.semantic_types(include_pending=include_pending)
        ]

    @r.get("/checks/{dataset:path}/effective", response_model=dict[str, list[CheckOut]])
    def effective_checks(dataset: str, g: Gloss = Depends(ledger)) -> Any:
        """The checks in effect per field of a dataset, including those inherited from semantic types."""
        from .checks import effective_checks as effective

        return guard(lambda: {
            f: [{"check": c, "origin": o} for c, o in items] for f, items in effective(g, dataset).items()
        })

    @r.get("/checks/results", response_model=list[CheckResultIO])
    def check_results(subject: str | None = None, dataset: str | None = None, run_id: str | None = None,
                      limit: int = 200, g: Gloss = Depends(ledger)) -> Any:
        """Logged check results, newest first, for one field, a dataset or one run."""
        rows = guard(lambda: g.check_history(subject, dataset=dataset, run_id=run_id, limit=min(limit, 5000)))
        return [{**row, "ts": row["ts"].isoformat()} for row in rows]

    @r.post("/checks/results", status_code=201, response_model=CheckRunOut)
    def record_checks(body: CheckRunIn, g: Gloss = Depends(ledger)) -> Any:
        """Record a run of check results, with the caller as runner; only for subjects the caller may read."""
        from .checks import CheckResult

        def go() -> Any:
            results = [CheckResult.from_dict(x.model_dump()) for x in body.results]
            for res in results:
                if not g.scope.covers_subject(res.subject):
                    raise PermissionDenied(f"{g.actor} may not record results for {res.subject}")
            run = g.record_check_results(results, runner=g.actor or "system:checks", run_id=body.run_id)
            return {"run_id": run, "recorded": len(results)}

        return guard(go)

    @r.post("/sync/{catalog}", response_model=SyncOut)
    def run_sync(catalog: str, g: Gloss = Depends(ledger)) -> Any:
        """Run a catalog sync as the catalog's own principal. Needs the right to re-key ledger-wide."""
        from .sync import sync

        if catalog not in (catalogs or {}):
            raise HTTPException(404, f"No catalog {catalog!r} is configured for sync")
        if not g.policy.can(g.actor or "", P.REKEY, None, g.context):
            raise HTTPException(403, f"{g.actor} may not run a catalog sync")
        root = g.as_actor(f"catalog:{catalogs[catalog].name}")  # type: ignore[index]
        return guard(lambda: sync(root, catalogs[catalog]).to_dict())  # type: ignore[index]

    @r.get("/export/ossie", response_class=PlainTextResponse)
    def export_ossie(include_pending: bool = False, name: str | None = None, namespace: str | None = None,
                     g: Gloss = Depends(ledger)) -> str:
        """The model as Ossie YAML."""
        return guard(lambda: g.to_yaml(include_pending=include_pending, name=name, namespace=namespace))

    @r.get("/viz/mermaid", response_class=PlainTextResponse)
    def viz_mermaid(include_pending: bool = True, g: Gloss = Depends(ledger)) -> str:
        """The model as a Mermaid ER diagram."""
        from .viz import mermaid_er

        return mermaid_er(g, include_pending=include_pending)

    @r.get("/viz/report", response_class=HTMLResponse)
    def viz_report(g: Gloss = Depends(ledger)) -> str:
        """An HTML page with the ER diagram and review progress per dataset."""
        from .viz import html_report

        return html_report(g)

    @r.get("/ai/context", response_model=dict[str, Any])
    def ai_context(dataset: list[str] | None = Param(default=None), g: Gloss = Depends(ledger)) -> Any:
        """The accepted model as compact JSON for an AI prompt, within the AI visibility limit."""
        from . import ai

        return ai.model_context(g, datasets=dataset)

    @r.get("/ai/search", response_model=list[SearchHit])
    def ai_search(q: str, limit: int = 20, g: Gloss = Depends(ledger)) -> Any:
        """Search as an AI tool sees it: the AI visibility limit applies."""
        from . import ai

        return ai.search(g, q, limit=limit)

    return r


def create_app(gloss: Gloss | Callable[..., Gloss], *, actor: Callable[..., str] = header_actor,
               catalogs: Mapping[str, Any] | None = None) -> FastAPI:
    """A standalone FastAPI app serving :func:`create_router` at the root."""
    app = FastAPI(title="Gloss", version=__import__("glosslayer").__version__)
    app.include_router(create_router(gloss, actor=actor, catalogs=catalogs))
    return app
