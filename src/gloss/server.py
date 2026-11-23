"""REST API. Run standalone (``gloss serve``) or mount the router inside a host app.

Every route, reads included, runs as an actor: the host says who is asking, the policy
decides what they see and may do. Mounted in a host with its own logins:

    from gloss.server import create_router

    def ledger(user=Depends(get_current_user)) -> gloss.Gloss:      # one bound view per request
        return g.as_actor(Human(user.id), context={"sign_in": user.sign_in})

    app.include_router(create_router(ledger), prefix="/semantic")

Or give the ledger and a dependency that returns the actor string:

    app.include_router(create_router(g, actor=current_actor), prefix="/semantic")

Standalone, the actor comes from the ``X-Gloss-Actor`` header. That is fine
behind an authenticating proxy and not for direct exposure.

Responses are typed, so the OpenAPI schema is a contract a client can be generated from
(:mod:`gloss.client` is one).
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
    raise RuntimeError("The server needs the 'server' extra: pip install gloss-semantics[server]") from exc

ACTIONS = ("review", "accept", "reject", "retract", "reopen", "restore")


class AssertionOut(BaseModel):
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
    items: list[AssertionOut]
    total: int
    limit: int
    offset: int


class EventOut(BaseModel):
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
    subject: str
    attribute: str
    value: Any
    confidence: float | None = None
    evidence: list[dict[str, Any]] = []
    comment: str | None = None
    group: str | None = None


class ProposeManyIn(BaseModel):
    proposals: list[ProposalIn]
    source: str | None = None
    accept: bool = False
    comment: str | None = None


class ProposeOut(BaseModel):
    outcome: str
    accepted: bool = False
    assertion: AssertionOut


class ActionIn(BaseModel):
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
    id: str
    error: str


class BulkOut(BaseModel):
    changed: list[AssertionOut]
    errors: list[BulkError]


class SubjectOut(BaseModel):
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
    subjects: dict[str, str]
    moved: int
    rewritten: int
    superseded: int
    staled: int


class SearchHit(BaseModel):
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
    check: dict[str, Any]
    origin: str | None = None


class ColumnOut(BaseModel):
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
    key: str
    source: str | None = None
    description: str | None = None
    primary_key: list[str] = []
    confidentiality: str | None = None
    owner: str | None = None
    columns: list[ColumnOut] = []
    metrics: list[MetricOut] = []


class RelationshipOut(BaseModel):
    name: str
    namespace: str | None = None
    from_table: str
    to_table: str
    from_columns: list[str]
    to_columns: list[str]
    description: str | None = None
    version: int | None = None


class TypeOut(BaseModel):
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
    results: list[CheckResultIO]
    run_id: str | None = Field(default=None, description="The host's id for the run that produced these results")


class CheckRunOut(BaseModel):
    run_id: str
    recorded: int


class SyncOut(BaseModel):
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
    ok: bool
    schema_version: int | None = None


def header_actor(x_gloss_actor: str | None = Header(default=None)) -> str:
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
        return {"ok": True, "schema_version": __import__("gloss").schema.LATEST}

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
        criteria = dict(
            subject=subject, subject_prefix=subject_prefix, dataset=dataset, namespace=namespace, kind=kind,
            attribute=attribute, statuses=status, source_prefix=source_prefix, group=group,
        )
        limit = max(1, min(limit, 5000))
        rows = guard(lambda: g.find(**criteria, limit=limit, offset=offset))
        return {"items": [out(a) for a in rows], "total": g.count(**criteria), "limit": limit, "offset": offset}

    @r.post("/assertions", status_code=201, response_model=ProposeOut)
    def propose(body: ProposeIn, g: Gloss = Depends(ledger)) -> Any:
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
        def go() -> Any:
            props = [Proposal(p.subject, p.attribute, p.value, p.confidence, p.evidence, p.comment, p.group)
                     for p in body.proposals]
            results = g.propose_many(props, source=body.source, accept=body.accept, comment=body.comment)
            return [{"outcome": x.outcome, "accepted": x.accepted, "assertion": out(x.assertion)} for x in results]

        return guard(go)

    @r.get("/assertions/{assertion_id}", response_model=AssertionOut)
    def get_assertion(assertion_id: str, g: Gloss = Depends(ledger)) -> Any:
        return guard(lambda: out(g.get(assertion_id)))

    @r.post("/assertions/{assertion_id}/{action}", response_model=AssertionOut)
    def act(
        assertion_id: str,
        action: str,
        body: ActionIn = Body(default_factory=ActionIn),
        g: Gloss = Depends(ledger),
    ) -> Any:
        if action not in ACTIONS:
            raise HTTPException(404, f"Unknown action {action!r}")
        return guard(lambda: out(getattr(g, action)(assertion_id, comment=body.comment)))

    @r.post("/bulk/{action}", response_model=BulkOut)
    def bulk(action: str, body: BulkIn, g: Gloss = Depends(ledger)) -> Any:
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
        if action not in ("accept", "reject"):
            raise HTTPException(404, f"Unknown action {action!r}")
        fn = g.accept_group if action == "accept" else g.reject_group
        return guard(lambda: [out(a) for a in fn(group, comment=body.comment)])

    @r.get("/subjects/{subject:path}/history", response_model=list[EventOut])
    def subject_history(subject: str, g: Gloss = Depends(ledger)) -> Any:
        return guard(lambda: [e.to_dict() for e in g.history(subject)])

    @r.get("/subjects/{subject:path}", response_model=SubjectOut)
    def subject_view(subject: str, g: Gloss = Depends(ledger)) -> Any:
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
        fn = g.rekey_namespace if body.namespace else g.rekey
        return guard(lambda: fn(body.old, body.new, comment=body.comment).__dict__)

    @r.get("/log", response_model=list[EventOut])
    def log(after_seq: int = 0, limit: int = 500, correlation: str | None = None,
            g: Gloss = Depends(ledger)) -> Any:
        return [e.to_dict() for e in g.log(after_seq=after_seq, limit=min(limit, 5000), correlation=correlation)]

    @r.get("/progress", response_model=dict[str, dict[str, int]])
    def progress(dataset_prefix: str | None = None, g: Gloss = Depends(ledger)) -> Any:
        return g.progress(dataset_prefix)

    @r.get("/search", response_model=list[SearchHit])
    def search(q: str, kind: list[str] | None = Param(default=None), dataset_prefix: str | None = None,
               limit: int = 20, g: Gloss = Depends(ledger)) -> Any:
        """Search as the caller: what they may read, redacted above their level."""
        return g.search(q, kinds=kind, dataset_prefix=dataset_prefix, limit=min(limit, 200))

    @r.get("/tables", response_model=list[TableOut])
    def tables(prefix: str | None = None, include_pending: bool = False, g: Gloss = Depends(ledger)) -> Any:
        return [_table(t) for t in g.tables(prefix, include_pending=include_pending)]

    @r.get("/tables/{key:path}", response_model=TableOut)
    def table(key: str, include_pending: bool = False, g: Gloss = Depends(ledger)) -> Any:
        return guard(lambda: _table(g.table(key, include_pending=include_pending)))

    @r.get("/relationships", response_model=list[RelationshipOut])
    def relationships(table: str | None = None, include_pending: bool = False, g: Gloss = Depends(ledger)) -> Any:
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
        return [_metric(m) for m in g.metrics(table, include_pending=include_pending)]

    @r.get("/types", response_model=list[TypeOut])
    def types(include_pending: bool = False, g: Gloss = Depends(ledger)) -> Any:
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
        from .checks import effective_checks as effective

        return guard(lambda: {
            f: [{"check": c, "origin": o} for c, o in items] for f, items in effective(g, dataset).items()
        })

    @r.get("/checks/results", response_model=list[CheckResultIO])
    def check_results(subject: str | None = None, dataset: str | None = None, run_id: str | None = None,
                      limit: int = 200, g: Gloss = Depends(ledger)) -> Any:
        rows = guard(lambda: g.check_history(subject, dataset=dataset, run_id=run_id, limit=min(limit, 5000)))
        return [{**row, "ts": row["ts"].isoformat()} for row in rows]

    @r.post("/checks/results", status_code=201, response_model=CheckRunOut)
    def record_checks(body: CheckRunIn, g: Gloss = Depends(ledger)) -> Any:
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
        return guard(lambda: g.to_yaml(include_pending=include_pending, name=name, namespace=namespace))

    @r.get("/viz/mermaid", response_class=PlainTextResponse)
    def viz_mermaid(include_pending: bool = True, g: Gloss = Depends(ledger)) -> str:
        from .viz import mermaid_er

        return mermaid_er(g, include_pending=include_pending)

    @r.get("/viz/report", response_class=HTMLResponse)
    def viz_report(g: Gloss = Depends(ledger)) -> str:
        from .viz import html_report

        return html_report(g)

    @r.get("/ai/context", response_model=dict[str, Any])
    def ai_context(dataset: list[str] | None = Param(default=None), g: Gloss = Depends(ledger)) -> Any:
        from . import ai

        return ai.model_context(g, datasets=dataset)

    @r.get("/ai/search", response_model=list[SearchHit])
    def ai_search(q: str, limit: int = 20, g: Gloss = Depends(ledger)) -> Any:
        from . import ai

        return ai.search(g, q, limit=limit)

    return r


def create_app(gloss: Gloss | Callable[..., Gloss], *, actor: Callable[..., str] = header_actor,
               catalogs: Mapping[str, Any] | None = None) -> FastAPI:
    app = FastAPI(title="Gloss", version=__import__("gloss").__version__)
    app.include_router(create_router(gloss, actor=actor, catalogs=catalogs))
    return app
