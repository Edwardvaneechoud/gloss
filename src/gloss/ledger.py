"""The Gloss facade: the one object hosts embed.

    import gloss
    g = gloss.open("sqlite:///gloss.db")
    a = g.propose("field:sales.orders/amount", "description", "Order value incl. VAT",
                  source="ai:claude", confidence=0.8)
    g.accept(a.id, actor="human:alice")
    g.current("field:sales.orders/amount")   # {'description': 'Order value incl. VAT'}

A ``Gloss`` opened like this is the host's own, trusted handle: it reads everything and
every write names its actor. ``g.as_actor(...)`` returns a view bound to one actor: its
reads are limited to what the policy lets that actor see, and its writes are made as them.
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import islice
from typing import Any

from . import policy as P
from .actors import ActorDirectory, IdentityDirectory, split_actor
from .ids import Kind, SubjectId, path_covers
from .model import (
    ATOMIC_KINDS,
    DATASET_REFERENCES,
    LIVE_STATUSES,
    OPEN_STATUSES,
    Assertion,
    Confidentiality,
    Event,
    EventType,
    InvalidTransition,
    Status,
    apply_event,
    can_apply,
    detail_attributes,
    inherited_attributes,
    new_id,
    validate_value,
    value_hash,
)
from .policy import Context, Decision, DefaultPolicy, PermissionDenied, Policy, Scope
from .store import ConcurrentWrite, Query, SQLStore

log = logging.getLogger("gloss")


@dataclass
class Proposal:
    """A fact a source wants Gloss to consider. Sources yield these; Gloss decides what becomes of them.

    Proposals sharing a ``group`` were made together. For relationships and metrics a group is
    accepted or rejected as a unit, because their facts only make sense together.
    """

    subject: str
    attribute: str
    value: Any
    confidence: float | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    comment: str | None = None
    group: str | None = None


@dataclass
class ProposeResult:
    assertion: Assertion
    outcome: str  # "created" | "corroborated" | "unchanged" | "previously_rejected"
    accepted: bool = False  # accepted in this same call (asked for, or by a policy rule)


@dataclass
class RekeyResult:
    """What a rename or move changed: subjects re-keyed, references rewritten, facts sent back."""

    subjects: dict[str, str] = field(default_factory=dict)  # old subject -> new subject
    moved: int = 0  # assertions whose subject changed
    rewritten: int = 0  # assertions whose value named a moved subject
    superseded: int = 0  # moved facts that lost to a value already accepted at the target
    staled: int = 0  # facts sent back to review because they may mention the old name


@dataclass(frozen=True)
class ReadRecord:
    """Which subjects were handed to which actor, for hosts that keep a record of what AI read."""

    actor: str | None
    via: str | None
    tool: str
    subjects: tuple[str, ...]


_ACTION_EVENTS = {
    P.REVIEW: EventType.REVIEWED,
    P.ACCEPT: EventType.ACCEPTED,
    P.REJECT: EventType.REJECTED,
    P.RETRACT: EventType.RETRACTED,
    P.REOPEN: EventType.REOPENED,
    P.RESTORE: EventType.RESTORED,
    P.MARK_STALE: EventType.MARKED_STALE,
}
# Actions that make a value the accepted one, replacing whatever was accepted before.
_ACCEPTING = (P.ACCEPT, P.RESTORE)
# Actions that apply to a whole proposal group for relationships and metrics.
_GROUPED = (P.ACCEPT, P.REJECT)

_UNSET: Any = object()


class _Shared:
    """State common to a ledger and every actor-bound view of it."""

    def __init__(self) -> None:
        self.subscribers: list[Callable[[list[Event]], None]] = []
        self.readers: list[Callable[[ReadRecord], None]] = []
        self.search_ready = False


class _Batch:
    """Events and the states they produce, collected for one atomic append."""

    def __init__(self, actor: str, correlation: str, via: str | None):
        self.actor, self.correlation, self.via = actor, correlation, via
        self.events: list[Event] = []
        self.states: dict[str, Assertion] = {}

    def now(self, a: Assertion) -> Assertion:
        return self.states.get(a.id, a)

    def emit(self, etype: EventType, assertion: Assertion | None, data: dict[str, Any],
             assertion_id: str | None = None) -> Assertion:
        ev = Event(etype, assertion.id if assertion else assertion_id or new_id(), self.actor,
                   {k: v for k, v in data.items() if v is not None},
                   correlation=self.correlation, via=self.via)
        state = apply_event(self.now(assertion) if assertion else None, ev)
        self.events.append(ev)
        self.states[state.id] = state
        return state

    def rollback(self, mark: tuple[int, dict[str, Assertion]]) -> None:
        n, states = mark
        del self.events[n:]
        self.states = states

    def mark(self) -> tuple[int, dict[str, Assertion]]:
        return len(self.events), dict(self.states)


class Gloss:
    def __init__(self, store: SQLStore, policy: Policy | None = None, *, directory: ActorDirectory | None = None):
        self.store = store
        self.policy: Policy = policy or DefaultPolicy()
        self.directory: ActorDirectory = directory or IdentityDirectory()
        self.actor: str | None = None
        self.context: Context | None = None
        self.correlation: str | None = None
        self._scope: Scope | None = None
        self._shared = _Shared()

    # ----------------------------------------------------------------- binding
    def as_actor(
        self,
        actor: str,
        *,
        via: str | None = None,
        context: Mapping[str, Any] | None = None,
        correlation: str | None = None,
    ) -> Gloss:
        """A view of this ledger for one actor: scoped reads, and writes made as them.

        ``via`` names the channel the actor came through (``"ai:claude"`` for a person working
        through an agent, ``"server:eu-1"``) and is recorded on every event. ``context`` is opaque
        host data handed to the policy. Bind once per request; the scope is resolved here.
        """
        split_actor(actor)
        view = copy.copy(self)
        view.actor = str(actor)
        view.context = Context(via=via, attrs=dict(context or {}))
        view.correlation = correlation or self.correlation
        view._scope = self.policy.scope(view.actor, view.context)
        return view

    def correlated(self, correlation: str | None = None) -> Gloss:
        """The same ledger with one correlation id on every event it writes (a request, a sync run)."""
        view = copy.copy(self)
        view.correlation = correlation or new_id()
        return view

    @property
    def scope(self) -> Scope:
        """What this handle reads: everything, unless it is bound to an actor."""
        return self._scope or Scope.everything()

    def _who(self, actor: str | None, fallback: str | None = None) -> str:
        if self.actor is not None:
            if actor is not None and str(actor) != self.actor:
                raise PermissionDenied(f"This view acts as {self.actor}, not {actor}")
            return self.actor
        who = actor or fallback
        if who is None:
            raise ValueError("Pass actor= (e.g. 'human:alice'), or bind one with g.as_actor(...)")
        split_actor(who)
        return str(who)

    def _batch(self, actor: str) -> _Batch:
        return _Batch(actor, self.correlation or new_id(), self.context.via if self.context else None)

    def _q(self, **criteria: Any) -> Query:
        return Query(scope=self._scope, **criteria)

    def _visible(self, a: Assertion) -> bool:
        return self._scope is None or self._scope.covers(a.scope_path)

    @property
    def _limit(self) -> Confidentiality | None:
        """The reader's confidentiality ceiling, or None when nothing has to be withheld."""
        if self._scope is None or self._scope.max_confidentiality is Confidentiality.RESTRICTED:
            return None
        return self._scope.max_confidentiality

    def _hidden_subjects(self, subjects: Iterable[str]) -> set[str]:
        limit = self._limit
        if limit is None:
            return set()
        from .visibility import levels

        return {s for s, lv in levels(self.store, subjects).items() if lv > limit}

    def _redact(self, rows: list[Assertion]) -> list[Assertion]:
        if self._limit is None or not rows:
            return rows
        hidden = self._hidden_subjects({a.subject for a in rows})
        detail = detail_attributes()
        return [
            replace(a, value=None, evidence=(), last_comment=None, redacted=True)
            if a.subject in hidden and a.attribute in detail
            else a
            for a in rows
        ]

    def _redact_events(self, events: list[Event]) -> list[Event]:
        if self._limit is None or not events:
            return events
        known = self.store.get_many({e.assertion_id for e in events})
        hidden = self._hidden_subjects({a.subject for a in known.values()})
        detail = detail_attributes()
        secret = ("value", "from_value", "evidence", "comment", "reason")
        out = []
        for e in events:
            a = known.get(e.assertion_id)
            if a is not None and a.subject in hidden and a.attribute in detail:
                e = replace(e, data={**{k: v for k, v in e.data.items() if k not in secret}, "redacted": True})
            out.append(e)
        return out

    # ------------------------------------------------------------------ hooks
    def subscribe(self, callback: Callable[[list[Event]], None]) -> Callable[[], None]:
        """Call ``callback(events)`` after every committed write, with ``seq`` filled in.

        Returns a function that unsubscribes. A failing subscriber is logged and never undoes the write.
        """
        self._shared.subscribers.append(callback)
        return lambda: self._shared.subscribers.remove(callback)

    def on_read(self, callback: Callable[[ReadRecord], None]) -> Callable[[], None]:
        """Call ``callback(ReadRecord)`` whenever an AI-facing view hands subjects to an actor."""
        self._shared.readers.append(callback)
        return lambda: self._shared.readers.remove(callback)

    def report_read(self, tool: str, subjects: Iterable[str]) -> None:
        if not self._shared.readers:
            return
        record = ReadRecord(self.actor, self.context.via if self.context else None, tool, tuple(subjects))
        for fn in list(self._shared.readers):
            try:
                fn(record)
            except Exception:  # a host's bookkeeping must not break a read
                log.exception("gloss read hook failed")

    def _notify(self, events: list[Event]) -> None:
        for fn in list(self._shared.subscribers):
            try:
                fn(events)
            except Exception:
                log.exception("gloss subscriber failed")

    def _commit(self, batch: _Batch, also_reindex: Iterable[str] = (),
                renames: dict[str, str] | None = None) -> None:
        """Append a batch and update the search index in one transaction, then tell subscribers."""
        if not batch.events:
            return
        wanted = bool(self._shared.subscribers)
        with self.store.transaction():
            stored = self.store.append(batch.events, list(batch.states.values()), returning=wanted)
            if renames:
                self.store.rename_check_subjects(renames)
            self._reindex_after(batch.events, batch.states, also_reindex)
        if wanted:
            self.store.after_commit(lambda: self._notify(stored))

    def _reindex_after(self, events: Sequence[Event], states: dict[str, Assertion],
                       extra: Iterable[str] = ()) -> None:
        """Update search rows for subjects whose accepted knowledge just changed.

        Rows that inherit from a changed subject follow only when what they inherit changed: a
        table's confidentiality, or anything about a semantic type.
        """
        changed = set(extra)
        cascade = set(extra)
        for e in events:
            state = states[e.assertion_id]
            subject = state.subject
            hit = e.type in (EventType.ACCEPTED, EventType.SUPERSEDED, EventType.RESTORED, EventType.REKEYED) or (
                e.type in (EventType.MARKED_STALE, EventType.RETRACTED)
                and e.data.get("from_status") == Status.ACCEPTED.value
            )
            if not hit:
                continue
            changed.add(subject)
            if e.type is EventType.REKEYED and e.data.get("from_subject"):
                changed.add(e.data["from_subject"])
            if subject.startswith("type:") or state.attribute == "confidentiality" or e.type is EventType.REKEYED:
                cascade.add(subject)
        if changed:
            from .search_index import reindex

            reindex(self, changed, cascade=cascade)

    # ------------------------------------------------------------------ write
    def propose(
        self,
        subject: str | SubjectId,
        attribute: str,
        value: Any,
        *,
        source: str | None = None,
        actor: str | None = None,
        confidence: float | None = None,
        evidence: Sequence[dict[str, Any]] | None = None,
        comment: str | None = None,
        force: bool = False,
        accept: bool = False,
        group: str | None = None,
    ) -> Assertion:
        return self.propose_many(
            [Proposal(str(subject), attribute, value, confidence, list(evidence or []), comment, group)],
            source=source,
            actor=actor,
            force=force,
            accept=accept,
        )[0].assertion

    def propose_many(
        self,
        proposals: Iterable[Proposal],
        *,
        source: str | None = None,
        actor: str | None = None,
        force: bool = False,
        accept: bool = False,
        auto_accept: bool = True,
        comment: str | None = None,
        batch_size: int = 500,
    ) -> list[ProposeResult]:
        """Propose many facts efficiently, with de-duplication.

        * the same value already live (open or accepted) → the source is recorded as *corroborating* it;
        * the same value previously rejected → nothing happens unless ``force`` (stops AI re-suggesting);
        * otherwise a new assertion is created in status *proposed*.

        What happens next is the policy's call (``Policy.on_propose``): the proposal is queued for
        review, or accepted at once under a named rule (a solo user's own edit, a catalog stating
        structure), or refused. ``auto_accept=False`` queues everything regardless. ``accept=True``
        asks to accept everything proposed here as the actor, which the policy must allow.

        Proposals are processed in chunks. Each chunk is one lookup and one transaction, and the
        proposed, superseded and accepted events of a chunk are written together.
        """
        actor = self._who(actor, source)
        source = str(source or actor)
        split_actor(source)
        ctx = self.context
        if not self.policy.can(actor, P.PROPOSE, None, ctx):
            raise PermissionDenied(f"{actor} may not propose")
        results: list[ProposeResult] = []
        it = iter(proposals)
        while chunk := list(islice(it, batch_size)):
            prepared = []
            for prop in chunk:
                sid = SubjectId.parse(prop.subject)
                if self._scope is not None and not self._scope.covers(sid.scope_path):
                    raise PermissionDenied(f"{actor} may not propose facts about {sid}")
                val = validate_value(sid, prop.attribute, prop.value)
                if prop.confidence is not None and not 0.0 <= prop.confidence <= 1.0:
                    raise ValueError("confidence must be between 0 and 1")
                decision = Decision.of(self.policy.on_propose(actor, source, sid, prop.attribute, ctx))
                if decision.outcome == "refuse":
                    raise PermissionDenied(
                        f"{actor} may not propose {prop.attribute!r} for {sid}"
                        + (f": {decision.rule}" if decision.rule else "")
                    )
                rule = decision.rule or "policy" if decision.outcome == "accept" and auto_accept else None
                prepared.append((sid, str(sid), val, value_hash(val), prop, rule))

            for attempt in (1, 2):
                try:
                    results += self._propose_chunk(prepared, source, actor, force, accept, comment)
                    break
                except ConcurrentWrite:
                    if attempt == 2:
                        raise
        return results

    def _propose_chunk(
        self, prepared: list[tuple[Any, ...]], source: str, actor: str, force: bool, accept: bool,
        comment: str | None,
    ) -> list[ProposeResult]:
        ctx = self.context
        index: dict[tuple[str, str, str], list[Assertion]] = defaultdict(list)
        accepted: dict[tuple[str, str], list[Assertion]] = defaultdict(list)
        for a in self.store.find_for_subjects({p[1] for p in prepared}):
            index[(a.subject, a.attribute, value_hash(a.value))].append(a)
            if a.status is Status.ACCEPTED:
                accepted[(a.subject, a.attribute)].append(a)

        batch = self._batch(actor)
        results: list[ProposeResult] = []
        for _sid, subj, val, vh, prop, rule in prepared:
            key = (subj, prop.attribute, vh)
            matches = [batch.now(m) for m in index[key]]
            live = [m for m in matches if m.status in LIVE_STATUSES]
            existing = live[0] if live else None
            if existing is None and not force:
                rejected = [m for m in matches if m.status is Status.REJECTED]
                if rejected:
                    results.append(ProposeResult(rejected[-1], "previously_rejected"))
                    continue
            if existing is not None:
                if source in existing.sources and prop.confidence in (None, existing.confidence):
                    state, outcome = existing, "unchanged"
                else:
                    state = batch.emit(
                        EventType.CORROBORATED,
                        existing,
                        {"source": source, "confidence": prop.confidence, "evidence": prop.evidence},
                    )
                    outcome = "corroborated"
            else:
                state = batch.emit(
                    EventType.PROPOSED,
                    None,
                    {
                        "subject": subj,
                        "attribute": prop.attribute,
                        "value": val,
                        "source": source,
                        "confidence": prop.confidence,
                        "evidence": prop.evidence,
                        "comment": prop.comment,
                        "group": prop.group,
                    },
                )
                index[key] = [state, *index[key]]
                outcome = "created"

            took = False
            if state.is_open and (rule or accept):
                if rule is None and not self.policy.can(actor, P.ACCEPT, state, ctx):
                    raise PermissionDenied(f"{actor} may not accept assertion {state.id}")
                extra = {"auto": True, "rule": rule} if rule else {"comment": comment}
                state = self._make_accepted(batch, state, EventType.ACCEPTED, accepted, extra)
                took = True
            results.append(ProposeResult(state, outcome, took))
        self._commit(batch)
        # results carry the state at the end of the batch (a later proposal may have superseded an earlier one)
        return [replace(r, assertion=batch.now(r.assertion)) for r in results]

    def _make_accepted(
        self, batch: _Batch, a: Assertion, etype: EventType, accepted: dict[tuple[str, str], list[Assertion]],
        data: dict[str, Any],
    ) -> Assertion:
        """Emit the events that make ``a`` the accepted value, superseding the one before it."""
        key = (a.subject, a.attribute)
        for prev in accepted[key]:
            prev = batch.now(prev)
            if prev.id != a.id and prev.status is Status.ACCEPTED:
                batch.emit(EventType.SUPERSEDED, prev, {"by": a.id})
        new = batch.emit(etype, a, {"from_status": batch.now(a).status.value, **data})
        accepted[key] = [new]
        return new

    def transition_many(
        self,
        action: str,
        assertion_ids: Iterable[str],
        *,
        actor: str | None = None,
        comment: str | None = None,
        batch_size: int = 1000,
        **data: Any,
    ) -> tuple[list[Assertion], list[tuple[str, Exception]]]:
        """Apply one action to many assertions with batched reads and one transaction per batch.

        Returns ``(changed, [(id, error), ...])``; failures do not stop the rest.
        Accepting (or restoring) supersedes the previously accepted value for the same subject and
        attribute. For relationships and metrics, accepting or rejecting one fact of a proposal
        group decides the whole group: all of it applies, or none of it does.
        """
        if action not in _ACTION_EVENTS:
            raise ValueError(f"Unknown action {action!r}")
        actor = self._who(actor)
        done: list[Assertion] = []
        errors: list[tuple[str, Exception]] = []
        it = iter(assertion_ids)
        while ids := list(islice(it, batch_size)):
            for attempt in (1, 2):
                try:
                    d, e = self._transition_batch(action, ids, actor, comment, data)
                    done += d
                    errors += e
                    break
                except ConcurrentWrite as exc:
                    if attempt == 2:
                        errors += [(aid, exc) for aid in ids]
        return done, errors

    def _transition_batch(
        self, action: str, ids: list[str], actor: str, comment: str | None, data: dict[str, Any]
    ) -> tuple[list[Assertion], list[tuple[str, Exception]]]:
        etype = _ACTION_EVENTS[action]
        ctx = self.context
        current = {k: a for k, a in self.store.get_many(ids).items() if self._visible(a)}

        # units: an assertion on its own, or every open member of its proposal group
        units: list[list[str]] = []
        seen: set[str] = set()
        if action in _GROUPED:
            grouped = [a for a in current.values() if a.group and a.subject_id.kind in ATOMIC_KINDS]
            members: dict[tuple[str, str], list[Assertion]] = defaultdict(list)
            if grouped:
                for m in self.store.find_for_subjects({a.subject for a in grouped}):
                    if m.group and can_apply(etype, m.status):
                        members[(m.subject, m.group)].append(m)
                        current.setdefault(m.id, m)
        for aid in ids:
            if aid in seen:
                continue
            a = current.get(aid)
            unit = [aid]
            if a is not None and action in _GROUPED and a.group and a.subject_id.kind in ATOMIC_KINDS:
                unit = [aid] + [m.id for m in members[(a.subject, a.group)] if m.id != aid]
            seen.update(unit)
            units.append(unit)

        accepted: dict[tuple[str, str], list[Assertion]] = defaultdict(list)
        if action in _ACCEPTING:
            for a in self.store.find_for_subjects({c.subject for c in current.values()}, [Status.ACCEPTED]):
                accepted[(a.subject, a.attribute)].append(a)

        batch = self._batch(actor)
        done: list[Assertion] = []
        errors: list[tuple[str, Exception]] = []
        requested = set(ids)
        for unit in units:
            mark, held = batch.mark(), {k: list(v) for k, v in accepted.items()}
            changed: list[Assertion] = []
            failure: Exception | None = None
            for aid in unit:
                a = current.get(aid)
                a = batch.now(a) if a is not None else None
                if a is None:
                    failure = KeyError(f"No assertion {aid!r}")
                elif not self.policy.can(actor, action, a, ctx):
                    failure = PermissionDenied(f"{actor} may not {action} assertion {aid}")
                elif not can_apply(etype, a.status):
                    failure = InvalidTransition(f"Cannot {action} assertion {aid}: it is {a.status.value}")
                if failure is not None:
                    break
                payload = {"comment": comment, **data}
                if len(unit) > 1:
                    payload["group"] = a.group
                if action in _ACCEPTING:
                    changed.append(self._make_accepted(batch, a, etype, accepted, payload))
                else:
                    changed.append(batch.emit(etype, a, {"from_status": a.status.value, **payload}))
            if failure is not None:
                batch.rollback(mark)
                accepted.clear()
                accepted.update(held)
                if len(unit) > 1 and not isinstance(failure, KeyError):
                    failure = type(failure)(f"{failure.args[0]} (decided as a group; nothing in the group changed)")
                errors += [(aid, failure) for aid in unit if aid in requested]
            else:
                done += changed
        self._commit(batch)
        return [batch.now(a) for a in done], errors

    def _transition(
        self, assertion_id: str, action: str, actor: str | None, comment: str | None = None, **data: Any
    ) -> Assertion:
        aid = self._require(assertion_id).id
        done, errors = self.transition_many(action, [aid], actor=actor, comment=comment, **data)
        if errors:
            raise errors[0][1]
        return next(a for a in done if a.id == aid)

    def review(self, assertion_id: str, *, actor: str | None = None, comment: str | None = None) -> Assertion:
        return self._transition(assertion_id, P.REVIEW, actor, comment)

    def accept(self, assertion_id: str, *, actor: str | None = None, comment: str | None = None) -> Assertion:
        return self._transition(assertion_id, P.ACCEPT, actor, comment)

    def reject(self, assertion_id: str, *, actor: str | None = None, comment: str | None = None) -> Assertion:
        return self._transition(assertion_id, P.REJECT, actor, comment)

    def retract(self, assertion_id: str, *, actor: str | None = None, comment: str | None = None) -> Assertion:
        return self._transition(assertion_id, P.RETRACT, actor, comment)

    def reopen(self, assertion_id: str, *, actor: str | None = None, comment: str | None = None) -> Assertion:
        return self._transition(assertion_id, P.REOPEN, actor, comment)

    def restore(self, assertion_id: str, *, actor: str | None = None, comment: str | None = None) -> Assertion:
        """Put a superseded value back as the accepted one (the value now accepted is superseded)."""
        return self._transition(assertion_id, P.RESTORE, actor, comment)

    def mark_stale(self, assertion_id: str, *, actor: str | None = None, reason: str) -> Assertion:
        return self._transition(assertion_id, P.MARK_STALE, actor, reason, reason=reason)

    def accept_group(self, group: str, *, actor: str | None = None, comment: str | None = None) -> list[Assertion]:
        """Accept every open fact of one proposal group, all or nothing."""
        return self._decide_group(P.ACCEPT, group, actor, comment)

    def reject_group(self, group: str, *, actor: str | None = None, comment: str | None = None) -> list[Assertion]:
        return self._decide_group(P.REJECT, group, actor, comment)

    def _decide_group(self, action: str, group: str, actor: str | None, comment: str | None) -> list[Assertion]:
        etype = _ACTION_EVENTS[action]
        ids = [a.id for a in self.store.find(self._q(group=group)) if can_apply(etype, a.status)]
        if not ids:
            raise KeyError(f"No open facts in group {group!r}")
        with self.store.transaction():
            done, errors = self.transition_many(action, ids, actor=actor, comment=comment)
            if errors:
                raise errors[0][1]
        return done

    def comment(self, assertion_id: str, *, actor: str | None = None, comment: str) -> Assertion:
        actor = self._who(actor)
        current = self._require(assertion_id)
        if not self.policy.can(actor, P.COMMENT, current, self.context):
            raise PermissionDenied(f"{actor} may not comment")
        batch = self._batch(actor)
        state = batch.emit(EventType.COMMENTED, current, {"comment": comment})
        self._commit(batch)
        return state

    def bulk(
        self, action: str, query: Query, *, actor: str | None = None, comment: str | None = None
    ) -> tuple[list[Assertion], list[tuple[str, str]]]:
        """Apply an action to every matching assertion that can take it. Returns (changed, [(id, error)])."""
        etype = _ACTION_EVENTS[action]
        if self._scope is not None:
            query = replace(query, scope=self._scope)
        ids = [a.id for a in self.store.find(query) if can_apply(etype, a.status)]
        done, errors = self.transition_many(action, ids, actor=actor, comment=comment)
        return done, [(aid, str(exc)) for aid, exc in errors]

    def add(
        self,
        *objects: Any,
        source: str | None = None,
        actor: str | None = None,
        accept: bool = False,
        comment: str | None = None,
    ) -> Any:
        """Save Table / Relationship / Metric / SemanticType objects as proposals.

        ``source`` defaults to the actor. With ``accept=True`` the actor accepts everything it just
        proposed, in the same transaction, which the policy must allow. Under a solo policy a
        person's own objects are accepted without asking.
        """
        from .modeling import AddResult

        props = [p for obj in objects for p in obj.proposals()]
        results = self.propose_many(props, source=source, actor=actor, accept=accept, comment=comment)
        out = AddResult()
        for r in results:
            setattr(out, r.outcome, getattr(out, r.outcome) + 1)
        out.assertion_ids = list(dict.fromkeys(r.assertion.id for r in results))
        out.accepted = len({r.assertion.id for r in results if r.accepted})
        return out

    def apply(
        self,
        *objects: Any,
        source: str | None = None,
        actor: str | None = None,
        accept: bool = False,
        comment: str | None = None,
    ) -> Any:
        """Like :meth:`add`, and converge: what this source no longer declares is retracted.

        For a model kept in code. Facts the same ``source`` stated earlier about these objects (a
        table with its columns and metrics, a relationship, a metric, a type) that the objects no
        longer declare are retracted, so removing a column from a ``Table`` removes what was said
        about it. Facts other sources also stated are left alone: they are no longer only yours.
        """
        who = self._who(actor, source)
        src = str(source or who)
        declared: set[tuple[str, str, str]] = set()
        covered: list[Query] = []
        for obj in objects:
            for p in obj.proposals():
                sid = SubjectId.parse(p.subject)
                declared.add((str(sid), p.attribute, value_hash(validate_value(sid, p.attribute, p.value))))
            key = getattr(obj, "key", None)
            covered.append(Query(dataset=key) if key else Query(subject=str(obj.subject)))
        with self.store.transaction():
            out = self.add(*objects, source=source, actor=actor, accept=accept, comment=comment)
            gone = [
                a.id
                for q in covered
                for a in self.store.find(replace(q, statuses=list(LIVE_STATUSES)))
                if a.sources == (src,) and (a.subject, a.attribute, value_hash(a.value)) not in declared
            ]
            done, errors = self.transition_many(
                P.RETRACT, list(dict.fromkeys(gone)), actor=who, comment=comment or "no longer declared"
            )
            if errors:
                raise errors[0][1]
        out.retracted = len(done)
        return out

    # ---------------------------------------------------------------- re-keying
    def rekey(
        self, old: str | SubjectId, new: str | SubjectId, *, actor: str | None = None, comment: str | None = None
    ) -> RekeyResult:
        """Record that a subject was renamed or moved, so everything known about it follows.

        ``old`` and ``new`` are subjects of the same kind:

        * a dataset: the dataset, its fields and its metrics move; relationships that join it are rewritten;
        * a field (same dataset): the column is renamed; keys and joins that list it are rewritten, and
          metrics that seem to use it go back to review;
        * a relationship, metric or semantic type: it is renamed or moved to another namespace; fields
          using a renamed type are rewritten.

        Nothing is lost and nothing is edited: each affected assertion gets a ``rekeyed`` event, so
        ``history()`` of the new subject shows its whole past. If the target already has an accepted
        value for an attribute, that one stays and the moved one is superseded.
        """
        o, n = SubjectId.parse(old), SubjectId.parse(new)
        if o.kind is not n.kind:
            raise ValueError(f"Cannot re-key a {o.kind.value} as a {n.kind.value}")
        if o == n:
            raise ValueError(f"{o} is already its own key")
        if o.kind is Kind.FIELD and o.dataset != n.dataset:
            raise ValueError("A field moves with its dataset; re-key the dataset instead")
        actor = self._who(actor)
        moves: dict[str, tuple[Assertion, str | None, Any]] = {}
        stale: list[tuple[Assertion, str]] = []

        if o.kind is Kind.DATASET:
            for a in self.store.find(Query(dataset=o.dataset)):
                moves[a.id] = (a, str(replace(a.subject_id, dataset=n.dataset)), _UNSET)
            for attr in DATASET_REFERENCES:
                for a in self.store.find(Query(attribute=attr, value_hash=value_hash(o.dataset))):
                    moves[a.id] = (a, None, n.dataset)
            self._stale_renamed_table(o.dataset, n.dataset, moves, stale)
        else:
            for a in self.store.find(Query(subject=str(o))):
                moves[a.id] = (a, str(n), _UNSET)
            if o.kind is Kind.TYPE:
                for a in self.store.find(Query(attribute="semantic_type", value_hash=value_hash(o.ref))):
                    moves[a.id] = (a, None, n.ref)
            if o.kind is Kind.FIELD:
                self._field_references(o, n, moves, stale)
        if not moves:
            raise KeyError(f"Nothing known about {o}")
        return self._apply_rekey(moves, stale, actor, comment)

    def rekey_namespace(
        self, old: str, new: str, *, actor: str | None = None, comment: str | None = None
    ) -> RekeyResult:
        """Record that a whole path was renamed or moved: a catalog, a schema, a namespace.

        Every subject at or under ``old`` (datasets by key, relationships, metrics and types by
        namespace) moves to the same place under ``new``, and facts naming them are rewritten.
        """
        old, new = old.rstrip("."), new.rstrip(".")
        if not old or not new:
            raise ValueError("Give the old and the new path")
        if path_covers(old, new) or path_covers(new, old):
            raise ValueError(f"Cannot move {old!r} to {new!r}: one path contains the other")
        actor = self._who(actor)
        moves: dict[str, tuple[Assertion, str | None, Any]] = {}
        types: dict[str, str] = {}
        for a in self.store.find(Query(namespace=old)):
            target = a.subject_id.moved(old, new)
            if target is None:
                continue
            moves[a.id] = (a, str(target), _UNSET)
            if target.kind is Kind.TYPE:
                types[a.subject_id.ref] = target.ref
        if not moves:
            raise KeyError(f"Nothing known under {old!r}")
        from .ids import replace_prefix

        for a in self.store.find(Query(kind=Kind.RELATIONSHIP.value, attributes=DATASET_REFERENCES)):
            if isinstance(a.value, str) and path_covers(old, a.value):
                subject = moves[a.id][1] if a.id in moves else None
                moves[a.id] = (a, subject, replace_prefix(a.value, old, new))
        for old_ref, new_ref in types.items():
            for a in self.store.find(Query(attribute="semantic_type", value_hash=value_hash(old_ref))):
                subject = moves[a.id][1] if a.id in moves else None
                moves[a.id] = (a, subject, new_ref)
        return self._apply_rekey(moves, [], actor, comment)

    def _field_references(
        self, o: SubjectId, n: SubjectId, moves: dict[str, tuple[Assertion, str | None, Any]],
        stale: list[tuple[Assertion, str]],
    ) -> None:
        """Value rewrites and stale candidates for a renamed column."""
        from .references import mentions_column, quote_ident

        ds, was, now = o.dataset or "", o.name or "", n.name or ""

        def swap(names: list[str]) -> list[str]:
            return [now if c == was else c for c in names]

        for a in self.store.find(Query(subject=str(SubjectId.for_dataset(ds)),
                                       attributes=("primary_key", "unique_keys"))):
            if a.attribute == "primary_key" and was in (a.value or []):
                moves[a.id] = (a, None, swap(a.value))
            elif a.attribute == "unique_keys" and any(was in k for k in a.value or []):
                moves[a.id] = (a, None, [swap(k) for k in a.value])
        # relationship columns: from_columns belong to the `from` dataset, to_columns to the `to` dataset
        ends: dict[str, set[str]] = defaultdict(set)
        for attr in DATASET_REFERENCES:
            for a in self.store.find(Query(attribute=attr, value_hash=value_hash(ds))):
                ends[a.subject].add(f"{attr}_columns")
        for a in self.store.find_for_subjects(ends, attributes=("from_columns", "to_columns")):
            if a.attribute in ends[a.subject] and was in (a.value or []):
                moves[a.id] = (a, None, swap(a.value))
        for a_id, (a, subject, _value) in list(moves.items()):
            if subject and a.attribute == "expression" and a.value == {"sql": quote_ident(was)}:
                moves[a_id] = (a, subject, {"sql": quote_ident(now)})
        reason = f"column {was!r} of {ds} was renamed to {now!r}"
        for a in self.store.find(Query(kind=Kind.METRIC.value, attribute="expression",
                                       statuses=[Status.PROPOSED, Status.REVIEWED, Status.ACCEPTED])):
            if mentions_column(a.value, ds, was, scoped=a.dataset == ds):
                stale.append((a, reason))

    def _stale_renamed_table(
        self, old: str, new: str, moves: dict[str, tuple[Assertion, str | None, Any]],
        stale: list[tuple[Assertion, str]],
    ) -> None:
        """Metrics outside the table that name it in SQL go back to review when its short name changed."""
        from .references import mentions_dataset, short_name

        if short_name(old) == short_name(new):
            return
        reason = f"table {old} was renamed to {new}"
        for a in self.store.find(Query(kind=Kind.METRIC.value, attribute="expression",
                                       statuses=[Status.PROPOSED, Status.REVIEWED, Status.ACCEPTED])):
            if mentions_dataset(a.value, old):
                stale.append((a, reason))

    def _apply_rekey(
        self, moves: dict[str, tuple[Assertion, str | None, Any]], stale: list[tuple[Assertion, str]],
        actor: str, comment: str | None,
    ) -> RekeyResult:
        ctx = self.context or Context()
        result = RekeyResult()
        for a, subject, _value in moves.values():
            if subject is None:
                continue
            if not self._visible(a):
                raise KeyError(f"Nothing known about {a.subject}")
            to_ctx = Context(via=ctx.via, attrs={**ctx.attrs, "to": subject})
            if not self.policy.can(actor, P.REKEY, a, to_ctx):
                raise PermissionDenied(f"{actor} may not move {a.subject} to {subject}")
            result.subjects[a.subject] = subject

        moving = set(moves)
        taken: dict[tuple[str, str], Assertion] = {}
        for a in self.store.find_for_subjects(set(result.subjects.values()), [Status.ACCEPTED]):
            if a.id not in moving:
                taken[(a.subject, a.attribute)] = a

        batch = self._batch(actor)
        for a, subject, value in moves.values():
            data: dict[str, Any] = {"comment": comment}
            if subject is not None:
                data.update(from_subject=a.subject, subject=subject)
                result.moved += 1
            if value is not _UNSET:
                data.update(from_value=a.value, value=value)
                result.rewritten += 1
            state = batch.emit(EventType.REKEYED, a, data)
            winner = taken.get((state.subject, state.attribute))
            if subject is not None and winner is not None and state.status is Status.ACCEPTED:
                batch.emit(EventType.SUPERSEDED, state, {"by": winner.id})
                result.superseded += 1
        for a, reason in stale:
            now = batch.now(a)
            if can_apply(EventType.MARKED_STALE, now.status):
                batch.emit(EventType.MARKED_STALE, now,
                           {"from_status": now.status.value, "comment": reason, "reason": reason})
                result.staled += 1
        fields = {o: n for o, n in result.subjects.items() if o.startswith("field:")}
        self._commit(batch, also_reindex=set(result.subjects) | set(result.subjects.values()), renames=fields)
        return result

    # ------------------------------------------------------------------ export
    def to_ossie(
        self,
        *,
        name: str | None = None,
        include_pending: bool = False,
        tables: str | None = None,
        annotate: bool = True,
        namespace: str | None = None,
    ) -> Any:
        """The model as an Apache Ossie document (``.document`` dict, ``.warnings``, ``.to_yaml()``).

        Accepted facts only unless ``include_pending``; ``tables`` limits it to keys with that prefix.
        """
        from .ossie.exporter import to_ossie

        return to_ossie(self, name=name, include_pending=include_pending, annotate=annotate,
                        dataset_prefix=tables, namespace=namespace)

    def to_yaml(
        self,
        path: Any = None,
        *,
        name: str | None = None,
        include_pending: bool = False,
        tables: str | None = None,
        annotate: bool = True,
        namespace: str | None = None,
    ) -> str:
        """The model as Ossie YAML text; also written to ``path`` when given."""
        text = self.to_ossie(name=name, include_pending=include_pending, tables=tables, annotate=annotate,
                             namespace=namespace).to_yaml()
        if path:
            from pathlib import Path

            Path(path).write_text(text)
        return text

    # ------------------------------------------------------------- read as objects
    def table(self, key: str, *, include_pending: bool = False) -> Any:
        """One table as a :class:`gloss.Table` (columns, keys, semantic types, its metrics).

        Readers return accepted facts by default; ``include_pending=True`` adds open proposals.
        """
        from .modeling import Table

        return Table.load(self, key, include_pending=include_pending)

    def __getitem__(self, key: str) -> Any:
        """``g["shop.customers"]`` → Table. Full subjects work too (accepted facts only):

        ``g["type:email_address"]`` → SemanticType, ``g["relationship:orders_customers"]`` → Relationship,
        ``g["metric:shop.orders/revenue"]`` → Metric, ``g["field:shop.customers/email"]`` → Column.
        Use ``g.table(key, include_pending=True)`` and friends to include proposals.
        """
        if not isinstance(key, str):
            raise TypeError("Gloss keys are strings: a table key like 'shop.customers' or a subject like 'type:x'")
        kind = key.split(":", 1)[0] if ":" in key else None
        if kind not in {k.value for k in Kind}:
            return self._get_or_hint(lambda: self.table(key), key)
        sid = SubjectId.parse(key)
        if sid.kind is Kind.DATASET:
            return self._get_or_hint(lambda: self.table(sid.dataset), key)
        if sid.kind is Kind.FIELD:
            table = self._get_or_hint(lambda: self.table(sid.dataset), key)
            if sid.name not in table:
                raise KeyError(f"Table {sid.dataset!r} has no accepted column {sid.name!r}")
            return table[sid.name]
        if sid.kind is Kind.TYPE:
            return self._get_or_hint(lambda: self.semantic_type(sid.ref), key)
        if sid.kind is Kind.RELATIONSHIP:
            return self._get_or_hint(lambda: self.relationship(sid.ref), key)
        if sid.kind is Kind.METRIC:
            return self._get_or_hint(
                lambda: self.metric(sid.name, table=sid.dataset, namespace=sid.namespace), key
            )
        raise KeyError(f"Use g.current({key!r}) for the model itself")

    def _get_or_hint(self, fetch: Any, key: str) -> Any:
        try:
            return fetch()
        except KeyError:
            pending = self.find(subject_prefix=_subject_prefix_for(key), statuses=list(OPEN_STATUSES), limit=1)
            if pending:
                raise KeyError(
                    f"{key!r} has only proposed facts so far; accept them, or read the proposals with "
                    f"include_pending=True (e.g. g.table(..., include_pending=True))"
                ) from None
            raise KeyError(f"Nothing known about {key!r}") from None

    def __contains__(self, key: object) -> bool:
        try:
            self[key]  # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def tables(self, prefix: str | None = None, *, include_pending: bool = False) -> list[Any]:
        """All tables (optionally those whose key starts with ``prefix``), sorted by key."""
        from .modeling import Table, read_statuses, types_from_state

        state = self.snapshot(read_statuses(include_pending), dataset_prefix=prefix, relationships=False)
        types = types_from_state(state)
        return [Table._from_state(key, state, types) for key in sorted(state.datasets)]

    def semantic_type(self, name: str, *, include_pending: bool = False) -> Any:
        """One semantic type as a :class:`gloss.SemanticType`; ``name`` may be ``"crm::email_address"``."""
        from .modeling import SemanticType

        return SemanticType.load(self, name, include_pending=include_pending)

    def semantic_types(self, *, include_pending: bool = False) -> list[Any]:
        """All semantic types, sorted by name."""
        from .modeling import read_statuses, types_from_state

        types = types_from_state(self.snapshot(read_statuses(include_pending), datasets=[]))
        return [types[n] for n in sorted(types)]

    def relationships(self, table: str | None = None, *, include_pending: bool = False) -> list[Any]:
        """All complete relationships (optionally only those touching ``table``), sorted by name.

        A relationship whose accepted facts do not yet form a valid join is left out, not raised.
        """
        from .modeling import read_statuses, relationship_from_attrs

        state = self.snapshot(read_statuses(include_pending), datasets=[table] if table else [])
        out = []
        for ref in sorted(state.relationships):
            rel = relationship_from_attrs(ref, state.relationships[ref].attrs)
            if rel is not None and (table is None or table in (rel.from_table, rel.to_table)):
                out.append(rel)
        return self._with_versions(out)

    def relationship(self, name: str, *, include_pending: bool = False) -> Any:
        """One relationship as a :class:`gloss.Relationship`; ``name`` may be ``"sales::orders_customers"``."""
        from .modeling import read_statuses, relationship_from_attrs

        sid = SubjectId.for_relationship(name)
        rel = relationship_from_attrs(sid.ref, self._attrs(sid, read_statuses(include_pending)))
        if rel is None:
            raise KeyError(f"No complete relationship {name!r}"
                           + ("" if include_pending else " (pass include_pending=True to see proposals)"))
        return self._with_versions([rel])[0]

    def metrics(self, table: str | None = None, *, include_pending: bool = False) -> list[Any]:
        """Metrics as :class:`gloss.Metric`: ledger-wide and namespace ones, plus every table's own.

        With ``table`` only that table's metrics are returned.
        """
        from .modeling import metric_from_attrs, read_statuses

        state = self.snapshot(read_statuses(include_pending), datasets=[table] if table else None)
        out = []
        if table is None:
            for ref in sorted(state.metrics):
                m = metric_from_attrs(ref, state.metrics[ref].attrs)
                if m is not None:
                    out.append(m)
        for key in sorted(state.datasets):
            if table is not None and key != table:
                continue
            for name in sorted(state.datasets[key].metrics):
                m = metric_from_attrs(name, state.datasets[key].metrics[name].attrs, table=key)
                if m is not None:
                    out.append(m)
        return self._with_versions(out)

    def metric(self, name: str, *, table: str | None = None, namespace: str | None = None,
               include_pending: bool = False) -> Any:
        """One metric; give ``table`` for a table's own metric, ``namespace`` for a namespace's."""
        from .modeling import metric_from_attrs, read_statuses

        sid = SubjectId.for_metric(name, table, namespace=namespace)
        m = metric_from_attrs(sid.name if table else sid.ref, self._attrs(sid, read_statuses(include_pending)),
                              table=table)
        if m is None:
            where = f" on {table!r}" if table else ""
            raise KeyError(f"No metric {name!r}{where}"
                           + ("" if include_pending else " (pass include_pending=True to see proposals)"))
        return self._with_versions([m])[0]

    def _attrs(self, subject: SubjectId, statuses: Sequence[Status]) -> dict[str, Any]:
        """Best value per attribute of one subject (accepted wins, then the most recent)."""
        rows = self._redact(self.store.find_for_subjects([str(subject)], statuses, scope=self._scope))
        return {a.attribute: a.value for a in _best(rows).values() if not a.redacted}

    def _with_versions(self, objects: list[Any]) -> list[Any]:
        versions = self.store.subject_versions(str(o.subject) for o in objects)
        for o in objects:
            o.version = versions.get(str(o.subject))
        return objects

    def version(self, subject: str | SubjectId) -> int | None:
        """A number that grows whenever the accepted definition of ``subject`` may have changed.

        It is the sequence number of the last accepting, superseding, restoring, retracting, staling
        or re-keying event on the subject's facts; None when it never had an accepted fact.
        """
        s = str(SubjectId.parse(subject))
        return self.store.subject_versions([s]).get(s)

    # ------------------------------------------------------------------- read
    def _require(self, assertion_id: str) -> Assertion:
        a = self.store.get(assertion_id)
        if a is None and 4 <= len(assertion_id) < 32:  # allow unambiguous short ids, git-style
            matches = [m for m in self.store.by_id_prefix(assertion_id) if self._visible(m)]
            if len(matches) > 1:
                raise KeyError(f"Assertion id prefix {assertion_id!r} is ambiguous")
            a = matches[0] if matches else None
        if a is None or not self._visible(a):
            raise KeyError(f"No assertion {assertion_id!r}")
        return a

    def get(self, assertion_id: str) -> Assertion:
        return self._redact([self._require(assertion_id)])[0]

    def find(self, **criteria: Any) -> list[Assertion]:
        if criteria.get("statuses") is not None:
            criteria["statuses"] = [Status(s) for s in criteria["statuses"]]
        return self._redact(self.store.find(self._q(**criteria)))

    def count(self, **criteria: Any) -> int:
        """How many assertions match, for paging through ``find(limit=..., offset=...)``."""
        if criteria.get("statuses") is not None:
            criteria["statuses"] = [Status(s) for s in criteria["statuses"]]
        criteria.pop("limit", None)
        criteria.pop("offset", None)
        return self.store.count(self._q(**criteria))

    def pending(self, **criteria: Any) -> list[Assertion]:
        """Everything that needs a human: proposed, reviewed and stale facts."""
        criteria.setdefault("statuses", list(OPEN_STATUSES))
        return self.find(**criteria)

    def current(self, subject: str | SubjectId) -> dict[str, Any]:
        """Accepted value per attribute for one subject."""
        rows = self.find(subject=str(SubjectId.parse(subject)), statuses=[Status.ACCEPTED])
        return {a.attribute: a.value for a in rows if not a.redacted}

    def current_many(
        self, subjects: Iterable[str | SubjectId], *, attributes: Sequence[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Accepted values for many subjects in one lookup: ``{subject: {attribute: value}}``."""
        rows = self.store.find_for_subjects(
            {str(SubjectId.parse(s)) for s in subjects}, [Status.ACCEPTED], attributes=attributes, scope=self._scope
        )
        out: dict[str, dict[str, Any]] = defaultdict(dict)
        for a in self._redact(rows):
            if not a.redacted:
                out[a.subject][a.attribute] = a.value
        return dict(out)

    def effective(self, subject: str | SubjectId) -> dict[str, Any]:
        """Accepted values for one subject, including what a field inherits from its semantic type."""
        own = self.current(subject)
        tname = own.get("semantic_type")
        if not tname:
            return own
        t = self.current(SubjectId.for_type(tname))
        out = dict(own)
        for attr in inherited_attributes():
            if attr not in out and attr in t:
                out[attr] = t[attr]
        if t.get("checks"):
            out["checks"] = list(t["checks"]) + [c for c in own.get("checks", []) if c not in t["checks"]]
        return out

    def type_usages(self, type_name: str) -> list[str]:
        """Subjects of every field that (accepted) uses a semantic type."""
        ref = SubjectId.for_type(type_name).ref
        rows = self.store.find(self._q(attribute="semantic_type", value_hash=value_hash(ref),
                                       statuses=[Status.ACCEPTED]))
        return sorted(a.subject for a in rows)

    def record_check_results(self, results: Iterable[Any], *, runner: str = "system:checks",
                             run_id: str | None = None) -> str:
        """Log a run of data quality checks (see gloss.checks). Returns the run id.

        Pass the host's own ``run_id`` to tie the results to the run that produced them.
        """
        run_id = run_id or new_id()
        self.store.append_check_results([
            {
                "ts": r.ts, "subject": r.subject, "check": r.check, "passed": r.passed,
                "failed_rows": r.failed_rows, "total_rows": r.total_rows, "origin": r.origin,
                "error": r.error, "runner": str(runner), "run_id": run_id,
            }
            for r in results
        ])
        return run_id

    def check_history(self, subject: str | SubjectId | None = None, *, dataset: str | None = None,
                      run_id: str | None = None, limit: int | None = 200) -> list[dict[str, Any]]:
        """Logged check results, newest first, for one field, a whole dataset or one run."""
        prefix = f"field:{dataset}/" if dataset else None
        rows = self.store.check_results(str(SubjectId.parse(subject)) if subject else None, prefix, limit,
                                        run_id=run_id)
        if self._scope is not None:
            rows = [r for r in rows if self._scope.covers_subject(r["subject"])]
        return rows

    def history(self, subject: str | SubjectId) -> list[Event]:
        """Every event of the facts now at ``subject``, including those from before it was re-keyed."""
        sid = SubjectId.parse(subject)
        if self._scope is not None and not self._scope.covers(sid.scope_path):
            return []
        return self._redact_events(self.store.events_for_subject(str(sid)))

    def log(self, after_seq: int = 0, limit: int | None = None, *, correlation: str | None = None) -> list[Event]:
        return self._redact_events(
            self.store.events(after_seq=after_seq, limit=limit, correlation=correlation, scope=self._scope)
        )

    def progress(self, dataset_prefix: str | None = None) -> dict[str, dict[str, int]]:
        """Review progress per dataset or namespace: {path: {status: count, ...}}."""
        counts = self.store.count_by_status(self._q())
        out: dict[str, dict[str, int]] = defaultdict(lambda: {s.value: 0 for s in Status})
        for (path, st), n in counts.items():
            key = path or "(model-level)"
            if dataset_prefix and not key.startswith(dataset_prefix):
                continue
            out[key][st.value] += n
        return dict(sorted(out.items()))

    def snapshot(
        self,
        statuses: Iterable[Status] = (Status.ACCEPTED,),
        *,
        datasets: Iterable[str] | None = None,
        dataset_prefix: str | None = None,
        relationships: bool = True,
    ) -> ModelState:
        """The model as nested dicts, built from assertions in the given statuses.

        With ``datasets`` only those tables are loaded (plus relationships touching them, semantic
        types and model-level facts); with ``dataset_prefix`` those whose key starts with it. Both
        keep per-table lookups fast on big models.

        When several assertions for the same subject+attribute qualify, accepted wins,
        then the most recent one. On an actor-bound view the snapshot holds only what that
        actor may read, with details above their confidentiality level removed.
        """
        wanted = [Status(s) for s in statuses]
        state = ModelState()
        find = self.store.find
        if datasets is None and dataset_prefix is None:
            source_rows = find(self._q(statuses=wanted))
        else:
            source_rows = []
            keys = list(datasets or [])
            for k in keys:
                source_rows += find(self._q(dataset=k, statuses=wanted))
            if dataset_prefix is not None:
                source_rows += find(self._q(dataset_prefix=dataset_prefix, statuses=wanted))
            if relationships and datasets is not None:
                if keys:
                    rel_subjects = {
                        a.subject
                        for k in keys
                        for attr in DATASET_REFERENCES
                        for a in find(self._q(attribute=attr, value_hash=value_hash(k), statuses=wanted))
                    }
                    source_rows += self.store.find_for_subjects(rel_subjects, wanted, scope=self._scope)
                else:
                    source_rows += find(self._q(kind=Kind.RELATIONSHIP.value, statuses=wanted))
            elif relationships:
                source_rows += find(self._q(kind=Kind.RELATIONSHIP.value, statuses=wanted))
            source_rows += find(self._q(kind=Kind.TYPE.value, statuses=wanted))
            source_rows += find(self._q(kind=Kind.MODEL.value, statuses=wanted))
        # insert in creation order so fields keep the order their catalog/source gave them
        for a in sorted(_best(source_rows).values(), key=lambda x: x.created_at):
            state.add(SubjectId.parse(a.subject), a.attribute, a)
        if self._scope is not None and not self._scope.all_paths:
            state.relationships = {
                ref: r for ref, r in state.relationships.items()
                if all(self._scope.covers(r.attrs[side]) for side in DATASET_REFERENCES
                       if isinstance(r.attrs.get(side), str))
            }
        state.apply_types()
        if self._limit is not None:
            state.redact(self._limit)
        return state

    # ------------------------------------------------------------- maintenance
    def rebuild_projection(self) -> int:
        """Recompute every assertion's state (and the search index) from the event log."""
        states: dict[str, Assertion] = {}
        for ev in self.store.events():
            states[ev.assertion_id] = apply_event(states.get(ev.assertion_id), ev)
        self.store.reset_projection(states.values())
        self.rebuild_search_index()
        return len(states)

    # ----------------------------------------------------------------- search
    def rebuild_search_index(self) -> int:
        from .search_index import rebuild

        return rebuild(self)

    def search(
        self,
        text: str,
        *,
        kinds: Sequence[str] | None = None,
        dataset_prefix: str | None = None,
        limit: int = 20,
        for_ai: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Ranked search over accepted knowledge: names, descriptions, synonyms, semantic types.

        Filler words ("per", "the", "by" …) are ignored, prefixes count ("rev" finds "revenue") and words
        are stemmed ("emails" finds "email"). Columns are also found by their semantic type's synonyms.
        Hits matching *every* word come first; if there are none, hits matching *any* word are returned
        with ``matched`` listing the words they cover; if still nothing, words are looked for *inside*
        names ("mail" finds "useremail"); then misspelt words are corrected from the index's own
        vocabulary and hits carry ``did_you_mean``.

        Who is asking decides what comes back. On an actor-bound view the policy's scope applies:
        only subjects the actor may read, with items above their confidentiality level found by
        name but returned redacted. On an unbound ledger the AI limit applies unless
        ``for_ai=False`` (the safe default for a result that may reach a model).
        """
        from .search_index import INDEX_VERSION, content_tokens, correct_spelling, tokens

        shared = self._shared
        if not shared.search_ready:
            outdated = self.store.get_meta("search_index_version") != INDEX_VERSION
            if (outdated or self.store.search_is_empty()) and self.store.accepted_subjects():
                self.rebuild_search_index()
            elif outdated:
                self.store.set_meta("search_index_version", INDEX_VERSION)
            shared.search_ready = True

        limit_level = self.scope.max_confidentiality
        if for_ai or (for_ai is None and self._scope is None):
            ai_level = self.policy.scope("ai:anonymous", None).max_confidentiality
            limit_level = min(limit_level, ai_level, key=lambda c: c.rank)
        names_only = limit_level is not Confidentiality.RESTRICTED
        words = content_tokens(text)
        filters = {"kinds": kinds, "dataset_prefix": dataset_prefix, "scope": self._scope, "limit": limit}

        def run(ws: list[str], any_word: bool) -> list[dict[str, Any]]:
            return self.store.search(ws, max_level=limit_level.rank, match_any=any_word, **filters)

        # 1. every word; 2. any word (natural questions span several subjects)
        how, corrected = "all", None
        rows = run(words, False)
        if not rows and len(words) > 1:
            rows, how = run(words, True), "any"
        if not rows and words:  # 3. the words inside names ("mail" in "useremail")
            rows = self.store.search_names_containing(words, **filters)
            if rows:
                how = "within"
        if not rows and words:  # 4. spelling correction
            # limited callers get suggestions from names only: description words may be confidential
            vocab = self.store.search_vocabulary(names_only=names_only)
            fixed = [self._unstem(w, names_only=names_only) for w in correct_spelling(words, vocab)]
            if fixed != words:
                corrected = " ".join(fixed)
                rows, how = run(fixed, False), "all"
                if not rows and len(fixed) > 1:
                    rows, how = run(fixed, True), "any"
                words = fixed

        rows = self._drop_hidden_joins(rows)
        out = []
        for r in rows:
            hit: dict[str, Any] = {"subject": r["subject"], "kind": r["kind"], "name": r["name"]}
            if how == "any":  # tell the caller which words this hit actually covers
                visible = r["name_text"] + (" " + r["detail_text"] if r["level"] <= limit_level.rank else "")
                have = set(tokens(visible))
                # prefix either way, so plurals and stems count ("emails" ~ "email")
                hit["matched"] = [w for w in words if any(t.startswith(w) or (len(t) > 3 and w.startswith(t))
                                                          for t in have)]
            if corrected:
                hit["did_you_mean"] = corrected
            if r["dataset"]:
                hit["dataset"] = r["dataset"]
            if r["datatype"]:
                hit["datatype"] = r["datatype"]
            if r["semantic_type"]:
                hit["semantic_type"] = r["semantic_type"]
            if r["level"] > limit_level.rank:
                hit["confidentiality"] = Confidentiality.from_rank(r["level"]).value
                hit["redacted"] = True
            elif r["description"]:
                hit["description"] = r["description"]
            out.append(hit)
        return out

    def _drop_hidden_joins(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """A relationship is only shown to a reader who may read both tables it joins."""
        scope = self._scope
        joins = [r["subject"] for r in rows if r["kind"] == Kind.RELATIONSHIP.value]
        if scope is None or scope.all_paths or not joins:
            return rows
        hidden = {
            a.subject
            for a in self.store.find_for_subjects(joins, [Status.ACCEPTED], attributes=DATASET_REFERENCES)
            if isinstance(a.value, str) and not scope.covers(a.value)
        }
        return [r for r in rows if r["subject"] not in hidden]

    def _unstem(self, word: str, *, names_only: bool = True) -> str:
        """The index stores stems ("revenu"); show a real word that has this stem ("revenue")."""
        from .search_index import tokens

        level = Confidentiality.PUBLIC if names_only else Confidentiality.RESTRICTED
        rows = self.store.search([word], max_level=level.rank, limit=1, scope=self._scope)
        if not rows:
            return word
        text = rows[0]["name_text"] + ("" if names_only else " " + rows[0]["detail_text"])
        candidates = [t for t in tokens(text) if t.startswith(word)]
        return min(candidates, key=len) if candidates else word

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Gloss:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _best(rows: Iterable[Assertion]) -> dict[tuple[str, str], Assertion]:
    """One assertion per subject and attribute: accepted wins, then the most recently updated."""
    best: dict[tuple[str, str], Assertion] = {}
    for a in rows:
        k = (a.subject, a.attribute)
        cur = best.get(k)
        if cur is None or (a.status is not Status.ACCEPTED, -a.updated_at.timestamp()) < (
            cur.status is not Status.ACCEPTED,
            -cur.updated_at.timestamp(),
        ):
            best[k] = a
    return best


@dataclass
class Entry:
    """Attributes of one subject in a snapshot, with the assertions behind them.

    ``inherited`` maps attributes that came from a semantic type to that type's name;
    ``check_origins`` maps positions in ``attrs["checks"]`` to ``"type:<name>"``;
    ``redacted`` is the subject's confidentiality level when its details were withheld.
    """

    attrs: dict[str, Any] = field(default_factory=dict)
    assertions: dict[str, Assertion] = field(default_factory=dict)
    inherited: dict[str, str] = field(default_factory=dict)
    check_origins: dict[int, str] = field(default_factory=dict)
    redacted: str | None = None

    @property
    def own_attrs(self) -> dict[str, Any]:
        return {k: v for k, v in self.attrs.items() if k not in self.inherited}

    def set(self, attribute: str, a: Assertion) -> None:
        self.attrs[attribute] = a.value
        self.assertions[attribute] = a

    @property
    def all_accepted(self) -> bool:
        return all(a.status is Status.ACCEPTED for a in self.assertions.values())

    def level(self, inherited: Confidentiality | None = None) -> Confidentiality:
        raw = self.attrs.get("confidentiality")
        return Confidentiality(raw) if raw else inherited or Confidentiality.INTERNAL

    def withhold(self, level: Confidentiality, limit: Confidentiality) -> None:
        if level > limit:
            for attr in detail_attributes() & self.attrs.keys():
                del self.attrs[attr]
                self.assertions.pop(attr, None)
                self.inherited.pop(attr, None)
            self.redacted = level.value


@dataclass
class DatasetEntry(Entry):
    fields: dict[str, Entry] = field(default_factory=dict)
    metrics: dict[str, Entry] = field(default_factory=dict)


@dataclass
class ModelState:
    """A snapshot. Relationships, metrics and types are keyed by reference (``sales::revenue`` or a bare name)."""

    model: Entry = field(default_factory=Entry)
    models: dict[str, Entry] = field(default_factory=dict)  # per namespace
    datasets: dict[str, DatasetEntry] = field(default_factory=dict)
    relationships: dict[str, Entry] = field(default_factory=dict)
    metrics: dict[str, Entry] = field(default_factory=dict)
    types: dict[str, Entry] = field(default_factory=dict)

    def apply_types(self) -> None:
        """Let fields inherit from their semantic type (own facts win; checks are combined)."""
        inherit = inherited_attributes()
        for d in self.datasets.values():
            for f in d.fields.values():
                tname = f.attrs.get("semantic_type")
                t = self.types.get(tname) if tname else None
                if t is None:
                    continue
                for attr in inherit:
                    if attr not in f.attrs and attr in t.attrs:
                        f.attrs[attr] = t.attrs[attr]
                        f.inherited[attr] = tname
                type_checks = t.attrs.get("checks") or []
                if type_checks:
                    own = [c for c in f.attrs.get("checks") or [] if c not in type_checks]
                    f.attrs["checks"] = list(type_checks) + own
                    f.check_origins = {i: f"type:{tname}" for i in range(len(type_checks))}

    def redact(self, limit: Confidentiality) -> None:
        """Remove the details of every subject above ``limit``; names and structure stay."""
        levels: dict[str, Confidentiality] = {}
        for key, d in self.datasets.items():
            levels[key] = dl = d.level()
            d.withhold(dl, limit)
            for f in d.fields.values():
                f.withhold(f.level(dl), limit)
            for m in d.metrics.values():
                m.withhold(m.level(dl), limit)
        for r in self.relationships.values():
            ends = [levels[x] for x in (r.attrs.get("from"), r.attrs.get("to")) if x in levels]
            r.withhold(max([r.level(), *ends], key=lambda c: c.rank), limit)
        for m in self.metrics.values():
            m.withhold(m.level(), limit)

    def usages(self, type_name: str) -> list[SubjectId]:
        return [
            SubjectId.for_field(dk, fk)
            for dk, d in self.datasets.items()
            for fk, f in d.fields.items()
            if f.attrs.get("semantic_type") == type_name
        ]

    def dataset(self, key: str) -> DatasetEntry:
        return self.datasets.setdefault(key, DatasetEntry())

    def add(self, sid: SubjectId, attribute: str, a: Assertion) -> None:
        if sid.kind is Kind.MODEL:
            (self.models.setdefault(sid.namespace, Entry()) if sid.namespace else self.model).set(attribute, a)
        elif sid.kind is Kind.DATASET:
            self.dataset(sid.dataset or "").set(attribute, a)
        elif sid.kind is Kind.FIELD:
            self.dataset(sid.dataset or "").fields.setdefault(sid.name or "", Entry()).set(attribute, a)
        elif sid.kind is Kind.RELATIONSHIP:
            self.relationships.setdefault(sid.ref, Entry()).set(attribute, a)
        elif sid.kind is Kind.METRIC:
            if sid.dataset:
                self.dataset(sid.dataset).metrics.setdefault(sid.name or "", Entry()).set(attribute, a)
            else:
                self.metrics.setdefault(sid.ref, Entry()).set(attribute, a)
        elif sid.kind is Kind.TYPE:
            self.types.setdefault(sid.ref, Entry()).set(attribute, a)

    def iter_subjects(self) -> Iterator[tuple[SubjectId, Entry]]:
        yield SubjectId.model(), self.model
        for ns, m in self.models.items():
            yield SubjectId.model(ns), m
        for dk, d in self.datasets.items():
            yield SubjectId.for_dataset(dk), d
            for fk, f in d.fields.items():
                yield SubjectId.for_field(dk, fk), f
            for mk, m in d.metrics.items():
                yield SubjectId.for_metric(mk, dk), m
        for rk, r in self.relationships.items():
            yield SubjectId.for_relationship(rk), r
        for mk, m in self.metrics.items():
            yield SubjectId.for_metric(mk), m
        for tk, t in self.types.items():
            yield SubjectId.for_type(tk), t


def _subject_prefix_for(key: str) -> str:
    """The subject prefix that covers everything about ``key`` (a table key or a subject)."""
    if ":" not in key or key.split(":", 1)[0] not in {k.value for k in Kind}:
        return f"dataset:{key}"
    return str(SubjectId.parse(key))


def open(  # noqa: A001
    url: str = "sqlite:///gloss.db", *, policy: Policy | None = None, directory: ActorDirectory | None = None
) -> Gloss:
    """Open (and create if needed) a Gloss ledger at a SQLAlchemy URL."""
    return Gloss(SQLStore(url), policy=policy, directory=directory)
