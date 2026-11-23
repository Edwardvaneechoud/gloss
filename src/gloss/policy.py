"""The Policy seam: who may do what, what they may read, and what happens to a proposal.

Gloss does not authenticate anyone. The host (an application, the REST server, a
script) tells Gloss who the actor is; the policy decides three things:

* ``can``         may this actor take this action on this fact;
* ``on_propose``  is a new proposal queued for review, accepted at once, or refused;
* ``scope``       which part of the ledger this actor reads, and up to which confidentiality.

All three receive a :class:`Context`: how the actor came in (``via`` an agent or a
server) plus whatever the host wants its own policy to see. Gloss passes it through
untouched and records ``via`` on the events.

Review modes are a policy choice, not a schema choice, so a ledger can start solo on a
laptop and move to team review without a migration:

    DefaultPolicy(mode="solo")                                   # your own edits are accepted at once
    DefaultPolicy(mode="team", modes={"prod.": "four_eyes"})     # stricter where it matters
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .actors import KNOWN_KINDS, actor_kind
from .ids import SubjectId, path_covers
from .model import Assertion, Confidentiality, is_structural


class PermissionDenied(PermissionError):
    pass


# Actions an actor can take on an assertion.
PROPOSE = "propose"
REVIEW = "review"
ACCEPT = "accept"
REJECT = "reject"
RETRACT = "retract"
REOPEN = "reopen"
RESTORE = "restore"
MARK_STALE = "mark_stale"
REKEY = "rekey"
COMMENT = "comment"

# Review modes.
SOLO = "solo"  # a person's own proposals are accepted at once; everything else is queued
TEAM = "team"  # everything is queued; any person may accept, including the author
FOUR_EYES = "four_eyes"  # everything is queued; nobody accepts their own work
MODES = frozenset({SOLO, TEAM, FOUR_EYES})


def actor_type(actor: str) -> str | None:
    """The actor's kind, or None for a string without a ``kind:`` prefix (which has no rights)."""
    return actor_kind(actor)


@dataclass(frozen=True)
class Context:
    """How an actor reached Gloss. Opaque to the core; meaningful to the host's policy.

    ``via``    the channel: ``"ai:claude"`` for a person acting through an agent, ``"server:eu-1"``;
    ``attrs``  anything else the host's policy needs, e.g. ``{"sign_in": "direct"}``.
    """

    via: str | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def via_ai(self) -> bool:
        return bool(self.via) and actor_kind(self.via) == "ai"


@dataclass(frozen=True)
class Scope:
    """What one actor reads: a set of path prefixes and a confidentiality ceiling.

    ``prefixes`` are dotted paths in the space of dataset keys and namespaces; ``None`` means
    every path. ``include_global`` covers ledger-wide subjects (``model:``, ``type:email``).
    Details of subjects above ``max_confidentiality`` are withheld; their names stay findable.
    """

    prefixes: tuple[str, ...] | None = None
    include_global: bool = True
    max_confidentiality: Confidentiality = Confidentiality.RESTRICTED

    @classmethod
    def everything(cls) -> Scope:
        return cls()

    @classmethod
    def nothing(cls) -> Scope:
        return cls(prefixes=(), include_global=False, max_confidentiality=Confidentiality.PUBLIC)

    @property
    def all_paths(self) -> bool:
        return self.prefixes is None and self.include_global

    @property
    def unrestricted(self) -> bool:
        return self.all_paths and self.max_confidentiality is Confidentiality.RESTRICTED

    def covers(self, path: str | None) -> bool:
        if path is None:
            return self.include_global
        return self.prefixes is None or any(path_covers(p, path) for p in self.prefixes)

    def covers_subject(self, subject: str | SubjectId) -> bool:
        return self.covers(SubjectId.parse(subject).scope_path)

    def intersect(self, other: Scope) -> Scope:
        if self.prefixes is None:
            prefixes = other.prefixes
        elif other.prefixes is None:
            prefixes = self.prefixes
        else:
            kept: list[str] = []
            for a in self.prefixes:
                for b in other.prefixes:
                    if path_covers(a, b.rstrip(".")):
                        kept.append(b)
                    elif path_covers(b, a.rstrip(".")):
                        kept.append(a)
            prefixes = tuple(dict.fromkeys(kept))
        level = min(self.max_confidentiality, other.max_confidentiality, key=lambda c: c.rank)
        return Scope(prefixes, self.include_global and other.include_global, level)


@dataclass(frozen=True)
class Decision:
    """What becomes of a proposal: ``queue`` it, ``accept`` it at once, or ``refuse`` it.

    ``rule`` names why (``"solo"``, ``"catalog-structure"``) and is recorded on the accepted event,
    so the history shows that nobody reviewed it.
    """

    outcome: str
    rule: str | None = None

    @classmethod
    def of(cls, value: Decision | str | None) -> Decision:
        if isinstance(value, Decision):
            return value
        if value is None:
            return QUEUE
        if value not in ("queue", "accept", "refuse"):
            raise ValueError(f"on_propose must return 'queue', 'accept' or 'refuse', not {value!r}")
        return cls(value)


QUEUE = Decision("queue")


def accept_now(rule: str) -> Decision:
    return Decision("accept", rule)


def refuse(reason: str) -> Decision:
    return Decision("refuse", reason)


@runtime_checkable
class Policy(Protocol):
    def can(
        self, actor: str, action: str, assertion: Assertion | None = None, context: Context | None = None
    ) -> bool:
        """Whether ``actor`` may take ``action``. Called once per assertion in a batch: answer from memory."""

    def on_propose(
        self, actor: str, source: str, subject: SubjectId, attribute: str, context: Context | None = None
    ) -> Decision | str | None:
        """Queue (the default), accept at once, or refuse one proposal."""

    def scope(self, actor: str, context: Context | None = None) -> Scope:
        """What ``actor`` may read."""


def _own_work(actor: str, assertion: Assertion) -> bool:
    """True when accepting would mean approving your own statement with no second person involved."""
    stated = {assertion.created_by, *assertion.sources}
    if actor not in stated:
        return False
    return not any(actor_kind(a) == "human" for a in stated - {actor})


@dataclass
class DefaultPolicy:
    """Sensible defaults for one person or one team.

    * every known kind of actor may propose and comment; strings with no ``kind:`` prefix and
      unknown kinds may do nothing;
    * AI never reviews, accepts or rejects, whether it acts on its own or a person acts through it;
    * a source may retract what it stated itself;
    * catalog and system actors may mark facts stale, re-key subjects, and accept *structural*
      facts only (a catalog's structural proposals are accepted as it makes them);
    * ``mode`` (and ``modes`` per path prefix, longest prefix wins) sets how people's work is
      reviewed: ``solo``, ``team`` or ``four_eyes``;
    * AI reads details up to ``ai_max``; people up to ``human_max``.
    """

    mode: str = TEAM
    modes: Mapping[str, str] = field(default_factory=dict)
    ai_max: Confidentiality = Confidentiality.INTERNAL
    human_max: Confidentiality = Confidentiality.RESTRICTED
    machine_kinds: frozenset[str] = frozenset({"catalog", "system"})
    kinds: frozenset[str] = KNOWN_KINDS
    refuse_attributes: frozenset[str] = frozenset()  # e.g. {"examples"} where no data values may be held

    def __post_init__(self) -> None:
        for m in (self.mode, *self.modes.values()):
            if m not in MODES:
                raise ValueError(f"Unknown review mode {m!r} (use one of {sorted(MODES)})")

    def mode_for(self, path: str | None) -> str:
        best, mode = -1, self.mode
        for prefix, m in self.modes.items():
            if path_covers(prefix, path) and len(prefix) > best:
                best, mode = len(prefix), m
        return mode

    def _is_ai(self, actor: str, context: Context | None) -> bool:
        return actor_kind(actor) == "ai" or bool(context and context.via_ai)

    def may_decide(self, actor: str, assertion: Assertion | None, context: Context | None) -> bool:
        """Whether this person has a say over this fact at all. Subclasses narrow it."""
        return True

    def may_review(self, actor: str, assertion: Assertion | None, context: Context | None) -> bool:
        return self.may_decide(actor, assertion, context)

    def can(
        self, actor: str, action: str, assertion: Assertion | None = None, context: Context | None = None
    ) -> bool:
        kind = actor_kind(actor)
        if kind is None or kind not in self.kinds:
            return False
        if action in (PROPOSE, COMMENT):
            return True
        if action == RETRACT and assertion is not None and actor in assertion.sources:
            return True
        if self._is_ai(actor, context):
            return False
        if kind in self.machine_kinds:
            if action in (MARK_STALE, REKEY):
                return True
            if action == ACCEPT:
                return assertion is not None and is_structural(assertion.subject_id, assertion.attribute)
            return False
        if kind != "human":
            return False
        if action == REVIEW:
            return self.may_review(actor, assertion, context)
        if action not in (ACCEPT, REJECT, REOPEN, RETRACT, RESTORE, MARK_STALE, REKEY):
            return False
        if not self.may_decide(actor, assertion, context):
            return False
        if action == ACCEPT and assertion is not None and self.mode_for(assertion.scope_path) == FOUR_EYES:
            return not _own_work(actor, assertion)
        return True

    def on_propose(
        self, actor: str, source: str, subject: SubjectId, attribute: str, context: Context | None = None
    ) -> Decision:
        kind = actor_kind(actor)
        if attribute in self.refuse_attributes:
            return refuse(f"this ledger does not hold {attribute!r}")
        if kind == "catalog" and is_structural(subject, attribute):
            return accept_now("catalog-structure")
        if (
            kind == "human"
            and source == actor
            and not self._is_ai(actor, context)
            and self.mode_for(subject.scope_path) == SOLO
        ):
            return accept_now("solo")
        return QUEUE

    def readable_prefixes(self, actor: str, context: Context | None) -> tuple[str, ...] | None:
        """Path prefixes ``actor`` may read; ``None`` for all. Subclasses narrow it."""
        return None

    def scope(self, actor: str, context: Context | None = None) -> Scope:
        kind = actor_kind(actor)
        if kind is None or kind not in self.kinds:
            return Scope.nothing()
        level = Confidentiality.RESTRICTED
        if kind == "human":
            level = self.human_max
        if self._is_ai(actor, context):
            level = min(level, self.ai_max, key=lambda c: c.rank)
        return Scope(prefixes=self.readable_prefixes(actor, context), max_confidentiality=level)

    def ai_visibility(self) -> Confidentiality:
        """Highest confidentiality level whose details may be shown to AI tools."""
        return self.ai_max


@dataclass
class RolePolicy(DefaultPolicy):
    """Per-domain stewardship on top of the defaults.

    ``stewards`` maps a path prefix (dataset keys and namespaces) to the actors who decide there:
    accept, reject, retract, restore, mark stale and re-key. ``reviewers`` may move facts to
    *reviewed* but not accept them. ``admins`` may do anything a person can, everywhere.
    ``readers`` maps a path prefix to who may read it; leave it out and everyone reads everything.

        RolePolicy(
            admins={"human:edward"},
            stewards={"finance.": {"human:alice"}, "sales.": {"human:bob"}},
            reviewers={"human:carol"},
            readers={"finance.": {"human:carol"}},          # stewards and admins read their own
        )
    """

    admins: set[str] = field(default_factory=set)
    stewards: Mapping[str, Iterable[str]] = field(default_factory=dict)
    reviewers: set[str] = field(default_factory=set)
    readers: Mapping[str, Iterable[str]] | None = None

    def _decides_path(self, actor: str, path: str | None) -> bool:
        if actor in self.admins:
            return True
        return any(path_covers(prefix, path) and actor in set(who) for prefix, who in self.stewards.items())

    def may_decide(self, actor: str, assertion: Assertion | None, context: Context | None) -> bool:
        if assertion is None:
            return actor in self.admins
        if not self._decides_path(actor, assertion.scope_path):
            return False
        target = (context.attrs.get("to") if context else None) or None
        if target:  # a move needs a say over where the subject is going, too
            return self._decides_path(actor, SubjectId.parse(target).scope_path)
        return True

    def may_review(self, actor: str, assertion: Assertion | None, context: Context | None) -> bool:
        return actor in self.reviewers or self.may_decide(actor, assertion, context)

    def readable_prefixes(self, actor: str, context: Context | None) -> tuple[str, ...] | None:
        if self.readers is None or actor in self.admins or actor_kind(actor) in self.machine_kinds:
            return None
        found = [p for p, who in self.readers.items() if actor in set(who)]
        found += [p for p, who in self.stewards.items() if actor in set(who)]
        return tuple(dict.fromkeys(found))
