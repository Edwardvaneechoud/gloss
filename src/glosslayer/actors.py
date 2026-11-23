"""Typed actors: who stated or did something.

    from glosslayer import Human, AI

    g.add(table, source=Human("edward"))
    g.accept(assertion_id, actor=Human("alice"))
    g.propose(..., source=AI("claude"))

Actors are ``str`` subclasses (``Human("edward") == "human:edward"``), so they work
everywhere a plain actor string does: storage, policies, the REST API, logs.

The part after the kind is an *identifier*, not a display name. Actors are written
into the event log, which is never edited, so a host with accounts should pass an id
that never changes and means the same on every machine (``Human("7f3a9c…")``), and
resolve it to a name when showing it: see :class:`ActorDirectory`. A person who has
left then resolves to nobody, and the log names no one.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable


class Actor(str):
    """Base of the typed actors: the string is ``"<kind>:<name>"`` and ``name`` is the id after the colon."""

    kind: ClassVar[str] = ""
    name: str

    def __new__(cls, name: str) -> Actor:
        if cls is Actor:
            raise TypeError("Use a concrete actor (Human, AI, Import, ...) or Actor.parse('kind:name')")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{cls.__name__} needs a non-empty name")
        prefix = f"{cls.kind}:"
        if name.startswith(prefix):  # Human("human:edward") is fine too
            name = name[len(prefix) :]
        obj = super().__new__(cls, prefix + name)
        obj.name = name
        return obj

    def __getnewargs__(self) -> tuple[str]:  # keeps copy/pickle from double-prefixing
        return (self.name,)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    @staticmethod
    def parse(value: str) -> Actor:
        """``"human:edward"`` → ``Human("edward")``. Unknown kinds become :class:`Other`.

        A string without a ``kind:`` prefix is refused: guessing that it is a person would hand
        an unknown caller a person's rights.
        """
        if isinstance(value, Actor):
            return value
        kind, name = split_actor(value)
        cls = _KINDS.get(kind)
        return cls(name) if cls else Other(value)


def split_actor(value: str) -> tuple[str, str]:
    """``"human:alice"`` → ``("human", "alice")``; raises ``ValueError`` for any other shape."""
    if not isinstance(value, str):
        raise ValueError(f"An actor is a string like 'human:alice', not {type(value).__name__}")
    kind, sep, name = value.partition(":")
    if not sep or not kind or not name.strip():
        raise ValueError(f"Invalid actor {value!r}: expected '<kind>:<id>', e.g. 'human:alice' or 'ai:claude'")
    return kind, name


def actor_kind(value: str) -> str | None:
    """The kind of a well-formed actor string, or None when it has no ``kind:`` prefix."""
    try:
        return split_actor(value)[0]
    except ValueError:
        return None


class Human(Actor):
    """A person. The default policy lets humans review and accept."""

    kind = "human"


class AI(Actor):
    """An AI model or agent. It can propose, but the default policy never lets it accept."""

    kind = "ai"


Agent = AI


class Import(Actor):
    """A file or tool the facts were imported from, e.g. ``Import("ossie")``, ``Import("dbt")``."""

    kind = "import"


class Catalog(Actor):
    """A catalog plugin acting during sync, e.g. ``Catalog("flowfile")``."""

    kind = "catalog"


class Inferred(Actor):
    """A rule that derived the fact, e.g. ``Inferred("flowfile-lineage")``."""

    kind = "inferred"


class System(Actor):
    """Gloss itself or an automation acting on its own authority."""

    kind = "system"


class Other(Actor):
    """An actor string whose kind Gloss doesn't know; kept verbatim."""

    kind = ""

    def __new__(cls, value: str) -> Other:
        obj = str.__new__(cls, value)
        obj.name = value
        return obj

    def __getnewargs__(self) -> tuple[str]:
        return (str(self),)


_KINDS: dict[str, type[Actor]] = {
    c.kind: c for c in (Human, AI, Import, Catalog, Inferred, System)
}

KNOWN_KINDS = frozenset(_KINDS)


@runtime_checkable
class ActorDirectory(Protocol):
    """How a host turns actor ids into something to show. Gloss never stores the answer."""

    def display_name(self, actor: str) -> str | None:
        """The name to show for ``actor``, or None when the host no longer knows them."""


class IdentityDirectory:
    """The default: the id is the name (fine for scripts, the CLI and single-user ledgers)."""

    def display_name(self, actor: str) -> str | None:
        return split_actor(actor)[1] if actor_kind(actor) else actor
