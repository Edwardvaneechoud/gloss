"""Wiring Gloss into the Flowfile backend: identity, permissions and catalog events.

Nothing here imports Flowfile. The host hands over plain values it already has (a user's id,
the namespace paths they manage), so this module is tested without a running Flowfile and the
core stays free of host imports.

    semantics = Embedded(g, catalog)

    # startup, and whenever in doubt
    semantics.sync()

    # from Flowfile's own table paths, right where the change is made
    semantics.table_written("main.sales.orders")
    semantics.table_renamed("main.sales.orders", "main.sales.order_lines")
    semantics.table_deleted("main.sales.orders")

    # per request
    view = semantics.view(user_id=user.id, admin=user.is_admin, manage=["main.sales"])
    view.pending()
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from gloss import Gloss, Human
from gloss.actors import actor_kind, split_actor
from gloss.ids import path_covers
from gloss.model import Assertion
from gloss.policy import SOLO, TEAM, Context, DefaultPolicy
from gloss.sync import SyncReport, asset_moved, asset_removed, sync, sync_asset, write_back


def actor_for(user_id: Any) -> Human:
    """The actor for a Flowfile user.

    Pass an id that never changes for that person. Flowfile's integer user id is stable on one
    machine; a ledger shared between machines needs an id every machine agrees on.
    """
    return Human(str(user_id))


@dataclass
class FlowfilePolicy(DefaultPolicy):
    """Flowfile's access rules as a Gloss policy, answered from memory.

    The host resolves a user's rights **once per request** and passes them in the view's context
    (``Embedded.view`` does this):

    * ``admin``   may decide and read everywhere;
    * ``manage``  namespace paths the user manages: there they accept, reject, retract, re-key;
    * ``read``    namespace paths the user may read; leave it out where the catalog is not private.

    Desktop mode has one user and no sharing: construct with ``mode="solo"`` and every table is
    theirs to decide.
    """

    def _attrs(self, context: Context | None) -> Mapping[str, Any]:
        return context.attrs if context else {}

    def _manages(self, context: Context | None, path: str | None) -> bool:
        attrs = self._attrs(context)
        if attrs.get("admin") or self.mode == SOLO:
            return True
        return any(path_covers(p, path) for p in attrs.get("manage") or ())

    def may_decide(self, actor: str, assertion: Assertion | None, context: Context | None) -> bool:
        """Admins and solo mode decide all; others must manage the fact's path (and a re-key's target)."""
        if assertion is None:
            return bool(self._attrs(context).get("admin")) or self.mode == SOLO
        if not self._manages(context, assertion.scope_path):
            return False
        target = self._attrs(context).get("to")
        if target:
            from gloss.ids import SubjectId

            return self._manages(context, SubjectId.parse(target).scope_path)
        return True

    def readable_prefixes(self, actor: str, context: Context | None) -> tuple[str, ...] | None:
        """A person's ``read`` plus ``manage`` paths; all for admins, non-humans or an unset ``read``."""
        attrs = self._attrs(context)
        if actor_kind(actor) != "human" or attrs.get("admin") or attrs.get("read") is None:
            return None
        return tuple(dict.fromkeys([*attrs["read"], *(attrs.get("manage") or ())]))


def policy_for(*, sharing: bool) -> FlowfilePolicy:
    """``sharing=False`` for the desktop app (one user, own edits accepted at once), else team review."""
    return FlowfilePolicy(mode=TEAM if sharing else SOLO)


class Directory:
    """Shows actors by name, using a lookup the host provides (``user id -> display name or None``)."""

    def __init__(self, lookup: Callable[[str], str | None]):
        self._lookup = lookup

    def display_name(self, actor: str) -> str | None:
        """A person's name from the host's lookup; other actors show their id."""
        kind, ident = split_actor(actor)
        return self._lookup(ident) if kind == "human" else ident


class Embedded:
    """The ledger plus the Flowfile catalog, with the calls the backend makes."""

    def __init__(self, gloss: Gloss, catalog: Any, *, own_descriptions: bool = True):
        """``own_descriptions`` keeps Flowfile's table description in step with the accepted one."""
        self.gloss = gloss
        self.catalog = catalog
        self._stop = write_back(gloss, catalog) if own_descriptions else None

    def close(self) -> None:
        """Stop writing accepted descriptions back to the catalog; safe to call twice."""
        if self._stop is not None:
            self._stop()
            self._stop = None

    # -- catalog events: run as the catalog's own principal, over everything ---------
    def sync(self) -> SyncReport:
        """Pull sync of the whole catalog: at startup, and whenever events may have been missed."""
        return sync(self.gloss, self.catalog)

    def table_written(self, key: str) -> SyncReport:
        """Call after Flowfile creates or overwrites a table."""
        return sync_asset(self.gloss, self.catalog, key)

    def table_renamed(self, old_key: str, new_key: str) -> SyncReport:
        """Call from Flowfile's table update path on a rename or move, so the table's facts follow it.

        Flowfile emits no table-change event, so a rename the host does not report looks like
        drop + add to a pull sync.
        """
        self.catalog.tables(refresh=True)
        return asset_moved(self.gloss, self.catalog, old_key, new_key)

    def table_deleted(self, key: str) -> SyncReport:
        """Call from Flowfile's table delete path: the table's facts are marked stale."""
        return asset_removed(self.gloss, self.catalog, key)

    # -- per request -----------------------------------------------------------------
    def view(
        self,
        *,
        user_id: Any,
        admin: bool = False,
        manage: Iterable[str] = (),
        read: Iterable[str] | None = None,
        via: str | None = None,
        **context: Any,
    ) -> Gloss:
        """A ledger view for one signed-in user, with their rights resolved by the host."""
        attrs = {"admin": admin, "manage": tuple(manage), "read": None if read is None else tuple(read), **context}
        return self.gloss.as_actor(actor_for(user_id), via=via, context=attrs)
