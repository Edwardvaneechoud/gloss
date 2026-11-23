"""Flowfile catalog seam.

Two ways to connect:

* **Embedded** (inside Flowfile's backend): pass a Flowfile ``CatalogService`` built *without*
  an access resolver, so the listing is the whole local catalog. No HTTP.

      FlowfileCatalog(service=CatalogService(repo))

* **Remote** (standalone Gloss, CLI): talk to a running Flowfile core over REST.

      gloss sync flowfile:http://localhost:63578      # token from FLOWFILE_TOKEN

Dataset keys are Flowfile's fully qualified ``catalog.schema.table`` names, which is also how
Flowfile itself addresses a table across machines.

A listing made for one user (a service with an access resolver, or a user's token) only shows
what that user may see. Such a catalog reports ``Coverage(complete=False)``, so a sync through it
discovers and updates tables and never concludes that a table it cannot see is gone.

Tables that belong to a connected hub are skipped: their meaning lives in the hub's ledger.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from typing import Any

from gloss.seams import AssetSchema, Column, Coverage, asset_fingerprint


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from a pydantic model (embedded) or a dict (REST)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def table_key(table: Any) -> str:
    return _get(table, "qualified_name") or _get(table, "full_table_name") or _get(table, "name")


def is_remote(table: Any) -> bool:
    """A row Flowfile shows from a connected hub (it carries ``source="hub"`` and a negated id)."""
    tid = _get(table, "id")
    return _get(table, "source", "local") != "local" or (isinstance(tid, int) and tid < 0)


class FlowfileCatalog:
    def __init__(
        self,
        *,
        service: Any | None = None,
        fetch_tables: Callable[[], list[Any]] | None = None,
        name: str = "flowfile",
        include_virtual: bool = True,
        complete: bool | None = None,
    ):
        """``complete`` says the listing is the whole local catalog.

        By default that is assumed only for an embedded service without an access resolver.
        """
        if fetch_tables is None:
            if service is None:
                raise ValueError("Pass a Flowfile CatalogService (embedded) or use FlowfileCatalog.from_spec(url)")
            fetch_tables = service.list_tables
            if complete is None:
                complete = getattr(service, "access", None) is None
        self._fetch = fetch_tables
        self._service = service
        self.name = name
        self.include_virtual = include_virtual
        self.complete = bool(complete)
        self._tables: dict[str, Any] | None = None

    # -- construction -------------------------------------------------------
    @classmethod
    def from_spec(cls, arg: str, **kwargs: Any) -> FlowfileCatalog:
        """``flowfile:<base-url>``; the bearer token comes from ``FLOWFILE_TOKEN``.

        A token is one user's view, so the listing is not taken as complete unless ``complete=True``.
        """
        base = arg.rstrip("/") if arg else "http://localhost:63578"
        token = kwargs.pop("token", None) or os.environ.get("FLOWFILE_TOKEN")

        def fetch() -> list[dict[str, Any]]:
            req = urllib.request.Request(f"{base}/catalog/tables")
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (user-supplied URL by design)
                return json.loads(resp.read())

        return cls(fetch_tables=fetch, **kwargs)

    # -- Catalog protocol ---------------------------------------------------
    def tables(self, refresh: bool = False) -> dict[str, Any]:
        if self._tables is None or refresh:
            out = {}
            for t in self._fetch():
                if is_remote(t):
                    continue
                if not self.include_virtual and _get(t, "table_type", "physical") != "physical":
                    continue
                out[table_key(t)] = t
            self._tables = out
        return self._tables

    def coverage(self) -> Coverage:
        return Coverage(complete=self.complete)

    def list_assets(self) -> list[str]:
        return list(self.tables(refresh=True))

    def describe(self, key: str) -> AssetSchema:
        tables = self.tables()
        if key not in tables:
            tables = self.tables(refresh=True)
        t = tables[key]
        cols = [Column(name=_get(c, "name"), dtype=str(_get(c, "dtype", ""))) for c in _get(t, "schema_columns") or []]
        tid = _get(t, "id")
        extra = {
            "flowfile_table_id": tid,
            "table_type": _get(t, "table_type", "physical"),
            "produced_by_flow": _get(t, "source_registration_name") or _get(t, "producer_registration_name"),
            "row_count": _get(t, "row_count"),
        }
        return AssetSchema(
            key=key,
            columns=cols,
            description=_get(t, "description"),
            source=key,
            extra={k: v for k, v in extra.items() if v is not None},
            # Flowfile's ids can be reused after a delete, so sync only trusts them together with
            # unchanged columns; report renames directly where possible (gloss.sync.asset_moved).
            uid=str(tid) if tid is not None else None,
        )

    def fingerprint(self, key: str) -> str:
        return asset_fingerprint(self.describe(key))

    def write_back(self, key: str, facts: dict[str, Any]) -> None:
        """Show accepted dataset descriptions in Flowfile's catalog (embedded mode only).

        Flowfile stores no description per column, so field descriptions stay in Gloss.
        """
        if self._service is None:
            raise NotImplementedError("write_back needs embedded mode (a CatalogService)")
        if "description" in facts:
            self._service.update_table(_get(self.tables()[key], "id"), description=facts["description"])
