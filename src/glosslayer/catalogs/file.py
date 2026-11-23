"""Built-in catalog backed by a YAML file. Useful on its own, for tests, and as a reference plugin.

    name: demo
    assets:
      sales.orders:
        source: warehouse.sales.orders
        description: One row per order
        primary_key: [order_id]
        columns:
          - {name: order_id, dtype: Int64}
          - {name: amount, dtype: "Decimal(18,2)"}
          - {name: created_at, dtype: "Datetime(time_unit='us', time_zone='UTC')"}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..seams import AssetSchema, Column, asset_fingerprint


class FileCatalog:
    """A catalog read from one YAML file in the format above; ``name`` defaults to the file's ``name:``."""

    def __init__(self, path: str | Path, name: str | None = None):
        self.path = Path(path)
        self._data = self._load()
        self.name = name or self._data.get("name") or f"file:{self.path.stem}"

    @classmethod
    def from_spec(cls, arg: str, **kwargs: Any) -> FileCatalog:
        """``arg`` is the YAML file's path."""
        return cls(arg, **kwargs)

    def _load(self) -> dict[str, Any]:
        data = yaml.safe_load(self.path.read_text()) or {}
        if not isinstance(data.get("assets", {}), dict):
            raise ValueError(f"{self.path}: 'assets' must be a mapping of dataset key -> definition")
        return data

    def reload(self) -> None:
        """Read the file again; the catalog keeps what it loaded until then."""
        self._data = self._load()

    def list_assets(self) -> list[str]:
        """The dataset keys under ``assets``, in file order."""
        return list((self._data.get("assets") or {}).keys())

    def describe(self, key: str) -> AssetSchema:
        """The asset as loaded; an ``id`` in its entry becomes the ``uid`` a sync uses to spot renames."""
        spec = (self._data.get("assets") or {})[key] or {}
        cols = [
            Column(
                name=c["name"],
                dtype=str(c.get("dtype", "")),
                nullable=c.get("nullable"),
                description=c.get("description"),
            )
            for c in spec.get("columns", [])
        ]
        return AssetSchema(
            key=key,
            columns=cols,
            description=spec.get("description"),
            source=spec.get("source"),
            primary_key=spec.get("primary_key"),
            uid=str(spec["id"]) if spec.get("id") is not None else None,
        )

    def fingerprint(self, key: str) -> str:
        """``asset_fingerprint`` of the described asset: columns, descriptions, primary key and source."""
        return asset_fingerprint(self.describe(key))

    def write_back(self, key: str, facts: dict[str, Any]) -> None:
        """Write accepted descriptions back into the YAML file."""
        assets = self._data.setdefault("assets", {})
        spec = assets.setdefault(key, {})
        if "description" in facts:
            spec["description"] = facts["description"]
        for col in spec.get("columns", []):
            f = (facts.get("fields") or {}).get(col["name"])
            if f and "description" in f:
                col["description"] = f["description"]
        self.path.write_text(yaml.safe_dump(self._data, sort_keys=False, allow_unicode=True))
