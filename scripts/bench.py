"""Benchmark Gloss on a synthetic catalog.

    python scripts/bench.py --tables 2000            # quick
    python scripts/bench.py --tables 10000 --db /tmp/big.db

Reports first sync, resync, drift, pending, bulk accept, search, per-table AI context and export.
Use it before and after changes to storage, sync, search or the ledger's write paths.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import gloss
from gloss import Human, ai
from gloss.ossie import to_ossie
from gloss.seams import AssetSchema, Column, schema_fingerprint
from gloss.sync import sync


class SyntheticCatalog:
    name = "bench"

    def __init__(self, n: int, cols: int = 10, changed: bool = False):
        self.n, self.cols, self.changed = n, cols, changed

    def list_assets(self) -> list[str]:
        return [f"db.t{i}" for i in range(self.n)]

    def describe(self, key: str) -> AssetSchema:
        cols = [Column(f"c{j}", "Int64") for j in range(self.cols)]
        if self.changed and key.endswith("7"):
            cols.append(Column("new_col", "Utf8"))
        return AssetSchema(key, cols, description=f"table {key}")

    def fingerprint(self, key: str) -> str:
        return schema_fingerprint(self.describe(key).columns)


def timed(label: str, fn, reps: int = 1):
    ts, result = [], None
    for _ in range(reps):
        t = time.perf_counter()
        result = fn()
        ts.append(time.perf_counter() - t)
    value = statistics.median(ts)
    unit, shown = ("ms", value * 1000) if value < 1 else ("s", value)
    print(f"{label:<42} {shown:9.2f} {unit}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tables", type=int, default=2000)
    p.add_argument("--db", default="/tmp/gloss-bench.db")
    args = p.parse_args()
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(args.db + suffix):
            os.remove(args.db + suffix)

    g = gloss.open(f"sqlite:///{args.db}")
    n = args.tables
    print(f"{n} tables x 10 columns -> {args.db}")
    timed("first sync", lambda: sync(g, SyntheticCatalog(n)))
    timed("resync, unchanged", lambda: sync(g, SyntheticCatalog(n)))
    timed("resync, 10% changed", lambda: sync(g, SyntheticCatalog(n, changed=True)))
    timed("list pending", lambda: g.pending(), reps=5)
    timed("bulk accept descriptions", lambda: g.bulk("accept", gloss.Query(attribute="description"), actor=Human("bench")))
    timed("search 'c7' (matches every table)", lambda: ai.search(g, "c7"), reps=30)
    timed("search 'db t42'", lambda: ai.search(g, "db t42"), reps=30)
    timed("search, no match", lambda: ai.search(g, "zebra"), reps=30)
    timed("AI context, one table", lambda: ai.model_context(g, datasets=["db.t42"]), reps=30)
    timed("export Ossie", lambda: to_ossie(g))
    timed("rebuild search index", g.rebuild_search_index)
    g.close()


if __name__ == "__main__":
    main()
