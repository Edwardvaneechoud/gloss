# CLAUDE.md

Guidance for working on Gloss. Read this before changing code.

## What this is

Gloss is a catalog-agnostic semantic layer built on Apache Ossie. It stores semantic metadata as small
**assertions** (one fact about one subject), each with its own review status, in an **append-only event
log**. Valid Ossie documents are generated from accepted facts. Catalogs, importers, exporters and AI
connect through plugin seams. The primary host is Flowfile; Gloss must stay usable without it, and
nothing host-specific belongs in `src/glosslayer`.

## Commands

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]" -e plugins/glosslayer-flowfile    # library + CLI + extras + the Flowfile plugin
pytest -q                                            # everything, incl. README examples (~5 s)
GLOSS_TEST_DATABASE_URL=postgresql+psycopg://user:pass@host/db pytest -q     # the same suite on Postgres
ruff check src plugins tests --select E,F,W,B --ignore E501,B008
python scripts/bench.py --tables 2000                # performance; use 10000 for real numbers
gloss --db sqlite:///demo.db sync file:examples/catalog.yaml   # quick manual check
```

No Postgres at hand: `pip install pgserver` gives a throwaway one (`pgserver.get_server(tmpdir).get_uri()`).

## Map

| Area | Files |
|---|---|
| Subjects (`field:shop.orders/amount`, `type:crm::email`, …), namespaces, path prefixes | `ids.py` |
| Assertions, events, statuses, transitions, the attribute registry, `apply_event` | `model.py` |
| The facade hosts use (`Gloss`): propose, transitions, groups, re-key, add/apply, readers, search, snapshot, actor-bound views, subscribe | `ledger.py` |
| Policy, `Scope`, `Context`, review modes, `RolePolicy` | `policy.py` |
| Storage (SQLAlchemy; SQLite + Postgres), transactions, search queries | `store/sql.py`, `store/base.py` |
| Schema creation and version ladder for hosts with migrations | `schema.py` |
| Object API: `Table`, `Column`, `Relationship`, `Metric`, `SemanticType` | `modeling.py` |
| Actors, precise dtypes, checks, type suggester | `actors.py`, `dtypes.py`, `checks.py`, `semantic.py` |
| Metric queries (compile, dry-run, what depends on a join), join measurement, rule suggestions | `query.py`, `joins.py`, `rules.py` |
| SQL text helpers and the contract of `execute` | `dialect.py` |
| The model with its review state beside it, the review queue | `modelview.py` |
| Confidentiality levels; which facts mention a column or table | `visibility.py`, `references.py` |
| Search index contents and incremental updates | `search_index.py` |
| Catalog sync, drift, renames, write-back | `sync.py`, `catalogs/file.py` |
| Ossie import/export, vendored JSON Schema | `ossie/` |
| AI views, MCP, REST, client + federation, CLI, visualisation | `ai.py`, `mcp_server.py`, `server.py`, `client.py`, `cli.py`, `viz.py` |
| Plugin protocols, registration and discovery | `seams.py`, `plugins.py` (+ entry points in `pyproject.toml`) |
| Flowfile plugin (separate package) | `plugins/glosslayer-flowfile/` |

Design rationale: `docs/architecture.md`.

## Invariants (do not break)

1. **The event log is append-only.** Never update or delete rows in `gloss_events`. Every state change
   is an `Event` folded by `model.apply_event`. `gloss_assertions` and `gloss_search` are projections:
   `Gloss.rebuild_projection()` must reproduce them exactly (`test_rebuild_projection_matches`,
   `test_rebuild_matches_incremental`, `test_rekey_is_rebuilt_from_the_log`).
2. **What is written into events is forever.** Subject strings, actor strings and event `data` keys cannot
   be changed once a ledger exists. A rename is a `rekeyed` event, never an edit. Changing the subject
   grammar (`ids.py`) or an event's shape needs a new event type or a new optional key, not a rewrite.
3. **All status changes go through `Gloss.transition_many`**, and all appends through `Gloss._commit`,
   which writes events, projection and search rows in one transaction and then tells subscribers.
   Don't call `store.append` from outside `ledger.py`.
4. **Sources only propose; the policy decides acceptance.** Importers, lineage, the type suggester and AI
   create `proposed` assertions. Auto-acceptance happens only when `Policy.on_propose` says so, and the
   accepted event then carries `{"auto": true, "rule": …}`. The default rules: a catalog actor's
   *structural* facts, and a person's own edits where the mode is `solo`. Never accept on a source's
   behalf anywhere else. Decide on the **actor**, not the source: callers choose `source` freely.
5. **AI never reviews or accepts** (`DefaultPolicy`), including a person acting `via` an agent.
   Everything AI-facing goes through `glosslayer.ai` or a view from `Gloss.as_actor`. Never hand an unbound
   ledger's raw snapshots or assertions to an AI tool.
6. **Reads are made as someone.** A `Gloss` from `as_actor` filters every read by `Policy.scope` (paths
   pushed into SQL, details above the reader's level redacted). A new reader must take its rows through
   `self._q(...)` / `scope=self._scope` and `self._redact(...)`; add a case to `tests/test_scoped_reads.py`.
   The unbound ledger is the host's own trusted handle.
7. **Gloss never touches a host's database uninvited.** No connection listeners, pragmas or DDL on an
   engine it was given; DDL only in `schema.upgrade`, only on `gloss_` tables. A schema change after a
   release is a new step in `schema._STEPS` plus a bumped `LATEST`.
8. **The core imports no catalog or tool.** Nothing in `src/glosslayer` may import Flowfile or any other host.
   Plugins depend on the core, never the reverse. Optional dependencies (polars, fastapi, mcp,
   jsonschema) are imported lazily inside functions.
9. **Exports must validate** against `ossie/ossie-schema.json`. Gloss-only data goes in
   `nonportable_extensions` with `vendor_name: gloss`. Re-importing an export must not duplicate facts.
10. **Build subjects with `SubjectId` helpers**, not string concatenation; compare paths with
    `ids.path_covers` (per segment), never `startswith`.
11. **Write paths stay batched.** Use `propose_many` / `transition_many` / `current_many`; no per-fact
    queries in loops, and `Policy.can` is called per fact so it must not query. A new query pattern on
    `gloss_assertions` needs a composite index. Check with `EXPLAIN QUERY PLAN` **after** `PRAGMA optimize`
    (statistics change the plan): an `ORDER BY` or a `LIKE 'prefix%'` makes SQLite scan, which is why
    `store.find` sorts in Python when unpaged and prefixes go through `SQLStore._starts`.
12. **One accepted value per subject and attribute.** The partial unique index
    `uq_gloss_assertions_accepted` enforces it; `store.append` writes states leaving `accepted` before
    states entering it. A violation surfaces as `ConcurrentWrite` and the ledger retries once.

13. **Gloss writes SQL and never runs it.** Anything that needs data takes the host's `execute` (`dialect.Execute`)
    and tells `QueryFailed` from `EngineUnavailable`; an unavailable engine never refuses anything. Names in
    compiled SQL are always quoted with `dialect.quote` (not `references.quote_ident`), values go through
    `dialect.literal`, and free SQL comes only from a stored metric. Rules read types and statistics, never names.
14. **Taking facts away goes through `Gloss.withdraw`** (and `withdraw_subject`, `clear`, `retract_left_out` on top
    of it): retract or reject by who stated the fact, all or nothing.

## Adding things

- **A new attribute:** `model.register_attribute(...)` in `_register_builtins` (hosts call the same
  function for their own). The flags decide the rest: `inherited` (fields take it from their semantic
  type), `detail` (hidden above the reader's level), `searchable` (indexed: in `detail_text` when
  `detail`, else `name_text`), `ossie` (has a native place in Ossie; otherwise it travels in the
  extension automatically), `structural` (a catalog may accept it). If Ossie has a native place, map it in
  the exporter and importer bodies. Add it to the relevant `modeling.py` dataclass and tests. If it changes
  what goes into index rows, bump `search_index.INDEX_VERSION`.
- **A new check:** subclass `checks.Check`, set `kind`, implement `polars()` and/or `sql()` (both return
  a *pass* condition; nulls pass unless the check is about nulls), decorate with `@register`. Add it to the
  Polars and DuckDB parametrised tests in `tests/test_semantic_types.py`.
- **A new plugin:** a plain class matching a protocol in `seams.py`, registered with
  `plugins.register(kind, name, cls)` or under the `gloss.catalogs` / `gloss.sources` / `gloss.sinks`
  entry points. Built-ins are also listed in `plugins._BUILTINS`. A catalog whose listing can be partial
  must implement `coverage()`.
- **A new action or event type:** add it to `policy.py` (constant + `DefaultPolicy.can`), to
  `model.EventType`, `EVENT_TARGET_STATUS` and `EVENT_FROM_STATUSES`, and to `ledger._ACTION_EVENTS`.
  Decide whether it changes accepted knowledge (`_reindex_after`, `store.VERSION_EVENTS`).
- **A new REST route:** give it a pydantic response model, depend on `ledger`, and mirror it in
  `client.Client`. **A new CLI command:** `cmd_<name>` plus a subparser in `cli.build_parser`; add a test
  that calls `main([...])`.

## Conventions

- Python 3.10+, `from __future__ import annotations`, type hints, dataclasses, line length 110.
- Docstrings carry the rationale. No comment banners and no multi-line `#` paragraphs; an inline comment
  is one short line.
- Errors: `PermissionDenied` (policy), `InvalidTransition` (status), `KeyError` (unknown id/subject, and
  anything outside the reader's scope: never reveal that it exists), `ValueError` (bad input),
  `ConcurrentWrite` (lost a race; retry), `schema.SchemaNotReady`. The CLI maps these to exit code 2 with
  a one-line message; the REST API to 403 / 422 / 404 / 422 / 409.
- Actors are `str` subclasses (`Human("x") == "human:x"`); store and compare them as plain strings. The
  part after the colon is an id, not a display name (`ActorDirectory` resolves names). A string without a
  `kind:` prefix is invalid everywhere.
- Timestamps are UTC (`model.utcnow`). SQLite drops tzinfo, and `store/sql._aware` restores it.
- Tests use the in-memory `g` fixture (`tests/conftest.py`) and the `examples/` data. `glosslayer.open("sqlite://")`
  in a test is redirected to a fresh Postgres schema when `GLOSS_TEST_DATABASE_URL` is set, so write tests
  that hold on both. The search tests run on the database's full-text search and the plain fallback.
- **The README's Python examples are executed by `tests/test_readme.py`.** If you change behaviour or
  output, update the README (including the `# …` output comments) and the assertions in that test.
  Mark host-dependent blocks with `<!-- not-tested -->` on the line before the code fence.

## Known gaps and gotchas

- Targets Ossie `0.3.0.dev0`, where extensions are `nonportable_extensions` (older drafts called them
  `custom_extensions`) and expressions are `{sql: …}` or `{dialects: [...]}`.
- **Export names:** dataset keys like `crm.orders` export as `orders`, but as `crm_orders` when two
  tables share a short name (same for namespaced relationships and metrics). Metric expressions written
  against `orders.…` then break. A per-dataset Ossie name override is not built yet.
- **Structured metric attributes have no native place in Ossie yet.** `aggregate`, `filter`, `time_column`,
  `dimensions`, `format` and a column's `role` travel in the extension. `query.compile_query` compiles a metric that
  belongs to a table; a namespace's metric has no rows to compute over and is refused.
- `glosslayer.query` and `glosslayer.rules` have no REST routes, MCP tools or CLI commands yet: they need a host's
  `execute`, which a served ledger does not have.
- An edit of a definition that leaves a part out only takes effect at once where the edit is accepted at once
  (`retract_left_out`). A queued edit keeps the accepted part, also after it is accepted later.
- Column renames are only known when reported (`Gloss.rekey` on a field). A pull sync sees a renamed
  column as drop + add; table renames are detected by catalog id plus unchanged columns.
- Drift detection of metric references is a word-match heuristic on SQL text (`references.py`): it can
  send a metric back to review, it never rewrites one.
- `mcp_server.py` supports MCP SDK 2.x (`MCPServer`) and 1.x (`FastMCP`).
- SQL regex functions differ per dialect (DuckDB `regexp_matches`, Postgres `~`, Snowflake `REGEXP_LIKE`,
  BigQuery `REGEXP_CONTAINS`). Only DuckDB is verified in tests.
- `glosslayer-flowfile` is tested against a fake `CatalogService` shaped like Flowfile's `CatalogTableOut`,
  not a running Flowfile. Flowfile has no event for table changes: its update and delete paths must call
  `Embedded.table_renamed` / `table_deleted`.
- Search index contents are versioned (`search_index.INDEX_VERSION`). Bump it whenever what goes into index
  rows or the FTS tokenizer changes: ledgers rebuild their index on the first search after upgrading.
- Spelling suggestions for limited readers come from the `name_text` vocabulary only; never widen that to
  detail text (it could reveal words from confidential descriptions). On Postgres the vocabulary is read
  with `ts_stat` over the whole index, which is fine on the rare no-results path and nowhere else.
- Postgres appends take a transaction-level advisory lock, so `seq` order is commit order and a consumer
  tailing `log(after_seq)` misses nothing. It also means one writer at a time per database.
- SQLite write contention relies on the default busy timeout; `SQLStore.using(connection)` lets a host
  with its own `BEGIN IMMEDIATE` transactions include Gloss's writes.
- Check results live outside the event log (`gloss_check_results`) and have no retention yet.
- `Federation` routes by path prefix and merges listings; it does not de-duplicate or rank across ledgers
  beyond interleaving search hits.
