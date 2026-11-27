# Changelog

## 0.2.1 (2026-11-27)

Only additions to `glosslayer.checks`; the event format, subject grammar and database schema are unchanged.

- `checks.plan(items, dataset, frame)`: plan checks that are not in the ledger, such as a list being proposed.
- `plan` and `polars_plan` take `schema=`, the column names the host already knows, so the frame's own schema is
  not resolved in the calling process.
- `checks.run_plan(plan, collect)`: results through the host's own `collect`, one check at a time when the single
  pass fails; `run_polars` is built on it. A check's error is the first line of what the engine said.
- `checks.validate`, applied to every proposed `checks` value: a registered check whose parameters do not build
  or cannot work (`Check.verify`) and a check listed twice are refused. Unknown kinds are kept as before.
- A stored check that does not build (a pattern that no longer compiles) is an error of its own in a plan, not an
  exception for the whole table.

## 0.2.0 (2026-11-25)

Only additions: the event format, subject grammar and database schema are unchanged.

### Queries and data
- `glosslayer.query`: compile a metric by dimensions, filters and time grain into one SQL statement, following joins
  many to one; `check_metric` (compile and dry-run a proposal) and `dependents` (what stops compiling without a join).
- `glosslayer.joins`: measure a join's two sides (`ok`, `reversed`, `many_to_many`, `unchecked`), one or a whole
  model's, and `validate` one without asking the data.
- `glosslayer.rules`: suggest column roles, joins and metrics from types and one aggregate per table, and store them
  as proposals with their evidence, optionally in an AI's words.
- `glosslayer.dialect`: SQL quoting, literals and date truncation, and the contract of a host's `execute`.

### Reading and reviewing
- `Gloss.model_view`: a namespace's model with its review state beside it; `Gloss.facts`, `Gloss.review_queue`,
  `Gloss.metric_names`.
- `visibility.above`: datasets and columns above a level by the strongest live claim.
- `Gloss.withdraw`, `withdraw_subject`, `can_withdraw`, `clear` and `retract_left_out`.

## 0.1.0 (2026-11-24)

First public version. The event format, subject grammar and schema are at version 1; ledgers created by
earlier development snapshots are not supported.

### Model
- Facts (assertions) with their own review status in an append-only event log; projections rebuildable
  from the log.
- Subjects for models, datasets, fields, relationships, metrics and semantic types, each with an owning
  namespace (`metric:sales::revenue`).
- Review modes as policy: `solo`, `team`, `four_eyes` (per path), and stewarded domains (`RolePolicy`).
  Auto-accepted facts are marked with the rule that accepted them.
- Proposal groups: relationships and metrics are accepted or rejected as a unit. Versions per subject.
- Restore a superseded value. Re-key events for renamed or moved tables, columns, namespaces and types.
- An attribute registry (`register_attribute`); structured metric attributes and column roles.
- Semantic types with inheritance; data quality checks as facts, with a single-pass lazy Polars plan.

### Embedding
- Actor-bound views (`Gloss.as_actor`): reads scoped by path and confidentiality, writes made as the
  actor, host context passed to the policy, `via` and correlation ids recorded on events.
- No listeners, pragmas or DDL on a host's engine; `glosslayer.schema.upgrade(connection)` with a version
  ladder; writes can join a host transaction.
- Post-commit subscribers, a read hook for AI-facing views, plugins and attributes registered in code.
- Typed REST API, a client for it, and `Federation` to use several ledgers as one.

### Storage
- SQLite (FTS5) and Postgres (`tsvector`), both covered by the test suite.
- At most one accepted value per subject and attribute, enforced by a partial unique index.
- Search index updated in the same transaction as the events.

### Integrations
- Catalog sync with drift detection, partial listings (`coverage`), push events for single assets,
  rename detection and description write-back.
- Apache Ossie 0.3.0.dev0 import and export, validated against the vendored schema.
- MCP server, CLI, Mermaid and HTML reports, a git sink.
- `glosslayer-flowfile`: catalog, lineage source and backend glue for Flowfile.
