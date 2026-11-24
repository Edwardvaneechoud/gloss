# Changelog

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
