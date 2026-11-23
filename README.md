# Gloss

**A catalog-agnostic semantic layer built on [Apache Ossie](https://github.com/apache/ossie).**
Track, review, search and serve the meaning of your data, with a full history of every change.

> Apache Ossie defines *what the data means*: datasets, fields, relationships, metrics.
> Gloss manages *what you know about those definitions*: who proposed them, how sure they are, whether
> they've been reviewed, how confidential they are, and what changed when.

Gloss is a Python library you embed in your own app (it's built to live inside [Flowfile](https://github.com/Edwardvaneechoud/Flowfile)),
or a small service you run next to any catalog. Catalogs, importers, exporters and AI agents connect
through plugins.

```python
import glosslayer
from glosslayer import Table, Column, SemanticType, Human, Int, String, checks

g = glosslayer.open("sqlite:///gloss.db")

Email = SemanticType("email_address", dtype=String(320), confidentiality="restricted",
                     checks=[checks.matches(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")])

customers = Table("crm.customers", primary_key=[Column("customer_id", Int(64))])
customers.add_columns(Column("email", Email), Column("country", String(2)))

g.add(customers, source=Human("edward"))     # proposals, waiting for review
```

---

**Contents:**
[Why](#why) ·
[Concepts](#concepts) ·
[Install](#install) ·
[Five-minute tour (CLI)](#five-minute-tour-cli) ·
[End-to-end example](#end-to-end-example) ·
[Python guide](#python-guide) ·
[Semantic types and checks](#semantic-types-and-checks) ·
[Catalogs and drift](#catalogs-and-drift) ·
[Search](#search) ·
[AI agents](#ai-agents) ·
[Embedding](#embedding) ·
[Ossie interop](#ossie-interop) ·
[Writing plugins](#writing-plugins) ·
[Policies and review modes](#policies-and-review-modes) ·
[Renames and moves](#renames-and-moves) ·
[Performance](#performance) ·
[Project layout](#project-layout) ·
[Roadmap](#roadmap)

## Why

With 10,000 tables there is no single "the semantic model". There are hundreds of thousands of small
facts, each in its own state: some described and accepted, some drafted by an AI and waiting for review,
some imported from an old model, some made stale by last night's schema change.

Gloss treats each fact as a unit with its own review status and an append-only history, and builds a
valid Ossie model from the facts that are accepted. You get:

- **Review per fact.** Accept a column's description without accepting the whole table.
- **Full history and provenance:** who stated what, when, from which source, how confident.
- **Many sources, reconciled.** Catalog, imports, lineage, people and AI. A second source stating the same
  value *confirms* it instead of creating a duplicate, and rejected suggestions don't come back.
- **Drift handling.** When a column is dropped or changes type, the facts that depend on it go back to review.
  When a table is renamed or moved, what you know about it moves with it.
- **One model, any size of team.** A single user whose own edits are accepted at once, a team that
  reviews each other, or stewards per domain: the same ledger, a different policy.
- **Reusable semantic types** (an e-mail address, a country code), defined once and used by any number of columns.
- **Data quality checks** stored as facts, run on Polars or as SQL, with logged results.
- **Safe AI access:** fast indexed search and context for agents, confidentiality enforced. AI can
  propose but never accept.
- **Built to be embedded.** It uses the host's database without touching the host's settings, the host's
  logins through one policy object, and every read is made as someone.

## Concepts

**Assertion**: one fact about one subject.

```
subject      field:shop.orders/amount
attribute    description
value        "Order value including VAT, in EUR"
source       human:alice | ai:claude | import:ossie | catalog:flowfile | inferred:type-names
confidence   0.8
status       proposed → reviewed → accepted …
evidence     [{"upstream": "raw.orders", "rule": "…"}]
```

**Subjects** are plain strings that don't depend on any catalog:

| Subject | Example |
|---|---|
| the model | `model:`, or `model:sales` for the model one namespace publishes |
| a dataset (table, view) | `dataset:shop.orders` |
| a field | `field:shop.orders/amount` |
| a relationship | `relationship:orders_customers`, or `relationship:sales::orders_customers` |
| a metric | `metric:total_revenue`, `metric:sales::revenue`, or `metric:shop.orders/avg_basket` for a table's own |
| a semantic type | `type:email_address`, or `type:crm::email_address` |

Dataset keys are dotted paths (`catalog.schema.table`). A **namespace** (`sales::…`) is a path in the same
space and says who owns a relationship, metric or type, so two teams can each define `revenue`, and one
prefix rule (`sales.`) covers access and stewardship for everything under it.

**Lifecycle**, per assertion:

```mermaid
stateDiagram-v2
    [*] --> proposed
    proposed --> reviewed
    proposed --> accepted
    reviewed --> accepted
    proposed --> rejected
    reviewed --> rejected
    rejected --> proposed: reopen
    accepted --> superseded: a newer value is accepted
    superseded --> accepted: restore
    accepted --> stale: the data changed
    proposed --> stale
    stale --> accepted: re-confirmed
    stale --> rejected
    accepted --> retracted
```

**The log is the truth.** Every change is an event appended to `gloss_events` and never edited. The
current state, and the search index, are projections you can rebuild from the log at any time
(`gloss rebuild`).

**Actors** say who did something: `Human("alice")`, `AI("claude")`, `Import("dbt")`, `Inferred("…")`,
`System("…")`. They're strings underneath (`Human("alice") == "human:alice"`). The part after the colon is
an *id*: actors are written into the log, which is never edited, so an application with accounts passes an
id that never changes (`Human("7f3a9c")`) and resolves it to a name when showing it.

**Groups.** Proposals made together share a group. For relationships and metrics a group is accepted or
rejected as a unit, because half a join is not a join.

## Install

```bash
git clone <this repo> gloss && cd gloss
pip install -e ".[dev]"           # library, CLI, REST, MCP, validation, Polars checks, tests
# or pick extras: pip install -e ".[server,mcp,validate,checks,postgres]"
```

Python 3.10+. The core needs only SQLAlchemy and PyYAML. The ledger is a SQLAlchemy URL:
`sqlite:///gloss.db` (default) or `postgresql+psycopg://…`. Both are tested: the suite runs on SQLite and,
with `GLOSS_TEST_DATABASE_URL` set, on Postgres.

## Five-minute tour (CLI)

The `examples/` folder has a small shop catalog and an Ossie model for it.

```bash
export GLOSS_ACTOR=human:you          # who you are; every action is logged under this name
export GLOSS_MODE=team                # or `solo`: your own edits are accepted without a second step
```

**1. Connect a catalog.** Gloss discovers tables and columns and accepts the structure automatically.

```
$ gloss sync file:examples/catalog.yaml
catalog shop: 2 new, 0 changed, 0 gone, 0 unchanged | 38 proposed, 37 accepted, 0 marked stale
```

**2. Import an existing model.** Its definitions arrive as *proposals*. Where it agrees with the catalog,
that's recorded as confirmation.

```
$ gloss import ossie examples/semantic_model.yaml --key-by source
48 facts from import:ossie: 22 corroborated, 26 created
```

**3. See what needs review.**

```
$ gloss progress
dataset         accepted  reviewed  proposed     stale  rejected  progress
(model-level)          0         0        13         0         0  ..........   0.0%
shop.customers        17         0         7         0         0  #######...  70.8%
shop.orders           20         0         7         0         0  #######...  74.1%

$ gloss pending --dataset shop.customers
842f0012  proposed   dataset:shop.customers  description = One row per registered customer   [import:ossie]
d6294b5a  proposed   field:shop.customers/email  confidentiality = restricted   [import:ossie]
bf30cbfd  proposed   field:shop.customers/email  description = Login e-mail address   [import:ossie]
…
```

**4. Review.** One by one (short ids are fine), or in bulk with filters.

```bash
gloss accept d6294b5a bf30cbfd --comment "checked with the DPO"
gloss reject 842f0012 --comment "also includes guest checkouts"
gloss accept --all --source import:ossie
```

**5. Search and export.**

```
$ gloss search turnover
field:shop.orders/amount  Order value including VAT, in EUR
metric:total_revenue  Sum of order value including VAT, excluding cancelled orders

$ gloss search email --as-ai
field:shop.customers/email [redacted]

$ gloss export --validate -o model.yaml
valid Ossie document
```

**6. The catalog changes.** A week later `status` was dropped and `customer_id` became a string:

```
$ gloss sync file:examples/catalog_v2.yaml
catalog shop: 0 new, 1 changed, 0 gone, 1 unchanged | 5 proposed, 5 accepted, 12 marked stale
  - column 'status' removed from shop.orders
  - type of shop.orders.customer_id changed from Integer to String

$ gloss pending --status stale --kind metric
d53eb71b  stale      metric:total_revenue  expression = {"sql": "SUM(orders.amount) FILTER (WHERE orders.status <> …
…
```

The revenue metric used the dropped column, so it went back for review, along with the join on the retyped column.

**7. Everything is history.**

```
$ gloss history field:shop.orders/status
#30     2026-11-23 17:42:58  proposed      6a7af4f3  catalog:shop     field:shop.orders/status expression = {"sql": "status"}
#31     2026-11-23 17:42:58  accepted      6a7af4f3  catalog:shop     by rule: catalog-structure
#93     2026-11-23 17:42:58  corroborated  6a7af4f3  human:you        also stated by import:ossie
#95     2026-11-23 17:42:58  proposed      33a518ac  human:you        field:shop.orders/status description = Fulfilment status (…)
#136    2026-11-23 17:42:59  accepted      33a518ac  human:you
#151    2026-11-23 17:43:00  marked_stale  33a518ac  catalog:shop     column 'status' removed from shop.orders
```

**8. A table is renamed.** Tell Gloss, and what is known about it follows:

```
$ gloss rename dataset:shop.customers dataset:shop.clients
6 subject(s) re-keyed: 24 facts moved, 1 references rewritten, 0 superseded, 0 marked stale
```

Other commands: `gloss show <subject>`, `gloss restore <id>`, `gloss types`, `gloss check`,
`gloss viz --html report.html`, `gloss serve`, `gloss mcp`, `gloss plugins`. Run `gloss --help` for the full list.

## End-to-end example

[`examples/e2e_shop.py`](examples/e2e_shop.py) tells one complete story over three weeks, with a data
engineer, a data steward and an AI agent working on the same shop data:

1. connect a catalog, so structure is accepted automatically;
2. add meaning in code: two semantic types, descriptions, a join, a metric;
3. the steward reviews and accepts;
4. Gloss suggests the types for columns nobody described yet;
5. an AI agent searches the model over MCP (restricted columns come back redacted) and proposes improvements;
6. the steward accepts one and rejects one, and the rejected one can't come back;
7. checks run on real data, with results logged;
8. the accepted model is exported as valid Ossie YAML;
9. the catalog changes: a column is dropped and another changes type, so dependent facts go stale;
10. the steward resolves them;
11. the full audit trail of one column;
12. an HTML report.

```bash
python examples/e2e_shop.py ./demo      # leaves demo/gloss.db, demo/shop.yaml, demo/report.html
```

## Python guide

The examples in this section run in order. They are executed by the test suite (`tests/test_readme.py`),
so they stay correct.

### Model in code

Objects are *descriptions*. Saving one proposes a fact per attribute, so they get the same review and
log as everything else.

```python
import glosslayer
from glosslayer import Table, Column, Relationship, Metric, SemanticType, Human, checks
from glosslayer import Int, Decimal, String, DateTime

g = glosslayer.open("sqlite:///gloss.db")

Email = SemanticType(
    "email_address",
    dtype=String(320),
    description="An e-mail address",
    confidentiality="restricted",
    classification=["pii.email"],
    synonyms=["e-mail", "mail address"],
    checks=[checks.matches(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"), checks.length(max=320)],
    name_patterns=[r"e[-_]?mail"],            # lets Gloss suggest this type for matching columns
)

customers = Table(
    "crm.customers",
    description="One row per customer account",
    primary_key=[Column("customer_id", Int(64))],   # Column objects are added to the table
)
customers.add_columns(
    Column("email", Email, description="Login e-mail address"),
    Column("country", String(2), checks=[checks.matches(r"^[A-Z]{2}$")]),
    Column("signed_up_at", DateTime(tz="UTC")),
)

orders = Table("crm.orders", description="One row per order", primary_key=[Column("order_id", Int(64))])
orders.add_columns(
    Column("customer_id", Int(64)),
    Column("amount", Decimal(18, 2), description="Order value incl. VAT, in EUR", synonyms=["revenue", "turnover"]),
    Column("status", String(16), checks=[checks.one_of(["open", "shipped", "cancelled"])]),
)
orders.add_metric(Metric("revenue", "SUM(orders.amount)", description="Total order value", datatype=Decimal))

result = g.add(
    customers,
    orders,
    Relationship("orders_customers", on={orders["customer_id"]: customers["customer_id"]}),  # tables inferred
    source=Human("edward"),
)
print(result)                                  # 40 new, 0 unchanged
```

Types can be precise Gloss types (`Int(32)`, `UInt(8)`, `Float(64)`, `Decimal(18, 2)`, `String(255)`,
`DateTime(tz="UTC")`, `Opaque("geometry")`), Ossie names (`"Integer"`), catalog names (`"Int64"`,
`"varchar(20)"`) or Python types (`int`, `datetime.date`). The Ossie datatype is derived from them, and
the exact type is kept as `physical_type`.

Saving the same objects again does nothing. Change one attribute and only that attribute is proposed.
`g.add` never removes anything; for a model kept in code, `g.apply(...)` also retracts what you stated
before and no longer declare (a column you deleted from the `Table`).

`Relationship(..., namespace="sales")`, `Metric(..., namespace="sales")` and
`SemanticType(..., namespace="crm")` put a definition in a namespace. A metric can also carry its
structured parts next to the SQL (`aggregate`, `filter`, `time_column`, `dimensions`, `format`), and a
column its `role` (`key`, `time`, `category`, `amount`), for hosts that build metric queries.

### Review

```python
from glosslayer import AI, PermissionDenied

print(len(g.pending()))                        # 40

amount = g.find(subject="field:crm.orders/amount", attribute="description")[0]
g.accept(amount.id, actor=Human("alice"), comment="matches the finance definition")

idea = g.propose("field:crm.orders/status", "description", "Fulfilment status",
                 source=AI("claude"), confidence=0.6)
try:
    g.accept(idea.id, actor=AI("claude"))       # AI may propose, never accept
except PermissionDenied as exc:
    print(exc)                                 # ai:claude may not accept assertion 17aa…
g.reject(idea.id, actor=Human("alice"), comment="it also covers returns")

done, failed = g.bulk("accept", glosslayer.Query(source_prefix="human:edward"), actor=Human("alice"))
print(len(done))                               # 39
```

`g.review(...)`, `g.retract(...)`, `g.reopen(...)` and `g.comment(...)` work the same way. Accepting a
value automatically supersedes the previously accepted one, and `g.restore(old_id, ...)` puts a superseded
value back. Accepting one fact of a relationship's or metric's proposal group accepts the group
(`g.accept_group(group, ...)` says so explicitly); if one part cannot be accepted, none is.

Who may do what is the policy's decision: see [Policies and review modes](#policies-and-review-modes).

### Read back

```python
t = g.table("crm.orders")                      # the accepted table as objects
print(t)          # Table('crm.orders', columns=[order_id:Int64, customer_id:Int64, amount:Decimal(18,2), status:String(16)])
print(repr(t["amount"].dtype))                 # Decimal(18, 2)

g.current("field:crm.customers/email")         # what this field itself states (accepted)
email = g.effective("field:crm.customers/email")   # … plus what it inherits from its semantic type
print(email["confidentiality"])                # restricted

for e in g.history("field:crm.orders/status"):
    print(e.seq, e.type.value, e.actor)        # 42 proposed ai:claude / 43 rejected human:alice / …

g["crm.orders"]                                # same as g.table("crm.orders")
g["field:crm.customers/email"]                 # a Column; also "type:…", "relationship:…", "metric:…"

# everything else as objects too
print([t.key for t in g.tables()])             # ['crm.customers', 'crm.orders']
print(g.semantic_types()[0].name)              # email_address
print(g.relationships()[0].from_table)         # crm.orders
print([(m.table, m.name) for m in g.metrics()])    # [('crm.orders', 'revenue')]
```

All readers return **accepted** facts. Facts you only proposed (`g.add(..., source=...)`) are waiting for
review, so pass `include_pending=True` to see them as well:
`g.table(key, include_pending=True)`, `g.tables(prefix, …)`, `g.semantic_type(name, …)`,
`g.semantic_types(…)`, `g.relationship(name, …)`, `g.relationships(table, …)`, `g.metric(name, table=…, …)`,
`g.metrics(table, …)`.

Relationships and metrics carry a `version` when read: a number that grows whenever their accepted
definition changes (`g.version(subject)` for anything else), so an answer computed from a metric can say
which definition it used.

## Semantic types and checks

A **semantic type** is a reusable meaning stored once as `type:<name>` and referenced by any number of fields.

- A field **inherits** what it doesn't state itself: datatype, description, confidentiality, classification,
  synonyms… The type's **checks are added** to the field's own.
- **Change the type once**, accept it, and every field that uses it follows. A pending change has no effect.
- **At scale**, `TypeSuggester` (CLI: `gloss import semantic-types`) proposes the type for every column whose
  name matches the type's `name_patterns` and whose datatype fits.

```bash
gloss import semantic-types                       # suggest types by column name, across all tables
gloss accept --all --source inferred:type-names
gloss types                                       # each type, its checks, and where it is used
```

**Checks** are declarations stored as facts, so they're reviewed and logged too: `not_null()`, `unique()`,
`matches(regex)`, `one_of([...])`, `between(min, max)`, `length(min, max)`, `sql("{column} >= 0")`, or your
own (see [Writing plugins](#writing-plugins)). Run them on a Polars DataFrame:

```python
import polars as pl

df = pl.DataFrame({
    "customer_id": [1, 2, 3],
    "email": ["ann@example.com", "not-an-email", None],
    "country": ["NL", "nl", "BE"],
})
results = glosslayer.checks.run_polars(g, "crm.customers", df)
for r in results:
    print(r)
# FAIL  field:crm.customers/email     matches(pattern='^[^@\\s]+@…')  1/3 rows fail  (from type:email_address)
# PASS  field:crm.customers/email     length(max=320)                 0/3 rows fail  (from type:email_address)
# FAIL  field:crm.customers/country   matches(pattern='^[A-Z]{2}$')   1/3 rows fail

g.record_check_results(results, runner=Human("edward"))     # quality history per field
history = g.check_history(dataset="crm.customers")

# For engines that never collect in the calling process: one lazy query counts every failure.
plan = glosslayer.checks.polars_plan(g, "crm.customers", df.lazy())
row = plan.frame.collect()                     # ship plan.frame to where the data is, collect it there
results = plan.results(row)                    # the same CheckResults, from that one row
g.record_check_results(results, runner="system:worker-1", run_id="run-2026-11-23-001")   # your own run id
```

Or generate SQL to run in your warehouse (`ansi`, `postgres`, `duckdb`, `snowflake`, `bigquery`):

```python
queries = glosslayer.checks.to_sql(g, "crm.customers", dialect="duckdb")
subject, check, sql = queries[0]               # each query returns failed_rows and total_rows
```

From the CLI: `gloss check crm.customers --data customers.parquet --record`, or `--sql --dialect postgres`.
Failing sample values are never shown for confidential or restricted fields, never stored, and never part
of a result's wire form (`CheckResult.to_dict()`).

## Catalogs and drift

Catalogs are plugins. The built-in `file` catalog reads a YAML file (see `examples/catalog.yaml`).
For Flowfile, see [`plugins/glosslayer-flowfile`](plugins/glosslayer-flowfile).

```python
from glosslayer.catalogs.file import FileCatalog
from glosslayer.sync import sync

report = sync(g, FileCatalog("examples/catalog.yaml"))
print(report.new)                              # ['shop.orders', 'shop.customers']

# shop.customers came from the catalog without meaning; let Gloss suggest semantic types
suggested = g.propose_many(glosslayer.TypeSuggester().proposals(g), source=glosslayer.TypeSuggester.source_id)
print([(r.assertion.subject, r.assertion.value) for r in suggested])
# [('field:shop.customers/email', 'email_address')]
```

On each sync Gloss compares each table's fingerprint with the last one it saw:

| Change in the catalog | What Gloss does |
|---|---|
| new table or column | structure (exists, datatype, physical type, expression) accepted as the catalog states it; descriptions only *proposed* |
| column removed | all its facts, and relationships and metrics that reference it, become **stale** |
| logical type changed (Integer → String) | new type recorded; the field's meaning and dependent joins and metrics become stale |
| precise type changed (Int32 → Int64) | new physical type recorded; nothing needs review |
| description or primary key changed | proposed again; nothing becomes stale |
| table renamed or moved | its facts are re-keyed and joins rewritten; nothing is lost (see [Renames and moves](#renames-and-moves)) |
| table gone | everything about it becomes stale |

Data-only changes don't affect the fingerprint, so they never trigger reviews.

**"Not listed" is not "gone".** A catalog says what its listing covers (`coverage()`). A listing filtered
by what one user may see is not complete, so a sync through it discovers and updates tables but never
concludes that a table it cannot see was removed. Run sync as the application itself, over everything.

**Push instead of pull.** A catalog that knows the moment something changes reports just that:

<!-- not-tested -->
```python
from glosslayer.sync import sync_asset, asset_moved, asset_removed, write_back

sync_asset(g, catalog, "shop.orders")                         # created, or its columns changed
asset_moved(g, catalog, "shop.orders", "shop.order_lines")    # renamed, or moved to another schema
asset_removed(g, catalog, "shop.orders")                      # deleted

stop = write_back(g, catalog)     # Gloss owns meaning: accepted descriptions flow back into the catalog
```

## Search

Accepted knowledge is kept in a search index that updates whenever a fact is accepted, superseded, marked
stale or retracted, including every field that inherits from a changed semantic type or table. The index
is updated in the same transaction as the events that change it.

```python
hits = g.search("turnover")
print(hits[0]["subject"], hits[0]["description"])  # field:crm.orders/amount Order value incl. VAT, in EUR

g.search("rev", kinds=["metric"])              # prefixes count; narrow by kind or dataset_prefix
print(g.search("email", dataset_prefix="crm.")[0])
# {'subject': 'field:crm.customers/email', 'kind': 'field', 'name': 'email', 'dataset': 'crm.customers',
#  'datatype': 'String', 'semantic_type': 'email_address', 'confidentiality': 'restricted', 'redacted': True}
```

How a query is answered:

1. Filler words are dropped ("what was the revenue **per** country" → revenue, country). Prefixes count,
   words are stemmed ("emails" finds `email`), and columns are also found by their semantic type's
   synonyms ("mail" finds the `email` columns through the type's "e-mail").
2. Items matching **every** word, ranked: exact name, then names starting with the query, then relevance
   (names weigh more than descriptions).
3. If nothing matches every word, items matching **any** word, each with `matched` listing which of your
   words it covers. Natural questions usually span several items, like a metric plus a column.
4. If still nothing, the words are looked for *inside* names ("mail" finds `useremail`).
5. Then misspelt words are corrected from the index's own vocabulary, and hits carry `did_you_mean`
   ("revnue" → revenue, "fmail" → mail, "cusotmer" → customer). For AI callers, suggestions only come from
   names, so a suggestion can't reveal a word from a confidential description.

What it doesn't do (yet) is match by *meaning*: "contact" won't find an e-mail column unless "contact" is
one of its synonyms. Adding synonyms to a semantic type fixes that for every column using it.

Who asks decides what comes back. On a view bound to an actor (`g.as_actor(...)`, see
[Embedding](#embedding)) the policy's scope applies: only what that actor may read, and items above their
confidentiality level are found by name but neither searched by description nor returned with one. On the
plain ledger object, which is the application's own handle, results are limited to what AI may see unless
you pass `for_ai=False`. Semantic type *definitions* stay readable; the columns that use a restricted type
are what's redacted.

On SQLite search uses FTS5; on Postgres `tsvector` with GIN indexes. Both stem words and feed the
spelling suggestions. Any other database falls back to plain matching.

## AI agents

### MCP

```bash
gloss --db sqlite:///gloss.db mcp --agent claude
```

Claude Desktop / Claude Code config:

```json
{"mcpServers": {"gloss": {"command": "gloss", "args": ["--db", "sqlite:////path/to/gloss.db", "mcp", "--agent", "claude"]}}}
```

| Tool | What it does |
|---|---|
| `search_model(text, limit, kinds)` | indexed search over accepted knowledge |
| `get_model_context(datasets)` | the accepted model (or some tables) as compact JSON, confidential fields redacted |
| `review_status(subject)` | every fact about a subject and its status, so the agent knows what is agreed |
| `propose_fact(subject, attribute, value, confidence, rationale)` | a proposal for human review |

Embedded in an application, build the server from a person's view
(`build_server(g.as_actor(Human(user.id)), agent="claude")`): the agent then reads what that person may
read, capped at the AI level, and its proposals are recorded as made by that person through the agent.

There's deliberately no tool to accept anything. `examples/mcp_client.py` is a real MCP client that
starts `gloss mcp` and calls these tools over stdio, which is handy for trying things out. A typical agent
session looks like this:

```
>>> search_model(text="revenue")
[{"subject": "field:crm.orders/amount", "kind": "field", "description": "Order value incl. VAT, in EUR", …}]

>>> get_model_context(datasets=["crm.customers"])
{"datasets": [{"name": "crm.customers", "fields": [
    {"name": "email", "datatype": "String", "confidentiality": "restricted", "redacted": true},
    {"name": "country", "datatype": "String", …}]}], …}

>>> propose_fact(subject="field:crm.orders/status", attribute="synonyms", value=["state"],
                 confidence=0.7, rationale="users ask for 'order state'")
{"outcome": "created", "status": "proposed"}
```

The proposal shows up in `gloss pending --source ai:` for a person to accept or reject. If it's
rejected, the same suggestion can't be proposed again.

### In your own prompts

```python
from glosslayer import ai

context = ai.model_context(g, datasets=["crm.orders", "crm.customers"])   # dict, ready to JSON-dump
hits = ai.search(g, "revenue")

seen = []
g.on_read(seen.append)                         # a record of what was handed to AI, for your own audit log
ai.search(g.as_actor(AI("claude")), "revenue")
print(seen[0].actor, seen[0].tool)             # ai:claude search
```

### REST

```bash
gloss serve                    # http://127.0.0.1:8765/docs
```

| | |
|---|---|
| Facts | `GET /assertions` (paged, with `total`), `GET /assertions/{id}`, `GET /subjects/{subject}`, `GET /subjects/{subject}/history`, `GET /log`, `GET /progress` |
| Proposing | `POST /assertions`, `POST /proposals` (many at once, with groups) |
| Deciding | `POST /assertions/{id}/{review\|accept\|reject\|retract\|reopen\|restore}`, `POST /bulk/{action}`, `POST /groups/{group}/{accept\|reject}`, `POST /rekey` |
| The model | `GET /tables`, `GET /tables/{key}`, `GET /relationships`, `GET /metrics`, `GET /types`, `GET /search?q=`, `GET /export/ossie`, `GET /viz/mermaid`, `GET /viz/report` |
| Checks | `GET /checks/{dataset}/effective`, `GET /checks/results`, `POST /checks/results` |
| AI | `GET /ai/search?q=`, `GET /ai/context` |
| Operations | `GET /health`, `POST /sync/{catalog}` |

Every route runs as an actor, reads included, and returns only what that actor may see. Responses are
typed, so `/openapi.json` is a contract; [`glosslayer.client.Client`](src/glosslayer/client.py) is a client for it.
Standalone, the actor comes from the `X-Gloss-Actor` header. That's meant for use behind an
authenticating proxy; embed it (below) to use your app's own logins.

## Embedding

Gloss is designed to live inside a host application: in its database, behind its logins, under its
permissions. Four things make that work.

**1. The host owns the database.** A store given the host's SQLAlchemy engine adds no connection
listeners, changes no settings and creates nothing. Gloss keeps its tables (all prefixed `gloss_`) at a
version it tracks itself, and the host creates or upgrades them from its own migrations:

<!-- not-tested -->
```python
# one revision in the host's Alembic chain; run it again whenever you upgrade Gloss
def upgrade():
    glosslayer.schema.upgrade(op.get_bind())        # idempotent: creates the tables or walks them to the latest

# and keep them out of the host's own autogenerate
context.configure(..., include_object=lambda obj, name, type_, *_:
                  not (type_ == "table" and glosslayer.schema.is_gloss_table(name)))
```

<!-- not-tested -->
```python
g = glosslayer.Gloss(glosslayer.SQLStore(host_engine), policy=HostPolicy())     # the application's own handle

with host_engine.begin() as conn, g.store.using(conn):                # or inside a transaction you own
    g.rekey("dataset:shop.orders", "dataset:shop.order_lines", actor=System("catalog"))
```

**2. Every request is someone.** `g` itself reads everything and is for the application's own work
(sync, migrations, jobs). For a user, bind a view once per request: its reads are limited to what the
policy lets that actor see, its writes are made as them, and `via` and `context` travel to the policy.

<!-- not-tested -->
```python
from fastapi import Depends, FastAPI
from glosslayer import Human
from glosslayer.server import create_router

def ledger(user=Depends(get_current_user)) -> glosslayer.Gloss:        # the host's auth dependency
    return g.as_actor(Human(user.id), context={"sign_in": user.sign_in})

app = FastAPI()
app.include_router(create_router(ledger), prefix="/semantic")
```

**3. The host hears about changes.** Subscribers are called after each committed write with the events
that were written, which is where UI refreshes, notifications and write-back to a catalog hang:

<!-- not-tested -->
```python
g.subscribe(lambda events: bus.publish("semantic_changed", [e.to_dict() for e in events]))
g.on_read(lambda record: audit.log(record.actor, record.via, record.tool, record.subjects))   # what AI read
```

Events written by one request, one sync run or one bulk action share a `correlation` id
(`g.log(correlation=...)`), so a UI can show them as one thing.

**4. No packaging surprises.** Register plugins and attributes in code, so nothing depends on entry-point
metadata being present (in a frozen application it often is not):

<!-- not-tested -->
```python
glosslayer.plugins.register("catalog", "mycatalog", MyCatalog)
glosslayer.register_attribute("trusted", ["dataset", "metric"], validate=bool)     # extend the vocabulary
```

The only data file is `glosslayer/ossie/ossie-schema.json`, needed for `validate()`; include it as package data
when freezing (PyInstaller: `collect_data_files("gloss")`).

### Two ledgers as one

An application can keep its own ledger for local tables and read an organisation's ledger next to it,
without copying one into the other:

<!-- not-tested -->
```python
from glosslayer.client import Client, Federation

shared = Client("https://hub.example/semantic", headers={"Authorization": f"Bearer {token}"})
both = Federation({"org.": shared}, default=g.as_actor(Human(user.id)))

both.pending()                                    # from both ledgers
both.propose("field:org.sales.orders/amount", "description", "Order value incl. VAT")   # goes to its owner
```

### Flowfile

The [`glosslayer-flowfile`](plugins/glosslayer-flowfile) plugin provides a catalog for Flowfile tables, a lineage
source that proposes confidentiality, classification and descriptions downstream, and the glue for
Flowfile's backend: catalog events, per-request views and a policy answered from the user's grants.

<!-- not-tested -->
```python
from glosslayer_flowfile import Embedded, FlowfileCatalog, LineageSource, policy_for

g = glosslayer.Gloss(glosslayer.SQLStore(engine), policy=policy_for(sharing=sharing_enabled()))
semantics = Embedded(g, FlowfileCatalog(service=unrestricted_catalog_service))

semantics.sync()                                           # as the backend itself, over the whole catalog
semantics.table_renamed("main.raw.orders", "main.raw.order_lines")     # from Flowfile's own update path
g.propose_many(LineageSource(semantics.catalog).proposals(g), source=LineageSource.source_id)

view = semantics.view(user_id=user.id, admin=user.is_admin, manage=managed_namespace_paths)
```

Standalone against a running Flowfile: `gloss sync flowfile:http://localhost:63578` (token in `FLOWFILE_TOKEN`).

## Ossie interop

```bash
gloss import ossie model.yaml [--key-by source] [--accept]
gloss export ossie -o model.yaml --validate [--include-pending] [--dataset crm.]
```

```python
yaml_text = g.to_yaml("crm.yaml", name="crm", tables="crm.")   # also written to crm.yaml
# g.to_yaml(include_pending=True)   # include facts still waiting for review

from glosslayer.ossie import validate

export = g.to_ossie(name="crm", tables="crm.")  # .document (dict), .warnings, .to_yaml()
print(validate(export.document))               # []  (valid against the Ossie JSON Schema)
```

- **Import:** every Ossie construct becomes proposals; `ai_context` splits into instructions, synonyms and examples.
- **Export:** only accepted facts by default. Fields carry their effective values (portable to any Ossie
  tool). What Ossie has no place for travels in `nonportable_extensions` under `vendor_name: gloss`.
  Re-importing restores it without copying inherited values onto fields.

```yaml
- name: email
  expression: {sql: email}
  description: Login e-mail address
  datatype: String
  ai_context: {synonyms: [e-mail, mail address]}
  nonportable_extensions:
  - vendor_name: gloss
    data:
      semantic_type: email_address
      confidentiality: restricted
      classification: [pii.email]
      physical_type: String(320)
      inherited: [classification, confidentiality, datatype, physical_type, synonyms]
```

- **Names:** Gloss keys may contain dots (`crm.orders`); exported Ossie names use the last part
  (`orders`), or the full key with underscores when two tables share a short name.

Targets Ossie `0.3.0.dev0`. The `git` sink (`gloss export git <repo>`) commits the export, recording the
log position in the commit message.

## Writing plugins

Plugins are plain classes. Register them as entry points and `pip install` is all it takes; `gloss plugins`
lists what's installed.

| Seam | Contract | Built in |
|---|---|---|
| Catalog | `name`, `list_assets()`, `describe(key) -> AssetSchema`, `fingerprint(key)`, optional `coverage()` and `write_back(key, facts)` | `file` |
| Source | `source_id`, `proposals(gloss) -> Iterable[Proposal]` (never accepted facts) | `ossie`, `semantic-types` |
| Sink | `name`, `publish(gloss, **options)` | `ossie`, `git` |
| Policy | `can(actor, action, assertion, context)`, `on_propose(actor, source, subject, attribute, context)`, `scope(actor, context)` | `DefaultPolicy`, `RolePolicy` |
| Attribute | `glosslayer.register_attribute(name, kinds, validate=…, inherited=…, detail=…, searchable=…)` | everything in the tables above |
| Check | subclass `checks.Check`, `@checks.register` | `not_null`, `unique`, `matches`, … |

Storage is not a seam: the store is SQLAlchemy, on SQLite or Postgres.

A catalog in a dozen lines:

```python
from glosslayer.seams import AssetSchema, Column as CatalogColumn, schema_fingerprint

class DictCatalog:
    """Tables described by a dict: {table: {column: dtype}}."""

    name = "dict"

    def __init__(self, tables):
        self.tables = tables

    def list_assets(self):
        return list(self.tables)

    def describe(self, key):
        return AssetSchema(key, [CatalogColumn(c, t) for c, t in self.tables[key].items()])

    def fingerprint(self, key):
        return schema_fingerprint(self.describe(key).columns)

sync(g, DictCatalog({"ops.tickets": {"ticket_id": "Int64", "opened_at": "Datetime"}}), detect_removed=False)
print(g.current("field:ops.tickets/ticket_id")["datatype"])     # Integer
```

```toml
# your package's pyproject.toml
[project.entry-points."gloss.catalogs"]
dict = "my_package:DictCatalog"
```

A custom check:

```python
from dataclasses import dataclass
from typing import ClassVar

@checks.register
@dataclass(frozen=True)
class Iban(checks.Check):
    kind: ClassVar[str] = "iban"

    def polars(self, col):
        return pl.col(col).str.contains(r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$") | pl.col(col).is_null()

IbanType = SemanticType("iban", dtype=String(34), checks=[Iban()], name_patterns=[r"iban"])
```

## Policies and review modes

Gloss doesn't authenticate anyone. The host says who the actor is, and the policy decides three things:
what they may **do** (`can`), what happens to a **proposal** (`on_propose`: queue it, accept it at once,
or refuse it), and what they may **read** (`scope`).

How work gets reviewed is a setting of the policy, not of the ledger, so a ledger that starts on one
person's laptop can move to a team later and keep its history:

| Mode | Your own edits | AI, imports, inferred | Who may accept |
|---|---|---|---|
| `solo` | accepted at once | queued | the user |
| `team` (default) | queued | queued | any person, including the author |
| `four_eyes` | queued | queued | any person except the author |
| stewarded (`RolePolicy`) | queued | queued | the stewards of that path |

```python
from glosslayer import DefaultPolicy

solo = glosslayer.open("sqlite:///solo.db", policy=DefaultPolicy(mode="solo"))
notes = Table("scratch.notes", description="Meeting notes").add_column("taken_at", DateTime())
print(solo.add(notes, source=Human("me")))        # 2 new, 0 unchanged, 2 accepted
print(solo.table("scratch.notes").description)    # Meeting notes  (no second step, readable at once)

solo.propose("dataset:scratch.notes", "synonyms", ["minutes"], source=AI("claude"))
print(len(solo.pending()))                        # 1  (you review the machine, not yourself)
print(solo.log()[1].data)                         # {'from_status': 'proposed', 'auto': True, 'rule': 'solo'}
```

An auto-accepted fact is still two events, written together, and the second one says which rule accepted
it: the history never pretends someone reviewed it. Modes can differ per path, longest prefix first:
`DefaultPolicy(mode="team", modes={"sandbox.": "solo", "prod.": "four_eyes"})`.

In every mode:

- AI never reviews or accepts, on its own or when a person works *through* an agent (`via="ai:…"`).
- Catalog actors accept **structural** facts only (a column exists, its type, a table's source and key),
  and those are accepted as the catalog states them. Meaning is always queued.
- A source may retract what it stated itself. Otherwise retracting, rejecting, restoring, marking stale
  and re-keying need the same say as accepting.
- An actor string without a known `kind:` prefix has no rights at all.
- Details above a reader's confidentiality level are withheld (`ai_max`, default `internal`; `human_max`,
  default `restricted`).

**`RolePolicy`** adds stewards per path, reviewers who may review but not accept, admins, and optionally
who may read what:

```python
from glosslayer import RolePolicy

policy = RolePolicy(
    admins={"human:edward"},
    stewards={"finance.": {"human:alice"}, "crm.": {"human:bob"}},   # paths: dataset keys and namespaces
    reviewers={"human:carol"},                                       # may review, not accept
    readers={"finance.": {"human:carol"}},                           # leave out: everyone reads everything
)
governed = glosslayer.open("sqlite:///governed.db", policy=policy)
governed.add(
    Table("finance.invoices", description="One row per invoice").add_column("total", Decimal(18, 2)),
    Table("crm.leads").add_column("score", Int(32)),
    source=Human("edward"), accept=True,
)

carol = governed.as_actor(Human("carol"))         # a view: reads scoped, writes made as her
print([t.key for t in carol.tables()])            # ['finance.invoices']
print(len(governed.tables()))                     # 2  (the application's own handle reads everything)
```

**Your own policy** subclasses `DefaultPolicy` and answers from what the host passed when it bound the
view. `can` is called once per fact in a batch, so resolve a user's rights once per request and answer
from memory:

```python
class GrantsPolicy(DefaultPolicy):
    """The host resolves `manages` per request: the paths this user may decide about."""

    def may_decide(self, actor, assertion, context):
        paths = context.attrs.get("manages", ()) if context else ()
        return assertion is not None and any(
            assertion.scope_path == p or (assertion.scope_path or "").startswith(p + ".") for p in paths
        )

granted = glosslayer.open("sqlite:///granted.db", policy=GrantsPolicy())
fact = granted.propose("dataset:crm.leads", "description", "Inbound leads", source=AI("claude"))
bob = granted.as_actor(Human("bob"), context={"manages": ["crm"]})
print(bob.accept(fact.id).status.value)           # accepted
```

A policy can also keep things out of a ledger altogether: `DefaultPolicy(refuse_attributes={"examples"})`
for a ledger hosted somewhere that must not hold data values.

## Renames and moves

Subjects are named after what they describe, so a renamed table would look like one table gone and a new
one with no history. Gloss records the rename instead, as events:

```python
moved = solo.rekey("dataset:scratch.notes", "dataset:scratch.minutes", actor=Human("me"))
print(moved.moved, solo.current("dataset:scratch.minutes")["description"])   # 3 Meeting notes
print(solo.history("dataset:scratch.minutes")[-1].type.value)                # rekeyed
```

- `g.rekey(old, new)` renames or moves one subject. A **dataset** takes its fields and metrics along, and
  relationships that join it are rewritten. A **field** (a renamed column) updates the keys and joins that
  list it. A **type** updates the fields that use it. Relationships and metrics can change name or namespace.
- `g.rekey_namespace("crm", "main.sales")` moves everything under a path: a renamed catalog or schema.
- Nothing is edited or lost: every affected fact gets a `rekeyed` event, `history()` of the new name shows
  the whole past, rejected suggestions stay rejected, and `rebuild_projection()` reproduces it all.
- A join that still joins the same two tables needs no new review. A metric whose SQL mentions the old
  name does, so it becomes stale.
- If the new name already has an accepted value for an attribute, that one stays and the moved one is
  superseded (and can be restored).

Catalog plugins report renames (`glosslayer.sync.asset_moved`); a pull sync also recognises one when the
catalog gives tables an id and the columns are unchanged.

## Performance

SQLite on a laptop, 10,000 tables × 10 columns (325k facts, 650k events). Reproduce with
`python scripts/bench.py --tables 10000`.

| Operation | Time |
|---|---|
| First catalog sync | ~55 s |
| Resync, nothing changed | 0.2 s |
| Resync, 1,000 tables changed | 1.7 s |
| Search, typical query | 1–15 ms |
| Search, a word present in every table | 18 ms |
| One table as objects, or as AI context | 1 ms |
| The same through an actor-scoped view | 1 ms |
| Accept one fact (incl. index update) | 17 ms |
| List 10,000 pending facts | 0.1 s |
| Export Ossie (10,000 datasets) | 6 s |
| Rebuild the search index from scratch | 9 s |

## Project layout

```
src/glosslayer/
  ids.py            subjects, namespaces, path prefixes
  model.py          assertions, events, statuses, the attribute registry, confidentiality
  ledger.py         the Gloss facade: propose / decide / re-key / read / search, and actor-bound views
  policy.py         Policy, Scope, Context, DefaultPolicy (review modes), RolePolicy
  actors.py         Human, AI, Import, Inferred, System; actor ids and the directory hook
  modeling.py       Table, Column, Relationship, Metric, SemanticType
  dtypes.py         Int(32), Decimal(18, 2), String(255), DateTime(tz=…)
  checks.py         data quality checks, single-pass Polars plan, SQL generation
  semantic.py       TypeSuggester
  visibility.py     confidentiality levels of subjects
  references.py     which facts mention a column or table
  search_index.py   search index rows, incremental updates, rebuild
  schema.py         the database schema and its version ladder, for hosts with migrations
  store/            SQLAlchemy store (SQLite and Postgres), transactions
  seams.py          Catalog / Source / Sink protocols
  plugins.py        registration in code and entry-point discovery
  sync.py           catalog sync, drift, renames, write-back
  catalogs/file.py  YAML catalog
  ossie/            importer, exporter, vendored JSON Schema
  sinks.py          git sink
  ai.py             AI views with redaction
  mcp_server.py     MCP tools
  server.py         REST API with a typed contract
  client.py         REST client, and several ledgers as one
  viz.py            Mermaid ER diagram, progress table, HTML report
  cli.py            the gloss command
plugins/glosslayer-flowfile/   Flowfile catalog, lineage source, backend glue
docs/architecture.md      design decisions
examples/                 catalog and Ossie model used in this README, and e2e_shop.py
```

Development: `pytest` runs everything, including the Python examples in this README;
`ruff check src plugins tests` for lint. To run the suite on Postgres, set
`GLOSS_TEST_DATABASE_URL=postgresql+psycopg://user:pass@host/db`. See [CONTRIBUTING.md](CONTRIBUTING.md)
and [CLAUDE.md](CLAUDE.md) for conventions.

## Roadmap

- Flowfile: column-level lineage from flow graphs, relationship proposals from joins
- Review UI as an embeddable component
- An AI suggester source: describe undocumented fields in batches, as proposals
- Table-level and cross-column checks; retention for check results
- Compiling structured metrics into queries (the host's engine does this today)
- A per-dataset Ossie name override for exports
- Semantic (embedding) search; an optional Rust search core for very large ledgers

## License

Apache License 2.0. Gloss is an independent project and is not affiliated with The Apache Software
Foundation; see [NOTICE](NOTICE).
