# glosslayer-flowfile

Gloss plugin for [Flowfile](https://github.com/Edwardvaneechoud/Flowfile)'s catalog.

| Part | Name | What it does |
|---|---|---|
| Catalog | `FlowfileCatalog` (entry point `flowfile`) | Lists Flowfile catalog tables as `catalog.schema.table`, with columns and types. Knows whether its listing is the whole catalog or one user's view. Writes accepted table descriptions back. |
| Source | `LineageSource` (entry point `flowfile-lineage`) | Uses table lineage (producing flow → tables it reads) to propose confidentiality, classification and descriptions downstream. |
| Backend glue | `Embedded`, `FlowfilePolicy`, `policy_for`, `actor_for`, `Directory` | Catalog events, per-request views, and a policy answered from the rights the backend already resolved. |

Nothing in this package imports Flowfile: it is handed Flowfile's objects and plain values. It is tested
against a fake `CatalogService` with the shape of Flowfile's `CatalogTableOut`, not a running Flowfile.

## Embedded in Flowfile's backend

```python
import glosslayer
from glosslayer_flowfile import Embedded, FlowfileCatalog, LineageSource, policy_for

# 1. Schema: from a Flowfile Alembic revision (and again whenever Gloss is upgraded)
glosslayer.schema.upgrade(op.get_bind())

# 2. One ledger on Flowfile's engine. Gloss adds no listeners and runs no DDL here.
g = glosslayer.Gloss(glosslayer.SQLStore(engine), policy=policy_for(sharing=sharing_enabled()))

# 3. The catalog, through a CatalogService built WITHOUT an access resolver: the whole local catalog.
semantics = Embedded(g, FlowfileCatalog(service=CatalogService(repo)))
semantics.sync()                                         # at startup, as the backend itself

# 4. From Flowfile's own table paths, where the change is made
semantics.table_written(qualified_name)                  # created, overwritten with new columns
semantics.table_renamed(old_qualified_name, new_qualified_name)      # update_table(name=…/namespace_id=…)
semantics.table_deleted(qualified_name)                  # delete_table

# 5. Per request: who is asking, and what the backend already knows about their rights
view = semantics.view(
    user_id=user.id,
    admin=user.is_admin,
    manage=paths_of(access.writable_namespace_ids()),    # namespace ids → "catalog.schema" paths
    read=None,                                           # or the readable paths where the catalog is private
)
app.include_router(create_router(lambda user=Depends(get_current_active_user): view_for(user)),
                   prefix="/semantic")

# 6. Proposals from lineage, whenever convenient
g.propose_many(LineageSource(semantics.catalog).proposals(g), source=LineageSource.source_id)
```

What the pieces take care of:

- **Never sync per user.** A `CatalogService` with an access resolver lists only what that user may see.
  `FlowfileCatalog` then reports its listing as incomplete, so a sync through it can discover and update
  tables but never marks a table it cannot see as gone. Build the service without a resolver for the
  backend's own sync.
- **Renames.** Flowfile's `update_table` can change a table's name and namespace, and table ids can be
  reused after a delete, so neither is an identity on its own. Call `table_renamed` from the update path
  and everything known about the table moves with it. A full `sync()` also recognises a rename when the id
  matches *and* the columns are unchanged.
- **Hub tables.** Rows Flowfile shows from a connected hub (`source="hub"`, negated ids) are skipped: their
  meaning lives in the hub's ledger. Use `glosslayer.client.Federation` to show both ledgers as one.
- **Who decides.** `policy_for(sharing=False)` is the desktop app: one user, their own edits accepted at
  once, AI suggestions still queued. With sharing on, people decide where they manage the namespace, and
  the policy answers from the `manage`/`read`/`admin` values passed to `view()`, so no database lookups
  happen inside a batch.
- **Descriptions.** Gloss owns meaning. When a table description is accepted, `Embedded` writes it to
  Flowfile's own `description` field (`own_descriptions=False` turns that off). Flowfile stores no
  description per column; those stay in Gloss.
- **Actors are ids.** `actor_for(user.id)` puts the user's id in the log, not their name; give
  `glosslayer.Gloss(..., directory=Directory(lookup))` a lookup to show names. Flowfile's integer id is stable
  on one machine. A ledger shared between machines needs an id every machine agrees on.

### Packaging (PyInstaller)

- Register plugins in code so nothing depends on distribution metadata:
  `glosslayer.plugins.register("catalog", "flowfile", FlowfileCatalog)`.
- Add `collect_data_files("gloss")` for `gloss/ossie/ossie-schema.json` (only needed for `validate()`).
- Gloss's core needs SQLAlchemy and PyYAML, both already Flowfile dependencies.

## Standalone

```bash
export FLOWFILE_TOKEN=...            # a Flowfile JWT: one user's view, so nothing is ever marked gone
gloss sync flowfile:http://localhost:63578
gloss import flowfile-lineage http://localhost:63578
gloss pending --source inferred:
```

## Roadmap

- Column-level lineage from the flow graph (select / rename / formula nodes)
- Relationship proposals from join nodes
- Metric candidates from group-by aggregations
- Profiling proposals (candidate keys, uniqueness) via Polars on the Delta tables
- A Vue review panel for the Flowfile catalog page
