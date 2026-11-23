# Contributing

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]" -e plugins/gloss-flowfile
pytest -q
ruff check src plugins tests --select E,F,W,B --ignore E501,B008
```

- The suite runs on in-memory SQLite. Set `GLOSS_TEST_DATABASE_URL=postgresql+psycopg://user:pass@host/db`
  to run the same tests on Postgres (CI does both). Without a server, `pip install pgserver` provides one.
- The README's Python examples are executed by `tests/test_readme.py`. Change behaviour, and you change
  the README and that test in the same pull request.
- [CLAUDE.md](CLAUDE.md) lists the invariants (append-only log, what may never change in an event, reads
  made as an actor, no host imports in the core) and how to add attributes, checks, plugins and routes.
  [docs/architecture.md](docs/architecture.md) explains why they are what they are.
- After touching storage, sync, search or the ledger's write paths, run `python scripts/bench.py` before
  and after, and look at `EXPLAIN QUERY PLAN` for any new query on `gloss_assertions`.
- A change to a table is a new step in `gloss.schema._STEPS` and a bumped `LATEST`, never an edit of an
  existing step.

Gloss is licensed under Apache 2.0; contributions are accepted under the same licence.
