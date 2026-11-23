"""Fixtures. The suite runs on in-memory SQLite by default.

Set ``GLOSS_TEST_DATABASE_URL`` to a Postgres URL (``postgresql+psycopg://user:pass@host/db``) and
every in-memory ledger the tests open becomes a fresh schema in that database instead, so the same
tests cover both backends. Tests that open a SQLite *file* keep doing so.
"""

import itertools
import os
from pathlib import Path

import pytest
import sqlalchemy as sa

import gloss
from gloss import ledger as ledger_module

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
POSTGRES_URL = os.environ.get("GLOSS_TEST_DATABASE_URL")
_counter = itertools.count()


@pytest.fixture(autouse=True)
def _ledgers_on_the_configured_database(monkeypatch):
    if not POSTGRES_URL:
        yield
        return
    admin = sa.create_engine(POSTGRES_URL, isolation_level="AUTOCOMMIT")
    schemas: list[str] = []
    real_open = ledger_module.open

    def open_ledger(url="sqlite:///gloss.db", **kwargs):
        if url not in ("sqlite://", "sqlite:///:memory:"):
            return real_open(url, **kwargs)
        name = f"gloss_test_{os.getpid()}_{next(_counter)}"
        with admin.connect() as conn:
            conn.exec_driver_sql(f'CREATE SCHEMA "{name}"')
        schemas.append(name)
        scoped = sa.engine.make_url(POSTGRES_URL).update_query_dict({"options": f"-csearch_path={name}"})
        return real_open(scoped.render_as_string(hide_password=False), **kwargs)

    monkeypatch.setattr(ledger_module, "open", open_ledger)
    monkeypatch.setattr(gloss, "open", open_ledger)
    yield
    with admin.connect() as conn:
        for name in schemas:
            conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
    admin.dispose()


@pytest.fixture
def g():
    with gloss.open("sqlite://") as ledger:
        yield ledger


@pytest.fixture
def examples() -> Path:
    return EXAMPLES
