"""Checks for hosts that never collect in the calling process, and results that travel."""

import pytest

import gloss
from gloss import Column, Human, SemanticType, String, Table, checks

pl = pytest.importorskip("polars")


@pytest.fixture
def model(g):
    Email = SemanticType("email", dtype=String(320), confidentiality="restricted",
                         checks=[checks.matches(r"^[^@\s]+@[^@\s]+$")])
    t = Table("crm.customers", primary_key=[Column("id", int, checks=[checks.not_null(), checks.unique()])])
    t.add_columns(Column("email", Email), Column("country", str, checks=[checks.sql("{column} ~ '^[A-Z]{2}$'")]),
                  Column("missing", str, checks=[checks.not_null()]))
    g.add(t, source=Human("a"), accept=True)
    return g


DATA = {"id": [1, 2, 2, None], "email": ["a@x.io", "nope", None, "b@x.io"], "country": ["NL", "nl", "BE", None]}


def test_one_lazy_pass_counts_every_failure(model):
    plan = checks.polars_plan(model, "crm.customers", pl.LazyFrame(DATA))
    assert isinstance(plan.frame, pl.LazyFrame)  # nothing has been collected
    row = plan.frame.collect()  # the host ships this plan to wherever the data is
    assert row.height == 1 and row["__total__"][0] == 4
    results = {(r.subject.rsplit("/", 1)[-1], r.check["check"]): r for r in plan.results(row)}
    assert results[("id", "not_null")].failed_rows == 1
    assert results[("id", "unique")].failed_rows == 2
    assert results[("email", "matches")].failed_rows == 1 and results[("email", "matches")].origin == "type:email"
    assert results[("country", "sql")].failed_rows == 2  # literal braces are fine; a SQL check decides on nulls
    assert results[("missing", "not_null")].error == "column 'missing' not in data"
    assert plan.results(row.row(0, named=True))[0].total_rows == 4  # a plain mapping works too

    failing = plan.failing(next(col for s, c, _o, col in plan.items if c["check"] == "sql")).collect()
    assert failing["country"].to_list() == ["nl", None]


def test_run_polars_samples_only_what_may_be_shown(model):
    results = {(r.subject.rsplit("/", 1)[-1], r.check["check"]): r for r in
               checks.run_polars(model, "crm.customers", pl.DataFrame(DATA))}
    assert results[("country", "sql")].sample == ["nl", None]
    assert results[("email", "matches")].failed_rows == 1 and results[("email", "matches")].sample == []


def test_a_check_that_cannot_run_does_not_stop_the_others(g):
    t = Table("t").add_columns(Column("n", int, checks=[checks.matches("^a"), checks.not_null()]))
    g.add(t, source=Human("a"), accept=True)
    results = checks.run_polars(g, "t", pl.DataFrame({"n": [1, None]}))
    by_kind = {r.check["check"]: r for r in results}
    assert by_kind["not_null"].failed_rows == 1 and by_kind["not_null"].error is None


def test_sql_check_with_braces_generates_sql(model):
    queries = {c["check"]: q for _s, c, q in checks.to_sql(model, "crm.customers", dialect="duckdb")}
    assert "\"country\" ~ '^[A-Z]{2}$'" in queries["sql"]


def test_results_travel_and_keep_the_callers_run_id(model):
    results = checks.run_polars(model, "crm.customers", pl.DataFrame(DATA))
    wire = [r.to_dict() for r in results]
    assert all("sample" not in w for w in wire)  # failing values are data and stay where they are
    back = [checks.CheckResult.from_dict(w) for w in wire]
    assert model.record_check_results(back, runner="system:worker-3", run_id="run-2026-11-23-001") == \
        "run-2026-11-23-001"
    logged = model.check_history(run_id="run-2026-11-23-001")
    assert len(logged) == len(results) and {r["runner"] for r in logged} == {"system:worker-3"}
    assert model.check_history(dataset="crm.customers", run_id="other") == []


def test_effective_checks_reads_one_table(model, monkeypatch):
    seen = []
    find = model.store.find

    def spy(query):
        seen.append(query)
        return find(query)

    monkeypatch.setattr(model.store, "find", spy)
    checks.effective_checks(model, "crm.customers")
    assert seen and all(q.dataset == "crm.customers" or q.kind in ("type", "model") for q in seen)
    seen.clear()
    model.semantic_type("email")
    model.relationships()
    assert all(q.kind in ("type", "model", "relationship") for q in seen)


def test_gloss_is_importable_without_polars_for_plans():
    assert callable(gloss.checks.polars_plan)
