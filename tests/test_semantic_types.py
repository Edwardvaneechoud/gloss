"""Semantic types (reusable meanings) and data quality checks."""

from dataclasses import dataclass
from typing import ClassVar

import pytest

import gloss
from gloss import Column, Human, Int, SemanticType, String, Table, TypeSuggester, ai, checks
from gloss.checks import Check, run_polars, to_sql
from gloss.ossie import OssieSource, to_ossie, validate

EMAIL_RX = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
ED = Human("edward")


def email_type(**kw):
    base = dict(
        dtype=String(320),
        description="An e-mail address",
        confidentiality="restricted",
        classification=["pii.email"],
        synonyms=["e-mail"],
        checks=[checks.matches(EMAIL_RX)],
        name_patterns=[r"e[-_]?mail"],
    )
    return SemanticType("email_address", **{**base, **kw})


@pytest.fixture
def model(g):
    Email = email_type()
    leads = Table("crm.leads", primary_key=[Column("lead_id", Int(64))]).add_columns(
        Column("contact_email", Email, checks=[checks.not_null()]),
    )
    news = Table("mkt.newsletter").add_columns(Column("subscriber_email", Email, description="Who signed up"))
    g.add(leads, news, actor=ED, accept=True)
    return g


# ------------------------------------------------------------------ definition and inheritance
def test_column_takes_type_as_datatype():
    Email = email_type()
    c = Column("e", Email)
    assert c.semantic_type is Email and c.datatype is None and c.dtype == String(320)
    with pytest.raises(ValueError):
        Column("e", Email, semantic_type=email_type(description="other"))
    with pytest.raises(ValueError):
        SemanticType("bad/name")


def test_type_saved_once_and_inherited(model):
    g = model
    assert g.current("type:email_address")["physical_type"] == "String(320)"
    eff = g.effective("field:crm.leads/contact_email")
    assert eff["confidentiality"] == "restricted" and eff["datatype"] == "String"
    assert eff["checks"] == [checks.matches(EMAIL_RX).to_dict(), checks.not_null().to_dict()]
    # own facts win over the type
    assert g.effective("field:mkt.newsletter/subscriber_email")["description"] == "Who signed up"
    # the field itself only stores what it states
    assert "confidentiality" not in g.current("field:crm.leads/contact_email")
    assert g.type_usages("email_address") == ["field:crm.leads/contact_email", "field:mkt.newsletter/subscriber_email"]


def test_changing_the_type_reaches_every_field_once_accepted(model):
    g = model
    a = g.propose("type:email_address", "confidentiality", "confidential", source=ED)
    assert g.effective("field:crm.leads/contact_email")["confidentiality"] == "restricted"  # pending: no effect
    g.accept(a.id, actor=ED)
    assert g.effective("field:crm.leads/contact_email")["confidentiality"] == "confidential"
    assert g.effective("field:mkt.newsletter/subscriber_email")["confidentiality"] == "confidential"


def test_snapshot_marks_inherited_values(model):
    f = model.snapshot().datasets["crm.leads"].fields["contact_email"]
    assert f.inherited["confidentiality"] == "email_address"
    assert f.check_origins == {0: "type:email_address"}
    assert "confidentiality" not in f.own_attrs


def test_ai_redaction_follows_the_type(model):
    ctx = ai.model_context(model, datasets=["crm.leads"])
    field = next(f for f in ctx["datasets"][0]["fields"] if f["name"] == "contact_email")
    assert field["redacted"] is True and field["confidentiality"] == "restricted"
    assert ai.subject_level(model, "field:crm.leads/contact_email").value == "restricted"


def test_load_round_trip(model):
    t = model.table("crm.leads")
    col = t["contact_email"]
    assert isinstance(col.semantic_type, SemanticType) and col.semantic_type.name == "email_address"
    assert col.dtype == String(320)
    assert col.checks == [checks.not_null().to_dict()]  # only its own; the type's stay on the type
    assert model.add(t, actor=ED).created == 0  # saving it back changes nothing
    st = SemanticType.load(model, "email_address")
    assert st.resolved_dtype == String(320) and st.name_patterns == [r"e[-_]?mail"]


def test_ossie_round_trip_keeps_types_without_duplicating(model):
    doc = to_ossie(model).document
    assert validate(doc) == []
    leads = next(d for d in doc["datasets"] if d["name"] == "leads")
    f = leads["fields"][1]
    assert f["datatype"] == "String"  # effective value exported for portability
    ext = f["nonportable_extensions"][0]["data"]
    assert ext["semantic_type"] == "email_address" and "confidentiality" in ext["inherited"]
    assert doc["nonportable_extensions"][0]["data"]["types"]["email_address"]["confidentiality"] == "restricted"

    with gloss.open("sqlite://") as g2:
        g2.propose_many(OssieSource(doc).proposals(), source="import:ossie")
        g2.bulk("accept", gloss.Query(), actor=ED)
        assert "confidentiality" not in g2.current("field:leads/contact_email")  # not copied onto the field
        assert g2.effective("field:leads/contact_email")["confidentiality"] == "restricted"


# ------------------------------------------------------------------ suggesting types at scale
def test_type_suggester(model, examples):
    from gloss.catalogs.file import FileCatalog
    from gloss.sync import sync

    g = model
    sync(g, FileCatalog(examples / "catalog.yaml"))
    g.add(Table("x.stats").add_columns(Column("email_count", Int(32))), actor=ED, accept=True)
    res = g.propose_many(TypeSuggester().proposals(g), source=TypeSuggester.source_id)
    got = {r.assertion.subject: r.assertion for r in res}
    assert set(got) == {"field:shop.customers/email"}  # not email_count (Integer), not already-typed fields
    assert got["field:shop.customers/email"].confidence == 0.7
    g.reject(got["field:shop.customers/email"].id, actor=ED)
    again = g.propose_many(TypeSuggester().proposals(g), source=TypeSuggester.source_id)
    assert [r.outcome for r in again] == ["previously_rejected"]


# ------------------------------------------------------------------ checks
pl = pytest.importorskip("polars")


@pytest.mark.parametrize(
    "check,values,failing",
    [
        (checks.not_null(), [1, None, 3], 1),
        (checks.unique(), [1, 2, 2, None, None], 2),
        (checks.matches(r"^[A-Z]{2}$"), ["NL", "nl", None], 1),
        (checks.one_of(["a", "b"]), ["a", "c", None], 1),
        (checks.between(0, 10), [-1, 5, 11, None], 2),
        (checks.length(2, 3), ["a", "ab", "abcd", None], 2),
        (checks.sql("{column} % 2 = 0"), [2, 3, 4], 1),
    ],
)
def test_builtin_checks_polars(g, check, values, failing):
    t = Table("t").add_columns(Column("c", checks=[check]))
    g.add(t, actor=ED, accept=True)
    [r] = run_polars(g, "t", pl.DataFrame({"c": values}))
    assert r.failed_rows == failing and r.passed is (failing == 0) and r.error is None


@pytest.mark.parametrize(
    "check,values,failing",
    [
        (checks.not_null(), [1, None, 3], 1),
        (checks.unique(), [1, 2, 2, None], 1),
        (checks.matches(r"^[A-Z]{2}$"), ["NL", "nl", None], 1),
        (checks.one_of(["a", "b"]), ["a", "c", None], 1),
        (checks.between(0, 10), [-1, 5, 11, None], 2),
        (checks.length(2, 3), ["a", "ab", "abcd", None], 2),
    ],
)
def test_builtin_checks_sql_on_duckdb(g, check, values, failing):
    duckdb = pytest.importorskip("duckdb")
    g.add(Table("t", source="t").add_columns(Column("c", checks=[check])), actor=ED, accept=True)
    con = duckdb.connect()
    sqltype = "VARCHAR" if any(isinstance(v, str) for v in values) else "INTEGER"
    con.execute(f"CREATE TABLE t (c {sqltype})")
    con.executemany("INSERT INTO t VALUES (?)", [[v] for v in values])
    [(_, _, q)] = to_sql(g, "t", dialect="duckdb")
    assert con.execute(q).fetchone()[0] == failing


def test_samples_hidden_for_confidential_fields(model):
    df = pl.DataFrame({"lead_id": [1, 2], "contact_email": ["ok@x.io", "broken"]})
    results = run_polars(model, "crm.leads", df)
    fmt = next(r for r in results if r.check["check"] == "matches")
    assert not fmt.passed and fmt.failed_rows == 1 and fmt.sample == [] and fmt.origin == "type:email_address"


def test_missing_column_is_an_error(model):
    results = run_polars(model, "crm.leads", pl.DataFrame({"lead_id": [1]}))
    assert all(r.error and not r.passed for r in results)


def test_record_and_history(model):
    df = pl.DataFrame({"lead_id": [1], "contact_email": ["ok@x.io"]})
    run_id = model.record_check_results(run_polars(model, "crm.leads", df), runner=ED)
    hist = model.check_history(dataset="crm.leads")
    assert len(hist) == 2 and all(h["run_id"] == run_id and h["passed"] for h in hist)
    assert model.check_history("field:crm.leads/contact_email")[0]["runner"] == "human:edward"


def test_custom_check(g):
    @checks.register
    @dataclass(frozen=True)
    class Iban(Check):
        kind: ClassVar[str] = "iban"

        def polars(self, col):
            return pl.col(col).str.contains(r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$") | pl.col(col).is_null()

    IbanType = SemanticType("iban", dtype=String(34), checks=[Iban()])
    g.add(Table("fin.accounts").add_columns(Column("iban", IbanType)), actor=ED, accept=True)
    [r] = run_polars(g, "fin.accounts", pl.DataFrame({"iban": ["NL91ABNA0417164300", "123"]}))
    assert r.failed_rows == 1


def test_unknown_check_kinds_are_kept(g):
    g.add(Table("t").add_columns(Column("c", checks=[{"check": "from_the_future", "x": 1}])), actor=ED, accept=True)
    assert g.current("field:t/c")["checks"] == [{"check": "from_the_future", "x": 1}]
    [r] = run_polars(g, "t", pl.DataFrame({"c": [1]}))
    assert r.error and "no Polars implementation" in r.error


def test_cli_types_and_check_file_db(tmp_path, capsys):
    from gloss.cli import main

    db = f"sqlite:///{tmp_path / 'g.db'}"
    with gloss.open(db) as g:
        g.add(Table("crm.leads").add_columns(Column("contact_email", email_type())), actor=ED, accept=True)
    csv = tmp_path / "leads.csv"
    csv.write_text("contact_email\nok@x.io\nbroken\n")
    assert main(["--db", db, "types"]) == 0
    assert "email_address  String(320)  [restricted]  used by 1 field(s)" in capsys.readouterr().out
    assert main(["--db", db, "--actor", "human:t", "check", "crm.leads", "--data", str(csv), "--record"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "recorded run" in out
    assert main(["--db", db, "check", "crm.leads", "--sql", "--dialect", "postgres"]) == 0
    assert '"contact_email" ~ ' in capsys.readouterr().out
