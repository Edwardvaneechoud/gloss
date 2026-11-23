"""The search index: ranking, prefixes, confidentiality, incremental updates, and the LIKE fallback."""

import pytest

import gloss
from gloss import Column, Human, SemanticType, String, Table
from gloss.ossie import OssieSource

ED = Human("edward")


@pytest.fixture(params=["fts", "like"])
def shop(request, examples):
    with gloss.open("sqlite://") as g:
        if request.param == "like":
            g.store.fts = False  # exercise the portable fallback used on Postgres & co
        g.propose_many(OssieSource(examples / "semantic_model.yaml").proposals(), source="import:ossie")
        g.bulk("accept", gloss.Query(), actor=ED)
        yield g


def subjects(hits):
    return [h["subject"] for h in hits]


def test_finds_by_name_description_and_synonym(shop):
    assert "field:orders/amount" in subjects(shop.search("amount"))
    assert "field:customers/country" in subjects(shop.search("ISO country"))  # description words
    assert set(subjects(shop.search("turnover"))) == {"field:orders/amount", "metric:total_revenue"}


def test_prefixes_and_all_words_first(shop):
    assert "metric:total_revenue" in subjects(shop.search("reven"))
    hits = shop.search("order count")  # both words on one item: exact answers only, no 'matched' marker
    assert hits[0]["subject"] == "metric:order_count" and "matched" not in hits[0]


def test_natural_questions_fall_back_to_any_word(shop):
    hits = shop.search("what was the revenue per country")
    got = {h["subject"]: h["matched"] for h in hits}
    assert got["metric:total_revenue"] == ["revenue"]
    assert got["field:customers/country"] == ["country"]
    assert shop.search("revenue zebra")[0]["matched"] == ["revenue"]


def test_spelling_is_corrected_from_the_index(shop):
    if not shop.store.fts:
        pytest.skip("spelling suggestions need SQLite FTS5")
    hits = shop.search("revnue")
    assert hits and hits[0]["did_you_mean"] == "revenue"
    assert shop.search("zzzzqqq") == []


def test_type_definitions_stay_visible_to_ai(g):
    from gloss import Column, SemanticType, String, Table

    Email = SemanticType("email_address", dtype=String(320), description="An e-mail address",
                         confidentiality="restricted")
    g.add(Table("crm.leads").add_columns(Column("contact", Email)), actor=ED, accept=True)
    hits = {h["subject"]: h for h in g.search("email address")}
    assert hits["type:email_address"]["description"] == "An e-mail address"  # the definition is readable
    assert hits["field:crm.leads/contact"]["redacted"] is True  # the data is not


def test_names_rank_above_descriptions(shop):
    hits = subjects(shop.search("orders"))
    assert hits[0] == "dataset:orders"


def test_filters(shop):
    assert subjects(shop.search("order", kinds=["metric"])) == ["metric:order_count", "metric:total_revenue"]
    assert subjects(shop.search("order count"))[0] == "metric:order_count"  # exact name first
    assert all(s.startswith("field:customers/") or s == "dataset:customers"
               for s in subjects(shop.search("customer", dataset_prefix="customers")))


def test_confidential_items_found_by_name_only(shop):
    [hit] = shop.search("email")
    assert hit["redacted"] and hit["confidentiality"] == "restricted" and "description" not in hit
    assert shop.search("login") == []  # 'Login e-mail address' is the restricted description
    assert "field:customers/email" in subjects(shop.search("login", for_ai=False))  # people see it


def test_index_follows_accepts_supersedes_and_stale(shop):
    a = shop.propose("field:orders/order_date", "synonyms", ["booking day"], source=ED)
    assert shop.search("booking") == []  # proposed only
    shop.accept(a.id, actor=ED)
    assert subjects(shop.search("booking")) == ["field:orders/order_date"]
    b = shop.propose("field:orders/order_date", "synonyms", ["purchase day"], source=ED)
    shop.accept(b.id, actor=ED)
    assert shop.search("booking") == [] and subjects(shop.search("purchase")) == ["field:orders/order_date"]
    shop.mark_stale(b.id, actor="catalog:x", reason="column changed")
    assert shop.search("purchase") == []


def test_dataset_confidentiality_reaches_its_fields(shop):
    a = shop.propose("dataset:orders", "confidentiality", "confidential", source=ED)
    shop.accept(a.id, actor=ED)
    hits = shop.search("amount")
    assert hits and all(h.get("redacted") for h in hits if h["subject"].startswith("field:orders"))
    assert "metric:total_revenue" in subjects(shop.search("turnover"))  # model-level metric unaffected


def test_semantic_type_changes_reach_every_field(g):
    Email = SemanticType("email_address", dtype=String(320), description="An e-mail address")
    g.add(Table("crm.leads").add_columns(Column("contact", Email)),
          Table("mkt.news").add_columns(Column("subscriber", Email)), actor=ED, accept=True)
    assert set(subjects(g.search("mail address", kinds=["field"]))) == {"field:crm.leads/contact", "field:mkt.news/subscriber"}
    assert subjects(g.search("email_address", kinds=["type"])) == ["type:email_address"]
    a = g.propose("type:email_address", "confidentiality", "restricted", source=ED)
    g.accept(a.id, actor=ED)
    assert all(h.get("redacted") for h in g.search("subscriber"))


def test_rebuild_matches_incremental(shop):
    before = shop.search("order", limit=100, for_ai=False)
    assert shop.rebuild_search_index() > 0
    assert shop.search("order", limit=100, for_ai=False) == before


def test_old_database_gets_indexed_on_first_search(tmp_path, examples):
    url = f"sqlite:///{tmp_path / 'g.db'}"
    with gloss.open(url) as g:
        g.propose_many(OssieSource(examples / "semantic_model.yaml").proposals(), source="import:ossie")
        g.bulk("accept", gloss.Query(), actor=ED)
        g.store.clear_search()  # simulate a ledger written before the index existed
    with gloss.open(url) as g:
        assert "field:orders/amount" in subjects(g.search("amount"))


def test_cli_search(tmp_path, examples, capsys):
    from gloss.cli import main

    db = ["--db", f"sqlite:///{tmp_path / 'g.db'}", "--actor", "human:t"]
    main(db + ["import", "ossie", str(examples / "semantic_model.yaml"), "--accept"])
    capsys.readouterr()
    assert main(db + ["search", "turnover"]) == 0
    out = capsys.readouterr().out
    assert "field:orders/amount" in out and "metric:total_revenue" in out
    assert main(db + ["search", "email", "--as-ai"]) == 0
    assert "[redacted]" in capsys.readouterr().out
    assert main(db + ["reindex"]) == 0


def test_mail_finds_email_columns_through_the_type(g):
    from gloss import Column, SemanticType, String, Table

    Email = SemanticType("email_address", dtype=String(320), confidentiality="restricted", synonyms=["e-mail"])
    g.add(Table("crm.leads").add_columns(Column("email", Email), Column("useremail", String(320))),
          Table("mkt.news").add_columns(Column("subscriber_email", Email)), actor=ED, accept=True)
    got = set(subjects(g.search("mail")))
    assert {"field:crm.leads/email", "field:mkt.news/subscriber_email", "type:email_address"} <= got
    assert all(h.get("redacted") for h in g.search("mail") if h["subject"].startswith("field:") and "email" == h["name"])


def test_plurals_are_stemmed(shop):
    hits = shop.search("customers emails")
    assert "field:customers/email" in subjects(hits)
    if shop.store.fts:
        assert not any("did_you_mean" in h for h in shop.search("emails"))  # no correction needed


def test_words_inside_names(g):
    from gloss import Column, String, Table

    g.add(Table("crm.users").add_columns(Column("useremail", String(320))), actor=ED, accept=True)
    assert subjects(g.search("mail")) == ["field:crm.users/useremail"]


def test_ai_spelling_suggestions_never_reveal_restricted_words(g):
    from gloss import Column, String, Table

    g.add(Table("hr.staff").add_columns(
        Column("note", String(100), description="Pending dismissal of Hendrickson", confidentiality="restricted")),
        actor=ED, accept=True)
    if not g.store.fts:
        pytest.skip("spelling suggestions need SQLite FTS5")
    assert g.search("hendriksen") == []  # the AI gets no hint that the word exists
    people = g.search("hendriksen", for_ai=False)
    assert people and people[0]["did_you_mean"] == "hendrickson"


@pytest.mark.parametrize("typo,word", [("fmail", "mail"), ("emial", "email"), ("cusotmer", "customer"), ("ordr", "order")])
def test_common_typos(g, typo, word):
    from gloss import Column, SemanticType, String, Table

    if not g.store.fts:
        pytest.skip("spelling suggestions need SQLite FTS5")
    Email = SemanticType("email_address", dtype=String(320), synonyms=["e-mail"])
    g.add(Table("shop.customers").add_columns(Column("email", Email), Column("customer_id")),
          Table("shop.orders").add_columns(Column("order_id")), actor=ED, accept=True)
    hits = g.search(typo)
    assert hits and hits[0]["did_you_mean"] == word
