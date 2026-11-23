"""Sync beyond the full pull: partial listings, single assets, renames, deletes and write-back."""

import gloss
from gloss import Human, Status
from gloss.catalogs.file import FileCatalog
from gloss.model import EventType
from gloss.seams import AssetSchema, Column, Coverage, asset_fingerprint
from gloss.sync import asset_moved, asset_removed, sync, sync_asset, write_back


class DictCatalog:
    """Tables as {key: {"columns": {name: dtype}, "description": ..., "id": ...}}; optionally one user's view."""

    name = "dict"

    def __init__(self, tables, *, visible=None, complete=True):
        self.tables, self.visible, self.complete = tables, visible, complete
        self.written = {}

    def coverage(self):
        return Coverage(complete=self.complete)

    def list_assets(self):
        return [k for k in self.tables if self.visible is None or k in self.visible]

    def describe(self, key):
        t = self.tables[key]
        return AssetSchema(key, [Column(c, d) for c, d in t["columns"].items()], description=t.get("description"),
                           uid=t.get("id"))

    def fingerprint(self, key):
        return asset_fingerprint(self.describe(key))

    def write_back(self, key, facts):
        self.written[key] = facts


def _tables():
    return {
        "main.alice.orders": {"columns": {"id": "Int64", "amount": "Float64"}, "id": "1"},
        "main.bob.payroll": {"columns": {"id": "Int64", "salary": "Float64"}, "id": "2"},
    }


def test_a_users_listing_does_not_mark_other_tables_gone(g):
    tables = _tables()
    sync(g, DictCatalog(tables))
    a = g.propose("dataset:main.bob.payroll", "description", "Bob's payroll", source=Human("bob"))
    g.accept(a.id, actor=Human("bob"))

    rep = sync(g, DictCatalog(tables, visible={"main.alice.orders"}, complete=False))  # what Alice may see
    assert rep.gone == [] and rep.staled == 0 and rep.unchanged == 1
    assert g.get(a.id).status is Status.ACCEPTED
    assert g.find(statuses=[Status.STALE]) == []

    rep = sync(g, DictCatalog({"main.alice.orders": tables["main.alice.orders"]}))  # the whole catalog says so
    assert rep.gone == ["main.bob.payroll"] and g.get(a.id).status is Status.STALE


def test_coverage_can_be_limited_to_prefixes():
    cov = Coverage(prefixes=("main.alice",))
    assert cov.speaks_for("main.alice.orders") and not cov.speaks_for("main.bob.payroll")
    assert not Coverage(complete=False).speaks_for("main.alice.orders")


def test_one_asset_at_a_time(g):
    tables = _tables()
    cat = DictCatalog(tables)
    rep = sync_asset(g, cat, "main.alice.orders")
    assert rep.new == ["main.alice.orders"] and rep.accepted == 7  # source, and three facts per column
    assert g.find(dataset="main.bob.payroll") == []  # nothing else was looked at
    assert sync_asset(g, cat, "main.alice.orders").unchanged == 1

    tables["main.alice.orders"]["columns"]["channel"] = "Utf8"
    rep = sync_asset(g, cat, "main.alice.orders")
    assert rep.changed == ["main.alice.orders"]
    assert g.current("field:main.alice.orders/channel")["datatype"] == "String"
    assert len({e.correlation for e in g.log() if e.correlation == rep.correlation}) == 1

    desc = g.propose("dataset:main.alice.orders", "description", "Orders", source=Human("a"))
    g.accept(desc.id, actor=Human("a"))
    rep = asset_removed(g, cat, "main.alice.orders")
    assert rep.gone == ["main.alice.orders"] and g.get(desc.id).status is Status.STALE
    assert g.store.get_asset("dict", "main.alice.orders") is None


def test_reported_rename_moves_the_facts(g):
    tables = _tables()
    cat = DictCatalog(tables)
    sync(g, cat)
    desc = g.propose("field:main.alice.orders/amount", "description", "Order value", source=Human("a"))
    g.accept(desc.id, actor=Human("a"))

    tables["main.alice.order_lines"] = tables.pop("main.alice.orders")
    rep = asset_moved(g, cat, "main.alice.orders", "main.alice.order_lines")
    assert rep.moved == {"main.alice.orders": "main.alice.order_lines"} and rep.staled == 0
    assert g.current("field:main.alice.order_lines/amount")["description"] == "Order value"
    assert g.current("dataset:main.alice.order_lines")["source"] == "main.alice.order_lines"
    rep = sync(g, cat)  # and a later full sync agrees: nothing new, nothing gone
    assert (rep.new, rep.gone, rep.unchanged) == ([], [], 2)


def test_pull_sync_spots_a_rename_by_id_and_columns(g):
    tables = _tables()
    sync(g, DictCatalog(tables))
    desc = g.propose("dataset:main.alice.orders", "description", "Orders", source=Human("a"))
    g.accept(desc.id, actor=Human("a"))

    tables["main.sales.orders"] = tables.pop("main.alice.orders")  # moved to another schema, same id
    rep = sync(g, DictCatalog(tables))
    assert rep.moved == {"main.alice.orders": "main.sales.orders"} and rep.gone == [] and rep.new == []
    assert g.current("dataset:main.sales.orders")["description"] == "Orders"
    assert g.get(desc.id).status is Status.ACCEPTED

    # an id reused by a different table is not a rename
    del tables["main.bob.payroll"]
    tables["main.eve.notes"] = {"columns": {"note": "Utf8"}, "id": "2"}
    rep = sync(g, DictCatalog(tables))
    assert rep.moved == {} and rep.gone == ["main.bob.payroll"] and rep.new == ["main.eve.notes"]


def test_a_changed_description_is_proposed_again_and_nothing_goes_stale(g, tmp_path, examples):
    p = tmp_path / "cat.yaml"
    p.write_text((examples / "catalog.yaml").read_text())
    sync(g, FileCatalog(p))
    before = len(g.find(subject="dataset:shop.orders", attribute="description"))
    text = p.read_text()
    assert "Orders placed in the web shop" in text
    p.write_text(text.replace("Orders placed in the web shop", "Order lines placed in the web shop"))
    rep = sync(g, FileCatalog(p))
    assert rep.changed == ["shop.orders"] and rep.staled == 0 and rep.proposed == 1
    rows = g.find(subject="dataset:shop.orders", attribute="description")
    assert len(rows) == before + 1 and {a.status for a in rows} == {Status.PROPOSED}


def test_sync_never_accepts_meaning_even_with_a_permissive_run(g, examples):
    rep = sync(g, FileCatalog(examples / "catalog.yaml"))
    assert rep.accepted and all(
        gloss.model.is_structural(a.subject_id, a.attribute) for a in g.find(statuses=[Status.ACCEPTED])
    )
    accepted = [e for e in g.log() if e.type is EventType.ACCEPTED]
    assert all(e.data["auto"] and e.data["rule"] == "catalog-structure" for e in accepted)
    with gloss.open("sqlite://") as h:
        rep = sync(h, FileCatalog(examples / "catalog.yaml"), auto_accept_structure=False)
        assert rep.accepted == 0 and h.find(statuses=[Status.ACCEPTED]) == []


def test_accepted_descriptions_are_written_back_to_the_catalog(g):
    cat = DictCatalog(_tables())
    sync(g, cat)
    stop = write_back(g, cat)
    d = g.propose("dataset:main.alice.orders", "description", "Orders", source=Human("a"))
    f = g.propose("field:main.alice.orders/amount", "description", "Order value", source=Human("a"))
    assert cat.written == {}  # a proposal is not meaning yet
    g.bulk("accept", gloss.Query(dataset="main.alice.orders", attribute="description"), actor=Human("a"))
    assert cat.written == {"main.alice.orders": {"description": "Orders",
                                                 "fields": {"amount": {"description": "Order value"}}}}
    assert {d.id, f.id} == {a.id for a in g.find(attribute="description")}
    stop()
    g.propose("dataset:main.bob.payroll", "description", "Payroll", source=Human("a"), accept=True)
    assert "main.bob.payroll" not in cat.written


def test_subscribers_get_committed_events_with_seq(g):
    seen = []
    stop = g.subscribe(seen.append)
    a = g.propose("field:t/a", "description", "x", source=Human("a"))
    g.accept(a.id, actor=Human("a"))
    assert [[e.type.value for e in batch] for batch in seen] == [["proposed"], ["accepted"]]
    assert [e.seq for batch in seen for e in batch] == [1, 2]

    def broken(events):
        raise RuntimeError("subscriber bug")

    g.subscribe(broken)
    assert g.propose("field:t/a", "label", "x", source=Human("a")).status is Status.PROPOSED  # write survives
    stop()
    g.propose("field:t/a", "synonyms", ["y"], source=Human("a"))
    assert len(seen) == 3


def test_correlation_groups_one_action(g):
    rows = g.propose_many([gloss.Proposal("field:t/a", "description", "x"),
                           gloss.Proposal("field:t/b", "description", "y")], source=Human("a"))
    one = {e.correlation for e in g.log()}
    assert len(one) == 1 and None not in one
    g.as_actor(Human("b"), correlation="req-42").transition_many("accept", [r.assertion.id for r in rows])
    assert [e.type.value for e in g.log(correlation="req-42")] == ["accepted", "accepted"]
