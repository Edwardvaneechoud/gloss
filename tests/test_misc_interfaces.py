"""CLI additions, programmatic plugins, actors, and the MCP server bound to a person."""

import pytest

import gloss
from gloss import AI, Human, Status, plugins
from gloss.actors import Actor, IdentityDirectory, Other
from gloss.cli import main


def test_cli_solo_mode_rename_and_restore(tmp_path, capsys):
    db = ["--db", f"sqlite:///{tmp_path / 'g.db'}", "--actor", "human:me", "--mode", "solo"]
    assert main(db + ["init"]) == 0
    assert "schema version 1" in capsys.readouterr().out
    assert main(db + ["propose", "dataset:shop.orders", "description", "Orders"]) == 0
    assert "accepted" in capsys.readouterr().out  # solo: no second step
    assert main(db + ["propose", "dataset:shop.orders", "description", "All orders"]) == 0
    assert main(db + ["rename", "dataset:shop.orders", "dataset:shop.order_lines"]) == 0
    assert "2 facts moved" in capsys.readouterr().out
    assert main(db + ["show", "dataset:shop.order_lines"]) == 0
    assert "All orders" in capsys.readouterr().out
    assert main(db + ["--json", "list", "--status", "superseded"]) == 0
    import json

    old = json.loads(capsys.readouterr().out)[0]["id"]
    assert main(db + ["restore", old]) == 0
    assert main(db + ["rename", "--namespace", "shop", "retail.shop"]) == 0
    assert main(db + ["show", "dataset:retail.shop.order_lines"]) == 0
    out = capsys.readouterr().out
    assert "description      Orders" in out
    assert main(db + ["rename", "dataset:nope", "dataset:other"]) == 2
    assert main(["--db", f"sqlite:///{tmp_path / 'g.db'}", "--actor", "nobody", "propose", "dataset:x",
                 "description", "y"]) == 2


def test_cli_accepting_one_fact_of_a_group_reports_the_rest(tmp_path, examples, capsys):
    db = ["--db", f"sqlite:///{tmp_path / 'g.db'}", "--actor", "human:me"]
    assert main(db + ["import", "ossie", str(examples / "semantic_model.yaml"), "--namespace", "shop"]) == 0
    capsys.readouterr()
    assert main(db + ["--json", "pending", "--kind", "relationship"]) == 0
    import json

    rows = json.loads(capsys.readouterr().out)
    assert rows and all(r["subject"].startswith("relationship:shop::") for r in rows)
    assert main(db + ["accept", rows[0]["id"], rows[1]["id"]]) == 0
    assert "(same group)" in capsys.readouterr().out
    assert main(db + ["pending", "--subject", rows[0]["subject"]]) == 0
    assert "nothing found" in capsys.readouterr().out


def test_plugins_can_be_registered_in_code(examples):
    class MemoryCatalog:
        name = "memory"

        def __init__(self, arg=None):
            self.arg = arg

    plugins.register("catalog", "memory", MemoryCatalog)
    try:
        assert plugins.load("catalog", "memory") is MemoryCatalog
        assert plugins.instantiate("catalog", "memory:x").arg == "x"
        assert "memory" in plugins.available("catalog")
        with pytest.raises(KeyError):
            plugins.register("nonsense", "x", MemoryCatalog)
    finally:
        plugins._REGISTERED["gloss.catalogs"].pop("memory")


def test_actor_parsing_is_strict():
    assert Actor.parse("human:alice") == Human("alice") and Actor.parse("ai:claude") == AI("claude")
    assert isinstance(Actor.parse("robot:r2"), Other) and Actor.parse("robot:r2") == "robot:r2"
    for bad in ("alice", "", ":x", "human:"):
        with pytest.raises(ValueError):
            Actor.parse(bad)
    assert IdentityDirectory().display_name("human:7f3a") == "7f3a"


def test_a_host_directory_names_actors(g):
    class Names:
        def display_name(self, actor):
            return {"human:7f3a": "Ann"}.get(actor)

    with gloss.open("sqlite://", directory=Names()) as h:
        a = h.propose("field:t/a", "description", "x", source=Human("7f3a"))
        assert a.created_by == "human:7f3a"  # the log holds the id
        assert h.directory.display_name(a.created_by) == "Ann"
        assert h.directory.display_name("human:gone") is None


def test_mcp_tools_act_for_the_bound_person(g):
    pytest.importorskip("mcp")
    from gloss.mcp_server import build_server

    g.propose("dataset:shop.orders", "description", "Orders", source=Human("a"), accept=True)
    reads = []
    g.on_read(reads.append)
    server = build_server(g.as_actor(Human("alice")), agent="claude")
    tools = getattr(server, "_tool_manager", None)
    if tools is None:
        pytest.skip("this MCP SDK does not expose its tool table")
    call = {t.name: t.fn for t in tools.list_tools()}
    assert call["search_model"]("orders")[0]["subject"] == "dataset:shop.orders"
    out = call["propose_fact"]("dataset:shop.orders", "synonyms", ["sales"], 0.7, "users say sales")
    a = g.get(out["id"])
    assert (a.source, a.created_by, a.status) == ("ai:claude", "human:alice", Status.PROPOSED)
    assert g.history("dataset:shop.orders")[-1].via == "ai:claude"
    assert reads and reads[0].actor == "human:alice" and reads[0].via == "ai:claude"
