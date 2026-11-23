"""CLI, REST server, visualisation, plugins and the git sink."""

import shutil
import subprocess

import pytest

import glosslayer
from glosslayer import plugins
from glosslayer.cli import main
from glosslayer.ossie import OssieSource
from glosslayer.viz import html_report, mermaid_er


def test_cli_end_to_end(tmp_path, examples, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = ["--db", f"sqlite:///{tmp_path / 'g.db'}", "--actor", "human:test"]
    assert main(db + ["sync", f"file:{examples / 'catalog.yaml'}"]) == 0
    assert main(db + ["import", "ossie", str(examples / "semantic_model.yaml"), "--key-by", "source"]) == 0
    assert main(db + ["accept", "--all"]) == 2  # refuses without a filter
    assert main(db + ["accept", "--all", "--source", "import:"]) == 0
    assert main(db + ["export", "--validate", "-o", "model.yaml"]) == 0
    assert (tmp_path / "model.yaml").read_text().startswith("version: 0.3.0.dev0")
    capsys.readouterr()
    assert main(db + ["show", "field:shop.orders/amount"]) == 0
    out = capsys.readouterr().out
    assert "Order value including VAT" in out
    assert main(db + ["propose", "field:shop.orders/amount", "label", "Amount", "--source-as", "ai:x"]) == 0
    assert main(db + ["--json", "pending", "--source", "ai:"]) == 0
    assert '"label"' in capsys.readouterr().out
    assert main(db + ["viz", "--html", "report.html"]) == 0
    assert main(db + ["rebuild"]) == 0
    assert main(db + ["accept", "nope1234"]) == 2


def test_viz(g, examples):
    g.propose_many(OssieSource(examples / "semantic_model.yaml").proposals(), source="import:ossie")
    g.bulk("accept", glosslayer.Query(dataset="orders"), actor="human:a")
    er = mermaid_er(g)
    assert er.startswith("erDiagram")
    assert "Decimal amount" in er
    assert '"pending' in er and "restricted" in er
    assert 'orders }o--|| customers : "orders_customers (pending)"' in er
    assert "<pre class=\"mermaid\">" in html_report(g)


def test_builtin_plugins_resolve(examples):
    cat = plugins.instantiate("catalog", f"file:{examples / 'catalog.yaml'}")
    assert "shop.orders" in cat.list_assets()
    assert plugins.load("sink", "ossie").name == "ossie"
    with pytest.raises(LookupError):
        plugins.load("catalog", "does-not-exist")


def test_rest_api(g, examples):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from glosslayer.server import create_app

    c = TestClient(create_app(g))
    H = {"X-Gloss-Actor": "human:alice"}
    r = c.post("/assertions", json={"subject": "field:a/x", "attribute": "description", "value": "hi"})
    assert r.status_code == 401
    r = c.post(
        "/assertions",
        json={"subject": "field:a/x", "attribute": "description", "value": "hi", "source": "ai:bot"},
        headers=H,
    )
    assert r.status_code == 201
    aid = r.json()["assertion"]["id"]
    assert c.post(f"/assertions/{aid}/accept", headers={"X-Gloss-Actor": "ai:bot"}).status_code == 403
    assert c.post(f"/assertions/{aid}/accept", headers=H, json={"comment": "ok"}).json()["status"] == "accepted"
    assert c.get("/subjects/field:a/x").status_code == 401  # reads need an actor too
    assert c.get("/subjects/field:a/x", headers=H).json()["current"] == {"description": "hi"}
    assert len(c.get("/subjects/field:a/x/history", headers=H).json()) == 2
    page = c.get("/assertions", params={"status": "accepted"}, headers=H).json()
    assert page["items"][0]["id"] == aid and page["total"] == 1
    assert c.get("/progress", headers=H).json()["a"]["accepted"] == 1
    assert c.get("/export/ossie", headers=H).text.startswith("version:")
    assert c.post("/assertions/zzzz9999/accept", headers=H).status_code == 404
    assert c.get("/health").json()["ok"] is True


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_sink(g, examples, tmp_path):
    from glosslayer.sinks import GitSink

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    g.propose_many(OssieSource(examples / "semantic_model.yaml").proposals(), source="import:ossie")
    g.bulk("accept", glosslayer.Query(), actor="human:a")
    GitSink(repo, "models/shop.yaml").publish(g)
    log = subprocess.run(["git", "-C", str(repo), "log", "--oneline"], capture_output=True, text=True).stdout
    assert "publish semantic model (log seq" in log
    assert (repo / "models" / "shop.yaml").exists()
