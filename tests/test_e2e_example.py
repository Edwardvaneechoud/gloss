"""examples/e2e_shop.py must keep working: it is the end-to-end story in the README."""

import importlib.util
from pathlib import Path

import pytest
import yaml

pytest.importorskip("polars")
pytest.importorskip("mcp")

ROOT = Path(__file__).resolve().parents[1]


def test_e2e_shop_story(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("e2e_shop", ROOT / "examples" / "e2e_shop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(tmp_path)
    out = capsys.readouterr().out

    assert "3 tables found" in out
    assert "suggested: field:mkt.newsletter/subscriber_email is a email_address" in out
    assert '"redacted": true' in out
    assert '"outcome": "previously_rejected"' in out
    assert "valid Ossie: True" in out
    assert "type of mkt.newsletter.country changed from String to Integer" in out
    assert "still open: 0 facts" in out
    for name in ("gloss.db", "shop.yaml", "report.html"):
        assert (tmp_path / name).exists()
    doc = yaml.safe_load((tmp_path / "shop.yaml").read_text())
    assert {d["name"] for d in doc["datasets"]} == {"orders", "customers", "newsletter"}
