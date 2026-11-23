from pathlib import Path

import pytest
import yaml

import gloss
from gloss.ossie import OssieSource, load_document, to_ossie, validate

HERE = Path(__file__).parent


def _accept_all(g):
    g.bulk("accept", gloss.Query(), actor="human:a")


def _strip_ext(doc):
    """Remove gloss annotations so documents can be compared with the original."""
    if isinstance(doc, dict):
        return {k: _strip_ext(v) for k, v in doc.items() if k != "nonportable_extensions"}
    if isinstance(doc, list):
        return [_strip_ext(v) for v in doc]
    return doc


def test_upstream_example_round_trips(g):
    """Import an example from the Apache Ossie repo, accept it, export it: same model, still valid."""
    original = yaml.safe_load((HERE / "ossie_renamed_fields.yaml").read_text())
    g.propose_many(OssieSource(original).proposals(), source="import:ossie")
    _accept_all(g)
    res = to_ossie(g)
    assert validate(res.document) == []
    assert not res.warnings

    got = _strip_ext(res.document)
    by_name = lambda items: {i["name"]: i for i in items}  # noqa: E731
    assert by_name(got["datasets"]) == by_name(original["datasets"])
    assert by_name(got["relationships"]) == by_name(original["relationships"])
    assert by_name(got["metrics"]) == by_name(original["metrics"])
    assert got["name"] == original["name"] and got["description"] == original["description"]
    # field order is preserved
    orders = by_name(got["datasets"])["orders"]
    assert [f["name"] for f in orders["fields"]] == [f["name"] for f in by_name(original["datasets"])["orders"]["fields"]]


def test_only_accepted_by_default(g, examples):
    g.propose_many(OssieSource(examples / "semantic_model.yaml").proposals(), source="import:ossie")
    with pytest.raises(ValueError):
        to_ossie(g)  # nothing accepted yet
    res = to_ossie(g, include_pending=True)
    assert validate(res.document) == []
    field = res.document["datasets"][0]["fields"][0]
    assert field["nonportable_extensions"][0]["data"]["pending"]


def test_confidentiality_travels_in_extensions(g, examples):
    g.propose_many(OssieSource(examples / "semantic_model.yaml").proposals(), source="import:ossie")
    _accept_all(g)
    assert g.current("field:customers/email")["confidentiality"] == "restricted"
    doc = to_ossie(g).document
    customers = next(d for d in doc["datasets"] if d["name"] == "customers")
    email = next(f for f in customers["fields"] if f["name"] == "email")
    assert email["nonportable_extensions"] == [
        {"vendor_name": "gloss", "data": {"confidentiality": "restricted", "classification": ["pii.email"]}}
    ]
    assert validate(doc) == []


def test_key_by_source_maps_relationships(g, examples):
    g.propose_many(OssieSource(examples / "semantic_model.yaml", key_by="source").proposals(), source="import:ossie")
    assert g.find(subject="relationship:orders_customers", attribute="from")[0].value == "shop.orders"
    _accept_all(g)
    doc = to_ossie(g).document
    # dotted keys export as simple Ossie names, and references follow
    assert {d["name"] for d in doc["datasets"]} == {"orders", "customers"}
    assert doc["relationships"][0]["from"] == "orders"
    assert validate(doc) == []


def test_incomplete_relationship_is_skipped_with_warning(g):
    g.propose("field:a/x", "datatype", "Integer", source="human:a")
    g.propose("relationship:r", "from", "a", source="human:a")
    _accept_all(g)
    res = to_ossie(g)
    assert "relationships" not in res.document
    assert any("relationship r" in w for w in res.warnings)
    assert validate(res.document) == []


def test_load_document_rejects_non_models(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("foo: bar\n")
    with pytest.raises(ValueError):
        load_document(p)
