import asyncio
import json

import pytest

from gloss import ai
from gloss.ossie import OssieSource
import gloss


@pytest.fixture
def shop(g, examples):
    g.propose_many(OssieSource(examples / "semantic_model.yaml").proposals(), source="import:ossie")
    g.bulk("accept", gloss.Query(), actor="human:a")
    return g


def test_context_redacts_confidential_fields(shop):
    ctx = ai.model_context(shop)
    customers = next(d for d in ctx["datasets"] if d["name"] == "customers")
    fields = {f["name"]: f for f in customers["fields"]}
    assert fields["email"] == {"name": "email", "datatype": "String", "confidentiality": "restricted", "redacted": True}
    assert fields["name"]["redacted"] is True
    assert fields["country"]["description"].startswith("ISO")
    assert ctx["visibility_limit"] == "internal"


def test_dataset_level_confidentiality_inherits(shop):
    a = shop.propose("dataset:orders", "confidentiality", "confidential", source="human:a")
    shop.accept(a.id, actor="human:a")
    ctx = ai.model_context(shop)
    orders = next(d for d in ctx["datasets"] if d["name"] == "orders")
    assert orders["redacted"] is True and "fields" not in orders
    assert ctx["relationships"] == []  # touches a redacted dataset
    assert all(h.get("redacted") for h in ai.search(shop, "amount"))


def test_search_uses_synonyms(shop):
    hits = ai.search(shop, "turnover")
    assert {h["subject"] for h in hits} == {"field:orders/amount", "metric:total_revenue"}


def test_review_status_hides_values_above_limit(shop):
    rows = ai.review_status(shop, "field:customers/email")
    desc = next(r for r in rows if r["attribute"] == "description")
    assert "value" not in desc


def test_mcp_tools(shop):
    pytest.importorskip("mcp")
    from gloss.mcp_server import build_server

    server = build_server(shop, agent="claude")
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"search_model", "get_model_context", "propose_fact", "review_status"}

    result = asyncio.run(
        server.call_tool(
            "propose_fact",
            {"subject": "field:orders/order_date", "attribute": "synonyms", "value": ["day"], "confidence": 0.7},
        )
    )
    text = json.dumps(result, default=str)
    assert "created" in text
    pending = shop.pending(source_prefix="ai:claude")
    assert len(pending) == 1 and pending[0].value == ["day"]
