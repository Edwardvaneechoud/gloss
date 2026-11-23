import pytest

from gloss.ids import Kind, SubjectId


@pytest.mark.parametrize(
    "text,kind,dataset,name",
    [
        ("model:", Kind.MODEL, None, None),
        ("dataset:sales.orders", Kind.DATASET, "sales.orders", None),
        ("field:sales.orders/amount", Kind.FIELD, "sales.orders", "amount"),
        ("relationship:orders_customers", Kind.RELATIONSHIP, None, "orders_customers"),
        ("metric:total_revenue", Kind.METRIC, None, "total_revenue"),
        ("metric:sales.orders/avg_price", Kind.METRIC, "sales.orders", "avg_price"),
    ],
)
def test_round_trip(text, kind, dataset, name):
    sid = SubjectId.parse(text)
    assert (sid.kind, sid.dataset, sid.name) == (kind, dataset, name)
    assert str(sid) == text


@pytest.mark.parametrize("bad", ["orders", "table:x", "field:orders", "dataset:", "field:/x"])
def test_invalid(bad):
    with pytest.raises(ValueError):
        SubjectId.parse(bad)
