"""End-to-end: three weeks of semantic modelling on a small web shop.

Cast
    Edward   data engineer: connects the catalog, models in code
    Alice    data steward: reviews and decides
    Claude   AI agent: reads the model over MCP and proposes improvements

Run it (needs the dev extras: pip install -e ".[dev]"):

    python examples/e2e_shop.py              # works in a fresh temporary folder
    python examples/e2e_shop.py ./demo       # or in a folder you choose, to inspect the results

It leaves behind: gloss.db (the ledger), shop.yaml (the Ossie model) and report.html.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import polars as pl
import yaml

import glosslayer
from glosslayer import Column, Human, Metric, Relationship, SemanticType, Status, String, Table, checks
from glosslayer.catalogs.file import FileCatalog
from glosslayer.mcp_server import build_server
from glosslayer.ossie import validate
from glosslayer.sync import sync
from glosslayer.viz import html_report, progress_table

EDWARD, ALICE = Human("edward"), Human("alice")


def step(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


def write_catalog(path: Path, *, week: int) -> FileCatalog:
    """The shop's catalog. In week 3 a column is dropped and another changes type."""
    orders = [
        {"name": "order_id", "dtype": "Int64"},
        {"name": "customer_id", "dtype": "Int64"},
        {"name": "order_date", "dtype": "Date"},
        {"name": "amount", "dtype": "Decimal(18,2)"},
    ]
    if week < 3:
        orders.append({"name": "status", "dtype": "Utf8"})
    newsletter_country = "Utf8" if week < 3 else "Int32"
    path.write_text(yaml.safe_dump({
        "name": "shop",
        "assets": {
            "shop.orders": {"primary_key": ["order_id"], "columns": orders},
            "shop.customers": {
                "primary_key": ["customer_id"],
                "columns": [
                    {"name": "customer_id", "dtype": "Int64"},
                    {"name": "email", "dtype": "Utf8"},
                    {"name": "country", "dtype": "Utf8"},
                    {"name": "signup_date", "dtype": "Date"},
                ],
            },
            "mkt.newsletter": {
                "columns": [
                    {"name": "subscriber_email", "dtype": "Utf8"},
                    {"name": "country", "dtype": newsletter_country},
                    {"name": "subscribed_at", "dtype": "Datetime(time_unit='us', time_zone='UTC')"},
                ],
            },
        },
    }, sort_keys=False))
    return FileCatalog(path)


class Agent:
    """Calls Gloss's MCP tools in-process, exactly as an MCP client such as Claude would."""

    def __init__(self, g: glosslayer.Gloss, name: str):
        self.server = build_server(g, agent=name)

    def __call__(self, tool: str, **args):
        result = asyncio.run(self.server.call_tool(tool, args))
        if isinstance(result, tuple):  # MCP SDK 1.x
            result = result[1] if result[1] is not None else result[0]
        data = getattr(result, "structured_content", None)
        if data is None:
            data = [json.loads(c.text) for c in getattr(result, "content", result)]
        data = data.get("result", data) if isinstance(data, dict) and set(data) == {"result"} else data
        print(f"  >>> {tool}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
        print("      " + json.dumps(data, ensure_ascii=False)[:300])
        return data


def main(workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    g = glosslayer.open(f"sqlite:///{workdir / 'gloss.db'}")
    catalog_file = workdir / "catalog.yaml"

    # ------------------------------------------------------------------ week 1
    step(1, "Week 1. Edward connects the catalog")
    report = sync(g, write_catalog(catalog_file, week=1))
    print(f"  {len(report.new)} tables found: {', '.join(report.new)}")
    print(f"  {report.accepted} structural facts accepted automatically (columns, types, keys)")
    print(f"  shop.orders/amount is now known as: {g.current('field:shop.orders/amount')}")

    step(2, "Edward adds meaning in code: two reusable types, descriptions, a join, a metric")
    Email = SemanticType(
        "email_address", dtype=String(320), description="An e-mail address",
        confidentiality="restricted", classification=["pii.email"], synonyms=["e-mail"],
        checks=[checks.matches(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")], name_patterns=[r"e[-_]?mail"],
    )
    Country = SemanticType(
        "country_code", dtype=String(2), description="ISO 3166-1 alpha-2 country code",
        checks=[checks.matches(r"^[A-Z]{2}$")], name_patterns=[r"^country(_code)?$"],
    )
    customers = Table("shop.customers", description="One row per customer account").add_columns(
        Column("customer_id"),
        Column("email", Email, description="Login e-mail address"),
        Column("country", Country),
    )
    orders = Table("shop.orders", description="One row per order placed in the web shop").add_columns(
        Column("customer_id"),
        Column("order_date", is_time=True, description="Date the order was placed"),
        Column("amount", description="Order value including VAT, in EUR", synonyms=["revenue", "turnover"]),
        Column("status", checks=[checks.one_of(["open", "shipped", "cancelled"])]),
    )
    orders.add_metric(Metric("revenue", "SUM(orders.amount)", description="Total order value incl. VAT"))
    result = g.add(
        customers, orders,
        Relationship("orders_customers", on={orders["customer_id"]: customers["customer_id"]}),
        source=EDWARD,
    )
    print(f"  saved: {result}   (corroborated = Edward confirmed what the catalog already said)")
    print("  nothing Edward wrote is live yet; it all waits for review:")
    print("  " + progress_table(g).replace("\n", "\n  "))

    step(3, "Alice reviews Edward's work")
    for a in g.pending(source_prefix="human:edward")[:4]:
        print(f"  {a!r}")
    print(f"  … {len(g.pending(source_prefix='human:edward'))} in total")
    done, _ = g.bulk("accept", glosslayer.Query(source_prefix="human:edward"), actor=ALICE,
                     comment="reviewed in the modelling session")
    print(f"  Alice accepted {len(done)} facts")
    email = g.effective("field:shop.customers/email")
    print(f"  the email column now inherits from its type: confidentiality={email['confidentiality']}, "
          f"classification={email['classification']}, {len(email['checks'])} check")

    step(4, "Gloss suggests the types for columns nobody has described yet")
    suggester = glosslayer.TypeSuggester()
    for r in g.propose_many(suggester.proposals(g), source=suggester.source_id):
        print(f"  suggested: {r.assertion.subject} is a {r.assertion.value} (confidence {r.assertion.confidence})")
    done, _ = g.bulk("accept", glosslayer.Query(source_prefix=suggester.source_id), actor=ALICE)
    print(f"  Alice accepted {len(done)} suggestions; mkt.newsletter/subscriber_email is now "
          f"{g.effective('field:mkt.newsletter/subscriber_email')['confidentiality']}")

    # ------------------------------------------------------------------ week 2
    step(5, "Week 2. An AI agent (Claude) uses the model over MCP")
    claude = Agent(g, "claude")
    claude("search_model", text="turnover")
    claude("get_model_context", datasets=["mkt.newsletter"])
    print("  Claude can see subscriber_email exists, but not what it contains: it is restricted.")
    claude("propose_fact", subject="dataset:mkt.newsletter", attribute="description",
           value="One row per newsletter subscription", confidence=0.8,
           rationale="table has a subscriber e-mail and a subscription timestamp")
    claude("propose_fact", subject="field:shop.orders/amount", attribute="synonyms",
           value=["revenue", "turnover", "profit"], confidence=0.4,
           rationale="users asked for profit per country")

    step(6, "Alice reviews what the agent proposed")
    for a in g.pending(source_prefix="ai:"):
        print(f"  {a!r}")
    desc, syn = sorted(g.pending(source_prefix="ai:"), key=lambda a: a.attribute)
    g.accept(desc.id, actor=ALICE, comment="correct")
    g.reject(syn.id, actor=ALICE, comment="amount is revenue, not profit: costs are not subtracted")
    print("  accepted the description, rejected 'profit' with a reason")
    again = claude("propose_fact", subject="field:shop.orders/amount", attribute="synonyms",
                   value=["revenue", "turnover", "profit"], confidence=0.9)
    assert again["outcome"] == "previously_rejected"
    print("  the same suggestion can't come back")

    step(7, "Data quality: run the checks on this week's data")
    customers_df = pl.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "email": ["ann@example.com", "bob@example", "cas@example.net", None],
        "country": ["NL", "BE", "nl", "DE"],
    })
    newsletter_df = pl.DataFrame({
        "subscriber_email": ["ann@example.com", "not an address"],
        "country": ["NL", "Netherlands"],
    })
    results = checks.run_polars(g, "shop.customers", customers_df) + checks.run_polars(g, "mkt.newsletter", newsletter_df)
    for r in results:
        print(f"  {r}")
        if r.sample:
            print(f"        e.g. {r.sample}")
    g.record_check_results(results, runner=EDWARD)
    print("  failing e-mail values are not shown: the type makes them restricted")

    step(8, "Publish: export the accepted model as Apache Ossie YAML")
    out = workdir / "shop.yaml"
    g.to_yaml(out, name="shop")
    doc = yaml.safe_load(out.read_text())
    print(f"  wrote {out.name}: {len(doc['datasets'])} datasets, {len(doc.get('relationships', []))} relationship, "
          f"valid Ossie: {validate(doc) == []}")
    amount = next(f for d in doc["datasets"] if d["name"] == "orders" for f in d["fields"] if f["name"] == "amount")
    print("  " + yaml.safe_dump({"amount": amount}, sort_keys=False).replace("\n", "\n  ").rstrip())

    # ------------------------------------------------------------------ week 3
    step(9, "Week 3. The catalog changes: 'status' is dropped, newsletter 'country' becomes Int32")
    report = sync(g, write_catalog(catalog_file, week=3))
    for note in report.notes:
        print(f"  - {note}")
    print(f"  {report.staled} facts went back to review:")
    for a in g.pending(statuses=[Status.STALE]):
        print(f"  {a!r}")

    step(10, "Alice resolves the stale facts")
    done, _ = g.bulk("retract", glosslayer.Query(subject="field:shop.orders/status", statuses=[Status.STALE]),
                     actor=ALICE, comment="column was dropped")
    print(f"  retracted {len(done)} facts about the dropped status column")
    st = g.find(subject="field:mkt.newsletter/country", attribute="semantic_type", statuses=[Status.STALE])[0]
    g.reject(st.id, actor=ALICE, comment="now a numeric country id, not an ISO code")
    print("  rejected country_code for mkt.newsletter/country: it is a numeric id now")
    print(f"  still open: {len(g.pending())} facts")

    step(11, "Audit: the full story of one column")
    for e in g.history("field:mkt.newsletter/country"):
        detail = e.data.get("comment") or e.data.get("reason") or ""
        if e.data.get("auto"):
            detail = f"by rule: {e.data['rule']}"  # nobody reviewed it, and the log says so
        what = f"{e.data['attribute']} = {e.data['value']}" if e.type.value == "proposed" else ""
        print(f"  #{e.seq:<4} {e.type.value:<13} {e.actor:<28} {what}{detail}")
    runs = g.check_history(dataset="mkt.newsletter")
    print(f"  plus {len(runs)} logged check results for mkt.newsletter")

    step(12, "Report")
    (workdir / "report.html").write_text(html_report(g, title="Shop semantic model"))
    print("  " + progress_table(g).replace("\n", "\n  "))
    print(f"\n  files in {workdir}: gloss.db, shop.yaml, report.html")
    g.close()


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="gloss-e2e-"))
    main(target)
