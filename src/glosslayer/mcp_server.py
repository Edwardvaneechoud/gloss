"""MCP server: lets AI agents read the accepted model and propose changes.

AI can *read* (with confidentiality enforced by :mod:`glosslayer.ai`) and *propose*.
It can never accept: proposals land in the review queue as ``ai:<agent>``.

    gloss mcp --db sqlite:///gloss.db --agent claude

Embedded in a host, pass a view bound to the person the agent works for and the agent
reads what that person may read, capped at the AI level:

    build_server(g.as_actor(Human(user.id)), agent="claude")
"""

from __future__ import annotations

from typing import Any

from . import ai
from .ids import SubjectId
from .ledger import Gloss, Proposal
from .model import ATTRIBUTES


def _propose_doc() -> str:
    return (
        "Propose a semantic fact for human review. It is NOT applied until a person accepts it. "
        "Subject examples: 'dataset:sales.orders', 'field:sales.orders/amount', 'metric:total_revenue', "
        "'metric:sales.orders/avg_price', 'metric:sales::revenue' (owned by the namespace 'sales'), "
        "'relationship:sales::orders_customers'. Valid attributes per subject kind: "
        + "; ".join(f"{k.value}: {', '.join(sorted(v))}" for k, v in ATTRIBUTES.items())
        + ". Give a confidence between 0 and 1 and a short rationale. Facts that belong together "
        "(the four parts of a relationship, the parts of a metric) should share a `group` string."
    )


def _server_class() -> Any:
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:  # mcp 1.x
        try:
            from mcp.server.fastmcp import FastMCP

            return FastMCP
        except ImportError as exc:
            raise RuntimeError("The MCP server needs the 'mcp' extra: pip install glosslayer[mcp]") from exc


def build_server(gloss: Gloss, *, agent: str = "assistant") -> Any:
    """An MCP server whose tools read and propose as ``ai:<agent>``, or as the bound person via that agent."""
    Server = _server_class()
    server = Server(
        name="gloss",
        instructions=(
            "Gloss holds the reviewed semantic model of this organisation's data: datasets, fields, "
            "relationships and metrics with their agreed meaning. Call get_model_context before writing "
            "queries or answering questions about data, and search_model to find the right dataset or metric. "
            "If a definition is missing or wrong, call propose_fact; a person will review it. "
            "Items marked redacted exist but are confidential: do not guess their contents."
        ),
    )
    source = f"ai:{agent}"
    # An unbound ledger: the agent acts for itself. A person's view: the person acts through the agent.
    view = gloss.as_actor(source) if gloss.actor is None else gloss.as_actor(gloss.actor, via=source)

    @server.tool()
    def search_model(text: str, limit: int = 20, kinds: list[str] | None = None) -> list[dict[str, Any]]:
        """Search the agreed data model: tables, columns, relationships, metrics and semantic types,
        by name, description or synonym. Plain questions work ("revenue per country"); filler words are
        ignored and prefixes count. Hits matching every word come first. If none do, hits matching some
        words are returned, each with `matched` listing which. Misspellings are corrected (`did_you_mean`).
        `kinds` narrows the search, e.g. ["metric"] or ["field", "dataset"]. Items marked redacted exist
        but are confidential: don't guess their contents. Follow up with get_model_context for details."""
        return ai.search(view, text, limit=limit, kinds=kinds)

    @server.tool()
    def get_model_context(datasets: list[str] | None = None) -> dict[str, Any]:
        """Return the accepted semantic model (optionally only the given dataset keys) as compact JSON."""
        return ai.model_context(view, datasets=datasets)

    @server.tool(description=_propose_doc())
    def propose_fact(
        subject: str,
        attribute: str,
        value: Any,
        confidence: float | None = None,
        rationale: str | None = None,
        group: str | None = None,
    ) -> dict[str, Any]:
        SubjectId.parse(subject)
        evidence = [{"rationale": rationale}] if rationale else []
        res = view.propose_many(
            [Proposal(subject, attribute, value, confidence, evidence, rationale, group)], source=source
        )[0]
        return {"outcome": res.outcome, "id": res.assertion.id, "status": res.assertion.status.value}

    @server.tool()
    def review_status(subject: str) -> list[dict[str, Any]]:
        """Every fact about one subject with its review status: what is agreed and what is still open."""
        return ai.review_status(view, subject)

    return server


def run(gloss: Gloss, *, agent: str = "assistant", transport: str = "stdio") -> None:
    """Build the MCP server and serve it over ``transport`` (stdio, sse or streamable-http) until stopped."""
    build_server(gloss, agent=agent).run(transport=transport)
