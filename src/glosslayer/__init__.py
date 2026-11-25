"""Gloss: a catalog-agnostic semantic layer built on Apache Ossie.

    import glosslayer
    from glosslayer import Table, Column, Human, Int, String

    g = glosslayer.open("sqlite:///gloss.db")
    t = Table("shop.returns", primary_key=[Column("return_id", Int(32))])
    t.add_columns(Column("reason", String(255)))
    g.add(t, source=Human("alice"))
"""

from . import checks, dtypes, plugins, schema
from .actors import AI, Actor, ActorDirectory, Agent, Human, Import, Inferred, System
from .dtypes import Boolean, Date, DateTime, Decimal, DType, Float, Int, Opaque, String, Time, UInt
from .ids import Kind, SubjectId, subject
from .ledger import Gloss, ModelState, Proposal, ProposeResult, ReadRecord, RekeyResult, open
from .model import (
    Assertion,
    AttributeSpec,
    Confidentiality,
    Event,
    EventType,
    InvalidTransition,
    Status,
    register_attribute,
)
from .modeling import AddResult, Column, Metric, Relationship, SemanticType, Table
from .policy import Context, Decision, DefaultPolicy, PermissionDenied, Policy, RolePolicy, Scope
from .seams import AssetSchema, Catalog, Coverage, Sink, Source
from .semantic import TypeSuggester
from .store import ConcurrentWrite, Query, SQLStore, Store

__version__ = "0.2.0"

__all__ = [
    "AI",
    "Actor",
    "ActorDirectory",
    "AddResult",
    "Agent",
    "Assertion",
    "AssetSchema",
    "AttributeSpec",
    "Boolean",
    "Catalog",
    "Column",
    "ConcurrentWrite",
    "Confidentiality",
    "Context",
    "Coverage",
    "DType",
    "Date",
    "DateTime",
    "Decimal",
    "Decision",
    "DefaultPolicy",
    "Event",
    "EventType",
    "Float",
    "Gloss",
    "Human",
    "Import",
    "Inferred",
    "Int",
    "InvalidTransition",
    "Kind",
    "Metric",
    "ModelState",
    "Opaque",
    "PermissionDenied",
    "Policy",
    "Proposal",
    "ProposeResult",
    "Query",
    "ReadRecord",
    "RekeyResult",
    "Relationship",
    "RolePolicy",
    "SQLStore",
    "Scope",
    "SemanticType",
    "Sink",
    "Source",
    "Status",
    "Store",
    "String",
    "SubjectId",
    "System",
    "Table",
    "Time",
    "TypeSuggester",
    "UInt",
    "checks",
    "dtypes",
    "open",
    "plugins",
    "register_attribute",
    "schema",
    "subject",
]
