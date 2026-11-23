"""Plugin discovery through Python entry points.

A package registers plugins in its ``pyproject.toml``:

    [project.entry-points."gloss.catalogs"]
    flowfile = "glosslayer_flowfile:FlowfileCatalog"

    [project.entry-points."gloss.sources"]
    flowfile-lineage = "glosslayer_flowfile:LineageSource"

Each entry point is a class (or factory). From the CLI a plugin is addressed
as ``name`` or ``name:argument``; the argument is passed to ``from_spec`` if
the class defines it, otherwise to the constructor.

A host that embeds Gloss can skip entry points altogether and register in code,
which also works where distribution metadata is missing (a frozen application):

    glosslayer.plugins.register("catalog", "flowfile", FlowfileCatalog)
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

GROUPS = {
    "catalog": "gloss.catalogs",
    "source": "gloss.sources",
    "sink": "gloss.sinks",
}

# Built-ins are also declared as entry points in pyproject.toml; this fallback
# keeps them available when running from a source checkout that is not installed.
_BUILTINS = {
    "gloss.catalogs": {"file": "glosslayer.catalogs.file:FileCatalog"},
    "gloss.sources": {
        "ossie": "glosslayer.ossie.importer:OssieSource",
        "semantic-types": "glosslayer.semantic:TypeSuggester",
    },
    "gloss.sinks": {"ossie": "glosslayer.ossie.exporter:OssieSink", "git": "glosslayer.sinks:GitSink"},
}


def _load_target(target: str) -> Any:
    import importlib

    module, _, attr = target.partition(":")
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


_REGISTERED: dict[str, dict[str, Any]] = {group: {} for group in GROUPS.values()}


def register(kind: str, name: str, plugin: Any) -> None:
    """Register a plugin class (or factory) in code. It wins over an entry point of the same name."""
    _REGISTERED[GROUPS[kind]][name] = plugin


def _entry_points(group: str) -> list[Any]:
    try:
        return list(entry_points(group=group))
    except Exception:  # metadata can be missing or unreadable inside a frozen application
        return []


def available(kind: str) -> dict[str, str]:
    """Plugin names of ``kind`` and where each comes from (``module:attr``).

    Code registrations win over entry points, which win over built-ins.
    """
    group = GROUPS[kind]
    found = dict(_BUILTINS.get(group, {}))
    for ep in _entry_points(group):
        found[ep.name] = ep.value
    for name, plugin in _REGISTERED[group].items():
        found[name] = f"{getattr(plugin, '__module__', '?')}:{getattr(plugin, '__qualname__', plugin)}"
    return found


def load(kind: str, name: str) -> Any:
    """The plugin class (or factory) called ``name``, in the order :func:`available` ranks them.

    ``LookupError`` names what is installed.
    """
    group = GROUPS[kind]
    if name in _REGISTERED[group]:
        return _REGISTERED[group][name]
    for ep in _entry_points(group):
        if ep.name == name:
            return ep.load()
    if name in _BUILTINS.get(group, {}):
        return _load_target(_BUILTINS[group][name])
    known = ", ".join(sorted(available(kind))) or "none"
    raise LookupError(f"No {kind} plugin named {name!r} (installed: {known})")


def instantiate(kind: str, spec: str, **kwargs: Any) -> Any:
    """Create a plugin from a ``name[:argument]`` spec."""
    name, sep, arg = spec.partition(":")
    cls = load(kind, name)
    if sep:
        if hasattr(cls, "from_spec"):
            return cls.from_spec(arg, **kwargs)
        return cls(arg, **kwargs)
    return cls(**kwargs)
