"""Command line interface.

    gloss init
    gloss import ossie model.yaml [--accept]
    gloss sync file:catalog.yaml
    gloss pending [--dataset sales.orders] [--source ai:]
    gloss accept <id>... | --all --dataset sales.orders
    gloss rename dataset:sales.orders dataset:sales.order_lines
    gloss show field:sales.orders/amount
    gloss history field:sales.orders/amount
    gloss progress
    gloss export ossie -o model.yaml [--validate]
    gloss viz [--html report.html]
    gloss serve | gloss mcp
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from typing import Any

from . import plugins
from .ledger import Gloss
from .ledger import open as open_gloss
from .model import OPEN_STATUSES, Status
from .policy import MODES, DefaultPolicy
from .store import Query

DEFAULT_DB = "sqlite:///gloss.db"


def _default_actor() -> str:
    return os.environ.get("GLOSS_ACTOR") or f"human:{getpass.getuser()}"


def _short(aid: str) -> str:
    return aid[:8]


def _fmt_value(v: Any, width: int = 60) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s if len(s) <= width else s[: width - 1] + "…"


def _print_assertions(rows: list[Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps([a.to_dict() for a in rows], indent=2, default=str))
        return
    if not rows:
        print("nothing found")
        return
    for a in rows:
        conf = f" {a.confidence:.2f}" if a.confidence is not None else ""
        print(
            f"{_short(a.id)}  {a.status.value:<10} {a.subject}  {a.attribute} = {_fmt_value(a.value)}"
            f"   [{a.source}{conf}]"
        )


def _query_from(args: argparse.Namespace, statuses: list[Status] | None) -> Query:
    return Query(
        subject=getattr(args, "subject", None),
        dataset=getattr(args, "dataset", None),
        namespace=getattr(args, "namespace", None),
        kind=getattr(args, "kind", None),
        attribute=getattr(args, "attribute", None),
        source_prefix=getattr(args, "source", None),
        group=getattr(args, "group", None),
        statuses=statuses,
        limit=getattr(args, "limit", None),
    )


# ---------------------------------------------------------------------------- commands
def cmd_init(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss init``: report the schema version; opening the ledger in ``main`` has created it."""
    from . import schema

    print(f"Gloss ledger ready at {args.db} (schema version {schema.current_version(g.store.engine)})")
    return 0


def cmd_import(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss import``: propose every fact from a source plugin; ``--accept`` accepts them as ``--actor``."""
    kwargs = {"key_by": args.key_by, "namespace": args.namespace} if args.plugin == "ossie" else {}
    src = plugins.instantiate("source", f"{args.plugin}:{args.target}" if args.target else args.plugin, **kwargs)
    results = g.propose_many(
        src.proposals(g), source=src.source_id, actor=args.actor, accept=args.accept,
        comment=f"bulk accepted on import by {args.actor}" if args.accept else None,
    )
    counts: dict[str, int] = {}
    for r in results:
        counts[r.outcome] = counts.get(r.outcome, 0) + 1
    print(f"{len(results)} facts from {src.source_id}: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    if args.accept:
        print(f"accepted {len({r.assertion.id for r in results if r.accepted})} facts as {args.actor}")
    return 0


def cmd_sync(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss sync``: sync with a catalog plugin and print what was new, changed, moved or gone."""
    from .sync import sync

    catalog = plugins.instantiate("catalog", args.catalog)
    rep = sync(g, catalog, auto_accept_structure=not args.no_auto_accept, detect_removed=not args.keep_removed)
    moved = f"{len(rep.moved)} moved, " if rep.moved else ""
    print(
        f"catalog {rep.catalog}: {len(rep.new)} new, {len(rep.changed)} changed, {moved}{len(rep.gone)} gone, "
        f"{rep.unchanged} unchanged | {rep.proposed} proposed, {rep.accepted} accepted, {rep.staled} marked stale"
    )
    for note in rep.notes:
        print(f"  - {note}")
    return 0


def cmd_propose(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss propose``: propose one fact; the value is read as JSON when it parses, else as text."""
    try:
        value = json.loads(args.value)
    except json.JSONDecodeError:
        value = args.value
    a = g.propose(
        args.subject,
        args.attribute,
        value,
        source=args.source_as or args.actor,
        actor=args.actor,
        confidence=args.confidence,
        comment=args.comment,
    )
    _print_assertions([a], args.json)
    return 0


def cmd_list(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss list`` and ``gloss pending``: print facts matching the filters (pending: open ones)."""
    statuses = [Status(s) for s in args.status] if args.status else None
    if args.command == "pending" and statuses is None:
        statuses = list(OPEN_STATUSES)
    _print_assertions(g.store.find(_query_from(args, statuses)), args.json)
    return 0


def cmd_action(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss accept`` and the other review actions: on the given ids, or with ``--all`` on every match."""
    action = args.command
    if args.all:
        statuses = [Status(s) for s in args.status] if args.status else list(OPEN_STATUSES)
        q = _query_from(args, statuses)
        if not any([q.subject, q.dataset, q.namespace, q.kind, q.attribute, q.source_prefix, q.group]) \
                and not args.yes:
            print("Refusing to act on everything without a filter. Add --dataset/--source/... or --yes.")
            return 2
        done, errors = g.bulk(action, q, actor=args.actor, comment=args.comment)
        print(f"{action}: {len(done)} changed, {len(errors)} failed")
        for aid, err in errors[:20]:
            print(f"  {_short(aid)}: {err}")
        return 1 if errors else 0
    if not args.ids:
        print("Give assertion ids or use --all with filters.")
        return 2
    fn = getattr(g, action)
    seen: set[str] = set()
    for aid in args.ids:
        full = g.get(aid).id
        if full in seen:  # already decided with its group
            continue
        before = g.store.last_seq()
        a = fn(full, actor=args.actor, comment=args.comment)
        print(f"{_short(a.id)}  -> {a.status.value}")
        if action in ("accept", "reject"):
            for e in g.log(after_seq=before):
                if e.type.value == a.status.value and e.assertion_id != a.id:
                    seen.add(e.assertion_id)
                    print(f"{_short(e.assertion_id)}  -> {a.status.value}  (same group)")
    return 0


def cmd_rename(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss rename``: re-key a subject, or a whole path with ``--namespace``, so its facts follow."""
    fn = g.rekey_namespace if args.namespace else g.rekey
    res = fn(args.old, args.new, actor=args.actor, comment=args.comment)
    print(
        f"{len(res.subjects)} subject(s) re-keyed: {res.moved} facts moved, {res.rewritten} references rewritten, "
        f"{res.superseded} superseded, {res.staled} marked stale"
    )
    return 0


def cmd_show(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss show``: a subject's accepted values, then its facts that are not accepted."""
    current = g.current(args.subject)
    rows = g.find(subject=str(args.subject))
    if args.json:
        print(json.dumps({"current": current, "assertions": [a.to_dict() for a in rows]}, indent=2, default=str))
        return 0
    print(args.subject)
    for k, v in current.items():
        print(f"  {k:<16} {_fmt_value(v, 80)}")
    other = [a for a in rows if a.status is not Status.ACCEPTED]
    if other:
        print("  not accepted:")
        for a in other:
            print(f"    {_short(a.id)} {a.status.value:<10} {a.attribute} = {_fmt_value(a.value)} [{a.source}]")
    return 0


def cmd_history(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss history``: the events of one subject, or the whole log after ``--after``."""
    events = g.history(args.subject) if args.subject else g.log(after_seq=args.after, limit=args.limit)
    if args.json:
        print(json.dumps([e.to_dict() for e in events], indent=2, default=str))
        return 0
    for e in events:
        detail = ""
        if e.type.value == "proposed":
            detail = f"{e.data['subject']} {e.data['attribute']} = {_fmt_value(e.data['value'], 50)}"
        elif e.type.value == "corroborated":
            detail = f"also stated by {e.data.get('source')}"
        elif e.type.value == "rekeyed":
            detail = f"was {e.data['from_subject']}" if e.data.get("from_subject") else "a subject it names moved"
        elif e.data.get("comment"):
            detail = e.data["comment"]
        elif e.data.get("auto"):
            detail = f"by rule: {e.data.get('rule')}"
        if e.via:
            detail = f"{detail}  (via {e.via})".strip()
        print(f"#{e.seq:<6} {e.ts:%Y-%m-%d %H:%M:%S}  {e.type.value:<13} {_short(e.assertion_id)}  {e.actor:<22} {detail}")
    return 0


def cmd_progress(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss progress``: review progress per dataset, as a table or JSON."""
    from .viz import progress_table

    if args.json:
        print(json.dumps(g.progress(args.dataset), indent=2))
    else:
        print(progress_table(g, args.dataset))
    return 0


def cmd_export(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss export``: write the model as Ossie YAML or publish it through a sink plugin."""
    opts: dict[str, Any] = {"include_pending": args.include_pending, "annotate": not args.no_annotate}
    if args.name:
        opts["name"] = args.name
    if args.dataset:
        opts["dataset_prefix"] = args.dataset
    if args.plugin == "ossie" and not args.target:
        from .ossie import to_ossie

        result = to_ossie(g, **opts)
        if args.output:
            with open(args.output, "w") as fh:
                fh.write(result.to_yaml())
        else:
            sys.stdout.write(result.to_yaml())
    else:
        sink = plugins.instantiate("sink", f"{args.plugin}:{args.target}" if args.target else args.plugin)
        if args.output:
            opts["path"] = args.output
        result = sink.publish(g, **opts)
    for w in getattr(result, "warnings", []):
        print(f"warning: {w}", file=sys.stderr)
    if args.validate and args.plugin in ("ossie", "git"):
        from .ossie import validate

        errors = validate(result.document)
        for err in errors:
            print(f"invalid: {err}", file=sys.stderr)
        if errors:
            return 1
        print("valid Ossie document", file=sys.stderr)
    return 0


def cmd_viz(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss viz``: print a Mermaid ER diagram, or write an HTML report with ``--html``."""
    from .viz import html_report, mermaid_er

    if args.html:
        with open(args.html, "w") as fh:
            fh.write(html_report(g))
        print(f"wrote {args.html}")
    else:
        print(mermaid_er(g, include_pending=not args.accepted_only, dataset_prefix=args.dataset))
    return 0


def cmd_types(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss types``: accepted semantic types and the fields that use them."""
    from .checks import Check

    state = g.snapshot([Status.ACCEPTED])
    if args.json:
        print(json.dumps({n: {**t.attrs, "used_by": [str(s) for s in state.usages(n)]}
                          for n, t in state.types.items()}, indent=2, default=str))
        return 0
    if not state.types:
        print("no accepted semantic types")
        return 0
    for name, t in sorted(state.types.items()):
        used = state.usages(name)
        a = t.attrs
        dtype = a.get("physical_type") or a.get("datatype") or "?"
        conf = f"  [{a['confidentiality']}]" if a.get("confidentiality") else ""
        print(f"{name}  {dtype}{conf}  used by {len(used)} field(s)")
        if a.get("description"):
            print(f"    {a['description']}")
        for c in a.get("checks") or []:
            print(f"    check: {Check.from_dict(c).describe()}")
        for s in used[: args.show]:
            print(f"    - {s}")
        if len(used) > args.show:
            print(f"    … and {len(used) - args.show} more")
    return 0


def cmd_check(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss check``: run a dataset's checks on a Parquet or CSV file (exit 1 if any fail), or print SQL."""
    from . import checks

    if args.sql:
        for subj, chk, query in checks.to_sql(g, args.dataset, dialect=args.dialect):
            print(f"-- {subj}: {checks.Check.from_dict(chk).describe()}\n{query};")
        return 0
    if not args.data:
        print("Give --data <file.parquet|.csv> to run the checks, or --sql to print them as SQL.")
        return 2
    import polars as pl

    df = pl.read_parquet(args.data) if args.data.endswith(".parquet") else pl.read_csv(args.data)
    results = checks.run_polars(g, args.dataset, df)
    for r in results:
        print(r)
        if r.sample:
            print(f"      e.g. {r.sample}")
    if args.record:
        run = g.record_check_results(results, runner=args.actor, run_id=args.run_id)
        print(f"recorded run {run[:8]}")
    return 0 if all(r.passed for r in results) else 1


def cmd_search(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss search``: search accepted knowledge; ``--as-ai`` applies the AI visibility limit."""
    hits = g.search(" ".join(args.text), kinds=args.kind, dataset_prefix=args.dataset, limit=args.limit,
                    for_ai=bool(args.as_ai))
    if args.json:
        print(json.dumps(hits, indent=2))
        return 0
    if not hits:
        print("nothing found")
    for h in hits:
        extra = " [redacted]" if h.get("redacted") else (f"  {_fmt_value(h['description'], 70)}" if h.get("description") else "")
        st = f" <{h['semantic_type']}>" if h.get("semantic_type") else ""
        print(f"{h['subject']}{st}{extra}")
    return 0


def cmd_reindex(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss reindex``: rebuild the search index."""
    print(f"indexed {g.rebuild_search_index()} subjects")
    return 0


def cmd_rebuild(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss rebuild``: recompute every assertion's state from the event log."""
    n = g.rebuild_projection()
    print(f"rebuilt {n} assertions from the event log")
    return 0


def cmd_plugins(g: Gloss | None, args: argparse.Namespace) -> int:
    """``gloss plugins``: list the installed plugins by kind (needs no ledger)."""
    for kind in plugins.GROUPS:
        for name, target in sorted(plugins.available(kind).items()):
            print(f"{kind:<8} {name:<20} {target}")
    return 0


def cmd_serve(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss serve``: run the REST API with uvicorn; the actor comes from the ``X-Gloss-Actor`` header."""
    import uvicorn

    from .server import create_app

    uvicorn.run(create_app(g), host=args.host, port=args.port)
    return 0


def cmd_mcp(g: Gloss, args: argparse.Namespace) -> int:
    """``gloss mcp``: run the MCP server, proposing as ``ai:<agent>``."""
    from .mcp_server import run

    run(g, agent=args.agent, transport=args.transport)
    return 0


# ---------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    """The ``gloss`` argument parser; each subcommand sets ``fn`` to its ``cmd_*`` function."""
    p = argparse.ArgumentParser(prog="gloss", description="Track, review and serve semantic metadata.")
    p.add_argument("--db", default=os.environ.get("GLOSS_DB", DEFAULT_DB), help="SQLAlchemy URL (env GLOSS_DB)")
    p.add_argument("--actor", default=_default_actor(), help="who is acting, e.g. human:alice (env GLOSS_ACTOR)")
    p.add_argument("--mode", default=os.environ.get("GLOSS_MODE", "team"), choices=sorted(MODES),
                   help="review mode: solo accepts your own edits at once (env GLOSS_MODE)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="command", required=True)

    def filters(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--subject")
        sp.add_argument("--dataset")
        sp.add_argument("--namespace", help="dotted path: everything at or under it")
        sp.add_argument("--kind", choices=["model", "dataset", "field", "relationship", "metric", "type"])
        sp.add_argument("--attribute")
        sp.add_argument("--source", help="source prefix, e.g. ai: or import:ossie")
        sp.add_argument("--group", help="a proposal group id")
        sp.add_argument("--status", action="append", choices=[s.value for s in Status])
        sp.add_argument("--limit", type=int)

    sub.add_parser("init", help="create the ledger").set_defaults(fn=cmd_init)

    sp = sub.add_parser("import", help="import proposals from a source plugin")
    sp.add_argument("plugin", help="source plugin, e.g. ossie")
    sp.add_argument("target", nargs="?", help="plugin argument, e.g. a file path")
    sp.add_argument("--accept", action="store_true", help="accept everything imported, as --actor")
    sp.add_argument("--key-by", default="name", choices=["name", "source"], help="(ossie) dataset key")
    sp.add_argument("--namespace", help="(ossie) namespace for the file's relationships and model-level metrics")
    sp.set_defaults(fn=cmd_import)

    sp = sub.add_parser("sync", help="sync with a catalog plugin and detect drift")
    sp.add_argument("catalog", help="catalog spec, e.g. file:catalog.yaml")
    sp.add_argument("--no-auto-accept", action="store_true", help="propose structural facts instead of accepting")
    sp.add_argument("--keep-removed", action="store_true", help="do not mark facts of missing assets stale")
    sp.set_defaults(fn=cmd_sync)

    sp = sub.add_parser("propose", help="propose one fact")
    sp.add_argument("subject")
    sp.add_argument("attribute")
    sp.add_argument("value", help="string or JSON")
    sp.add_argument("--confidence", type=float)
    sp.add_argument("--comment")
    sp.add_argument("--source-as", help="record a different source than the actor, e.g. ai:claude")
    sp.set_defaults(fn=cmd_propose)

    for name, helptext in (("pending", "facts awaiting review"), ("list", "query facts")):
        sp = sub.add_parser(name, help=helptext)
        filters(sp)
        sp.set_defaults(fn=cmd_list)

    for name in ("review", "accept", "reject", "retract", "reopen", "restore"):
        sp = sub.add_parser(name, help=f"{name} facts")
        sp.add_argument("ids", nargs="*", help="assertion ids (unique prefixes are fine)")
        sp.add_argument("--all", action="store_true", help="apply to all facts matching the filters")
        sp.add_argument("--yes", action="store_true", help="allow --all without filters")
        sp.add_argument("--comment")
        filters(sp)
        sp.set_defaults(fn=cmd_action)

    sp = sub.add_parser("rename", help="record that a subject was renamed or moved; its facts follow")
    sp.add_argument("old", help="subject, e.g. dataset:sales.orders (or a path with --namespace)")
    sp.add_argument("new")
    sp.add_argument("--namespace", action="store_true", help="old and new are paths: move everything under old")
    sp.add_argument("--comment")
    sp.set_defaults(fn=cmd_rename)

    sp = sub.add_parser("show", help="current state of one subject")
    sp.add_argument("subject")
    sp.set_defaults(fn=cmd_show)

    sp = sub.add_parser("history", help="event log, for one subject or everything")
    sp.add_argument("subject", nargs="?")
    sp.add_argument("--after", type=int, default=0)
    sp.add_argument("--limit", type=int, default=200)
    sp.set_defaults(fn=cmd_history)

    sp = sub.add_parser("progress", help="review progress per dataset")
    sp.add_argument("--dataset", help="dataset key prefix")
    sp.set_defaults(fn=cmd_progress)

    sp = sub.add_parser("export", help="publish through a sink plugin")
    sp.add_argument("plugin", nargs="?", default="ossie", help="sink plugin: ossie (default) or git")
    sp.add_argument("target", nargs="?", help="plugin argument, e.g. repo path for git")
    sp.add_argument("-o", "--output")
    sp.add_argument("--name")
    sp.add_argument("--dataset", help="only datasets with this key prefix")
    sp.add_argument("--include-pending", action="store_true")
    sp.add_argument("--no-annotate", action="store_true", help="omit gloss review annotations")
    sp.add_argument("--validate", action="store_true", help="validate against the Ossie JSON Schema")
    sp.set_defaults(fn=cmd_export)

    sp = sub.add_parser("viz", help="Mermaid ER diagram or an HTML report")
    sp.add_argument("--html", help="write an HTML report to this path")
    sp.add_argument("--accepted-only", action="store_true")
    sp.add_argument("--dataset")
    sp.set_defaults(fn=cmd_viz)

    sp = sub.add_parser("types", help="semantic types and where they are used")
    sp.add_argument("--show", type=int, default=5, help="how many usages to list per type")
    sp.set_defaults(fn=cmd_types)

    sp = sub.add_parser("check", help="run data quality checks for a dataset")
    sp.add_argument("dataset")
    sp.add_argument("--data", help="Parquet or CSV file with the data (needs polars)")
    sp.add_argument("--record", action="store_true", help="log the results")
    sp.add_argument("--run-id", help="record under this run id (default: a new one)")
    sp.add_argument("--sql", action="store_true", help="print the checks as SQL instead of running them")
    sp.add_argument("--dialect", default="ansi", choices=["ansi", "postgres", "duckdb", "snowflake", "bigquery"])
    sp.set_defaults(fn=cmd_check)

    sp = sub.add_parser("search", help="search accepted knowledge (names, descriptions, synonyms)")
    sp.add_argument("text", nargs="+")
    sp.add_argument("--kind", action="append", choices=["model", "dataset", "field", "relationship", "metric", "type"])
    sp.add_argument("--dataset", help="dataset key prefix")
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--as-ai", action="store_true", help="apply the AI visibility limit")
    sp.set_defaults(fn=cmd_search)

    sub.add_parser("reindex", help="rebuild the search index").set_defaults(fn=cmd_reindex)
    sub.add_parser("rebuild", help="rebuild current state from the event log").set_defaults(fn=cmd_rebuild)
    sub.add_parser("plugins", help="list installed plugins").set_defaults(fn=cmd_plugins, no_db=True)

    sp = sub.add_parser("serve", help="run the REST API")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8765)
    sp.set_defaults(fn=cmd_serve)

    sp = sub.add_parser("mcp", help="run the MCP server for AI agents")
    sp.add_argument("--agent", default="assistant", help="recorded as source ai:<agent>")
    sp.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    sp.set_defaults(fn=cmd_mcp)
    return p


def main(argv: list[str] | None = None) -> int:
    """Run ``gloss`` with ``argv`` and return the exit code (2, with a one-line message, on a known error)."""
    from .model import InvalidTransition
    from .policy import PermissionDenied

    args = build_parser().parse_args(argv)
    if getattr(args, "no_db", False):
        return args.fn(None, args)
    g = open_gloss(args.db, policy=DefaultPolicy(mode=args.mode))
    try:
        return args.fn(g, args)
    except (PermissionDenied, InvalidTransition, KeyError, ValueError, LookupError, RuntimeError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {msg}", file=sys.stderr)
        return 2
    finally:
        g.close()


if __name__ == "__main__":
    raise SystemExit(main())
