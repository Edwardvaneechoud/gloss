"""A client for a Gloss ledger served over REST, and a way to use two ledgers as one.

    remote = gloss.client.Client("https://hub.example/semantic", headers={"Authorization": "Bearer …"})
    remote.search("revenue")
    remote.propose("field:main.sales.orders/amount", "description", "Order value incl. VAT")

``Client`` mirrors the reading and proposing half of :class:`gloss.Gloss` against the routes of
:mod:`gloss.server`. Who you are is decided by the server from your credentials, so no call takes
an actor.

``Federation`` presents several ledgers as one, each owning a path prefix: an application's own
ledger for local tables next to an organisation's ledger for shared ones.

    both = Federation({"main.": remote}, default=local.as_actor(Human("me")))
    both.pending()                         # from both
    both.current("dataset:main.sales.orders")     # from the remote, by its key

Only the standard library is needed. Pass ``send`` to use another HTTP stack (or a test client).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .ids import SubjectId, path_covers
from .ledger import Proposal, ProposeResult
from .model import OPEN_STATUSES, Assertion, Event, InvalidTransition
from .policy import PermissionDenied

Send = Callable[[str, str, Mapping[str, Any] | None, Any], tuple[int, Any]]


class RemoteError(RuntimeError):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"Gloss server answered {status}: {detail}")
        self.status, self.detail = status, detail


def _urllib_send(base: str, headers: Mapping[str, str], timeout: float) -> Send:
    def send(method: str, path: str, params: Mapping[str, Any] | None, body: Any) -> tuple[int, Any]:
        url = base + path
        if params:
            url += "?" + urllib.parse.urlencode(params, doseq=True)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)  # noqa: S310 (the caller names the server)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        for k, v in headers.items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                raw, status = resp.read(), resp.status
        except urllib.error.HTTPError as exc:
            raw, status = exc.read(), exc.code
        text = raw.decode() if raw else ""
        try:
            return status, json.loads(text) if text else None
        except json.JSONDecodeError:
            return status, text

    return send


def _clean(params: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple, set, frozenset)):
            v = [getattr(x, "value", x) for x in v]
        out[k] = getattr(v, "value", v)
    return out


class Client:
    def __init__(self, base_url: str = "", *, headers: Mapping[str, str] | None = None,
                 send: Send | None = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._send = send or _urllib_send(self.base_url, dict(headers or {}), timeout)

    def _call(self, method: str, path: str, params: Mapping[str, Any] | None = None, body: Any = None) -> Any:
        status, data = self._send(method, path, _clean(params or {}) or None, body)
        if status < 400:
            return data
        detail = data.get("detail") if isinstance(data, dict) else data
        if status == 403:
            raise PermissionDenied(detail)
        if status == 404:
            raise KeyError(detail)
        if status == 422 and isinstance(detail, str):
            raise InvalidTransition(detail) if detail.startswith("Cannot ") else ValueError(detail)
        raise RemoteError(status, detail)

    @staticmethod
    def _subject(subject: str | SubjectId) -> str:
        return urllib.parse.quote(str(SubjectId.parse(subject)), safe="/:")

    # -- reads ------------------------------------------------------------------
    def find(self, **criteria: Any) -> list[Assertion]:
        params = dict(criteria)
        if "statuses" in params:
            params["status"] = params.pop("statuses")
        return [Assertion.from_dict(a) for a in self._call("GET", "/assertions", params)["items"]]

    def count(self, **criteria: Any) -> int:
        params = {**criteria, "limit": 1}
        if "statuses" in params:
            params["status"] = params.pop("statuses")
        return self._call("GET", "/assertions", params)["total"]

    def pending(self, **criteria: Any) -> list[Assertion]:
        criteria.setdefault("statuses", list(OPEN_STATUSES))
        return self.find(**criteria)

    def get(self, assertion_id: str) -> Assertion:
        return Assertion.from_dict(self._call("GET", f"/assertions/{assertion_id}"))

    def current(self, subject: str | SubjectId) -> dict[str, Any]:
        return self._call("GET", f"/subjects/{self._subject(subject)}")["current"]

    def version(self, subject: str | SubjectId) -> int | None:
        return self._call("GET", f"/subjects/{self._subject(subject)}")["version"]

    def history(self, subject: str | SubjectId) -> list[Event]:
        return [Event.from_dict(e) for e in self._call("GET", f"/subjects/{self._subject(subject)}/history")]

    def log(self, after_seq: int = 0, limit: int | None = None, *, correlation: str | None = None) -> list[Event]:
        params = {"after_seq": after_seq, "limit": limit, "correlation": correlation}
        return [Event.from_dict(e) for e in self._call("GET", "/log", params)]

    def progress(self, dataset_prefix: str | None = None) -> dict[str, dict[str, int]]:
        return self._call("GET", "/progress", {"dataset_prefix": dataset_prefix})

    def search(self, text: str, *, kinds: Sequence[str] | None = None, dataset_prefix: str | None = None,
               limit: int = 20, for_ai: bool | None = None) -> list[dict[str, Any]]:
        path = "/ai/search" if for_ai else "/search"
        params = {"q": text, "limit": limit} if for_ai else {
            "q": text, "kind": kinds, "dataset_prefix": dataset_prefix, "limit": limit}
        return [{k: v for k, v in hit.items() if v not in (None, False)} for hit in self._call("GET", path, params)]

    def tables(self, prefix: str | None = None, *, include_pending: bool = False) -> list[dict[str, Any]]:
        """Tables as plain dicts (the server's ``TableOut``)."""
        return self._call("GET", "/tables", {"prefix": prefix, "include_pending": include_pending})

    def table(self, key: str, *, include_pending: bool = False) -> dict[str, Any]:
        return self._call("GET", f"/tables/{urllib.parse.quote(key)}", {"include_pending": include_pending})

    def relationships(self, table: str | None = None, *, include_pending: bool = False) -> list[dict[str, Any]]:
        return self._call("GET", "/relationships", {"table": table, "include_pending": include_pending})

    def metrics(self, table: str | None = None, *, include_pending: bool = False) -> list[dict[str, Any]]:
        return self._call("GET", "/metrics", {"table": table, "include_pending": include_pending})

    def semantic_types(self, *, include_pending: bool = False) -> list[dict[str, Any]]:
        return self._call("GET", "/types", {"include_pending": include_pending})

    def model_context(self, datasets: list[str] | None = None) -> dict[str, Any]:
        return self._call("GET", "/ai/context", {"dataset": datasets})

    def to_yaml(self, *, name: str | None = None, include_pending: bool = False,
                namespace: str | None = None) -> str:
        return self._call("GET", "/export/ossie",
                          {"name": name, "include_pending": include_pending, "namespace": namespace})

    # -- writes -----------------------------------------------------------------
    def propose(self, subject: str | SubjectId, attribute: str, value: Any, *, source: str | None = None,
                confidence: float | None = None, evidence: Sequence[dict[str, Any]] | None = None,
                comment: str | None = None, accept: bool = False, group: str | None = None) -> Assertion:
        body = {"subject": str(subject), "attribute": attribute, "value": value, "source": source,
                "confidence": confidence, "evidence": list(evidence or []), "comment": comment,
                "accept": accept, "group": group}
        return Assertion.from_dict(self._call("POST", "/assertions", body=body)["assertion"])

    def propose_many(self, proposals: Iterable[Proposal], *, source: str | None = None, accept: bool = False,
                     comment: str | None = None) -> list[ProposeResult]:
        body = {
            "proposals": [
                {"subject": p.subject, "attribute": p.attribute, "value": p.value, "confidence": p.confidence,
                 "evidence": p.evidence, "comment": p.comment, "group": p.group}
                for p in proposals
            ],
            "source": source, "accept": accept, "comment": comment,
        }
        return [ProposeResult(Assertion.from_dict(r["assertion"]), r["outcome"], r["accepted"])
                for r in self._call("POST", "/proposals", body=body)]

    def _act(self, action: str, assertion_id: str, comment: str | None) -> Assertion:
        return Assertion.from_dict(
            self._call("POST", f"/assertions/{assertion_id}/{action}", body={"comment": comment})
        )

    def review(self, assertion_id: str, *, comment: str | None = None) -> Assertion:
        return self._act("review", assertion_id, comment)

    def accept(self, assertion_id: str, *, comment: str | None = None) -> Assertion:
        return self._act("accept", assertion_id, comment)

    def reject(self, assertion_id: str, *, comment: str | None = None) -> Assertion:
        return self._act("reject", assertion_id, comment)

    def retract(self, assertion_id: str, *, comment: str | None = None) -> Assertion:
        return self._act("retract", assertion_id, comment)

    def reopen(self, assertion_id: str, *, comment: str | None = None) -> Assertion:
        return self._act("reopen", assertion_id, comment)

    def restore(self, assertion_id: str, *, comment: str | None = None) -> Assertion:
        return self._act("restore", assertion_id, comment)

    def transition_many(self, action: str, assertion_ids: Iterable[str], *, comment: str | None = None
                        ) -> tuple[list[Assertion], list[tuple[str, str]]]:
        res = self._call("POST", f"/bulk/{action}", body={"ids": list(assertion_ids), "comment": comment})
        return [Assertion.from_dict(a) for a in res["changed"]], [(e["id"], e["error"]) for e in res["errors"]]

    def accept_group(self, group: str, *, comment: str | None = None) -> list[Assertion]:
        return [Assertion.from_dict(a) for a in self._call("POST", f"/groups/{group}/accept",
                                                           body={"comment": comment})]

    def reject_group(self, group: str, *, comment: str | None = None) -> list[Assertion]:
        return [Assertion.from_dict(a) for a in self._call("POST", f"/groups/{group}/reject",
                                                           body={"comment": comment})]

    def record_check_results(self, results: Iterable[Any], *, run_id: str | None = None) -> str:
        body = {"results": [r.to_dict() for r in results], "run_id": run_id}
        return self._call("POST", "/checks/results", body=body)["run_id"]


class Federation:
    """Several ledgers as one, each owning the subjects under its path prefix.

    ``ledgers`` maps a path prefix to a ledger: a ``Client`` or an actor-bound ``Gloss`` view.
    ``default`` takes everything no prefix covers, including ledger-wide subjects. Reads about
    one subject go to its owner; listings ask every ledger and combine the answers; a write goes
    to the owner of its subject. Assertion ids are looked for in each ledger in turn.
    """

    def __init__(self, ledgers: Mapping[str, Any], *, default: Any = None):
        self.ledgers = dict(sorted(ledgers.items(), key=lambda kv: -len(kv[0])))
        self.default = default

    def _all(self) -> list[Any]:
        seen = list(dict.fromkeys(self.ledgers.values()))
        return seen + ([self.default] if self.default is not None and self.default not in seen else [])

    def owner(self, subject: str | SubjectId) -> Any:
        path = SubjectId.parse(subject).scope_path
        for prefix, ledger in self.ledgers.items():
            if path is not None and path_covers(prefix, path):
                return ledger
        if self.default is None:
            raise KeyError(f"No ledger owns {subject}")
        return self.default

    def _owner_of_key(self, key: str) -> Any:
        return self.owner(SubjectId.for_dataset(key))

    def _holding(self, assertion_id: str) -> Any:
        for ledger in self._all():
            try:
                ledger.get(assertion_id)
                return ledger
            except KeyError:
                continue
        raise KeyError(f"No assertion {assertion_id!r}")

    def _targets(self, criteria: Mapping[str, Any]) -> list[Any]:
        if criteria.get("subject"):
            return [self.owner(criteria["subject"])]
        if criteria.get("dataset"):
            return [self._owner_of_key(criteria["dataset"])]
        return self._all()

    # -- reads ------------------------------------------------------------------
    def find(self, **criteria: Any) -> list[Assertion]:
        rows = [a for ledger in self._targets(criteria) for a in ledger.find(**criteria)]
        return sorted(rows, key=lambda a: (a.subject, a.attribute, a.created_at))

    def pending(self, **criteria: Any) -> list[Assertion]:
        criteria.setdefault("statuses", list(OPEN_STATUSES))
        return self.find(**criteria)

    def count(self, **criteria: Any) -> int:
        return sum(ledger.count(**criteria) for ledger in self._targets(criteria))

    def get(self, assertion_id: str) -> Assertion:
        return self._holding(assertion_id).get(assertion_id)

    def current(self, subject: str | SubjectId) -> dict[str, Any]:
        return self.owner(subject).current(subject)

    def version(self, subject: str | SubjectId) -> int | None:
        return self.owner(subject).version(subject)

    def history(self, subject: str | SubjectId) -> list[Event]:
        return self.owner(subject).history(subject)

    def progress(self, dataset_prefix: str | None = None) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for ledger in self._all():
            out.update(ledger.progress(dataset_prefix))
        return dict(sorted(out.items()))

    def search(self, text: str, *, limit: int = 20, **options: Any) -> list[dict[str, Any]]:
        """Hits from every ledger, interleaved so each ledger's best come first."""
        lists = [ledger.search(text, limit=limit, **options) for ledger in self._all()]
        out: list[dict[str, Any]] = []
        for i in range(max((len(x) for x in lists), default=0)):
            out += [x[i] for x in lists if i < len(x)]
        return out[:limit]

    def table(self, key: str, **options: Any) -> Any:
        return self._owner_of_key(key).table(key, **options)

    def tables(self, prefix: str | None = None, **options: Any) -> list[Any]:
        return [t for ledger in self._all() for t in ledger.tables(prefix, **options)]

    # -- writes -----------------------------------------------------------------
    def propose(self, subject: str | SubjectId, attribute: str, value: Any, **options: Any) -> Assertion:
        return self.owner(subject).propose(subject, attribute, value, **options)

    def propose_many(self, proposals: Iterable[Proposal], **options: Any) -> list[ProposeResult]:
        """Each proposal goes to the owner of its subject; results come back in the order given."""
        items = list(proposals)
        by_owner: dict[int, tuple[Any, list[int]]] = {}
        for i, p in enumerate(items):
            owner = self.owner(p.subject)
            by_owner.setdefault(id(owner), (owner, []))[1].append(i)
        out: list[ProposeResult | None] = [None] * len(items)
        for owner, positions in by_owner.values():
            for pos, res in zip(positions, owner.propose_many([items[i] for i in positions], **options),
                                strict=True):
                out[pos] = res
        return [r for r in out if r is not None]

    def _act(self, action: str, assertion_id: str, **options: Any) -> Assertion:
        return getattr(self._holding(assertion_id), action)(assertion_id, **options)

    def review(self, assertion_id: str, **options: Any) -> Assertion:
        return self._act("review", assertion_id, **options)

    def accept(self, assertion_id: str, **options: Any) -> Assertion:
        return self._act("accept", assertion_id, **options)

    def reject(self, assertion_id: str, **options: Any) -> Assertion:
        return self._act("reject", assertion_id, **options)

    def retract(self, assertion_id: str, **options: Any) -> Assertion:
        return self._act("retract", assertion_id, **options)

    def reopen(self, assertion_id: str, **options: Any) -> Assertion:
        return self._act("reopen", assertion_id, **options)

    def restore(self, assertion_id: str, **options: Any) -> Assertion:
        return self._act("restore", assertion_id, **options)
