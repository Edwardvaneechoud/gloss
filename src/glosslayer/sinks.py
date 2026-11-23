"""Extra built-in sinks."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .ossie.exporter import ExportResult, to_ossie

if TYPE_CHECKING:
    from .ledger import Gloss


class GitSink:
    """Write the accepted model as Ossie YAML into a git working tree and commit it.

    Git is an *output* here: a reviewable, versioned publication of the accepted
    model. The ledger remains the source of truth. The commit message records the
    log position the export reflects, so any published version traces back to events.
    """

    name = "git"

    def __init__(self, repo: str | Path, path: str = "semantic_model.yaml", *, author: str | None = None):
        self.repo = Path(repo)
        self.path = path
        self.author = author

    @classmethod
    def from_spec(cls, arg: str, **kwargs: Any) -> GitSink:
        """``repo`` or ``repo#path/in/repo.yaml``; the path defaults to ``semantic_model.yaml``."""
        repo, _, path = arg.partition("#")
        return cls(repo, path or "semantic_model.yaml", **kwargs)

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.repo), *args], check=True, capture_output=True, text=True
        ).stdout

    def publish(self, gloss: Gloss, *, message: str | None = None, **options: Any) -> ExportResult:
        """Export, write the file and commit it; no commit when the exported text is unchanged."""
        result = to_ossie(gloss, **options)
        target = self.repo / self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.to_yaml())
        self._git("add", self.path)
        if not self._git("status", "--porcelain", "--", self.path).strip():
            return result  # nothing changed
        seq = gloss.store.last_seq()
        msg = message or f"gloss: publish semantic model (log seq {seq})"
        args = ["commit", "-m", msg]
        if self.author:
            args += ["--author", self.author]
        self._git(*args)
        return result
