"""The README's Python examples must run.

* Python blocks before "## Python guide" run on their own (each in a fresh directory).
* Blocks from "## Python guide" on run in order in one namespace, like a reader following along.
* A block directly preceded by ``<!-- not-tested -->`` is skipped (it needs a host app).
"""

import io
import re
import shutil
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
BLOCK = re.compile(r"(<!-- not-tested -->\s*\n)?```python\n(.*?)```", re.S)


def _blocks(text: str) -> list[str]:
    return [code for skip, code in BLOCK.findall(text) if not skip]


def _split() -> tuple[list[str], list[str]]:
    text = README.read_text()
    head, _, guide = text.partition("## Python guide")
    return _blocks(head), _blocks(guide)


def _run(blocks: list[str], workdir: Path) -> str:
    shutil.copytree(ROOT / "examples", workdir / "examples")
    ns: dict = {"__name__": "readme"}
    out = io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(workdir)
        with redirect_stdout(out):
            for i, code in enumerate(blocks):
                try:
                    exec(compile(code, f"README block {i + 1}", "exec"), ns)
                except Exception as exc:  # pragma: no cover - the message is the point
                    raise AssertionError(f"README block {i + 1} failed: {exc!r}\n---\n{code}") from exc
    g = ns.get("g")
    if g is not None:
        g.close()
    return out.getvalue()


def test_intro_example_runs(tmp_path):
    intro, _ = _split()
    assert intro, "README should open with an example"
    _run(intro, tmp_path)


def test_python_guide_runs_in_order(tmp_path):
    pytest.importorskip("polars")
    _, guide = _split()
    assert len(guide) >= 8
    out = _run(guide, tmp_path)
    # spot-check the outputs the README shows in comments
    assert "40 new, 0 unchanged" in out
    assert "ai:claude may not accept assertion" in out
    assert "Decimal(18, 2)" in out and "restricted" in out
    assert "['shop.orders', 'shop.customers']" in out
    assert "[('field:shop.customers/email', 'email_address')]" in out
    assert "field:crm.orders/amount Order value incl. VAT, in EUR" in out
    assert "FAIL  field:crm.customers/email" in out
    assert "'subject': 'field:crm.customers/email'" in out and "'redacted': True" in out
    assert "\n[]\n" in out  # Ossie validation errors: none
    assert "Integer" in out  # DictCatalog example
    assert "['crm.customers', 'crm.orders']" in out and "[('crm.orders', 'revenue')]" in out
    assert "ai:claude search" in out  # the read hook
    assert "2 new, 0 unchanged, 2 accepted\nMeeting notes\n1\n" in out  # solo mode
    assert "{'from_status': 'proposed', 'auto': True, 'rule': 'solo'}" in out
    assert "['finance.invoices']\n2\naccepted\n" in out  # scoped view, own policy
    assert "3 Meeting notes\nrekeyed" in out  # rename
