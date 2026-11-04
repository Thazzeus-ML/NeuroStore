"""Tiny percent-format <-> ipynb tooling (no jupytext dependency).

Source format: cells start with a line `# %%` (code) or `# %% [markdown]` (markdown), optionally followed by
`id=<cell-id>` after the marker. Markdown cell lines are comment lines (`# text`); the leading `# ` is stripped.
"""
from __future__ import annotations
import json, re, hashlib
from pathlib import Path

_MARK = re.compile(r"^# %%(?P<md>\s*\[markdown\])?(?:\s+id=(?P<id>[\w\-]+))?\s*$")


def _read_source(path):
    path = Path(path)
    if path.is_dir():
        return "\n".join(f.read_text() for f in sorted(path.glob("*.py")))
    return path.read_text()


def load_pct(path):
    """`path` is a percent-format file or a directory of them (concatenated in sorted filename order)."""
    cells, cur = [], None
    for line in _read_source(path).splitlines():
        m = _MARK.match(line)
        if m:
            if cur is not None:
                cells.append(cur)
            cur = {"kind": "markdown" if m.group("md") else "code", "id": m.group("id"), "lines": []}
        elif cur is not None:
            cur["lines"].append(line)
    if cur is not None:
        cells.append(cur)
    out = []
    for i, c in enumerate(cells):
        lines = c["lines"]
        while lines and not lines[0].strip():
            lines = lines[1:]
        while lines and not lines[-1].strip():
            lines = lines[:-1]
        if c["kind"] == "markdown":
            lines = [re.sub(r"^# ?", "", l) for l in lines]
        cid = c["id"] or f"ns-{c['kind'][:2]}-{i:02d}"
        out.append({"kind": c["kind"], "id": cid, "source": "\n".join(lines)})
    ids = [c["id"] for c in out]
    assert len(ids) == len(set(ids)), "duplicate cell ids"
    return out


def to_notebook(cells, title="NeuroStore benchmarks"):
    nb_cells = []
    for c in cells:
        src = c["source"].splitlines(keepends=True)
        if c["kind"] == "markdown":
            nb_cells.append({"cell_type": "markdown", "id": c["id"], "metadata": {}, "source": src})
        else:
            nb_cells.append({"cell_type": "code", "id": c["id"], "metadata": {}, "execution_count": None,
                             "outputs": [], "source": src})
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "colab": {"name": title, "provenance": []}},
            "cells": nb_cells}


def build(src_path, out_path, title="NeuroStore benchmarks"):
    cells = load_pct(src_path)
    nb = to_notebook(cells, title)
    Path(out_path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return cells


def cell_source(cells, cid):
    for c in cells:
        if c["id"] == cid:
            return c["source"]
    raise KeyError(cid)


def exec_cells(cells, ids, ns=None):
    ns = {"__name__": "__nb__"} if ns is None else ns
    for cid in ids:
        exec(compile(cell_source(cells, cid), f"<cell {cid}>", "exec"), ns)
    return ns
