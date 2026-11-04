"""Static checks of the Colab notebook sources (no JAX needed) and an optional end-to-end smoke run (needs JAX + nbclient)."""
import ast, io, json, re, subprocess, sys, tokenize
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "colab"))
import nbtools  # noqa: E402

CELLS = nbtools.load_pct(ROOT / "colab" / "src")
CODE = "\n".join(c["source"] for c in CELLS if c["kind"] == "code")


def fstring_portability_issues(src):
    """Constructs that only parse on Python 3.12+ (Colab often runs 3.10-3.12): an f-string reusing its own quote character inside
    a replacement field, a backslash inside one, or a comment inside one. Uses the 3.12+ tokenizer, so it must be run on 3.12+."""
    issues, stack = [], []

    def quote_of(tok):
        tok = tok.lstrip("rRbBuUfF")
        return tok[:3] if tok[:3] in ('"""', "'''") else tok[:1]

    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.FSTRING_START:
            q = quote_of(t.string)
            if stack and stack[-1] == q:
                issues.append((t.start[0], "nested f-string reuses the enclosing quote"))
            stack.append(q)
        elif t.type == tokenize.FSTRING_END:
            stack.pop()
        elif stack and t.type == tokenize.STRING:
            if quote_of(t.string) == stack[-1]:
                issues.append((t.start[0], "string inside a replacement field reuses the enclosing quote"))
            if "\\" in t.string:
                issues.append((t.start[0], "backslash inside a replacement field"))
        elif stack and t.type == tokenize.COMMENT:
            issues.append((t.start[0], "comment inside a replacement field"))
    return issues


@pytest.mark.skipif(sys.version_info < (3, 12), reason="needs the 3.12+ tokenizer to see f-string structure")
def test_fstring_checker_catches_known_bad_patterns_and_notebook_is_clean():
    bad = ['x = f"{d["k"]}"\n', "y = f'{a[\'k\']}'\n", 'z = f"{f"{1}"}"\n']
    for src in bad:
        assert fstring_portability_issues(src), src
    assert not fstring_portability_issues('x = f"{d[\'k\']} and {n:.2f}"\n')
    for c in CELLS:
        if c["kind"] == "code":
            assert not fstring_portability_issues(c["source"]), (c["id"], fstring_portability_issues(c["source"]))


def test_every_code_cell_parses_and_ids_are_unique():
    ids = [c["id"] for c in CELLS]
    assert len(ids) == len(set(ids))
    for c in CELLS:
        assert c["source"].strip(), c["id"]
        if c["kind"] == "code":
            ast.parse(c["source"], filename=c["id"])


def test_committed_notebook_is_built_from_the_sources():
    built = nbtools.to_notebook(CELLS, "NeuroStore GPU/TPU benchmark lab (v2)")
    committed = json.loads((ROOT / "colab" / "NeuroStore_Colab_Benchmarks_v2.ipynb").read_text())
    assert committed == built, "colab/src changed: run `python colab/build_notebook.py`"
    for c in committed["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None      # never commit outputs / run counters


def test_profiles_are_complete_and_consistent():
    ns = nbtools.exec_cells(CELLS, ["config"])
    keys = set(ns["PROFILES"]["quick"])
    for name, p in ns["PROFILES"].items():
        assert set(p) == keys, (name, set(p) ^ keys)
        assert p["seeds"] >= 2 and p["Q"] >= 8 and p["lat_samples"] >= 30
    assert ns["PROFILES"]["extended"]["seeds"] >= 5
    assert ns["PROFILES"]["extended"]["lat_samples"] >= 1000          # needed for p99.9
    assert ns["D"] % 32 == 0


def test_reproducibility_and_methodology_rules():
    assert 'jax.config.update("jax_default_matmul_precision"' not in CODE, "v2 sets precision per kernel, never globally"
    assert "RESULTS.append" not in CODE, "recording must be idempotent (use record())"
    assert "np.random.seed" not in CODE and "np.random.rand" not in CODE and "np.random.normal" not in CODE
    for m in re.finditer(r"default_rng\(([^)]*)\)", CODE):
        assert m.group(1).strip(), "every generator needs an explicit seed"
    assert "jnp.linalg" not in CODE
    # form cells keep the Colab `#@title` on their first line
    for c in CELLS:
        if c["kind"] == "code" and "#@title" in c["source"]:
            assert c["source"].lstrip().startswith("#@title"), c["id"]


def test_every_section_registers_a_self_check_or_documents_why_not():
    checks = set(re.findall(r'check\(\s*f?"(S\d+)', CODE))
    assert {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11"} <= checks


def test_smoke_run_end_to_end(tmp_path):
    pytest.importorskip("jax")
    for mod in ("nbclient", "ipykernel", "pandas", "matplotlib", "nbformat"):
        pytest.importorskip(mod)
    out = tmp_path / "v.json"
    proc = subprocess.run([sys.executable, str(ROOT / "colab" / "validate_v2.py"), "--runs", "smoke", "--workdir", str(tmp_path / "w"), "--out", str(out)],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    run = json.loads(out.read_text())["runs"][0]
    assert not run["errors"] and run["checks"] and all(c["ok"] for c in run["checks"])
