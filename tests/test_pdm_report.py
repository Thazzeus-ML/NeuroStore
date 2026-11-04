"""The PdM report is reproducible: REPORT.md is exactly what make_report.py writes from the committed result files, and its gate table
agrees with the mechanical evaluator."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "studies" / "pdm_cmapss"
RES = ROOT / "results" / "pdm_cmapss"
sys.path.insert(0, str(STUDY))
import evaluate_gates as EG  # noqa: E402

pytestmark = pytest.mark.skipif(not (RES / "test_panel.json").exists() or not (STUDY / "REPORT.md").exists(), reason="no committed PdM results / report")


def test_report_regenerates_byte_for_byte(tmp_path):
    out = tmp_path / "REPORT.md"
    r = subprocess.run([sys.executable, "-I", str(STUDY / "make_report.py"), "--out", str(out)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.read_text() == (STUDY / "REPORT.md").read_text(), "REPORT.md is stale or was edited by hand; run `make pdm-report`"


def test_gate_table_matches_the_evaluator():
    import fetch_data
    text = (STUDY / "REPORT.md").read_text()
    rows = EG.evaluate(RES, data_sums=fetch_data.FILES)
    for g in rows:
        m = re.search(rf"^\| {re.escape(g['id'])} \| \*\*([A-Z ]+)\*\* \|", text, re.M)
        assert m and m.group(1) == g["verdict"], (g["id"], m and m.group(1), g["verdict"])


def test_every_result_file_comes_from_the_committed_protocol_and_clean_code():
    g0 = EG.g0_provenance(RES, __import__("hashlib").sha256((STUDY / "PROTOCOL.md").read_bytes()).hexdigest(), __import__("fetch_data").FILES)
    assert g0["verdict"] == "HELD", g0["evidence"]
