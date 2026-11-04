"""Result serialisation, report/scorecard generation, the runner's failure isolation and the CLI."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neurostore import cli
from neurostore.report import _verdict, md_table, num, pct, render_report, scorecard, sig
from neurostore.results import SuiteResult, WorkloadResult, to_jsonable
from neurostore.runner import parse_workloads, run_suite, selfcheck_failures
from neurostore.workloads import WORKLOADS


# ----------------------------------------------------------------------------- serialisation
def test_to_jsonable_handles_numpy_nan_paths_and_dataclasses():
    out = to_jsonable({"a": np.float32(1.5), "b": np.int64(3), "c": np.array([1, 2]), "d": float("nan"), "e": float("inf"), "f": Path("/x"), "g": (1, 2), "h": np.bool_(True)})
    assert out == {"a": 1.5, "b": 3, "c": [1, 2], "d": None, "e": None, "f": "/x", "g": [1, 2], "h": True}
    json.dumps(out, allow_nan=False)
    wr = WorkloadResult("a", "t", "quick", 0, tables={"x": [{"v": np.float64("nan")}]})
    assert to_jsonable(wr)["tables"]["x"][0]["v"] is None


def test_suite_result_roundtrips_through_json(quick_suite, tmp_path):
    path = quick_suite.save(tmp_path / "sub" / "r.json")
    loaded = SuiteResult.load(path)
    assert [w.workload for w in loaded.workloads] == [w.workload for w in quick_suite.workloads]
    assert loaded.to_dict() == quick_suite.to_dict()
    json.loads(path.read_text(), parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))  # strict JSON
    assert loaded.get("a").tables["thresholds"] == quick_suite.get("a").to_dict()["tables"]["thresholds"]
    assert loaded.get("zzz") is None


def test_add_check_classifies_kinds():
    w = WorkloadResult("a", "t", "quick", 0)
    w.add_check("Benchmark self-check: foo", "t", "m", True)
    w.add_check("A claim", "t", "m", False)
    w.add_check("Just info", "t", "m", None)
    assert [c["kind"] for c in w.checks] == ["selfcheck", "spec", "info"]
    w.add_skip("x", "why")
    w.add_skip("x", "why")
    assert w.skipped == [{"system": "x", "reason": "why"}]  # de-duplicated


# ----------------------------------------------------------------------------- formatting and verdicts
def test_markdown_tables_escape_pipes_and_handle_empty_data():
    t = md_table(["a|b", "c"], [["x|y", "line1\nline2"]])
    assert "a\\|b" in t and "x\\|y" in t and "line1 line2" in t
    assert t.count("\n") == 3 and all(line.count("|") - line.count("\\|") == 3 for line in t.strip().split("\n"))
    assert "no data" in md_table(["a"], [])


def test_number_formatters_never_print_nan():
    for f in (pct, num, sig):
        assert f(None) == "n/a" and f(float("nan")) == "n/a"
    assert pct(0.1234) == "12.3%" and num(1234567.0) == "1.23e+06" and num(12345.6) == "12,346" and sig(0.000123456) == "0.000123"


@pytest.mark.parametrize("ns,base,hib,expected", [
    (10, 5, True, "WIN"), (5, 10, True, "LOSS"), (10.2, 10, True, "TIE"),
    (5, 10, False, "WIN"), (10, 5, False, "LOSS"), (10, 10.2, False, "TIE"),
    (1.0, 0.0, True, "WIN"), (0.0, 0.0, True, "TIE"), (None, 1.0, True, "n/a"),
])
def test_verdict_logic(ns, base, hib, expected):
    assert _verdict(ns, base, hib) == expected


def _suite_with(headlines):
    w = WorkloadResult("a", "T", "quick", 0, headlines=headlines)
    return SuiteResult("0", "quick", 0, {}, [w])


def test_scorecard_compares_best_neurostore_against_each_competing_family():
    hs = [
        {"metric": "recall", "unit": "%", "higher_is_better": True, "context": "c",
         "systems": {"ns1": {"value": 90, "family": "neurostore"}, "ns2": {"value": 99, "family": "neurostore"},
                     "kv1": {"value": 0, "family": "kv"}, "ann1": {"value": 99.5, "family": "ann"}, "ann2": {"value": 50, "family": "ann"},
                     "lsh1": {"value": 60, "family": "lsh"}, "th": {"value": 100, "family": "theory"}}},
        {"metric": "latency", "unit": "us", "higher_is_better": False, "context": "c",
         "systems": {"ns": {"value": 30, "family": "neurostore"}, "kv": {"value": 0.2, "family": "kv"}}},
        {"metric": "only ours", "unit": "x", "higher_is_better": True, "context": "c", "systems": {"ns": {"value": 1, "family": "neurostore"}}},
    ]
    rows, tally = scorecard(_suite_with(hs))
    assert len(rows) == 3
    assert "ns2" in rows[0][3] and "kv: 0 (kv1) **WIN**" in rows[0][4] and "ann: 99.5 (ann1) **TIE**" in rows[0][4] and "lsh: 60 (lsh1) **WIN**" in rows[0][4]
    assert "theory" not in rows[0][4]  # the analytic curve is a reference line, not a competitor
    assert "**LOSS**" in rows[1][4] and rows[2][4] == "no baseline"
    assert tally == {"WIN": 2, "TIE": 1, "LOSS": 1}  # row 1: kv WIN, lsh WIN, ann TIE; row 2: kv LOSS


# ----------------------------------------------------------------------------- the rendered report
def test_report_has_every_section_and_no_nan(quick_suite):
    text = render_report(quick_suite)
    for needle in ("# NeuroStore benchmark report", "## Scorecard", "## Checks against the evaluation spec", "Tally over", *[WORKLOADS[k].TITLE for k in WORKLOADS]):
        assert needle in text, needle
    assert "nan" not in text.lower().replace("financial", "") and "None" not in text
    assert text.count("MODELED") >= 1  # CIM latency is labelled as a model wherever it appears
    for w in quick_suite.workloads:
        for c in w.checks:
            assert c["name"].replace("|", "\\|") in text


def test_report_is_regenerable_from_the_stored_json(quick_suite, tmp_path):
    path = quick_suite.save(tmp_path / "r.json")
    assert render_report(SuiteResult.load(path)) == render_report(quick_suite)


def test_report_survives_a_crashed_workload_and_a_renderer_bug(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    with monkeypatch.context() as m:
        m.setattr(WORKLOADS["capacity"], "run", boom)
        suite = run_suite(["capacity", "cim"], profile="quick", include_external=False)
    text = render_report(suite)
    assert "WORKLOAD CRASHED: RuntimeError: synthetic failure" in text and "No results" in text
    assert WORKLOADS["cim"].TITLE in text and "**FAIL**" in text  # the other workload and the failed self-check are still reported

    from neurostore import report

    monkeypatch.setitem(report.RENDERERS, "cim", lambda w: 1 / 0)
    assert "could not be rendered (ZeroDivisionError" in render_report(suite)


def test_environment_records_the_code_revision_and_the_report_shows_it(quick_suite):
    env = quick_suite.environment
    assert "git_commit" in env and "git_dirty" in env
    if env["git_commit"]:  # running from a git checkout
        assert len(env["git_commit"]) == 40 and env["git_commit"][:10] in render_report(quick_suite)


def test_progress_is_persisted_after_every_workload_and_writes_are_atomic(tmp_path):
    seen = []
    out = tmp_path / "run"

    def persist(partial):
        partial.save(out / "results.json")
        seen.append([w.workload for w in SuiteResult.load(out / "results.json").workloads])

    run_suite(["capacity", "cim"], profile="quick", include_external=False, on_progress=persist)
    assert seen == [["capacity"], ["capacity", "cim"]]  # a crash after the first workload would have kept it
    assert not list(out.glob("*.tmp"))  # no stray temp file


def test_cli_leaves_partial_results_when_a_later_workload_dies(monkeypatch, tmp_path):
    def die(*a, **k):
        raise KeyboardInterrupt  # not caught by the per-workload handler (it only catches Exception)

    monkeypatch.setattr(WORKLOADS["cim"], "run", die)
    out = tmp_path / "o"
    with pytest.raises(KeyboardInterrupt):
        cli.main(["run", "-w", "capacity,cim", "--profile", "quick", "--quiet", "--no-external", "--out", str(out)])
    assert [w.workload for w in SuiteResult.load(out / "results.json").workloads] == ["capacity"]


def test_report_says_so_when_single_thread_timing_was_not_enforced(quick_suite):
    import copy

    assert "enforced with threadpoolctl" in render_report(quick_suite)
    without = copy.deepcopy(quick_suite)
    without.environment["optional_libs"]["threadpoolctl"] = None
    assert "NOT enforced" in render_report(without) and "enforced with threadpoolctl" not in render_report(without)


def test_report_marks_skipped_systems():
    w = WorkloadResult("a", "T", "quick", 0)
    w.add_skip("redis", "redis-server binary not found on PATH")
    text = render_report(SuiteResult("0", "quick", 0, {"optional_libs": {}}, [w]))
    assert "Skipped systems" in text and "redis-server binary not found" in text


# ----------------------------------------------------------------------------- runner
def test_parse_workloads():
    assert parse_workloads("all") == ["a", "b", "c", "d", "capacity", "cim"]
    assert parse_workloads("cim, a") == ["a", "cim"]  # canonical order, whitespace tolerated
    assert parse_workloads(["d", "b"]) == ["b", "d"] and parse_workloads("") == parse_workloads("all")
    with pytest.raises(ValueError, match="unknown workload"):
        parse_workloads("a,zzz")


def test_overrides_reach_the_workload_and_unknown_overrides_are_rejected():
    s = run_suite("b", profile="quick", overrides={"b": {"n_queries": 21}}, include_external=False)
    assert s.get("b").params["n_queries"] == 21 and {r["n"] for r in s.get("b").tables["accuracy"]} == {21}
    with pytest.raises(KeyError):
        run_suite("b", profile="quick", overrides={"nope": {}})


def test_thread_override_is_applied_to_every_workload():
    s = run_suite(["capacity", "cim"], profile="quick", threads=2, include_external=False)
    assert all(w.params["threads"] == 2 for w in s.workloads)


def test_a_crashing_workload_is_recorded_not_fatal(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(WORKLOADS["capacity"], "run", boom)
    s = run_suite(["capacity", "cim"], profile="quick", include_external=False)
    assert s.get("cim").tables  # later workloads still ran
    bad = selfcheck_failures(s)
    assert len(bad) == 1 and bad[0][0] == "capacity" and "synthetic failure" in bad[0][1]["measured"]
    assert any("CRASHED" in n for n in s.get("capacity").notes)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_suite(["capacity"], profile="quick", fail_fast=True)


# ----------------------------------------------------------------------------- CLI
def test_cli_help_and_version(capsys):
    assert cli.main([]) == 0 and "scaffold" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and "neurostore" in capsys.readouterr().out
    p = cli.build_parser()
    assert {"scaffold", "run", "report", "list"} <= set(p._subparsers._group_actions[0].choices)


def test_cli_list_shows_workloads_and_baseline_availability(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    for k in WORKLOADS:
        assert k in out
    assert "python_dict" in out and "available" in out and "Profiles:" in out


def test_cli_scaffold_is_non_destructive(tmp_path, capsys):
    d = tmp_path / "ws"
    assert cli.main(["scaffold", "--dir", str(d), "--profile", "quick"]) == 0
    cfg = json.loads((d / "config.json").read_text())
    assert set(cfg) == set(WORKLOADS) and cfg["a"]["trials"] == 80 and (d / "results").is_dir() and (d / "README.md").exists()
    (d / "config.json").write_text('{"edited": true}')
    cli.main(["scaffold", "--dir", str(d)])
    assert json.loads((d / "config.json").read_text()) == {"edited": True} and "keep" in capsys.readouterr().out
    cli.main(["scaffold", "--dir", str(d), "--profile", "quick", "--force"])
    assert set(json.loads((d / "config.json").read_text())) == set(WORKLOADS)


def test_cli_run_writes_results_and_report_and_report_command_matches(tmp_path, capsys):
    out = tmp_path / "run"
    rc = cli.main(["run", "-w", "b,capacity", "--profile", "quick", "--seed", "4", "--no-external", "--quiet", "--strict", "--out", str(out)])
    assert rc == 0 and (out / "results.json").exists()
    assert "Scorecard tally" in capsys.readouterr().out
    suite = SuiteResult.load(out / "results.json")
    assert [w.workload for w in suite.workloads] == ["b", "capacity"] and suite.seed == 4 and suite.run_config["include_external"] is False
    assert (out / "REPORT.md").read_text() == render_report(suite)
    assert cli.main(["report", str(out / "results.json"), "-o", str(tmp_path / "again.md")]) == 0
    assert (tmp_path / "again.md").read_text() == (out / "REPORT.md").read_text()


def test_cli_run_streams_each_workload_as_it_finishes(capsys):
    assert cli.main(["run", "-w", "cim", "--profile", "quick", "--no-external"]) == 0
    out = capsys.readouterr().out
    assert "Section 5" in out and "MODELED" in out and "[PASS] Benchmark self-check" in out


def test_cli_config_file_overrides(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"b": {"n_queries": 17}}))
    out = tmp_path / "o"
    assert cli.main(["run", "-w", "b", "--profile", "quick", "--config", str(cfg), "--quiet", "--no-external", "--out", str(out)]) == 0
    assert SuiteResult.load(out / "results.json").get("b").params["n_queries"] == 17


def test_cli_rejects_unknown_workloads_and_strict_fails_on_a_broken_self_check(monkeypatch, capsys):
    assert cli.main(["run", "-w", "nonsense", "--quiet"]) == 2
    assert "unknown workload" in capsys.readouterr().err

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(WORKLOADS["cim"], "run", boom)
    assert cli.main(["run", "-w", "cim", "--profile", "quick", "--quiet", "--no-external"]) == 0
    assert cli.main(["run", "-w", "cim", "--profile", "quick", "--quiet", "--no-external", "--strict"]) == 2
    assert "SELF-CHECK FAILED" in capsys.readouterr().err


@pytest.mark.external
def test_terminating_a_run_does_not_orphan_the_redis_server_it_spawned():
    """SIGTERM used to bypass the `finally` blocks and leave redis-server running."""
    import os
    import signal as sg
    import time

    from conftest import _redis_ok

    if not _redis_ok():
        pytest.skip("redis client or redis-server missing")

    def children(pid):
        r = subprocess.run(["pgrep", "-P", str(pid), "-f", "redis-server"], capture_output=True, text=True)
        return [int(x) for x in r.stdout.split()]

    def alive(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"  # a zombie is dead
        except OSError:
            return False

    proc = subprocess.Popen([sys.executable, "-m", "neurostore", "run", "-w", "a", "--profile", "quick", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 60
        kids = []
        while not kids and time.time() < deadline and proc.poll() is None:
            kids = children(proc.pid)
            time.sleep(0.05)
        assert kids, "the run never started a redis-server"
        proc.send_signal(sg.SIGTERM)
        proc.wait(timeout=30)
        assert proc.returncode == 128 + sg.SIGTERM
        time.sleep(0.5)
        assert not [k for k in kids if alive(k)], "redis-server was orphaned by SIGTERM"
    finally:
        if proc.poll() is None:
            proc.kill()
        for k in children(proc.pid) if proc.poll() is None else []:
            os.kill(k, sg.SIGKILL)


def test_REGRESSION_module_entry_point_parses_and_runs():
    """The original neurostore_cli.py could not even be imported (`run_parser subparsers.add_parser(...)`, missing '=')."""
    for args in (["--help"], ["run", "--help"], ["list"]):
        r = subprocess.run([sys.executable, "-m", "neurostore", *args], capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, (args, r.stderr)
    r = subprocess.run([sys.executable, "-m", "neurostore", "run", "-w", "d", "--profile", "quick", "--quiet"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "Scorecard tally" in r.stdout
    with pytest.raises(SyntaxError):
        compile('run_parser subparsers.add_parser("run")', "<original>", "exec")  # the defect, for the record
