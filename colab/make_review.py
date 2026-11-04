#!/usr/bin/env python3
"""Build colab/REVIEW.md from colab/review_template.md, colab/review_evidence.json and colab/validation_cpu.json.
Every number in the review is a {{name}} placeholder resolved here from those two files; nothing is typed in."""
from __future__ import annotations
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV = json.loads((HERE / "review_evidence.json").read_text())
VAL = json.loads((HERE / "validation_cpu.json").read_text()) if (HERE / "validation_cpu.json").exists() else {"runs": []}
RUN = {r["name"]: r for r in VAL["runs"]}


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def build_values():
    V = {}
    e0 = EV["E0_saturation_in_original_output"]
    V["jax"] = EV["environment"]["jax"]; V["backend"] = EV["environment"]["backend"]
    V["fault_rows"] = e0["fault_rows"]; V["fault_rows_one"] = e0["fault_rows_recall_1.0"]
    V["ood_auroc"] = f"{e0['ood_auroc']:.3f}"
    V["noisy_rows"] = e0["noisy_recall_rows"]; V["noisy_rows_one"] = e0["noisy_recall_rows_with_recall_1.0"]
    ana = EV["E4_analytic_vs_measured_exact_recall"]
    V["sat_cells"] = sum(1 for r in ana if r["analytic"] > 1 - 1e-9); V["all_cells"] = len(ana)
    V["max_err_lowcells"] = f"{max(1 - r['analytic'] for r in ana if r['noise'] <= 0.25):.0e}"
    V["max_abs_z"] = f"{max(abs(r['z']) for r in ana if r['analytic'] < 1 - 1e-9):.2f}"
    s = EV["E5a_ood_separation"]
    V["ood_known_min"] = f"{s['known_min']:.2f}"; V["ood_max"] = f"{s['ood_max']:.2f}"; V["ood_gap"] = f"{s['gap_between_worst_known_and_best_ood']:.2f}"
    sd = e0["sdm_rows_where_output_ber_exceeds_input_noise"]
    V["sdm_worse_cells"] = len(sd); V["sdm_ber_example"] = f"{sd[-1][2]:.2f}"; V["sdm_example_n"] = sd[-1][0]; V["sdm_noise_example"] = sd[-1][1]
    e1 = EV["E1_published_point_paired"]
    V["p_n128"] = f"{e1['n128_noise0.40']['mcnemar_p']:.2f}"; V["p_n512"] = f"{e1['n512_noise0.40']['mcnemar_p']:.2f}"
    V["pub_n128_ex"] = f"{e1['n128_noise0.40']['exact']:.3f}"; V["pub_n128_hop"] = f"{e1['n128_noise0.40']['hopfield']:.3f}"
    V["pub_n512_ex"] = f"{e1['n512_noise0.40']['exact']:.3f}"; V["pub_n512_hop"] = f"{e1['n512_noise0.40']['hopfield']:.3f}"
    e2 = EV["E2_run_to_run_spread"]
    for k, tag in (("n128_noise0.40", "128"), ("n512_noise0.40", "512")):
        V[f"sd_{tag}"] = f"{100 * e2[k]['exact_sd']:.1f}"; V[f"diff_{tag}"] = f"{100 * e2[k]['diff_mean']:+.1f}"
        V[f"ex_better_{tag}"] = e2[k]["batches_exact_better"]; V[f"hop_better_{tag}"] = e2[k]["batches_hopfield_better"]; V[f"eq_{tag}"] = e2[k]["batches_equal"]
    V["batches"] = e2["n512_noise0.40"]["batches"]
    e3 = EV["E3_exact_vs_hopfield_agreement"]
    V["dis_512_25"] = pct(e3["n512_noise0.25"]["disagreement_rate"], 1); V["dis_512_40"] = pct(e3["n512_noise0.4"]["disagreement_rate"], 1)
    V["dis_2048_40"] = pct(e3["n2048_noise0.4"]["disagreement_rate"], 1)
    V["ex_right_512"] = e3["n512_noise0.4"]["of_which_exact_right"]; V["hop_right_512"] = e3["n512_noise0.4"]["of_which_hopfield_right"]
    e6 = EV["E6_percentile_reliability_cpu"]
    V["p99_8_low"] = pct(1 - e6["p99_from_8_samples"]["estimate_median_ms"] / e6["p99_from_8_samples"]["reference_from_2000_samples_ms"])
    V["p99_8_spread"] = f"{e6['p99_from_8_samples']['ratio_p95_over_p5']:.1f}"
    V["p99_20_low"] = pct(1 - e6["p99_from_20_samples"]["estimate_median_ms"] / e6["p99_from_20_samples"]["reference_from_2000_samples_ms"])
    e7 = EV["E7_launch_bound"]
    V["floor_p50"] = f"{e7['empty_jit_call_p50_us_this_machine']:.0f}"; V["floor_p99"] = f"{e7['empty_jit_call_p99_us_this_machine']:.0f}"
    q = [w for w in e7["workloads"] if w["profile"] == "quick"][-1]; x = [w for w in e7["workloads"] if w["profile"] == "extended"][-1]
    V["q_flops"] = f"{q['flops'] / 1e6:.0f}"; V["q_t1"] = f"{q['time_at_1_TFLOPs_us']:.0f}"; V["q_t10"] = f"{q['time_at_10_TFLOPs_us']:.0f}"
    V["x_flops"] = f"{x['flops'] / 1e6:.0f}"; V["x_t1"] = f"{x['time_at_1_TFLOPs_us']:.0f}"
    V["cpu_exact_ms"] = f"{e7['original_cpu_batch_p50_ms']['exact_flat_n512_noise0.25']:.2f}"
    e8 = EV["E8_seed_derivation"]; V["seed_pair"] = " and ".join(f"(n={n}, noise={z})" for n, z in e8["colliding_pairs"][0]); V["seed_val_offset"] = e8["colliding_seeds"][0] - 2026
    e9 = EV["E9_rerun_duplicates"]; V["rows_before"] = e9["noisy_recall_rows_after_first_run"]; V["rows_after"] = e9["after_re_executing_the_same_cell"]
    e10 = EV["E10_vsa_top1_by_construction"]
    V["vsa_top1"] = f"{e10['top1_accuracy']:.4f}"; V["vsa_oracle"] = f"{e10['oracle_distinct_over_N']:.4f}"; V["vsa_maxdistinct"] = e10["max_distinct_predictions_within_a_tuple_group"]
    V["vsa_tuple_acc"] = f"{e10['tuple_level_accuracy']:.2f}"; V["vsa_groups"] = e10["distinct_cue_tuples"]; V["vsa_n"] = e10["records"]
    e11 = EV["E11_precision_exactness_cpu"]
    V["bf16_err"] = e11["bf16_inputs_f32_accumulate"]["max_abs_score_error"]; V["i8_err"] = e11["int8_inputs_int32_accumulate"]["max_abs_score_error"]
    V["bf16out_err"] = e11["bf16_inputs_bf16_output"]["max_abs_score_error"]
    e12 = EV["E12_flyhash_ties"]
    V["fly_tie_frac"] = pct(e12["fraction_inputs_with_tie_across_the_WTA_boundary"], 1); V["fly_tied_units"] = f"{e12['mean_units_tied_at_boundary_level']:.0f}"
    V["fly_above"] = f"{e12['mean_units_strictly_above_boundary']:.0f}"; V["fly_distinct"] = e12["distinct_activation_values_over_5000_inputs"]
    V["fly_index_decided"] = f"{64 - e12['mean_units_strictly_above_boundary']:.0f}"
    rec = e12["recall_at_noise"]
    V["fly_bin_35"] = f"{rec['0.35']['binary_pm1_over_sqrt16 (original)']:.2f}"; V["fly_gauss_35"] = f"{rec['0.35']['gaussian_weights (continuous activations)']:.2f}"
    V["fly_bin_40"] = f"{rec['0.4']['binary_pm1_over_sqrt16 (original)']:.2f}"; V["fly_gauss_40"] = f"{rec['0.4']['gaussian_weights (continuous activations)']:.2f}"
    e13 = EV["E13_bipolar_host_memory_gb"]; V["bip_gb_2e20"] = f"{e13['n=1048576']:.1f}"
    e14 = EV["E14_sdm_dense_vs_scatter"]; big = e14["M=32768"]
    V["sdm_equal"] = "yes" if all(v["counters_identical"] and v["reads_identical"] for v in e14.values()) else "NO"
    V["sdm_write_x"] = f"{big['build_dense_ms'] / big['build_scatter_ms']:.1f}"; V["sdm_read_x"] = f"{big['read_dense_ms'] / big['read_gather_ms']:.1f}"
    V["sdm_mask_mb"] = f"{big['dense_mask_bytes'] / 1e6:.0f}"
    V["evidence_runtime"] = f"{EV['runtime_s']:.0f}"
    # validation of v2
    for name in ("smoke", "multidev", "quick"):
        r = RUN.get(name)
        V[f"{name}_ok"] = bool(r) and not r["errors"] and all(c["ok"] for c in r["checks"])
        V[f"{name}_wall"] = f"{r['wall_seconds']:.0f}" if r else "not run"
        V[f"{name}_checks"] = f"{sum(c['ok'] for c in r['checks'])}/{len(r['checks'])}" if r else "n/a"
        V[f"{name}_rows"] = r["rows"] if r else "n/a"
        V[f"{name}_plots"] = r.get("plots", "n/a") if r else "n/a"
        V[f"{name}_errors"] = len(r["errors"]) if r else "n/a"
    r = RUN.get("multidev")
    V["multidev_devices"] = r["environment"]["local_device_count"] if r else "n/a"
    r = RUN.get("quick") or RUN.get("smoke")
    V["v2_experiments"] = len(r["experiments"]) if r else "n/a"; V["v2_columns"] = r["columns"] if r else "n/a"
    V["checks_list"] = "\n".join(f"| {c['name'].replace('|', '\\|')} | {'PASS' if c['ok'] else 'FAIL'} |" for c in (RUN.get("multidev") or RUN.get("smoke") or {"checks": []})["checks"])
    V["quick_findings"] = "\n".join(f"- {f}" for f in (RUN.get("quick") or {"findings": []})["findings"])
    V["env_jax"] = (r or {}).get("environment", {}).get("jax", V["jax"]); V["env_numpy"] = (r or {}).get("environment", {}).get("numpy", "")
    V["env_python"] = (r or {}).get("environment", {}).get("python", "")
    return V


def main():
    V = build_values()
    text = (HERE / "review_template.md").read_text()
    missing = set(re.findall(r"\{\{(\w+)\}\}", text)) - set(V)
    if missing:
        raise SystemExit(f"template placeholders without values: {sorted(missing)}")
    (HERE / "REVIEW.md").write_text(re.sub(r"\{\{(\w+)\}\}", lambda m: str(V[m.group(1)]), text))
    print("wrote", HERE / "REVIEW.md")


if __name__ == "__main__":
    main()
