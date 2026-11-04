#!/usr/bin/env python3
"""Figures for the PdM report (matplotlib; optional dependency). Reads results/pdm_cmapss/*.json, writes studies/pdm_cmapss/figures/*.png.

Colour jobs: retrieval backends in one series colour and context baselines in gray (accuracy bars; the axis names the category), a validated
one-hue ordinal ramp for the ordered chimera bands, a one-hue sequential ramp for degradation, and the first five validated categorical slots
(in order) for the five scale-up lines, which carry direct labels because three of those slots are below 3:1 contrast on the light surface.
All charts sit on the light surface #fcfcfb; text is never series-coloured."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, GRAY = "#2a78d6", "#a9a8a1"                      # retrieval backends (one series -> one colour) vs context baselines
SCALE_COLOR = {"exact": "#2a78d6", "ann": "#eb6834", "ann_sq8": "#1baf7a", "flyhash": "#eda100", "vsa": "#e87ba4"}   # slots 1-5 in validated order
BAND = {"same": "#86b6ef", "mid": "#3987e5", "far": "#184f95"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]       # blue 100..700
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8})


def jl(d, n):
    p = Path(d) / n
    return json.loads(p.read_text()) if p.exists() else None


def style(ax, grid="x"):
    ax.grid(axis=grid, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_accuracy(d, out):
    t = jl(d, "test_panel.json")
    if not t:
        return
    order = ["const", "ridge", "exact", "exact_l2", "hopfield", "ann", "ann_sq8", "flyhash", "vsa"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    for ax, (fd, e) in zip(axes.ravel(), t["subsets"].items()):
        ref = e["reference"]
        ys = np.arange(len(order))[::-1]
        for y, n in zip(ys, order):
            v = e["variants"][n]
            ax.barh(y, v["rmse"], height=0.55, color=GRAY if n in ("const", "ridge") else BLUE, zorder=3)
            ax.plot([v["rmse_lo"], v["rmse_hi"]], [y, y], color=INK2, lw=1.0, zorder=4, solid_capstyle="butt")
            ax.text(0.012 * max(e["variants"][m]["rmse_hi"] for m in order), y, f"{v['rmse']:.1f}", va="center", ha="left", fontsize=8.5,
                    color=INK if n in ("const", "ridge") else "#ffffff", zorder=5)
        xm = e["variants"][ref]["rmse"] * 1.10
        ax.axvline(xm, color=INK, lw=0.9, zorder=2)
        ax.set_yticks(ys)
        ax.set_yticklabels([n + (" (ref)" if n == ref else "") for n in order])
        ax.set_xlim(0, max(e["variants"][n]["rmse_hi"] for n in order) * 1.18)
        ax.set_title(f"{fd}  ({e['n_test']} test engines)", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("RMSE vs official RUL (cycles), 95 % engine-bootstrap CI")
        style(ax, "x")
    fig.legend(handles=[Patch(color=BLUE, label="retrieval-based stores"), Patch(color=GRAY, label="context baselines (constant, ridge)"),
                        Line2D([0], [0], color=INK, lw=0.9, label="reference RMSE x 1.10 (non-inferiority margin)")], loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(Path(out) / "fig_accuracy.png", dpi=170)
    plt.close(fig)


def fig_robustness(d, out):
    s = jl(d, "sweeps.json")
    if not s:
        return
    lv = "L50" if "L50" in next(iter(s["levels"].values())) else next(iter(next(iter(s["levels"].values()))))
    fams = list(s["levels"].keys())
    cols = ["exact_l2", "exact", "hopfield", "ann", "ann_sq8", "flyhash", "vsa", "ridge"]
    cols = [c for c in cols if c in s["variants"]]
    M = np.array([[s["variants"][c]["families"][f][lv]["ratio"] for c in cols] for f in fams])
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    vmax = max(M.max(), 2.2)
    edges = np.linspace(1.0, vmax, len(SEQ) + 1)
    for i, f in enumerate(fams):
        for j, c in enumerate(cols):
            r = M[i, j]
            k = int(np.clip(np.searchsorted(edges, r, side="right") - 1, 0, len(SEQ) - 1))
            ax.add_patch(plt.Rectangle((j, len(fams) - 1 - i), 1, 1, facecolor=SEQ[k], edgecolor=SURFACE, linewidth=2))
            vr = s["variants"][c]["families"][f][lv].get("vs_ref")
            mark = ""
            if vr and c != s["reference"]:
                mark = " \u25bc" if vr["hi"] < 0 else (" \u25b2" if vr["lo"] > 0 else "")
            ax.text(j + 0.5, len(fams) - 1 - i + 0.5, f"{r:.2f}{mark}", ha="center", va="center", fontsize=8.5, color="#ffffff" if k >= 3 else INK)
    ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(fams))
    ax.set_xticks(np.arange(len(cols)) + 0.5); ax.set_xticklabels([c + ("\n(ref)" if c == s["reference"] else ("\n(baseline)" if c == "ridge" else "")) for c in cols])
    ax.set_yticks(np.arange(len(fams)) + 0.5); ax.set_yticklabels(fams[::-1])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"RMSE(corrupted) / RMSE(clean) at level {lv} (severities calibrated on the reference; discrete families overshoot)", loc="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.01, 0.02, "\u25bc = significantly LESS degradation than the reference, \u25b2 = significantly MORE (paired 95 % CI). Darker = worse. ridge is a context baseline.", fontsize=8, color=INK2)
    fig.savefig(Path(out) / "fig_robustness.png", dpi=170)
    plt.close(fig)


def fig_chimera(d, out):
    o = jl(d, "ood.json")
    if not o:
        return
    names = [n for n in ["exact_l2", "exact", "hopfield", "ann", "flyhash", "vsa"] if n in o["variants"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, kind in zip(axes, ("channel", "time")):
        ys = np.arange(len(names))[::-1]
        for y, n in zip(ys, names):
            for off, band in zip((0.2, 0.0, -0.2), ("same", "mid", "far")):
                key = f"chimera_{kind}_{band}"
                if key not in o["variants"][n]:
                    continue
                v = o["variants"][n][key]["max_sim"]
                ax.plot([v["lo"], v["hi"]], [y + off, y + off], color=BAND[band], lw=1.0, zorder=3, solid_capstyle="butt")
                ax.scatter([v["auroc"]], [y + off], s=26, color=BAND[band], edgecolor=SURFACE, linewidth=1.2, zorder=4)
        ax.axvline(0.5, color=AXIS, lw=0.9, zorder=2)
        ax.set_yticks(ys); ax.set_yticklabels([n + (" (ref)" if n == o["reference"] else "") for n in names])
        ax.set_xlim(0.4, 1.02); ax.set_xlabel("AUROC (held-out windows vs chimeras), max-similarity score")
        ax.set_title(f"{kind} chimeras", loc="left", fontsize=10, color=INK)
        ax.text(0.505, ys[0] + 0.42, "chance", fontsize=7.5, color=INK2, ha="left", va="center")
        style(ax, "x")
    axes[0].legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=BAND[b], markeredgecolor=SURFACE, markersize=6, label=f"{b} stage gap") for b in ("same", "mid", "far")],
                   loc="lower left", frameon=False, fontsize=8, ncol=3, bbox_to_anchor=(0.0, -0.36))
    fig.tight_layout()
    fig.savefig(Path(out) / "fig_chimera.png", dpi=170)
    plt.close(fig)


def fig_scale(d, out):
    t = jl(d, "timing.json")
    if not t:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    order = ["exact", "ann", "ann_sq8", "flyhash", "vsa"]
    for key, ax, ylabel in (("latency", axes[0], "single-query p50 latency (ms), k = 10"), ("index_bytes", axes[1], "index size (MB)")):
        ends = []
        for n in order:
            rows = t["scale"].get(n, [])
            if not rows:
                continue
            xs = [r["n"] for r in rows]
            ys = [r["latency"]["p50_ms"] if key == "latency" else r["index_bytes"] / 1e6 for r in rows]
            ax.plot(xs, ys, color=SCALE_COLOR[n], lw=1.8, marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            ends.append((n, xs[-1], ys[-1]))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([2e4, 1e5, 5e5]); ax.set_xticklabels(["20k", "100k", "500k"]); ax.minorticks_off()
        ax.set_xlabel("stored vectors n"); ax.set_ylabel(ylabel)
        style(ax, "both")
        lo, hi = ax.get_ylim()
        placed = []
        for n, x, y in sorted(ends, key=lambda e: e[2]):
            yy = y
            for p in placed:
                if abs(np.log10(yy) - np.log10(p)) < 0.09:
                    yy = p * 10 ** 0.09
            placed.append(yy)
            ax.annotate(n, (x, y), xytext=(x * 1.12, yy), fontsize=8.5, color=INK2, va="center", annotation_clip=False)
        ax.margins(x=0.2)
    axes[0].legend(handles=[Line2D([0], [0], color=SCALE_COLOR[n], lw=2, marker="o", ms=5, label=n) for n in order if n in t["scale"]], frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(Path(out) / "fig_scale.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "pdm_cmapss"))
    ap.add_argument("--out", default=str(HERE / "figures"))
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for f in (fig_accuracy, fig_robustness, fig_chimera, fig_scale):
        f(a.results, a.out)
    print("figures written to", a.out)
