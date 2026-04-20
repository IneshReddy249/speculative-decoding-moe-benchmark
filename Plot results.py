#!/usr/bin/env python3
"""
plot_results.py — parse benchmark JSONs and generate comparison plots
usage: python3 plot_results.py [--results-dir results/] [--output-dir plots/]
"""

import json
import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# configs we benchmarked and their display names
CONFIGS = {
    "baseline":  {"label": "Baseline",           "color": "#6b7280", "marker": "o"},
    "mtp":       {"label": "MTP-v1",             "color": "#94a3b8", "marker": "s"},
    "mtpv2":     {"label": "MTP-v2 (SpecV2)",    "color": "#2563eb", "marker": "s"},
    "eagle3":    {"label": "EAGLE3-v1",           "color": "#f59e0b", "marker": "^"},
    "eagle3v2":  {"label": "EAGLE3-v2 (big tree)","color": "#d97706", "marker": "^"},
    "eagle3v3":  {"label": "EAGLE3-v3 (SpecV2)",  "color": "#dc2626", "marker": "^"},
}

CONCURRENCIES = [1, 4, 8, 16, 32]


def load_results(results_dir):
    """load all benchmark JSONs into {config: {concurrency: data}}"""
    data = {}
    for config_name in CONFIGS:
        data[config_name] = {}
        for c in CONCURRENCIES:
            path = Path(results_dir) / f"{config_name}_c{c}.json"
            if path.exists():
                with open(path) as f:
                    data[config_name][c] = json.load(f)
    return data


def get_metric(data, config, concurrency, key):
    """pull a metric from nested benchmark JSON"""
    try:
        return data[config][concurrency][key]
    except (KeyError, TypeError):
        return None


def plot_throughput(data, output_dir):
    """output tok/s vs concurrency for all configs"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for config_name, style in CONFIGS.items():
        xs, ys = [], []
        for c in CONCURRENCIES:
            val = get_metric(data, config_name, c, "output_throughput")
            if val is not None:
                xs.append(c)
                ys.append(val)
        if xs:
            ax.plot(xs, ys, label=style["label"], color=style["color"],
                    marker=style["marker"], linewidth=2, markersize=8)

    ax.set_xlabel("Concurrency", fontsize=12)
    ax.set_ylabel("Output Throughput (tok/s)", fontsize=12)
    ax.set_title("GLM-4.7-Flash (30B MoE, H100 PCIe) — Throughput vs Concurrency", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(CONCURRENCIES)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "throughput_comparison.png", dpi=150)
    plt.close()
    print(f"saved throughput_comparison.png")


def plot_tpot(data, output_dir):
    """TPOT (ms) vs concurrency — lower is better"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for config_name, style in CONFIGS.items():
        xs, ys = [], []
        for c in CONCURRENCIES:
            val = get_metric(data, config_name, c, "mean_tpot_ms")
            if val is not None:
                xs.append(c)
                ys.append(val)
        if xs:
            ax.plot(xs, ys, label=style["label"], color=style["color"],
                    marker=style["marker"], linewidth=2, markersize=8)

    ax.set_xlabel("Concurrency", fontsize=12)
    ax.set_ylabel("Mean TPOT (ms)", fontsize=12)
    ax.set_title("GLM-4.7-Flash (30B MoE, H100 PCIe) — Per-Token Latency vs Concurrency", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(CONCURRENCIES)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "tpot_comparison.png", dpi=150)
    plt.close()
    print(f"saved tpot_comparison.png")


def plot_speedup(data, output_dir):
    """speedup over baseline at each concurrency — bar chart"""
    # only plot the best configs
    best = ["mtpv2", "eagle3", "eagle3v3"]
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(CONCURRENCIES))
    width = 0.25

    for i, config_name in enumerate(best):
        speedups = []
        for c in CONCURRENCIES:
            base = get_metric(data, "baseline", c, "output_throughput")
            val = get_metric(data, config_name, c, "output_throughput")
            if base and val:
                speedups.append(val / base)
            else:
                speedups.append(0)
        bars = ax.bar(x + i * width, speedups, width,
                      label=CONFIGS[config_name]["label"],
                      color=CONFIGS[config_name]["color"])
        # put values on bars
        for bar, s in zip(bars, speedups):
            if s > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{s:.2f}x", ha="center", va="bottom", fontsize=8)

    ax.axhline(y=1.0, color="#6b7280", linestyle="--", alpha=0.5, label="Baseline (1.0x)")
    ax.set_xlabel("Concurrency", fontsize=12)
    ax.set_ylabel("Speedup vs Baseline", fontsize=12)
    ax.set_title("GLM-4.7-Flash — Speedup: MTP-v2 vs EAGLE3-v1 vs EAGLE3-v3 (SpecV2)", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(c) for c in CONCURRENCIES])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "speedup_comparison.png", dpi=150)
    plt.close()
    print(f"saved speedup_comparison.png")


def plot_accept_length(data, output_dir):
    """accept length by config — horizontal bar"""
    configs_with_accept = ["mtp", "mtpv2", "eagle3", "eagle3v2", "eagle3v3"]
    fig, ax = plt.subplots(figsize=(8, 5))

    labels, values, colors = [], [], []
    for config_name in configs_with_accept:
        # grab accept length from c=1 run
        val = get_metric(data, config_name, 1, "accept_length")
        if val is not None:
            labels.append(CONFIGS[config_name]["label"])
            values.append(val)
            colors.append(CONFIGS[config_name]["color"])

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, height=0.5)
    for i, v in enumerate(values):
        ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=11)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Accept Length (tokens/step)", fontsize=12)
    ax.set_title("Draft Accept Length by Configuration (c=1)", fontsize=13)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "accept_length.png", dpi=150)
    plt.close()
    print(f"saved accept_length.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--output-dir", default="plots/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_results(args.results_dir)

    # check we have at least baseline
    if not data.get("baseline"):
        print(f"no baseline results found in {args.results_dir}")
        return

    plot_throughput(data, args.output_dir)
    plot_tpot(data, args.output_dir)
    plot_speedup(data, args.output_dir)
    plot_accept_length(data, args.output_dir)
    print(f"\nall plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
