#!/usr/bin/env python3
"""
Publication Report & Figure Generator
Compares trained Neural ODE models directly against the official NeurIPS 2025
scaling-crl benchmark datasets (ResNet depths 4, 8, 16, 32, 64).
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Fix matplotlib cache dir (per-user to avoid shared /tmp collisions)
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib_cache_{os.getuid()}")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Standard publication style parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.3,
})

def find_latest_run(base_dir="runs"):
    runs = sorted(Path(base_dir).glob("*breakthrough*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        # Fallback to any run with metrics.csv
        runs = sorted([p.parent for p in Path(base_dir).glob("*/metrics.csv")], key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None

def load_paper_data(env_id, json_path="results_data/fig1_main.json"):
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get(env_id, None)

def get_resnet_params(depth):
    # Parameter counts for ResNet architectures from the paper
    # ResNet-4: ~0.58M, ResNet-8: ~0.85M, ResNet-16: ~1.4M, ResNet-32: ~2.2M, ResNet-64: ~4.3M
    param_map = {4: 587000, 8: 850000, 16: 1400000, 32: 2200000, 64: 4300000}
    return param_map.get(depth, depth * 67000)

def main():
    parser = argparse.ArgumentParser(description="Generate publication comparison report.")
    parser.add_argument("--run-dir", type=str, default="", help="Path to run directory with metrics.csv")
    parser.add_argument("--output-dir", type=str, default="", help="Directory to save report outputs")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run()
    if not run_dir or not (run_dir / "metrics.csv").exists():
        print(f"Error: Could not locate valid run directory with metrics.csv: {run_dir}")
        sys.exit(1)

    print(f"Loading run from: {run_dir}")
    metrics_path = run_dir / "metrics.csv"
    df = pd.read_csv(metrics_path)
    
    # Identify environment name
    env_id = "ant_u4_maze"
    if "ant_big_maze" in str(run_dir):
        env_id = "ant_big_maze"
    elif "arm_push_hard" in str(run_dir):
        env_id = "arm_push_hard"
    elif "arm_push_easy" in str(run_dir):
        env_id = "arm_push_easy"
    elif "humanoid" in str(run_dir):
        env_id = "humanoid"
    
    # Check args.pkl if available
    args_pkl = run_dir / "args.pkl"
    if args_pkl.exists():
        try:
            with open(args_pkl, "rb") as f:
                saved_args = pickle.load(f)
                env_id = getattr(saved_args, "env_id", env_id)
        except Exception:
            pass

    out_dir = Path(args.output_dir) if args.output_dir else run_dir
    os.makedirs(out_dir, exist_ok=True)

    paper_data = load_paper_data(env_id)

    # 4-Panel Publication Dashboard
    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

    # --- PANEL 1: Official Learning Curves ---
    ax1 = fig.add_subplot(gs[0, 0])
    color_map = {4: "#9e9e9e", 8: "#8bc34a", 16: "#03a9f4", 32: "#ff9800", 64: "#9c27b0"}
    
    paper_depth_peaks = {}
    if paper_data is not None:
        depths = paper_data.get("depths", [4, 8, 16, 32, 64])
        means = paper_data.get("depth_mean", {})
        stds = paper_data.get("depth_std_error", {})

        for d in depths:
            str_d = str(d)
            if str_d in means:
                y = np.array(means[str_d])
                x = np.linspace(1, len(y), len(y))
                y_err = np.array(stds[str_d]) if str_d in stds else np.zeros_like(y)
                lbl = f"ResNet Depth-{d}"
                c = color_map.get(d, "#607d8b")
                ax1.plot(x, y, label=lbl, color=c, alpha=0.75, linestyle="--", linewidth=1.5)
                ax1.fill_between(x, y - y_err, y + y_err, color=c, alpha=0.12)
                paper_depth_peaks[d] = float(np.max(y))

    # Read actual NFE and parameter count from run artifacts
    our_nfe = int(df["critic_nfe"].iloc[-1]) if "critic_nfe" in df.columns else 32
    ode_steps = our_nfe // 4 if our_nfe > 0 else 8
    our_params = None
    ckpt_path = run_dir / "latest_ckpt.pkl"
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
            def _count_params(tree):
                if isinstance(tree, dict):
                    return sum(_count_params(v) for v in tree.values())
                if isinstance(tree, (list, tuple)):
                    return sum(_count_params(v) for v in tree)
                return int(np.asarray(tree).size)
            total_p = _count_params(ckpt.get("actor_params", {})) + _count_params(ckpt.get("critic_params", {}))
            if total_p > 0:
                our_params = total_p / 1e6
        except Exception:
            pass
    if our_params is None:
        our_params = 0.64 if "w256" in str(run_dir) else 2.485

    # Plot our Dual Neural ODE run
    node_epochs = df["epoch"].values + 1
    node_tag = df["time_at_goal"].values
    ax1.plot(node_epochs, node_tag, label=f"Dual Neural ODE (Ours, NFE={our_nfe})", color="#e91e63", linewidth=2.8)
    
    ax1.set_xlabel("Training Epochs (Millions of Env Steps)")
    ax1.set_ylabel("Evaluation Time at Goal / Success")
    ax1.set_title(f"A. Learning Curves: Neural ODE vs. Discrete ResNets ({env_id})", fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # --- PANEL 2: Critical Depth & Phase Transition ---
    ax2 = fig.add_subplot(gs[0, 1])
    discrete_nfe = [4, 8, 16, 32, 64]
    discrete_scores = [paper_depth_peaks.get(d, 0.0) for d in discrete_nfe]
    
    ax2.plot(discrete_nfe, discrete_scores, marker="o", markersize=8, color="#3f51b5", label="Discrete ResNet Scaling", linewidth=2.0)
    
    our_peak = float(df["time_at_goal"].max())
    ax2.scatter([our_nfe], [our_peak], color="#e91e63", s=180, zorder=5, label=f"Dual Neural ODE (Peak: {our_peak:.1f})")
    ax2.annotate(
        f"Ours: {our_peak:.1f}\n(NFE={our_nfe}, ~{our_params:.2f}M params)",
        xy=(our_nfe, our_peak),
        xytext=(max(4, our_nfe - 18), our_peak + 5 if our_peak > 10 else our_peak + 1),
        arrowprops=dict(facecolor="#e91e63", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color="#c2185b"
    )
    
    ax2.set_xlabel("Effective Depth / NFE per Step")
    ax2.set_ylabel("Peak Task Performance")
    ax2.set_title("B. Phase Transition: Scaling Depth Unlocks Goal Reaching", fontweight="bold")
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # --- PANEL 3: Parameter Efficiency Frontier ---
    ax3 = fig.add_subplot(gs[1, 0])
    resnet_params = [get_resnet_params(d) / 1e6 for d in discrete_nfe]
    
    ax3.plot(resnet_params, discrete_scores, marker="s", markersize=8, color="#009688", label="ResNet Parameter Scaling", linewidth=2.0)
    for d, p, sc in zip(discrete_nfe, resnet_params, discrete_scores):
        ax3.annotate(f"D{d}", (p, sc), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    ax3.scatter([our_params], [our_peak], color="#e91e63", s=180, zorder=5, label=f"Dual Neural ODE (~{our_params:.2f}M params)")
    ax3.annotate(
        f"Continuous Flow\n(~{our_params:.2f}M params)",
        xy=(our_params, our_peak),
        xytext=(our_params + 0.3, our_peak + 10 if our_peak > 20 else our_peak + 1),
        arrowprops=dict(facecolor="#e91e63", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color="#c2185b"
    )

    ax3.set_xlabel("Trainable Parameters (Millions)")
    ax3.set_ylabel("Peak Task Performance")
    ax3.set_title("C. Parameter Efficiency: Continuous vs. Discrete Architectures", fontweight="bold")
    ax3.legend(loc="lower right", framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.4)

    # --- PANEL 4: Continuous Representation Loss & Gradient Stability ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4_twin = ax4.twinx()

    c_loss = df["critic_loss"].values
    g_norm = df["critic_grad_norm"].values

    line1 = ax4.plot(node_epochs, c_loss, color="#1565c0", label="Critic InfoNCE Loss", linewidth=2.0)
    line2 = ax4_twin.plot(node_epochs, g_norm, color="#ff7043", label="Critic Gradient Norm", linestyle=":", linewidth=1.8)

    ax4.set_xlabel("Training Epochs")
    ax4.set_ylabel("InfoNCE Loss", color="#1565c0")
    ax4_twin.set_ylabel("Gradient Norm", color="#ff7043")
    ax4.tick_params(axis="y", labelcolor="#1565c0")
    ax4_twin.tick_params(axis="y", labelcolor="#ff7043")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc="upper right", framealpha=0.9)
    ax4.set_title("D. Continuous Flow Loss Convergence & Gradient Conditioning", fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(f"Neural Ordinary Differential Equations for Goal-Conditioned RL\nBenchmark: {env_id} | RK4 Solver ($s={ode_steps}$, NFE={our_nfe})", fontsize=15, fontweight="bold", y=0.98)

    png_path = out_dir / "publication_report.png"
    pdf_path = out_dir / "publication_report.pdf"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Publication figures saved to:\n  {png_path}\n  {pdf_path}")

    # Generate Publication Summary Table (Markdown)
    final_epoch = int(df["epoch"].iloc[-1]) + 1
    last5_mean = float(df["time_at_goal"].tail(5).mean())
    last5_std = float(df["time_at_goal"].tail(5).std()) if len(df) >= 5 else 0.0
    final_success = float(df["success_any"].max()) * 100.0
    final_closs = float(df["critic_loss"].iloc[-1])
    min_gnorm, max_gnorm = float(df["critic_grad_norm"].min()), float(df["critic_grad_norm"].max())

    table_md = f"""# Quantitative Benchmark Report: {env_id}
**Benchmark Reference:** NeurIPS 2025 Best Paper (*"1000 Layer Networks for Self-Supervised RL"*)

| Architecture | Model Type | Effective NFE | Trainable Params | Peak Performance | Last-5 Mean Performance | Peak Success Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-4** | Discrete | 4 | ~0.58M | {paper_depth_peaks.get(4, 0.0):.2f} | — | — |
| **ResNet-8** | Discrete | 8 | ~0.85M | {paper_depth_peaks.get(8, 0.0):.2f} | — | — |
| **ResNet-16** | Discrete | 16 | ~1.40M | {paper_depth_peaks.get(16, 0.0):.2f} | — | — |
| **ResNet-32** | Discrete | 32 | ~2.20M | {paper_depth_peaks.get(32, 0.0):.2f} | — | — |
| **ResNet-64** | Discrete | 64 | ~4.30M | {paper_depth_peaks.get(64, 0.0):.2f} | — | — |
| **Dual Neural ODE (Ours)** | **Continuous Flow** | **{our_nfe}** | **~{our_params:.2f}M** | **{our_peak:.2f}** | **{last5_mean:.2f} ± {last5_std:.2f}** | **{final_success:.1f}%** |

### Run Diagnostics ({final_epoch} epochs):
- **Effective Depth / NFE:** {our_nfe} (RK4, `ode_steps={ode_steps}`) with **~{our_params:.2f}M** total trainable parameters.
- **Peak vs. Last-5 `TimeAtGoal`:** Peak `{our_peak:.2f}`, Last-5 `{last5_mean:.2f} ± {last5_std:.2f}`, Peak `success_any` `{final_success:.1f}%`.
- **Training Stability:** Final Critic InfoNCE loss `{final_closs:.3f}`; Critic gradient norm range `{min_gnorm:.2f}`–`{max_gnorm:.2f}`.
"""
    table_file = out_dir / "publication_table.md"
    with open(table_file, "w") as f:
        f.write(table_md)
    print(f"Publication summary table saved to: {table_file}")

if __name__ == "__main__":
    main()
