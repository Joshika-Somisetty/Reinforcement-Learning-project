"""Plot training curves and policy comparison results for TSA-SAC."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

Path("results/plots").mkdir(parents=True, exist_ok=True)


def smooth(arr, window=10):
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_training(history_path="results/training_history.json"):
    with open(history_path) as f:
        h = json.load(f)

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("TSA-SAC Training Curves - Irrigation RL", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    eps = h["episode"]

    def ax_plot(pos, y_key, title, ylabel, color):
        ax = fig.add_subplot(gs[pos])
        raw = np.array(h[y_key])
        ax.plot(eps, raw, alpha=0.25, color=color, linewidth=0.8)
        s = smooth(raw)
        ax.plot(eps[len(eps)-len(s):], s, color=color, linewidth=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        return ax

    ax_plot((0,0), "reward",      "Episode Reward",      "Reward",    "#2563EB")
    ax_plot((0,1), "profit",      "Episode Profit ($)",  "USD",       "#16A34A")
    ax_plot((0,2), "irrigation",  "Total Irrigation (mm)","mm",       "#9333EA")
    ax_plot((1,0), "yield",       "Final Yield (kg/ha)", "kg/ha",     "#059669")
    ax_plot((1,1), "critic_loss", "Critic Loss",         "MSE Loss",  "#DC2626")
    ax_plot((1,2), "alpha",       "Entropy Temperature", "alpha",     "#0891B2")

    plt.savefig("results/plots/training_curves.png", dpi=150, bbox_inches="tight")
    print("Saved: results/plots/training_curves.png")
    plt.show()


def plot_comparison(comparison_path="results/baseline_comparison.json"):
    with open(comparison_path) as f:
        data = json.load(f)

    names   = list(data.keys())
    profits = [data[n]["profit_mean"] for n in names]
    stds    = [data[n]["profit_std"]  for n in names]
    yields  = [data[n]["yield_mean"] for n in names]
    irrs    = [data[n]["irrigation_mean"] for n in names]
    iwues   = [data[n]["iwue_mean"] for n in names]
    stress  = [data[n]["stress_days_mean"] for n in names]

    colors = ["#94A3B8", "#94A3B8", "#94A3B8", "#16A34A"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Policy Comparison", fontsize=13, fontweight="bold")
    ax1, ax2, ax3, ax4 = axes.flatten()

    bars = ax1.bar(names, profits, yerr=stds, capsize=5, color=colors,
                   edgecolor="white", linewidth=1.2)
    ax1.set_title("Mean Season Profit (USD/ha)")
    ax1.set_ylabel("Profit ($)")
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, profits):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f"${val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    bars2 = ax2.bar(names, irrs, color=colors, edgecolor="white", linewidth=1.2)
    ax2.set_title("Mean Total Irrigation (mm/season)")
    ax2.set_ylabel("Irrigation (mm)")
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, irrs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val:.0f}mm", ha="center", va="bottom", fontsize=9)

    bars3 = ax3.bar(names, iwues, color=colors, edgecolor="white", linewidth=1.2)
    ax3.set_title("Irrigation Water Use Efficiency")
    ax3.set_ylabel("kg/ha/mm")
    ax3.set_xticklabels(names, rotation=15, ha="right")
    ax3.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars3, iwues):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    bars4 = ax4.bar(names, stress, color=colors, edgecolor="white", linewidth=1.2)
    ax4.set_title("Stress Days")
    ax4.set_ylabel("days/season")
    ax4.set_xticklabels(names, rotation=15, ha="right")
    ax4.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars4, stress):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("results/plots/policy_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: results/plots/policy_comparison.png")
    plt.show()


def plot_episode_rollout(agent, seq_len=7, seed=999):
    """Visualise a single rollout for the sequence-based TSA-SAC agent."""
    from environment import CropIrrigationEnv
    from collections import deque

    env = CropIrrigationEnv()
    obs, _ = env.reset(seed=seed)
    window = deque(maxlen=seq_len)
    window.append(obs.copy())
    done = False

    days, soil, irr, rainfall, ks = [], [], [], [], []
    while not done:
        seq = agent.build_seq(window)
        action = agent.select_action(seq, deterministic=True)
        obs, _, done, _, info = env.step(action)
        window.append(obs.copy())
        days.append(info["day"])
        soil.append(info["soil_moisture"])
        irr.append(info["irr_applied_mm"])
        rainfall.append(info["rainfall_mm"])
        ks.append(info["Ks"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Single-Season Rollout (TSA-SAC Agent)", fontsize=13, fontweight="bold")

    axes[0].plot(days, soil, color="#2563EB", linewidth=2, label="Soil Moisture")
    axes[0].axhline(0.35, color="green",  linestyle="--", alpha=0.6, label="Field Capacity")
    axes[0].axhline(0.12, color="red",    linestyle="--", alpha=0.6, label="Wilting Point")
    axes[0].set_ylabel("vol/vol")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].bar(days, irr, color="#9333EA", alpha=0.8, label="Irrigation (mm)")
    axes[1].bar(days, rainfall, color="#38BDF8", alpha=0.6, label="Rainfall (mm)")
    axes[1].set_ylabel("Water (mm)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(days, ks, color="#16A34A", linewidth=2, label="Water Stress (Ks)")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Ks")
    axes[2].set_xlabel("Day of Season")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/episode_rollout.png", dpi=150, bbox_inches="tight")
    print("Saved: results/plots/episode_rollout.png")
    plt.show()


if __name__ == "__main__":
    if Path("results/training_history.json").exists():
        plot_training()
    if Path("results/baseline_comparison.json").exists():
        plot_comparison()
