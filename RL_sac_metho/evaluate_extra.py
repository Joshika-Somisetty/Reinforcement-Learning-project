"""
Hyperparameter sensitivity & unseen-environment generalization analysis.

Produces:
  results/hyperparameter_analysis.json
  results/unseen_environment_analysis.json
  results/plots/hyperparameter_sensitivity.png
  results/plots/unseen_environment.png

Run:
  python evaluate_extra.py
"""

import json, os, sys, copy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── project imports ──────────────────────────────────────────────────
from environment import CropIrrigationEnv
from sac_agent import SACAgent
from train import evaluate_policy, set_global_seed

CKPT = "checkpoints/tsa_sac_improved_best.pt"
RESULTS = "results"
PLOTS   = "results/plots"
Path(RESULTS).mkdir(exist_ok=True)
Path(PLOTS).mkdir(exist_ok=True)

def load_agent(obs_dim, lstm_hidden=256, seq_len=7, encoder="tsa"):
    """Load trained agent from checkpoint."""
    agent = SACAgent(obs_dim=obs_dim, action_dim=1, lstm_hidden=lstm_hidden,
                     seq_len=seq_len, encoder_type=encoder, lr=3e-4)
    agent.load(CKPT)
    return agent

def eval_on_env(agent, env, n_episodes=30, seq_len=7):
    """Evaluate agent on given env."""
    return evaluate_policy(env, agent, n_episodes=n_episodes,
                           deterministic=True, seq_len=seq_len)


# ══════════════════════════════════════════════════════════════════════
#  1. UNSEEN ENVIRONMENT TESTING
# ══════════════════════════════════════════════════════════════════════
def run_unseen_env_test():
    print("\n" + "="*70)
    print("  UNSEEN ENVIRONMENT GENERALIZATION TEST")
    print("="*70)

    scenarios = {
        "Trained (arid, generous)": dict(climate="arid", water_budget_level="generous", reservoir_capacity_mm=2000),
        "Best: humid, generous":    dict(climate="humid", water_budget_level="generous", reservoir_capacity_mm=2000),
        "Avg: semi-arid, moderate": dict(climate="semi_arid", water_budget_level="moderate", reservoir_capacity_mm=1000),
        "Worst: arid, scarce":      dict(climate="arid", water_budget_level="scarce", reservoir_capacity_mm=500),
    }

    # Build agent from trained env
    ref_env = CropIrrigationEnv(crop="cotton", climate="arid",
                                water_budget_level="generous",
                                reservoir_capacity_mm=2000,
                                weather_source="synthetic")
    obs_dim = ref_env.observation_space.shape[0]
    agent = load_agent(obs_dim)
    del ref_env

    results = {}
    for label, kwargs in scenarios.items():
        print(f"\n  Testing: {label}")
        env = CropIrrigationEnv(crop="cotton", weather_source="synthetic", **kwargs)
        metrics = eval_on_env(agent, env)
        results[label] = {
            "profit_mean": metrics["profit_mean"],
            "profit_std":  metrics["profit_std"],
            "yield_mean":  metrics["yield_mean"],
            "irrigation_mean": metrics["irrigation_mean"],
            "iwue_mean":   metrics["iwue_mean"],
            "stress_days_mean": metrics["stress_days_mean"],
        }
        print(f"    Profit: ${metrics['profit_mean']:.1f} ± {metrics['profit_std']:.1f}")
        print(f"    Yield:  {metrics['yield_mean']:.0f} kg/ha | Irrigation: {metrics['irrigation_mean']:.0f}mm")
        print(f"    IWUE:   {metrics['iwue_mean']:.2f} | Stress Days: {metrics['stress_days_mean']:.1f}")

    # Save
    with open(f"{RESULTS}/unseen_environment_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {RESULTS}/unseen_environment_analysis.json")

    # Plot
    labels  = list(results.keys())
    profits = [results[l]["profit_mean"] for l in labels]
    p_std   = [results[l]["profit_std"] for l in labels]
    yields  = [results[l]["yield_mean"] for l in labels]
    iwues   = [results[l]["iwue_mean"] for l in labels]
    stress  = [results[l]["stress_days_mean"] for l in labels]

    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("TSA-SAC Generalization — Unseen Environments", fontsize=16, fontweight="bold")

    for ax, data, std, title, ylabel in [
        (axes[0, 0], profits, p_std, "Mean Season Profit ($)", "USD"),
        (axes[0, 1], yields, None, "Mean Yield (kg/ha)", "kg/ha"),
        (axes[1, 0], iwues, None, "IWUE (kg/ha/mm)", "kg/ha/mm"),
        (axes[1, 1], stress, None, "Stress Days", "days"),
    ]:
        bars = ax.bar(range(len(labels)), data, color=colors,
                      yerr=std if std else None, capsize=5)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        for b, v in zip(bars, data):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(f"{PLOTS}/unseen_environment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS}/unseen_environment.png")

    return results


# ══════════════════════════════════════════════════════════════════════
#  2. HYPERPARAMETER SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════
def run_hyperparameter_analysis():
    print("\n" + "="*70)
    print("  HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("="*70)

    # We'll test sensitivity to key hyperparameters by training SHORT
    # ablation runs (100 episodes each) with different settings
    from train import train
    import argparse

    base_args = argparse.Namespace(
        crop="cotton", climate="arid", weather_source="synthetic",
        water_budget_level="moderate", water_budget=400.0,
        reservoir=2000, seed=42,
        episodes=100, warmup=500, eval_every=50, eval_episodes=10,
        batch_size=64, lr=3e-4, gamma=0.99, tau=0.005,
        lstm_hidden=64, lstm_layers=2,
        seq_len=7, encoder_type="tsa", fixed_reward=False,
        algorithm="sac", buffer_size=100000, amp=True,
        cuda=torch.cuda.is_available(),
        checkpoint_path="checkpoints/_hp_temp.pt",
        history_path="results/_hp_temp_history.json",
        transfer_from=None, freeze_encoder_epochs=0,
        compare_episodes=10, model=None, eval_only=False,
        run_ablation=False,
        update_every=2, gradient_steps=1,
        terminal_reward_scale=25.0,
        alpha_min=0.02, alpha_max=0.5,
        critic_loss="huber",
    )

    experiments = {
        "Learning Rate": {
            "param": "lr",
            "values": [1e-4, 3e-4, 1e-3],
            "labels": ["1e-4", "3e-4 (default)", "1e-3"],
        },
        "LSTM Hidden Size": {
            "param": "lstm_hidden",
            "values": [32, 64, 128],
            "labels": ["32", "64 (default)", "128"],
        },
        "Sequence Length": {
            "param": "seq_len",
            "values": [1, 3, 7],
            "labels": ["1 (no history)", "3 days", "7 days (default)"],
        },
    }

    results = {}
    for exp_name, config in experiments.items():
        print(f"\n  Experiment: {exp_name}")
        results[exp_name] = {"values": [], "labels": config["labels"],
                             "profits": [], "yields": [], "iwues": []}

        for val, label in zip(config["values"], config["labels"]):
            print(f"    Testing {label}...")
            args = copy.deepcopy(base_args)
            setattr(args, config["param"], val)
            args.checkpoint_path = f"checkpoints/_hp_{config['param']}_{val}.pt"
            args.history_path = f"results/_hp_{config['param']}_{val}_history.json"

            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                agent, history = train(args)

                eval_env = CropIrrigationEnv(
                    crop="cotton", climate="arid", weather_source="synthetic",
                    water_budget_level="moderate",
                    reservoir_capacity_mm=2000)
                metrics = eval_on_env(agent, eval_env, n_episodes=10,
                                      seq_len=args.seq_len)

                results[exp_name]["values"].append(val)
                results[exp_name]["profits"].append(metrics["profit_mean"])
                results[exp_name]["yields"].append(metrics["yield_mean"])
                results[exp_name]["iwues"].append(metrics["iwue_mean"])

                print(f"      Profit: ${metrics['profit_mean']:.1f} | "
                      f"Yield: {metrics['yield_mean']:.0f}")

                del agent, eval_env
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"      ERROR: {e}")
                results[exp_name]["values"].append(val)
                results[exp_name]["profits"].append(0)
                results[exp_name]["yields"].append(0)
                results[exp_name]["iwues"].append(0)

            # Cleanup temp files
            for f in [args.checkpoint_path, args.history_path]:
                if os.path.exists(f):
                    os.remove(f)

    # Save
    with open(f"{RESULTS}/hyperparameter_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {RESULTS}/hyperparameter_analysis.json")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Hyperparameter Sensitivity Analysis (100 episodes, moderate budget)",
                 fontsize=14, fontweight="bold")
    bar_colors = ["#42A5F5", "#4CAF50", "#FF9800"]

    for ax, (exp_name, data) in zip(axes, results.items()):
        bars = ax.bar(data["labels"], data["profits"], color=bar_colors[:len(data["labels"])])
        ax.set_title(exp_name, fontweight="bold", fontsize=13)
        ax.set_ylabel("Profit ($)")
        ax.tick_params(axis="x", rotation=15)
        for b, v in zip(bars, data["profits"]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"${v:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(f"{PLOTS}/hyperparameter_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS}/hyperparameter_sensitivity.png")

    # Print summary table
    print("\n  HYPERPARAMETER SENSITIVITY SUMMARY")
    print("  " + "-"*60)
    for exp_name, data in results.items():
        print(f"  {exp_name}:")
        for label, profit in zip(data["labels"], data["profits"]):
            print(f"    {label:25s} → ${profit:.1f}")
    print("  " + "-"*60)

    return results


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    set_global_seed(42)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  TSA-SAC Extended Evaluation                               ║")
    print("║  1. Unseen Environment Generalization                      ║")
    print("║  2. Hyperparameter Sensitivity Analysis                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # 1. Unseen environment (fast — just evaluation, ~2 min)
    unseen = run_unseen_env_test()

    # 2. Hyperparameter sensitivity (trains 9 short runs, ~30-45 min)
    hp = run_hyperparameter_analysis()

    print("\n" + "="*70)
    print("  ALL DONE! Generated:")
    print(f"    {RESULTS}/unseen_environment_analysis.json")
    print(f"    {RESULTS}/hyperparameter_analysis.json")
    print(f"    {PLOTS}/unseen_environment.png")
    print(f"    {PLOTS}/hyperparameter_sensitivity.png")
    print("="*70)
