"""
Training + evaluation pipeline for the TSA-SAC style irrigation agent.

Run:
    python train.py
    python train.py --crop cotton --climate arid --episodes 1000
    python train.py --eval-only --model checkpoints/tsa_sac_best.pt
"""

import argparse
import csv
import json
import random
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from environment import CropIrrigationEnv
from sac_agent import SACAgent
from baselines import RandomPolicy, FixedSchedulePolicy, ThresholdPolicy


# ──────────────────────────────────────────────────────────────────────────────
def evaluate_policy(env, policy, n_episodes=30, deterministic=True,
                    seed_start=1000, seq_len=7):
    """
    Evaluate any policy over n_episodes.
    SAC receives a sequence window; baselines receive a flat obs.
    """
    profits, irrigations, yields = [], [], []
    rainfalls, total_waters, stress_days = [], [], []
    iwues, wues = [], []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + i)
        if hasattr(policy, "reset"):
            policy.reset()

        # Rolling window for SAC
        window = deque(maxlen=seq_len)
        window.append(obs.copy())

        done = False
        ep_reward = 0.0

        while not done:
            if isinstance(policy, SACAgent):
                seq    = policy.build_seq(window)
                action = policy.select_action(seq, deterministic=deterministic)
            else:
                action = policy.select_action(obs)

            obs, reward, done, _, info = env.step(action)
            window.append(obs.copy())
            ep_reward += reward

        profits.append(info.get("episode_profit", ep_reward))
        irrigation = info.get("total_irrigation_mm", 0.0)
        final_yield = info.get("final_yield_kg_ha", 0.0)
        rainfall = info.get("total_rainfall_mm", 0.0)
        total_water = info.get("total_water_input_mm", irrigation + rainfall)
        stress = info.get("stress_days", 0.0)

        irrigations.append(irrigation)
        yields.append(final_yield)
        rainfalls.append(rainfall)
        total_waters.append(total_water)
        stress_days.append(stress)
        iwues.append(final_yield / max(irrigation, 1e-6))
        wues.append(final_yield / max(total_water, 1e-6))

    return {
        "profit_mean":     float(np.mean(profits)),
        "profit_std":      float(np.std(profits)),
        "profit_min":      float(np.min(profits)),
        "profit_max":      float(np.max(profits)),
        "yield_std":       float(np.std(yields)),
        "irrigation_mean": float(np.mean(irrigations)),
        "irrigation_std":  float(np.std(irrigations)),
        "yield_mean":      float(np.mean(yields)),
        "rainfall_mean":   float(np.mean(rainfalls)),
        "total_water_mean":float(np.mean(total_waters)),
        "stress_days_mean":float(np.mean(stress_days)),
        "stress_days_std": float(np.std(stress_days)),
        "iwue_mean":       float(np.mean(iwues)),
        "iwue_std":        float(np.std(iwues)),
        "wue_mean":        float(np.mean(wues)),
        "wue_std":         float(np.std(wues)),
    }


def build_env(args, seed_override=None):
    return CropIrrigationEnv(
        crop=args.crop,
        climate=args.climate,
        reservoir_capacity_mm=args.reservoir,
        dynamic_reward=not args.fixed_reward,
        terminal_reward_scale=getattr(args, "terminal_reward_scale", 100.0),
        seed=args.seed if seed_override is None else seed_override,
    )


def build_agent(args, obs_dim, action_dim):
    return SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        seq_len=args.seq_len,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        mlp_hidden=(256, 256),
        lr=args.lr,
        gamma=0.99,
        tau=0.005,
        auto_alpha=True,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        device="cuda" if torch.cuda.is_available() and args.cuda else "cpu",
        use_amp=args.amp,
        encoder_type=args.encoder_type,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        critic_loss_type=args.critic_loss,
    )


def variant_slug(label: str) -> str:
    slug = label.lower().replace("+", "plus").replace("/", "_")
    for ch in [" ", "(", ")", ".", "-", ","]:
        slug = slug.replace(ch, "_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def run_name(args):
    base = f"{args.encoder_type.upper()}"
    reward_name = "fixed-reward" if args.fixed_reward else "dynamic-reward"
    return f"{base} | seq={args.seq_len} | hidden={args.lstm_hidden} | {reward_name}"


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_checkpoint_overrides(args, model_path):
    """Use saved architecture metadata when evaluating a current-code checkpoint."""
    path = Path(model_path)
    if not path.exists():
        return

    ckpt = torch.load(path, map_location="cpu")
    key_map = {
        "seq_len": "seq_len",
        "encoder_type": "encoder_type",
        "lstm_hidden": "lstm_hidden",
        "lstm_layers": "lstm_layers",
        "critic_loss_type": "critic_loss",
        "alpha_min": "alpha_min",
        "alpha_max": "alpha_max",
    }
    changed = []
    for ckpt_key, arg_key in key_map.items():
        if ckpt_key not in ckpt:
            continue
        saved = ckpt[ckpt_key]
        if saved is None:
            continue
        if getattr(args, arg_key) != saved:
            setattr(args, arg_key, saved)
            changed.append(f"{arg_key}={saved}")

    if changed:
        print("Loaded checkpoint settings: " + ", ".join(changed))


# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    set_global_seed(args.seed)
    env = build_env(args, seed_override=args.seed)
    eval_env = build_env(args, seed_override=args.seed + 101)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = build_agent(args, obs_dim, action_dim)
    checkpoint_path = getattr(args, "checkpoint_path", "checkpoints/tsa_sac_improved_best.pt")
    history_path = getattr(args, "history_path", "results/training_history.json")

    print(f"\n{'='*68}")
    print("  TSA-SAC Irrigation Agent")
    print(f"  Crop: {args.crop.upper()} | Climate: {args.climate}")
    print(f"  {run_name(args)}")
    print(f"  obs_dim={obs_dim} | seq_len={args.seq_len} | lstm={args.lstm_hidden}x{args.lstm_layers}")
    print(f"  Device: {agent.device}")
    print(f"  Mixed precision: {'on' if agent.use_amp else 'off'}")
    print(f"{'='*68}\n")

    history = {
        "episode": [], "reward": [], "profit": [], "irrigation": [],
        "yield": [], "critic_loss": [], "actor_loss": [], "alpha": [],
    }
    best_profit = -np.inf
    total_steps = 0
    start_time  = time.time()

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()

        # Rolling window: holds the last seq_len observations
        window = deque(maxlen=args.seq_len)
        window.append(obs.copy())

        done      = False
        ep_reward = 0.0
        ep_losses = {"critic_loss": [], "actor_loss": [], "alpha": []}

        while not done:
            seq = agent.build_seq(window)   # (seq_len, obs_dim)

            # Warm-up with random actions
            if total_steps < args.warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(seq)

            next_obs, reward, done, _, info = env.step(action)

            # Build next_seq by appending next_obs to the window
            next_window = deque(window, maxlen=args.seq_len)
            next_window.append(next_obs.copy())
            next_seq = agent.build_seq(next_window)

            agent.remember(seq, action, reward, next_seq, float(done))

            # Advance window
            window.append(next_obs.copy())
            obs = next_obs
            ep_reward += reward
            total_steps += 1

            if total_steps >= args.warmup and total_steps % args.update_every == 0:
                for _ in range(args.gradient_steps):
                    losses = agent.update()
                    if losses:
                        for k in ep_losses:
                            ep_losses[k].append(losses.get(k, 0))

        # ── Logging ──────────────────────────────────────────────────────────
        profit = info.get("episode_profit", ep_reward)
        irr    = info.get("total_irrigation_mm", 0.0)
        final_yield = info.get("final_yield_kg_ha", 0.0)
        history["episode"].append(ep)
        history["reward"].append(ep_reward)
        history["profit"].append(profit)
        history["irrigation"].append(irr)
        history["yield"].append(final_yield)
        for k in ["critic_loss", "actor_loss", "alpha"]:
            history[k].append(np.mean(ep_losses[k]) if ep_losses[k] else 0.0)

        # ── Eval & checkpoint ─────────────────────────────────────────────────
        if ep % args.eval_every == 0:
            eval_result = evaluate_policy(
                eval_env, agent, n_episodes=args.eval_episodes, seq_len=args.seq_len
            )
            elapsed = time.time() - start_time
            print(
                f"Ep {ep:4d}/{args.episodes} | "
                f"Reward={ep_reward:7.1f} | "
                f"Profit=${eval_result['profit_mean']:6.1f}"
                f"+/-{eval_result['profit_std']:4.1f} | "
                f"Yield={eval_result['yield_mean']:6.0f}kg/ha | "
                f"Irr={eval_result['irrigation_mean']:5.0f}mm | "
                f"alpha={history['alpha'][-1]:.3f} | "
                f"Steps={total_steps:6d} | {elapsed:.0f}s"
            )
            if eval_result["profit_mean"] > best_profit:
                best_profit = eval_result["profit_mean"]
                agent.save(checkpoint_path)
                print(f"  New best checkpoint saved: ${best_profit:.1f}")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best eval profit: ${best_profit:.1f}")
    return agent, history


# ──────────────────────────────────────────────────────────────────────────────
def compare_baselines(args, agent=None):
    env = build_env(args, seed_override=args.seed + 202)

    if agent is None:
        obs_dim    = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = build_agent(args, obs_dim, action_dim)
        agent.load(args.model)

    policies = {
        "Random":            RandomPolicy(seed=42),
        "Farmer (10-day)":   FixedSchedulePolicy(interval=10, amount_mm=45.0),
        "Threshold (0.45)":  ThresholdPolicy(threshold=0.45, refill_mm=30.0),
        "TSA-SAC (ours)":    agent,
    }

    print("\n" + "="*72)
    print(f"  POLICY COMPARISON — {args.crop.upper()} | {args.climate}")
    print("="*72)
    print(
        f"{'Policy':<24} {'Profit ($)':>12} {'Yield':>10} {'Irr':>8} "
        f"{'IWUE':>8} {'WUE':>8} {'Stress':>8}"
    )
    print("-"*72)

    results = {}
    for name, policy in policies.items():
        r = evaluate_policy(env, policy, n_episodes=args.compare_episodes,
                            seed_start=2000, seq_len=args.seq_len)
        results[name] = r
        print(
            f"{name:<24} "
            f"${r['profit_mean']:>10.1f} "
            f"{r['yield_mean']:>10.0f} "
            f"{r['irrigation_mean']:>8.0f} "
            f"{r['iwue_mean']:>8.2f} "
            f"{r['wue_mean']:>8.2f} "
            f"{r['stress_days_mean']:>8.1f}"
        )
    print("="*72)

    with open("results/baseline_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    metric_order = [
        "profit_mean", "profit_std", "yield_mean", "yield_std",
        "irrigation_mean", "irrigation_std", "rainfall_mean",
        "total_water_mean", "iwue_mean", "iwue_std",
        "wue_mean", "wue_std", "stress_days_mean", "stress_days_std",
    ]
    with open("results/baseline_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", *metric_order])
        for name, metrics in results.items():
            writer.writerow([name] + [metrics.get(k, "") for k in metric_order])
    return results


def run_ablation(args):
    Path("results/ablation").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/ablation").mkdir(parents=True, exist_ok=True)

    variants = [
        ("SAC-MLP", {"encoder_type": "mlp", "seq_len": 1, "fixed_reward": True}),
        ("SAC-BiLSTM", {"encoder_type": "bilstm", "seq_len": args.seq_len, "fixed_reward": True}),
        ("TSA-SAC w/o DynReward", {"encoder_type": "tsa", "seq_len": args.seq_len, "fixed_reward": True}),
        ("TSA-SAC Full", {"encoder_type": "tsa", "seq_len": args.seq_len, "fixed_reward": False}),
    ]

    results = {}
    for label, overrides in variants:
        variant_args = deepcopy(args)
        variant_args.episodes = args.ablation_episodes
        variant_args.warmup = args.ablation_warmup
        variant_args.eval_every = args.ablation_eval_every
        for key, value in overrides.items():
            setattr(variant_args, key, value)

        slug = variant_slug(label)
        variant_args.checkpoint_path = f"checkpoints/ablation/{slug}.pt"
        variant_args.history_path = f"results/ablation/{slug}_training_history.json"

        print(f"\n{'#'*76}")
        print(f"Running ablation variant: {label}")
        print(f"{'#'*76}\n")

        agent, _ = train(variant_args)
        eval_env = build_env(variant_args)
        metrics = evaluate_policy(
            eval_env,
            agent,
            n_episodes=args.ablation_eval_episodes,
            seed_start=4000,
            seq_len=variant_args.seq_len,
        )
        metrics["encoder_type"] = variant_args.encoder_type
        metrics["dynamic_reward"] = not variant_args.fixed_reward
        metrics["seq_len"] = variant_args.seq_len
        metrics["lstm_hidden"] = variant_args.lstm_hidden
        results[label] = metrics

    json_path = "results/ablation/ablation_results.json"
    csv_path = "results/ablation/ablation_results.csv"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    metric_order = [
        "profit_mean", "profit_std", "yield_mean", "yield_std",
        "irrigation_mean", "iwue_mean", "wue_mean", "stress_days_mean",
        "encoder_type", "dynamic_reward", "seq_len", "lstm_hidden",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", *metric_order])
        for name, metrics in results.items():
            writer.writerow([name] + [metrics.get(k, "") for k in metric_order])

    print("\nAblation summary")
    print("=" * 88)
    print(f"{'Variant':<24} {'Profit':>10} {'Yield':>10} {'IWUE':>8} {'Stress':>8}")
    print("-" * 88)
    for name, metrics in results.items():
        print(
            f"{name:<24} "
            f"{metrics['profit_mean']:>10.1f} "
            f"{metrics['yield_mean']:>10.0f} "
            f"{metrics['iwue_mean']:>8.2f} "
            f"{metrics['stress_days_mean']:>8.1f}"
        )
    print("=" * 88)
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop",       default="cotton",     choices=["cotton","wheat","maize"])
    parser.add_argument("--climate",    default="arid", choices=["semi_arid","humid","arid"])
    parser.add_argument("--seed",       default=42, type=int)
    parser.add_argument("--episodes",   default=1000,  type=int)
    parser.add_argument("--warmup",     default=10000, type=int)
    parser.add_argument("--batch-size", default=256,  type=int)
    parser.add_argument("--buffer-size",default=1_000_000, type=int)
    parser.add_argument("--lr",         default=1e-4, type=float)
    parser.add_argument("--reservoir",  default=800.0,type=float)
    parser.add_argument("--terminal-reward-scale", default=100.0, type=float,
                        help="Scale terminal profit before adding it to the RL reward")
    parser.add_argument("--eval-every", default=50,   type=int)
    parser.add_argument("--eval-episodes", default=20, type=int)
    parser.add_argument("--compare-episodes", default=30, type=int)
    parser.add_argument("--update-every", default=2, type=int,
                        help="Run gradient updates every N environment steps")
    parser.add_argument("--gradient-steps", default=1, type=int,
                        help="Number of gradient updates per update event")
    parser.add_argument("--seq-len",    default=7,    type=int,
                        help="Days of observation history fed to BiLSTM")
    parser.add_argument("--lstm-hidden",default=128,  type=int)
    parser.add_argument("--lstm-layers",default=2,    type=int)
    parser.add_argument("--encoder-type", default="tsa", choices=["tsa", "bilstm", "mlp"])
    parser.add_argument("--fixed-reward", action="store_true",
                        help="Use a fixed reward weighting instead of the stage-aware dynamic reward")
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--model",      default="checkpoints/tsa_sac_improved_best.pt")
    parser.add_argument("--checkpoint-path", default="checkpoints/tsa_sac_improved_best.pt")
    parser.add_argument("--history-path", default="results/training_history.json")
    parser.add_argument("--cuda",       action="store_true")
    parser.add_argument("--amp",        action="store_true",
                        help="Enable mixed precision on CUDA to reduce GPU memory usage")
    parser.add_argument("--alpha-min",  default=0.02, type=float)
    parser.add_argument("--alpha-max",  default=0.5, type=float)
    parser.add_argument("--critic-loss", default="huber", choices=["huber", "mse"])
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--ablation-episodes", default=120, type=int)
    parser.add_argument("--ablation-warmup", default=1000, type=int)
    parser.add_argument("--ablation-eval-every", default=30, type=int)
    parser.add_argument("--ablation-eval-episodes", default=12, type=int)
    args = parser.parse_args()

    if args.run_ablation:
        run_ablation(args)
    elif args.eval_only:
        set_global_seed(args.seed)
        apply_checkpoint_overrides(args, args.model)
        env = build_env(args, seed_override=args.seed + 303)
        obs_dim    = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = build_agent(args, obs_dim, action_dim)
        agent.load(args.model)
        compare_baselines(args, agent)
    else:
        agent, history = train(args)
        compare_baselines(args, agent)


if __name__ == "__main__":
    main()
