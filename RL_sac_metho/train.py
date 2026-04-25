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
import time
from collections import deque
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


# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    env = CropIrrigationEnv(
        crop=args.crop,
        climate=args.climate,
        reservoir_capacity_mm=args.reservoir,
    )
    eval_env = CropIrrigationEnv(
        crop=args.crop,
        climate=args.climate,
        reservoir_capacity_mm=args.reservoir,
    )

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(
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
    )

    print(f"\n{'='*68}")
    print("  TSA-SAC Irrigation Agent")
    print(f"  Crop: {args.crop.upper()} | Climate: {args.climate}")
    print(f"  obs_dim={obs_dim} | seq_len={args.seq_len} | "
          f"lstm={args.lstm_hidden}x{args.lstm_layers}")
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

            if total_steps >= args.warmup:
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
                eval_env, agent, n_episodes=10, seq_len=args.seq_len
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
                agent.save("checkpoints/tsa_sac_best.pt")
                print(f"  New best checkpoint saved: ${best_profit:.1f}")

    with open("results/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best eval profit: ${best_profit:.1f}")
    return agent, history


# ──────────────────────────────────────────────────────────────────────────────
def compare_baselines(args, agent=None):
    env = CropIrrigationEnv(
        crop=args.crop, climate=args.climate,
        reservoir_capacity_mm=args.reservoir,
    )

    if agent is None:
        obs_dim    = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = SACAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            seq_len=args.seq_len,
            lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers,
            device="cuda" if torch.cuda.is_available() and args.cuda else "cpu",
            use_amp=args.amp,
        )
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
        r = evaluate_policy(env, policy, n_episodes=30,
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


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop",       default="cotton",     choices=["cotton","wheat","maize"])
    parser.add_argument("--climate",    default="arid", choices=["semi_arid","humid","arid"])
    parser.add_argument("--episodes",   default=1000,  type=int)
    parser.add_argument("--warmup",     default=10000, type=int)
    parser.add_argument("--batch-size", default=64,  type=int)
    parser.add_argument("--buffer-size",default=1_000_000, type=int)
    parser.add_argument("--lr",         default=3e-4, type=float)
    parser.add_argument("--reservoir",  default=300.0,type=float)
    parser.add_argument("--eval-every", default=50,   type=int)
    parser.add_argument("--seq-len",    default=7,    type=int,
                        help="Days of observation history fed to BiLSTM")
    parser.add_argument("--lstm-hidden",default=64,  type=int)
    parser.add_argument("--lstm-layers",default=2,    type=int)
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--model",      default="checkpoints/tsa_sac_best.pt")
    parser.add_argument("--cuda",       action="store_true")
    parser.add_argument("--amp",        action="store_true",
                        help="Enable mixed precision on CUDA to reduce GPU memory usage")
    args = parser.parse_args()

    if args.eval_only:
        env = CropIrrigationEnv(crop=args.crop, climate=args.climate)
        obs_dim    = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = SACAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            seq_len=args.seq_len,
            lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers,
            device="cuda" if torch.cuda.is_available() and args.cuda else "cpu",
            use_amp=args.amp,
        )
        agent.load(args.model)
        compare_baselines(args, agent)
    else:
        agent, history = train(args)
        compare_baselines(args, agent)


if __name__ == "__main__":
    main()
