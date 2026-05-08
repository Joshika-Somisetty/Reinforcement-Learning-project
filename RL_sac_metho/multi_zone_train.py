"""
Train a single SAC policy on the shared-resource multi-zone farm environment.

This is a compact proof-of-concept for centralized training over a joint
observation/action space.  Each action dimension controls one zone.
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from multi_zone_env import MultiZoneFarmEnv
from sac_agent import SACAgent
from train import set_global_seed


def build_env(args, seed=None):
    zones = [{"crop": crop.strip(), "weight": 1.0} for crop in args.zones.split(",")]
    weight = 1.0 / max(len(zones), 1)
    for zone in zones:
        zone["weight"] = weight
    return MultiZoneFarmEnv(
        zone_configs=zones,
        shared_reservoir_mm=args.shared_reservoir,
        shared_budget_mm=args.shared_budget,
        climate=args.climate,
        seed=args.seed if seed is None else seed,
    )


def evaluate(env, agent, episodes=10, seq_len=7):
    profits, irrigations, yields = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=5000 + ep)
        window = deque(maxlen=seq_len)
        window.append(obs.copy())
        done = False
        total_reward = 0.0
        while not done:
            seq = agent.build_seq(window)
            action = agent.select_action(seq, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            window.append(obs.copy())
            total_reward += reward
        profits.append(info.get("farm_profit", total_reward))
        irrigations.append(info.get("farm_irrigation_mm", 0.0))
        yields.append(info.get("farm_yield_kg_ha", 0.0))
    return {
        "profit_mean": float(np.mean(profits)),
        "irrigation_mean": float(np.mean(irrigations)),
        "yield_mean": float(np.mean(yields)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones", default="cotton,maize")
    parser.add_argument("--climate", default="arid", choices=["semi_arid", "humid", "arid"])
    parser.add_argument("--shared-reservoir", default=600.0, type=float)
    parser.add_argument("--shared-budget", default=500.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--episodes", default=200, type=int)
    parser.add_argument("--warmup", default=1000, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--buffer-size", default=500_000, type=int)
    parser.add_argument("--seq-len", default=7, type=int)
    parser.add_argument("--eval-every", default=25, type=int)
    parser.add_argument("--eval-episodes", default=5, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    set_global_seed(args.seed)

    env = build_env(args)
    eval_env = build_env(args, seed=args.seed + 100)
    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        seq_len=args.seq_len,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        device="cuda" if torch.cuda.is_available() and args.cuda else "cpu",
    )

    history = {"episode": [], "reward": [], "profit": [], "irrigation": [], "yield": []}
    best_profit = -np.inf
    total_steps = 0
    start = time.time()

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        window = deque(maxlen=args.seq_len)
        window.append(obs.copy())
        done = False
        ep_reward = 0.0

        while not done:
            seq = agent.build_seq(window)
            action = env.action_space.sample() if total_steps < args.warmup else agent.select_action(seq)
            next_obs, reward, done, _, info = env.step(action)
            next_window = deque(window, maxlen=args.seq_len)
            next_window.append(next_obs.copy())
            agent.remember(seq, action, reward, agent.build_seq(next_window), float(done))
            window.append(next_obs.copy())
            obs = next_obs
            ep_reward += reward
            total_steps += 1
            if total_steps >= args.warmup:
                agent.update()

        history["episode"].append(ep)
        history["reward"].append(ep_reward)
        history["profit"].append(info.get("farm_profit", ep_reward))
        history["irrigation"].append(info.get("farm_irrigation_mm", 0.0))
        history["yield"].append(info.get("farm_yield_kg_ha", 0.0))

        if ep % args.eval_every == 0:
            metrics = evaluate(eval_env, agent, args.eval_episodes, args.seq_len)
            print(
                f"Ep {ep:4d}/{args.episodes} | "
                f"Profit={metrics['profit_mean']:7.1f} | "
                f"Irr={metrics['irrigation_mean']:6.1f} | "
                f"Yield={metrics['yield_mean']:7.0f} | "
                f"{time.time() - start:.0f}s"
            )
            if metrics["profit_mean"] > best_profit:
                best_profit = metrics["profit_mean"]
                agent.save("checkpoints/multi_zone_sac_best.pt")

    with open("results/multi_zone_training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best farm profit: {best_profit:.1f}")


if __name__ == "__main__":
    main()
