"""
Multi-zone farming environment with shared reservoir.

Extends the single-zone CropIrrigationEnv to N independent crop zones
that compete for water from a single shared reservoir/budget.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, List

from environment import CropIrrigationEnv, CROP_PROFILES, WATER_BUDGET_PRESETS


class MultiZoneFarmEnv(gym.Env):
    """
    Multi-zone irrigation environment where N zones share a single
    water reservoir and seasonal budget.

    Observation: concatenation of per-zone obs + shared resource info.
    Action: per-zone irrigation amounts (N-dimensional).
    Reward: sum of per-zone daily rewards − coordination penalty.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        zone_configs: Optional[List[dict]] = None,
        shared_reservoir_mm: float = 600.0,
        shared_budget_mm: float = 500.0,
        climate: str = "arid",
        coordination_penalty_weight: float = 0.1,
        seed: Optional[int] = None,
        render_mode=None,
    ):
        """
        Args:
            zone_configs: List of dicts with keys {crop, weight}.
                Default: 2 zones, cotton + maize.
            shared_reservoir_mm: Total shared reservoir capacity.
            shared_budget_mm: Total shared seasonal water budget.
            climate: Climate preset for all zones.
            coordination_penalty_weight: Penalty for exceeding shared budget.
        """
        super().__init__()

        if zone_configs is None:
            zone_configs = [
                {"crop": "cotton", "weight": 0.6},
                {"crop": "maize", "weight": 0.4},
            ]

        self.n_zones = len(zone_configs)
        self.zone_configs = zone_configs
        self.shared_reservoir_cap = shared_reservoir_mm
        self.shared_budget_cap = shared_budget_mm
        self.climate = climate
        self.coord_penalty_w = coordination_penalty_weight
        self.render_mode = render_mode

        self.rng = np.random.default_rng(seed)

        # Create per-zone environments (they don't manage their own reservoir)
        self.zones: List[CropIrrigationEnv] = []
        for i, cfg in enumerate(zone_configs):
            zone = CropIrrigationEnv(
                crop=cfg["crop"],
                reservoir_capacity_mm=1e6,  # effectively unlimited per-zone
                climate=climate,
                water_budget_mm=1e6,        # budget managed at farm level
                seed=(seed + i * 1000) if seed else None,
            )
            self.zones.append(zone)

        # Find max season length across zones
        self.max_T = max(z.T for z in self.zones)

        # Observation: per-zone obs concatenated + 2 shared dims
        self.per_zone_obs_dim = self.zones[0].observation_space.shape[0]
        total_obs_dim = self.per_zone_obs_dim * self.n_zones + 2  # +shared reservoir, +shared budget
        self.observation_space = spaces.Box(
            low=np.zeros(total_obs_dim, dtype=np.float32),
            high=np.ones(total_obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        # Action: per-zone irrigation [0, 60] each
        self.action_space = spaces.Box(
            low=np.zeros(self.n_zones, dtype=np.float32),
            high=np.full(self.n_zones, 60.0, dtype=np.float32),
            dtype=np.float32,
        )

        self._reset_shared()

    def _reset_shared(self):
        self.shared_reservoir = self.shared_reservoir_cap * self.rng.uniform(0.85, 1.0)
        self.shared_budget_remaining = self.shared_budget_cap
        self.day = 0
        self.zone_done = [False] * self.n_zones

    def _get_obs(self):
        obs_parts = []
        for zone in self.zones:
            obs_parts.append(zone._get_obs())

        shared = np.array([
            np.clip(self.shared_reservoir / max(self.shared_reservoir_cap, 1e-6), 0, 1),
            np.clip(self.shared_budget_remaining / max(self.shared_budget_cap, 1e-6), 0, 1),
        ], dtype=np.float32)

        return np.concatenate(obs_parts + [shared])

    def step(self, action):
        """
        action: array of shape (n_zones,) — irrigation per zone.
        """
        action = np.clip(action, 0.0, 60.0)

        # ── Allocate shared water across zones ──────────────────────
        total_requested = float(np.sum(action))
        available = min(self.shared_reservoir, self.shared_budget_remaining)

        if total_requested > available and total_requested > 0:
            # Scale down proportionally
            scale = available / total_requested
            action = action * scale

        total_allocated = float(np.sum(action))
        self.shared_reservoir -= total_allocated
        self.shared_budget_remaining -= total_allocated

        # ── Step each zone ──────────────────────────────────────────
        total_reward = 0.0
        all_info = {}
        all_terminated = True

        for i, (zone, cfg) in enumerate(zip(self.zones, self.zone_configs)):
            if self.zone_done[i]:
                continue

            zone_action = np.array([action[i]], dtype=np.float32)

            # Override zone's own budget/reservoir to let farm manage them
            zone.water_budget_remaining = 1e6
            zone.reservoir = 1e6

            obs, reward, terminated, truncated, info = zone.step(zone_action)

            weight = cfg.get("weight", 1.0 / self.n_zones)
            total_reward += weight * reward

            self.zone_done[i] = terminated
            all_info[f"zone_{i}"] = info

            if not terminated:
                all_terminated = False

        # Coordination penalty: penalise days when total demand exceeds supply
        if total_requested > available * 1.1:
            excess_ratio = (total_requested - available) / max(total_requested, 1e-6)
            total_reward -= self.coord_penalty_w * excess_ratio

        self.day += 1
        terminated = all_terminated or self.day >= self.max_T

        obs = self._get_obs()
        all_info["shared_reservoir_mm"] = self.shared_reservoir
        all_info["shared_budget_remaining_mm"] = self.shared_budget_remaining
        all_info["day"] = self.day

        if terminated:
            # Aggregate farm-level metrics
            total_profit = 0.0
            total_irr = 0.0
            total_yield = 0.0
            for i in range(self.n_zones):
                zi = all_info.get(f"zone_{i}", {})
                total_profit += zi.get("episode_profit", 0.0)
                total_irr += zi.get("total_irrigation_mm", 0.0)
                total_yield += zi.get("final_yield_kg_ha", 0.0)
            all_info["farm_profit"] = total_profit
            all_info["farm_irrigation_mm"] = total_irr
            all_info["farm_yield_kg_ha"] = total_yield

        return obs, total_reward, terminated, False, all_info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._reset_shared()
        for i, zone in enumerate(self.zones):
            zone.reset(seed=(seed + i * 1000) if seed else None)

        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            print(f"=== Farm Day {self.day} ===")
            print(f"  Shared reservoir: {self.shared_reservoir:.0f}mm")
            print(f"  Shared budget:    {self.shared_budget_remaining:.0f}mm")
            for i, zone in enumerate(self.zones):
                cfg = self.zone_configs[i]
                print(f"  Zone {i} ({cfg['crop']}): "
                      f"θ={zone.theta:.3f} Ks={zone._water_stress():.2f} "
                      f"BM={zone.biomass:.0f}")
