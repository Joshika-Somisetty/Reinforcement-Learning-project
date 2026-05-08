"""
baselines.py
============
Baseline irrigation policies to compare against SAC.

Policies
--------
1. RandomPolicy      — uniformly random irrigation
2. FixedSchedule     — irrigate fixed amount every N days
3. ThresholdPolicy   — irrigate when soil moisture drops below threshold
                       (best heuristic, common in practice)
4. FarmerExpert      — stage-aware heuristic matching the paper's FE baseline
"""

import numpy as np


class RandomPolicy:
    """Uniformly random irrigation in [0, max_irr] mm."""
    def __init__(self, max_irr: float = 50.0, seed: int = 0):
        self.max_irr = max_irr
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        return np.array([self.rng.uniform(0, self.max_irr)], dtype=np.float32)


class FixedSchedulePolicy:
    """
    Irrigate a fixed amount every `interval` days regardless of conditions.
    Mimics traditional calendar-based irrigation.
    """
    def __init__(self, interval: int = 7, amount_mm: float = 25.0):
        self.interval = interval
        self.amount   = amount_mm
        self.day      = 0

    def reset(self):
        self.day = 0

    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        irr = self.amount if self.day % self.interval == 0 else 0.0
        self.day += 1
        return np.array([irr], dtype=np.float32)


class ThresholdPolicy:
    """
    Irrigate when the normalised available soil water (obs[3])
    drops below `threshold`. This is a strong agronomic heuristic.
    """
    def __init__(self, threshold: float = 0.45, refill_mm: float = 30.0):
        self.threshold = threshold
        self.refill    = refill_mm

    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        soil_water_avail_norm = float(obs[3])
        irr = self.refill if soil_water_avail_norm < self.threshold else 0.0
        return np.array([irr], dtype=np.float32)


class FarmerExpert:
    """
    Stage-aware expert heuristic matching the paper's FE baseline.

    Rules:
      - During reproductive/flowering stage (obs[12]=1): irrigate 40mm
        if soil water < 0.60, conserving water during less critical stages.
      - During vegetative stage (obs[11]=1): irrigate 30mm if soil < 0.45.
      - During emergence/maturity: irrigate 20mm only if soil < 0.35.
      - Also considers budget remaining (obs[15]) — reduces irrigation
        when budget is low.
    """
    def __init__(self):
        self.day = 0

    def reset(self):
        self.day = 0

    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        soil = float(obs[3])      # soil_water_avail_norm
        budget = float(obs[15]) if len(obs) > 15 else 1.0   # budget_remaining_norm

        # Growth stage one-hot: indices 10-14
        stage_idx = int(np.argmax(obs[10:15]))

        # Reduce irrigation amounts when budget is low
        budget_factor = np.clip(budget / 0.3, 0.3, 1.0)

        if stage_idx == 2:  # reproductive — most critical
            if soil < 0.60:
                irr = 40.0 * budget_factor
            else:
                irr = 0.0
        elif stage_idx == 1:  # vegetative
            if soil < 0.45:
                irr = 30.0 * budget_factor
            else:
                irr = 0.0
        elif stage_idx == 3:  # boll fill / grain fill
            if soil < 0.40:
                irr = 25.0 * budget_factor
            else:
                irr = 0.0
        else:  # emergence or maturity
            if soil < 0.35:
                irr = 20.0 * budget_factor
            else:
                irr = 0.0

        self.day += 1
        return np.array([irr], dtype=np.float32)
