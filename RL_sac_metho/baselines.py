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
