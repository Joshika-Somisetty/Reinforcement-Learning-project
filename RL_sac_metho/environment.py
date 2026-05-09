"""
Crop irrigation environment used to approximate the TSA-SAC paper setup.

This is still a custom Gymnasium environment rather than a DSSAT wrapper, but
it now mirrors the paper more closely:
  - stage-aware reward weights
  - forecast-aware preprocessed state with reservoir awareness
  - continuous irrigation action in [0, 60] mm
  - soil / crop / weather variables that proxy the paper's DSSAT features
  - seasonal water budget constraint for scarcity-driven decision making
  - optional real weather data support
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional


# ──────────────────────────────────────────────
# Crop parameter presets
# ──────────────────────────────────────────────
CROP_PROFILES = {
    "cotton": dict(
        season_days=170,
        max_yield_kg=7600,
        Kc_ini=0.35, Kc_mid=1.15, Kc_end=0.70,
        fc=0.33,
        wp=0.15,
        root_depth_mm=1200,
        price_per_kg=0.22,
        water_cost_per_mm=0.55,
        base_temp_c=14.0,
        lai_max=5.5,
    ),
    "wheat": dict(
        season_days=120,
        max_yield_kg=6000,          # kg / ha at no stress
        Kc_ini=0.4, Kc_mid=1.15, Kc_end=0.4,   # FAO crop coefficients
        fc=0.35,                    # field capacity (vol/vol)
        wp=0.12,                    # wilting point
        root_depth_mm=600,          # effective rooting depth
        price_per_kg=0.25,          # USD / kg
        water_cost_per_mm=0.45,     # USD / mm applied
        base_temp_c=5.0,
        lai_max=6.0,
    ),
    "maize": dict(
        season_days=100,
        max_yield_kg=9000,
        Kc_ini=0.3, Kc_mid=1.20, Kc_end=0.6,
        fc=0.32, wp=0.11,
        root_depth_mm=700,
        price_per_kg=0.18,
        water_cost_per_mm=0.45,
        base_temp_c=10.0,
        lai_max=6.5,
    ),
}

# Default seasonal water budgets (mm) per crop — calibrated so that
# the "moderate" budget forces real trade-offs (roughly 50-60% of
# unconstrained optimal irrigation).
WATER_BUDGET_PRESETS = {
    "cotton":  {"generous": 600, "moderate": 400, "scarce": 250},
    "wheat":   {"generous": 400, "moderate": 280, "scarce": 180},
    "maize":   {"generous": 450, "moderate": 300, "scarce": 200},
}

STATE_KEYS = [
    "lai_norm",
    "biomass_norm",
    "root_depth_norm",
    "soil_water_avail_norm",
    "reservoir_norm",
    "water_stress_norm",
    "et0_norm",
    "rain_norm",
    "rain_forecast_3d_norm",
    "et0_forecast_3d_norm",
    "stage_emergence",
    "stage_vegetative",
    "stage_reproductive",
    "stage_boll_fill",
    "stage_maturity",
    "budget_remaining_norm",     # NEW: seasonal water budget fraction
]


class CropIrrigationEnv(gym.Env):
    """
    Single-season crop irrigation environment with stochastic weather.

    Enhancement over baseline Q-learning paper:
      1. Continuous action space (SAC-compatible)
      2. Physics-based soil-water balance
      3. Stochastic Markov weather chain
      4. Multi-objective reward function
      5. Seasonal water budget constraint
      6. Optional real weather data
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        crop: str = "cotton",
        reservoir_capacity_mm: float = 800.0,
        climate: str = "arid",    # "semi_arid" | "humid" | "arid"
        dynamic_reward: bool = True,
        terminal_reward_scale: float = 25.0,
        water_budget_mm: Optional[float] = None,
        water_budget_level: str = "generous",
        weather_source: str = "synthetic",   # "synthetic" | "real"
        seed: Optional[int] = None,
        render_mode=None,
    ):
        super().__init__()
        self.crop_name = crop
        self.crop_params = CROP_PROFILES[crop]
        self.T = self.crop_params["season_days"]
        self.reservoir_cap = reservoir_capacity_mm
        self.climate = climate
        self.dynamic_reward = dynamic_reward
        self.terminal_reward_scale = max(float(terminal_reward_scale), 1e-6)
        self.render_mode = render_mode
        self.weather_source = weather_source

        # ── Water budget ─────────────────────────────────────────────
        if water_budget_mm is not None:
            self.water_budget_cap = float(water_budget_mm)
        else:
            self.water_budget_cap = float(
                WATER_BUDGET_PRESETS[crop][water_budget_level]
            )

        # ── action / observation spaces ──────────────────────────────
        # Continuous irrigation depth in mm [0, 60] as in the paper.
        self.action_space = spaces.Box(
            low=np.float32([0.0]),
            high=np.float32([60.0]),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.float32(np.zeros(len(STATE_KEYS))),
            high=np.float32(np.ones(len(STATE_KEYS))),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)

        # Climate presets (mean daily rainfall prob, mean temp, temp std)
        self._climate_params = {
            "semi_arid": dict(p_rain=0.15, rain_mean=6, rain_std=4, T_mean=28, T_std=5, solar_mean=22, wind_mean=2.8),
            "humid":     dict(p_rain=0.40, rain_mean=10, rain_std=6, T_mean=24, T_std=4, solar_mean=18, wind_mean=2.2),
            "arid":      dict(p_rain=0.05, rain_mean=3,  rain_std=2, T_mean=34, T_std=6, solar_mean=25, wind_mean=3.4),
        }[climate]

        self._reset_state()

    # ──────────────────────────────────────────────────────────────────
    def _reset_state(self):
        cp = self.crop_params
        self.day = 0
        # Soil moisture: start at field capacity − small random deficit
        self.theta = cp["fc"] - self.rng.uniform(0.02, 0.06)
        self.biomass = 0.0
        self.reservoir = self.reservoir_cap * self.rng.uniform(0.85, 1.0)
        self.water_budget_remaining = self.water_budget_cap
        self.cumulative_irrigation = 0.0
        self.cumulative_effective_irrigation = 0.0
        self.cumulative_deficit = 0.0
        self.cumulative_rainfall = 0.0
        self.stress_days = 0
        self.episode_profit = 0.0
        self.leaf_area_index = 0.05
        self.stage_idx = 0
        self.wasted_actions = 0   # times agent tried to irrigate past budget

        # Pre-generate season weather for this episode
        self._generate_season_weather()

    def _generate_season_weather(self):
        """
        Generate stochastic daily weather for one season using a
        two-state Markov chain for rainfall occurrence.
        Falls back to real weather if weather_source == "real".
        """
        if self.weather_source == "real":
            self._load_real_weather()
            return

        cp = self._climate_params
        p = cp["p_rain"]
        # Markov transition: P(rain|dry), P(rain|wet)
        p_rw = min(p * 2.5, 0.85)   # rain → rain persistence
        p_dr = p * 0.5               # dry  → rain

        rain_occurs = np.zeros(self.T, dtype=bool)
        state = self.rng.random() < p   # initial state
        for t in range(self.T):
            rain_occurs[t] = state
            if state:
                state = self.rng.random() < p_rw
            else:
                state = self.rng.random() < p_dr

        # Rainfall amounts (mm) — gamma distributed on wet days
        rain_mm = np.where(
            rain_occurs,
            self.rng.gamma(2.0, cp["rain_mean"] / 2.0, size=self.T),
            0.0,
        )
        rain_mm = np.clip(rain_mm, 0, 80)

        # Temperature: sine seasonal curve + noise
        day_idx = np.arange(self.T)
        seasonal = 4 * np.sin(np.pi * day_idx / self.T)   # peak mid-season
        self.temps = (
            cp["T_mean"] + seasonal
            + self.rng.normal(0, cp["T_std"], size=self.T)
        )
        self.rainfall = rain_mm
        self.wind = np.clip(
            self.rng.normal(cp["wind_mean"], 0.8, size=self.T), 0.5, 8.0
        )
        self.solar_rad = np.clip(
            cp["solar_mean"] + 3.5 * np.sin(np.pi * day_idx / self.T)
            + self.rng.normal(0, 2.0, size=self.T),
            8.0, 32.0,
        )
        self.et0_series = np.array([
            self._et0(self.temps[t], self.solar_rad[t], self.wind[t])
            for t in range(self.T)
        ], dtype=np.float32)
        self.rain_forecast_3d = np.zeros(self.T, dtype=np.float32)
        self.et0_forecast_3d = np.zeros(self.T, dtype=np.float32)
        for t in range(self.T):
            t_end = min(t + 3, self.T)
            rain_sum = float(np.sum(self.rainfall[t:t_end]))
            et0_mean = float(np.mean(self.et0_series[t:t_end]))
            self.rain_forecast_3d[t] = np.clip(rain_sum * self.rng.normal(1.0, 0.12), 0.0, 90.0)
            self.et0_forecast_3d[t] = np.clip(et0_mean * self.rng.normal(1.0, 0.05), 0.0, 12.0)

    def _load_real_weather(self):
        """Load real weather data from weather_data module."""
        try:
            from weather_data import load_weather_season
            data = load_weather_season(
                crop=self.crop_name,
                season_days=self.T,
                climate=self.climate,
                rng=self.rng,
            )
            self.temps = data["temperature"]
            self.rainfall = data["rainfall"]
            self.wind = data["wind"]
            self.solar_rad = data["solar_rad"]
            self.et0_series = np.array([
                self._et0(self.temps[t], self.solar_rad[t], self.wind[t])
                for t in range(self.T)
            ], dtype=np.float32)
            self.rain_forecast_3d = np.zeros(self.T, dtype=np.float32)
            self.et0_forecast_3d = np.zeros(self.T, dtype=np.float32)
            for t in range(self.T):
                t_end = min(t + 3, self.T)
                rain_sum = float(np.sum(self.rainfall[t:t_end]))
                et0_mean = float(np.mean(self.et0_series[t:t_end]))
                self.rain_forecast_3d[t] = np.clip(
                    rain_sum * self.rng.normal(1.0, 0.12), 0.0, 90.0
                )
                self.et0_forecast_3d[t] = np.clip(
                    et0_mean * self.rng.normal(1.0, 0.05), 0.0, 12.0
                )
        except Exception as e:
            print(f"[WARNING] Real weather load failed ({e}), using synthetic.")
            self.weather_source = "synthetic"
            self._generate_season_weather()

    # ──────────────────────────────────────────────────────────────────
    def _get_kc(self) -> float:
        """Linear Kc interpolation across three growth stages."""
        cp = self.crop_params
        frac = self.day / self.T
        if frac < 0.25:
            return cp["Kc_ini"] + (cp["Kc_mid"] - cp["Kc_ini"]) * (frac / 0.25)
        elif frac < 0.75:
            return cp["Kc_mid"]
        else:
            return cp["Kc_mid"] + (cp["Kc_end"] - cp["Kc_mid"]) * ((frac - 0.75) / 0.25)

    def _et0(self, temp_c: float, solar_rad: float, wind: float) -> float:
        """Approximate daily ET0 using temperature, radiation, and wind."""
        temp_term = max(0.0, temp_c - 5.0)
        return np.clip(
            0.11 * temp_term + 0.08 * solar_rad + 0.35 * wind,
            0.0,
            12.0,
        )

    def _water_stress(self) -> float:
        """
        FAO-56 Ks: linear between TAW and RAW thresholds.
        Ks = 1 (no stress) when theta > p_factor * TAW
        """
        cp = self.crop_params
        root_depth_mm = self._root_depth_mm()
        TAW = (cp["fc"] - cp["wp"]) * root_depth_mm        # mm
        RAW = 0.5 * TAW  # depletion fraction p = 0.5
        Dr = max(0.0, (cp["fc"] - self.theta) * root_depth_mm)
        if Dr <= RAW:
            return 1.0
        elif Dr >= TAW:
            return 0.0
        else:
            return (TAW - Dr) / (TAW - RAW)

    def _root_depth_mm(self) -> float:
        """Root depth grows through the season instead of staying constant."""
        cp = self.crop_params
        frac = self.day / max(self.T - 1, 1)
        growth = 0.18 + 0.82 / (1.0 + np.exp(-8.0 * (frac - 0.30)))
        return float(cp["root_depth_mm"] * growth)

    def _effective_rainfall(self, rain_mm: float) -> float:
        """Large storms lose a larger share to runoff and surface losses."""
        efficiency = np.clip(0.95 - 0.004 * max(rain_mm - 8.0, 0.0), 0.55, 0.95)
        return rain_mm * efficiency

    def _effective_irrigation(self, irrigation_mm: float) -> float:
        """
        Smaller, targeted irrigation events are more efficient than large pulses.
        This gives the RL policy a meaningful advantage over coarse heuristics.
        """
        efficiency = np.clip(0.90 - 0.005 * max(irrigation_mm - 18.0, 0.0), 0.60, 0.90)
        return irrigation_mm * efficiency

    def _crop_stage(self) -> float:
        frac = self.day / self.T
        if frac < 0.12:
            self.stage_idx = 0
        elif frac < 0.35:
            self.stage_idx = 1
        elif frac < 0.65:
            self.stage_idx = 2
        elif frac < 0.85:
            self.stage_idx = 3
        else:
            self.stage_idx = 4
        return self.stage_idx / 4.0

    def _stage_one_hot(self):
        one_hot = np.zeros(5, dtype=np.float32)
        one_hot[self.stage_idx] = 1.0
        return one_hot

    def _dynamic_reward_weights(self):
        #                  (yield_w, water_w, stress_w)
        weights = [
            (0.8, 0.4, 0.8),  # emergence — low stress penalty, crop is hardy
            (1.0, 0.3, 1.0),  # vegetative — moderate
            (1.2, 0.2, 1.8),  # flowering / reproductive — stress matters most here
            (0.6, 0.5, 0.6),  # boll opening / grain fill — winding down
            (0.2, 0.8, 0.3),  # maturity — mostly about water cost now
        ]
        return weights[self.stage_idx]

    def _scarcity_cost_multiplier(self) -> float:
        """
        Water becomes progressively more expensive as the seasonal budget
        depletes.  This non-linear pricing forces the agent to plan ahead.

        At 100% budget remaining → 1.0× cost
        At  50% budget remaining → 1.25× cost
        At  20% budget remaining → 1.64× cost
        At   0% budget remaining → 2.0× cost  (but irrigation is blocked)
        """
        if self.water_budget_cap <= 0:
            return 1.0
        frac = np.clip(self.water_budget_remaining / self.water_budget_cap, 0, 1)
        return 1.0 + 1.0 * (1.0 - frac) ** 2

    # ──────────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        cp = self.crop_params
        self._crop_stage()
        day = min(self.day, self.T - 1)
        root_depth_mm = self._root_depth_mm()
        taw = (cp["fc"] - cp["wp"]) * root_depth_mm
        available = np.clip((self.theta - cp["wp"]) * root_depth_mm, 0.0, taw)
        budget_norm = np.clip(
            self.water_budget_remaining / max(self.water_budget_cap, 1e-6), 0, 1
        )
        obs = np.array([
            np.clip(self.leaf_area_index / cp["lai_max"], 0, 1),
            np.clip(self.biomass / cp["max_yield_kg"], 0, 1),
            np.clip(root_depth_mm / cp["root_depth_mm"], 0, 1),
            np.clip(available / max(taw, 1e-6), 0, 1),
            np.clip(self.reservoir / max(self.reservoir_cap, 1e-6), 0, 1),
            np.clip(1.0 - self._water_stress(), 0, 1),
            np.clip(self._et0(self.temps[day], self.solar_rad[day], self.wind[day]) / 12.0, 0, 1),
            np.clip(self.rainfall[day] / 30.0, 0, 1),
            np.clip(self.rain_forecast_3d[day] / 60.0, 0, 1),
            np.clip(self.et0_forecast_3d[day] / 12.0, 0, 1),
            *self._stage_one_hot(),
            budget_norm,
        ], dtype=np.float32)
        return obs

    # ──────────────────────────────────────────────────────────────────
    def step(self, action):
        cp = self.crop_params
        irr_mm = float(np.clip(action[0], 0.0, 60.0))

        # ── 1. Weather for today ──────────────────────────────────────
        temp  = self.temps[self.day]
        rain  = self.rainfall[self.day]
        wind  = self.wind[self.day]
        solar = self.solar_rad[self.day]
        et0   = self._et0(temp, solar, wind)
        kc    = self._get_kc()
        potential_etc = et0 * kc  # crop evapotranspiration (mm/day)

        # ── 2. Budget + Reservoir update ─────────────────────────────
        # Water budget is the binding constraint; reservoir is secondary.
        budget_limited = min(irr_mm, self.water_budget_remaining)
        actual_irr = min(budget_limited, self.reservoir)

        # Track wasted actions (agent tried to irrigate but budget empty)
        budget_waste = irr_mm - actual_irr
        if budget_waste > 1.0:
            self.wasted_actions += 1

        self.reservoir -= actual_irr
        self.water_budget_remaining -= actual_irr
        self.cumulative_irrigation += actual_irr
        self.cumulative_rainfall += rain
        effective_rain = self._effective_rainfall(rain)
        effective_irr = self._effective_irrigation(actual_irr)
        self.cumulative_effective_irrigation += effective_irr

        # ── 3. Soil water balance ─────────────────────────────────────
        # Apply rain and irrigation before stress is evaluated so today's action
        # can influence today's crop response.
        root_depth_mm = self._root_depth_mm()
        self.theta += (effective_rain + effective_irr) / root_depth_mm
        # Drainage (percolation) when above field capacity
        deep_drainage_mm = 0.0
        if self.theta > cp["fc"]:
            deep_drainage_mm = (self.theta - cp["fc"]) * root_depth_mm
            self.theta = cp["fc"]   # free drainage

        self._crop_stage()
        ks = self._water_stress()
        actual_etc = potential_etc * ks
        self.theta -= actual_etc / root_depth_mm
        self.theta = np.clip(self.theta, cp["wp"] * 0.5, cp["fc"])
        if ks < 0.5:
            self.stress_days += 1

        # Water deficit accumulation (for reward)
        deficit = max(0.0, potential_etc - actual_etc)
        self.cumulative_deficit += deficit

        # ── 4. Biomass accumulation ───────────────────────────────────
        # Biomass gain is used as the paper-style yield gain term.
        temp_eff = np.clip((temp - cp["base_temp_c"]) / (30.0 - cp["base_temp_c"]), 0.0, 1.0)
        radiation_eff = np.clip(solar / 26.0, 0.2, 1.2)
        biomass_gain = 55.0 * temp_eff * radiation_eff * ks
        self.biomass += biomass_gain

        # Approximate LAI from biomass with stage-sensitive saturation.
        stage_factor = [0.45, 0.9, 1.0, 0.7, 0.35][self.stage_idx]
        lai_target = cp["lai_max"] * stage_factor * np.clip(self.biomass / (0.55 * cp["max_yield_kg"]), 0.0, 1.0)
        self.leaf_area_index += 0.35 * (lai_target - self.leaf_area_index)
        self.leaf_area_index = np.clip(self.leaf_area_index, 0.05, cp["lai_max"])

        # ── 5. Reward (multi-objective) ───────────────────────────────
        if self.dynamic_reward:
            wy, ww, ws = self._dynamic_reward_weights()
        else:
            wy, ww, ws = 1.0, 0.4, 1.5
        yield_gain = biomass_gain / 55.0

        # Scarcity-adjusted water cost
        scarcity = self._scarcity_cost_multiplier()
        water_cost = (actual_irr / 45.0) * scarcity

        stress_penalty = (1.0 - ks) ** 2
        daily_reward = wy * yield_gain - ww * water_cost - ws * stress_penalty

        # Reward baseline offset (standard RL reward shaping).
        # Adding a constant to all timesteps does NOT change the optimal policy
        # or gradient — it only shifts the displayed reward to be more intuitive.
        daily_reward += 0.5

        # Small penalty for trying to irrigate with exhausted budget
        if budget_waste > 1.0:
            daily_reward -= 0.05

        self.day += 1
        terminated = self.day >= self.T

        # Terminal reward: harvest profit
        terminal_reward = 0.0
        if terminated:
            bm_factor = np.clip(self.biomass / (self.T * 42.0), 0, 1)
            final_yield = bm_factor * cp["max_yield_kg"]
            gross = final_yield * cp["price_per_kg"]
            total_water_cost = self.cumulative_irrigation * cp["water_cost_per_mm"]
            self.episode_profit = gross - total_water_cost
            terminal_reward = self.episode_profit / self.terminal_reward_scale

        reward = daily_reward + terminal_reward
        obs = self._get_obs()
        info = {
            "day": self.day,
            "soil_moisture": self.theta,
            "Ks": ks,
            "biomass": self.biomass,
            "lai": self.leaf_area_index,
            "rainfall_mm": rain,
            "effective_rainfall_mm": effective_rain,
            "irr_applied_mm": actual_irr,
            "effective_irr_mm": effective_irr,
            "reservoir_mm": self.reservoir,
            "budget_remaining_mm": self.water_budget_remaining,
            "root_depth_mm": root_depth_mm,
            "deep_drainage_mm": deep_drainage_mm,
            "et0": et0,
            "wind": wind,
            "solar_rad": solar,
            "reward_weights": {"yield": wy, "water": ww, "stress": ws},
            "growth_stage_idx": self.stage_idx,
        }
        if terminated:
            info["episode_profit"] = self.episode_profit
            info["total_irrigation_mm"] = self.cumulative_irrigation
            info["total_effective_irrigation_mm"] = self.cumulative_effective_irrigation
            info["final_yield_kg_ha"] = final_yield
            info["total_rainfall_mm"] = self.cumulative_rainfall
            info["total_water_input_mm"] = self.cumulative_irrigation + self.cumulative_rainfall
            info["stress_days"] = self.stress_days
            info["terminal_reward_scaled"] = terminal_reward
            info["water_budget_cap_mm"] = self.water_budget_cap
            info["water_budget_used_mm"] = self.water_budget_cap - self.water_budget_remaining
            info["wasted_actions"] = self.wasted_actions

        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            print(
                f"Day {self.day:3d} | θ={self.theta:.3f} | "
                f"Ks={self._water_stress():.2f} | "
                f"Biomass={self.biomass:.1f} | "
                f"Reservoir={self.reservoir:.0f}mm | "
                f"Budget={self.water_budget_remaining:.0f}mm"
            )
