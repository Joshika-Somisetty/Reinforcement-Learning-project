# Irrigation RL with TSA-SAC Style Attention
## Paper-aligned irrigation scheduling with BiLSTM, temporal attention, and stage-aware rewards

---

## 📄 Anchor Paper

**"Smart Irrigation Scheduling for Crop Production Using a Crop Model and Improved Deep Reinforcement Learning"**  
*MDPI Agriculture, December 2025*  
Key ideas adopted:
- DRL over Q-learning for high-dimensional, continuous control
- Crop growth model as RL training environment
- Multi-objective reward (yield profit − water cost)

**Current project status**
| Feature | Paper | This Project |
|---|---|---|
| Algorithm | TSA-SAC | TSA-SAC-style SAC + BiLSTM + temporal attention + feature attention |
| Additional algorithms | — | PPO, DDPG (for comparison) |
| Environment | DSSAT (external simulator) | Physics-based surrogate environment |
| Weather | Historical seasons | Stochastic Markov generator + NASA POWER real data |
| Action space | Continuous [0–60 mm] | Continuous [0–60 mm] |
| Baselines | FE / SAC / DDPG / PPO / LSTM-SAC | Random / Farmer / Threshold / FarmerExpert / PPO / DDPG / TSA-SAC |
| Crop models | Cotton | Cotton default, plus wheat and maize presets |
| Water constraint | — | Seasonal water budget (generous / moderate / scarce) |
| Multi-zone | — | Shared-reservoir multi-zone farming |
| Transfer learning | — | Cross-crop encoder transfer with freeze/unfreeze |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                CropIrrigationEnv                    │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────┐ │
│  │ Stochastic   │  │  Soil-Water   │  │  Crop    │ │
│  │  Weather     │  │  Balance      │  │  Growth  │ │
│  │  (Markov)    │  │  (FAO-56)     │  │  (RUE)   │ │
│  └──────────────┘  └───────────────┘  └──────────┘ │
│         ↓                  ↓                ↓       │
│              16-dim observation vector              │
└─────────────────────────────────────────────────────┘
                         ↕
┌─────────────────────────────────────────────────────┐
│                    SAC Agent                        │
│  ┌──────────────────┐   ┌─────────────────────────┐ │
│  │  Gaussian Actor  │   │  Twin Q-Networks        │ │
│  │  (256-256 MLP)   │   │  (256-256 MLP × 2)      │ │
│  │  → continuous    │   │  + target networks      │ │
│  │    action [0,60] │   │                         │ │
│  └──────────────────┘   └─────────────────────────┘ │
│         Auto-tuned entropy temperature (α)          │
└─────────────────────────────────────────────────────┘
```

---

## RL Formulation

### State Space (16-dimensional)
| Index | Variable | Description |
|---|---|---|
| 0 | `lai_norm` | Leaf area index proxy [0,1] |
| 1 | `biomass_norm` | Accumulated biomass / max [0,1] |
| 2 | `root_depth_norm` | Effective root depth / crop maximum [0,1] |
| 3 | `soil_water_avail_norm` | Available root-zone water [0,1] |
| 4 | `reservoir_norm` | Remaining seasonal irrigation supply [0,1] |
| 5 | `water_stress_norm` | Stress factor proxy [0,1] |
| 6 | `et0_norm` | Reference ET / 12 mm |
| 7 | `rain_norm` | Rainfall today / 30 mm |
| 8 | `rain_forecast_3d_norm` | Noisy 3-day rainfall forecast [0,1] |
| 9 | `et0_forecast_3d_norm` | Noisy 3-day ET0 forecast [0,1] |
| 10-14 | `stage_*` | One-hot critical growth stage indicators |
| 15 | `budget_remaining_norm` | Seasonal water budget fraction [0,1] |

### Action Space
- **Continuous**: irrigation depth in mm ∈ [0, 60] per day

### Reward (Stage-Aware Multi-Objective)
``` 
Daily:    r_t = w_y * yield_gain − w_w * irrigation_cost × scarcity − w_s * stress_penalty
Terminal: r_T = (yield_revenue - total_water_cost) / terminal_reward_scale
```

### Algorithm: TSA-SAC Style Soft Actor-Critic
- 2-layer BiLSTM sequence encoder with 7-day history
- Temporal attention over the sequence hidden states
- Feature attention before actor and critic heads
- Twin Q-networks with target critics
- Auto-tuned entropy temperature `alpha`

---

## 🌦️ Realistic Environment

### Weather Model
- **Synthetic** (default): Two-state Markov chain for rainfall, sinusoidal temperature, Gaussian noise
- **Real**: NASA POWER API historical weather (Lubbock TX, Hyderabad, Gainesville FL)
- Three climate presets: `semi_arid`, `humid`, `arid`

### Soil Water Balance (FAO-56)
```
θ_{t+1} = θ_t + (rain + irrigation) / root_depth − ET_c / root_depth
ET_c     = ET_0 × Kc × Ks
Ks       = water stress coefficient (1 = no stress)
```

### Water Budget Constraint
```
Seasonal budget presets (cotton):
  generous: 600mm  → easy
  moderate: 400mm  → challenging, forces smart timing
  scarce:   250mm  → very hard, forces real trade-offs
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train TSA-SAC with moderate water budget (cotton, arid)
python train.py --crop cotton --climate arid --episodes 1000

# Train with specific water budget
python train.py --water-budget 400 --episodes 500

# Train PPO or DDPG baseline
python train.py --algorithm ppo --episodes 500
python train.py --algorithm ddpg --episodes 500

# Train with real weather data
python weather_data.py --generate-offline   # first time: generate CSV
python train.py --weather-source real --episodes 500

# Transfer learning: pre-train on cotton, fine-tune on wheat
python train.py --crop cotton --episodes 500 --checkpoint-path checkpoints/cotton_pretrained.pt
python train.py --crop wheat --transfer-from checkpoints/cotton_pretrained.pt --freeze-encoder-epochs 50 --episodes 300

# Run ablation study (3 seeds, 6 variants)
python train.py --run-ablation --ablation-episodes 500 --water-budget 400

# Multi-zone farming
python multi_zone_train.py --zones cotton,maize --shared-budget 500 --episodes 200

# Evaluate saved model and compare all baselines
python train.py --eval-only --model checkpoints/tsa_sac_improved_best.pt

# Plot training curves, comparisons, and agent metrics
python visualize.py --no-show
```

---

## 📊 Expected Results

| Policy | Profit ($/ha) | Irrigation (mm) |
|---|---|---|
| Random | ~$300 | ~1200 |
| Fixed 7-day | ~$700 | ~700 |
| Threshold | ~$900 | ~550 |
| Farmer Expert | ~$1000 | ~450 |
| **SAC (ours)** | **~$1100** | **~420** |

SAC learns to:
1. Skip irrigation when rainfall is forecast
2. Apply heavy irrigation at critical flowering stage
3. Conserve water budget for dry spells
4. Trade off stress risk vs water cost dynamically

---

## 📁 File Structure

```
irrigation_rl/
├── environment.py      # Gymnasium env: weather, soil, crop, reward, water budget
├── sac_agent.py        # TSA-SAC: BiLSTM + attention + SAC + transfer learning
├── ppo_agent.py        # PPO baseline agent
├── ddpg_agent.py       # DDPG baseline agent
├── baselines.py        # Random / Fixed / Threshold / FarmerExpert policies
├── train.py            # Training loop, eval, ablation, baseline comparison
├── visualize.py        # Plots: training, comparison, ablation, robustness
├── weather_data.py     # NASA POWER real weather + offline CSV
├── multi_zone_env.py   # Multi-zone shared-reservoir environment
├── multi_zone_train.py # Multi-zone SAC training script
├── data/weather/       # Cached weather CSVs
├── requirements.txt
└── README.md
```

---

## 🔬 Enhancement Ideas (for report/presentation)

1. ~~BiLSTM temporal state~~ ✅ Implemented
2. ~~Multi-crop transfer learning~~ ✅ Implemented
3. ~~PPO comparison~~ ✅ Implemented
4. ~~Water budget constraint~~ ✅ Implemented
5. ~~Multi-zone farming~~ ✅ Implemented
6. ~~Real weather data~~ ✅ Implemented
7. **Nitrogen management** — add soil nitrogen as state variable
8. **Market price uncertainty** — stochastic crop prices
9. **Climate change scenarios** — train under projected future weather
