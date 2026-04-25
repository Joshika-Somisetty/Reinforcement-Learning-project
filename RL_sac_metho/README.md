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
| Environment | DSSAT (external simulator) | Physics-based surrogate environment |
| Weather | Historical seasons | Stochastic seasonal weather generator |
| Action space | Continuous [0–60 mm] | Continuous [0–60 mm] |
| Baselines | FE / SAC / DDPG / PPO / LSTM-SAC | Random / Farmer / Threshold / TSA-SAC |
| Crop models | Cotton | Cotton default, plus wheat and maize presets |

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
│              10-dim observation vector              │
└─────────────────────────────────────────────────────┘
                         ↕
┌─────────────────────────────────────────────────────┐
│                    SAC Agent                        │
│  ┌──────────────────┐   ┌─────────────────────────┐ │
│  │  Gaussian Actor  │   │  Twin Q-Networks        │ │
│  │  (256-256 MLP)   │   │  (256-256 MLP × 2)      │ │
│  │  → continuous    │   │  + target networks      │ │
│  │    action [0,50] │   │                         │ │
│  └──────────────────┘   └─────────────────────────┘ │
│         Auto-tuned entropy temperature (α)          │
└─────────────────────────────────────────────────────┘
```

---

## RL Formulation

### State Space (12-dimensional)
| Index | Variable | Description |
|---|---|---|
| 0 | `lai_norm` | Leaf area index proxy [0,1] |
| 1 | `biomass_norm` | Accumulated biomass / max [0,1] |
| 2 | `root_depth_norm` | Effective root depth / 1500 mm |
| 3 | `soil_water_avail_norm` | Available root-zone water [0,1] |
| 4 | `water_stress_norm` | Stress factor proxy [0,1] |
| 5 | `et0_norm` | Reference ET / 12 mm |
| 6 | `rain_norm` | Rainfall today / 30 mm |
| 7-11 | `stage_*` | One-hot critical growth stage indicators |

### Action Space
- **Continuous**: irrigation depth in mm ∈ [0, 60] per day

### Reward (Stage-Aware Multi-Objective)
``` 
Daily:    r_t = w_y * yield_gain − w_w * irrigation_cost − w_s * stress_penalty
Terminal: r_T = yield_revenue − total_water_cost
```

### Algorithm: TSA-SAC Style Soft Actor-Critic
- 2-layer BiLSTM sequence encoder with 7-day history
- Temporal attention over the sequence hidden states
- Feature attention before actor and critic heads
- Twin Q-networks with target critics
- Auto-tuned entropy temperature `alpha`

---

## 🌦️ Realistic Stochastic Environment

### Weather Model
- Two-state Markov chain for rainfall occurrence
- Gamma-distributed rainfall amounts on wet days
- Sinusoidal seasonal temperature + Gaussian noise
- Three climate presets: `semi_arid`, `humid`, `arid`

### Soil Water Balance (FAO-56)
```
θ_{t+1} = θ_t + (rain + irrigation) / root_depth − ET_c / root_depth
ET_c     = ET_0 × Kc × Ks
Ks       = water stress coefficient (1 = no stress)
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train TSA-SAC style agent (cotton, arid, 1000 episodes)
python train.py --crop cotton --climate arid --episodes 1000

# Train on maize in arid conditions
python train.py --crop maize --climate arid --episodes 500

# Evaluate saved model and compare baselines
python train.py --eval-only --model checkpoints/tsa_sac_best.pt

# Plot training curves and comparisons
python visualize.py
```

---

## 📊 Expected Results

| Policy | Profit ($/ha) | Irrigation (mm) |
|---|---|---|
| Random | ~$300 | ~1200 |
| Fixed 7-day | ~$700 | ~700 |
| Threshold | ~$900 | ~550 |
| **SAC (ours)** | **~$1100** | **~420** |

SAC learns to:
1. Skip irrigation when rainfall is forecast
2. Apply heavy irrigation at critical flowering stage
3. Conserve reservoir for dry spells
4. Trade off stress risk vs water cost dynamically

---

## 📁 File Structure

```
irrigation_rl/
├── environment.py    # Stochastic crop-irrigation Gymnasium env
├── sac_agent.py      # SAC with auto-α, twin-Q, replay buffer
├── baselines.py      # Random / Fixed / Threshold policies
├── train.py          # Training loop + evaluation
├── visualize.py      # Plots: training curves, policy comparison
├── requirements.txt
└── README.md
```

---

## 🔬 Enhancement Ideas (for report/presentation)

1. **BiLSTM temporal state** — encode last 7 days of obs (paper's key trick)
2. **Multi-crop transfer learning** — pre-train on maize, fine-tune on wheat
3. **PPO comparison** — on-policy vs off-policy analysis
4. **Water budget constraint** — add seasonal water limit as hard constraint
5. **Multi-zone farming** — extend to multi-agent with shared reservoir
