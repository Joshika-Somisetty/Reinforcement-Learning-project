# Smart Irrigation Scheduling with TSA-SAC

Temporal-State-Aware Soft Actor-Critic (TSA-SAC) for water-efficient irrigation scheduling in crop production. The project trains a deep reinforcement learning agent to decide daily irrigation depth using weather, crop state, soil-water status, growth stage, and seasonal water-budget constraints.

The implementation is inspired by the paper **"Smart Irrigation Scheduling for Crop Production Using a Crop Model and Improved Deep Reinforcement Learning"** (MDPI Agriculture, 2025), but uses a lightweight custom simulation environment instead of DSSAT so experiments can be trained and ablated quickly.

## Highlights

- TSA-SAC-style agent with BiLSTM temporal encoding, temporal attention, feature attention, twin critics, and automatic entropy tuning.
- Custom Gymnasium-compatible crop irrigation environment.
- FAO-56-inspired soil-water balance and RUE crop growth surrogate.
- Continuous irrigation action space: `0-60 mm/day`.
- Stage-aware multi-objective reward balancing yield, water cost, and stress.
- Baselines: Random, fixed farmer schedule, soil-moisture threshold, farmer expert, PPO, and DDPG.
- Experiments include policy comparison, ablation study, hyperparameter sensitivity, and unseen-environment generalization.
- Streamlit demo app for visualizing results and running simulations.

## Project Status Compared with Anchor Paper

| Feature | Anchor paper | This project |
|---|---|---|
| Main algorithm | TSA-SAC | TSA-SAC-style SAC with BiLSTM + temporal/feature attention |
| Additional algorithms | SAC, PPO, DDPG, LSTM-SAC | PPO, DDPG, SAC ablations |
| Crop simulator | DSSAT | Custom physics-based surrogate environment |
| Weather | Historical seasons | Synthetic Markov weather + NASA POWER real-data support |
| Action space | Continuous irrigation | Continuous irrigation `[0, 60] mm/day` |
| Reward | Yield profit - water cost | Yield gain - water cost - stress penalty |
| Crop | Cotton | Cotton, wheat, maize presets |
| Constraints | Irrigation scheduling | Seasonal water budgets: generous, moderate, scarce |
| Extra experiments | Baselines | Ablation, hyperparameter sensitivity, unseen environments |

## System Architecture

![TSA-SAC smart irrigation system architecture](system_architecture.png)

The system is a closed loop. Weather, crop state, and water constraints feed a crop-water simulation environment. The environment returns a 16-dimensional state, reward, and next state. The TSA-SAC agent processes a 7-day state window and outputs a continuous irrigation action. Transitions are stored in replay memory for off-policy actor, critic, and entropy-temperature updates.

## Reinforcement Learning Formulation

### State Space

The observation vector has 16 normalized features.

| Index | Variable | Meaning |
|---|---|---|
| 0 | `lai_norm` | Leaf area index proxy |
| 1 | `biomass_norm` | Accumulated biomass / max yield |
| 2 | `root_depth_norm` | Effective root depth / crop maximum |
| 3 | `soil_water_avail_norm` | Available root-zone water |
| 4 | `reservoir_norm` | Remaining seasonal irrigation supply |
| 5 | `water_stress_norm` | Water stress proxy |
| 6 | `et0_norm` | Reference evapotranspiration |
| 7 | `rain_norm` | Current-day rainfall |
| 8 | `rain_forecast_3d_norm` | Noisy 3-day rainfall forecast |
| 9 | `et0_forecast_3d_norm` | Noisy 3-day ET0 forecast |
| 10-14 | `stage_*` | One-hot crop growth stage |
| 15 | `budget_remaining_norm` | Seasonal water-budget fraction |

### Action Space

The agent chooses a continuous irrigation amount:

```text
a_t in [0, 60] mm/day
```

### Reward

The daily reward combines crop growth, water use, and stress:

```text
r_t = w_y * yield_gain - w_w * water_cost * scarcity - w_s * stress_penalty
```

At harvest, a terminal reward is added from final yield revenue, irrigation cost, and water-use-efficiency bonus.

## Environment

`environment.py` implements `CropIrrigationEnv`, a Gymnasium-compatible simulator with:

- Synthetic weather from a two-state rainfall Markov chain.
- Optional real weather slices from NASA POWER CSV data.
- FAO-56-style soil-water balance.
- Dynamic root growth and field-capacity drainage.
- RUE crop growth and stage-sensitive LAI dynamics.
- Water budget constraints and nonlinear scarcity pricing.

Supported presets:

| Category | Options |
|---|---|
| Crop | `cotton`, `wheat`, `maize` |
| Climate | `arid`, `semi_arid`, `humid` |
| Water budget level | `generous`, `moderate`, `scarce` |
| Weather source | `synthetic`, `real` |

## Installation

From the project folder:

```bash
cd RL_sac_metho
pip install -r requirements.txt
```

Dependencies:

- `torch`
- `gymnasium`
- `numpy`
- `matplotlib`
- `scipy`
- `streamlit`
- `pandas`

## Quick Start

Run commands from `RL_sac_metho`.

### Train TSA-SAC

```bash
python train.py --crop cotton --climate arid --episodes 1000
```

Useful options:

```bash
python train.py --water-budget-level moderate --episodes 500
python train.py --water-budget 400 --episodes 500
python train.py --encoder-type tsa --seq-len 7 --lstm-hidden 128
python train.py --cuda
```

### Evaluate a Saved Model

```bash
python train.py --eval-only --model checkpoints/tsa_sac_improved_best.pt
```

### Train PPO or DDPG

```bash
python train.py --algorithm ppo --episodes 500
python train.py --algorithm ddpg --episodes 500
```

### Run Ablation Study

```bash
python train.py --run-ablation --ablation-episodes 500 --water-budget 400
```

### Generate Extra Evaluation Results

This creates unseen-environment and hyperparameter-analysis outputs.

```bash
python evaluate_extra.py
```

### Generate Plots

```bash
python visualize.py --plot all --no-show
```

Available plot modes:

```text
all, training, agent, comparison, robustness, tradeoff, ablation
```

### Run Demo App

```bash
streamlit run demo_app.py
```

The app includes result dashboards, policy comparison, ablation plots, unseen-environment views, live simulation, environment explanation, and architecture view.

### Weather Data

Generate offline fallback weather:

```bash
python weather_data.py --generate-offline
```

Download NASA POWER weather for a climate preset:

```bash
python weather_data.py --download --climate arid
python weather_data.py --download --climate semi_arid
python weather_data.py --download --climate humid
```

Train with real weather:

```bash
python train.py --weather-source real --climate arid --episodes 500
```

### Multi-Zone Training

```bash
python multi_zone_train.py --zones cotton,maize --shared-budget 500 --episodes 200
```

## Main Results

### Policy Comparison

Generous budget, 30 evaluation episodes.

| Policy | Profit ($/ha) | Yield (kg/ha) | Irrigation (mm) | IWUE | Stress days |
|---|---:|---:|---:|---:|---:|
| Random | 237 | 4,825 | 1,500 | 3.22 | 93.2 |
| Farmer (10-day) | 836 | 5,713 | 765 | 7.47 | 81.0 |
| Threshold (0.45) | 946 | 7,600 | 1,320 | 5.76 | 0.0 |
| Farmer Expert | 990 | 7,600 | 1,240 | 6.13 | 0.0 |
| **TSA-SAC** | **1,039** | **7,600** | **1,152** | **6.60** | **16.0** |

TSA-SAC achieves the highest profit while using the least irrigation among the high-yield policies. It accepts minor non-critical stress to conserve water for more important crop stages.

### Ablation Study

Moderate budget, 400 mm, 500 training episodes.

| Variant | Profit ($) | Yield (kg/ha) | IWUE | Stress days |
|---|---:|---:|---:|---:|
| **TSA-SAC fixed reward** | **817** | **4,714** | 11.79 | 94.8 |
| SAC-MLP | 804 | 4,652 | 11.63 | 95.8 |
| SAC-BiLSTM | 803 | 4,651 | 11.63 | 95.8 |
| DDPG-MLP | 784 | 4,563 | 11.41 | 97.2 |
| TSA-SAC dynamic reward | 783 | 4,557 | 11.39 | 97.2 |
| PPO-MLP | 340 | 1,880 | 14.05 | 153.2 |

Key findings:

- SAC strongly outperforms PPO for this long-horizon continuous-control problem.
- SAC performs slightly better than DDPG due to entropy-regularized exploration and twin critics.
- Temporal context improves scheduling by capturing rainfall and ET trends.
- Fixed reward is more stable than dynamic reward under tight water constraints.

### Unseen Environment Generalization

| Scenario | Profit ($) | Yield (kg/ha) | IWUE | Stress days |
|---|---:|---:|---:|---:|
| Trained: arid, generous | 1,201 | 7,600 | 8.88 | 46.0 |
| Humid, generous | 1,000 | 5,875 | 11.05 | 0.0 |
| Semi-arid, moderate | 860 | 4,908 | 12.27 | 77.5 |
| Arid, scarce | 588 | 3,296 | 13.18 | 117.3 |

The trained policy degrades gracefully on unseen climates and water budgets, suggesting it learns useful irrigation behavior rather than memorizing a single training condition.

### Hyperparameter Sensitivity

Most important observed factor: sequence length. A 7-day observation window outperforms a 1-day memoryless setting by roughly 15% profit in the moderate-budget sensitivity test.

## Outputs

Generated artifacts are stored under:

```text
checkpoints/                         trained models
results/training_history.json         main training history
results/baseline_comparison.json      baseline comparison metrics
results/ablation/                     ablation histories and summaries
results/plots/                        generated visualizations
data/weather/                         cached weather CSVs
```

Important plots:

| Plot | Path |
|---|---|
| Training curves | `results/plots/training_curves.png` |
| Agent metrics | `results/plots/agent_metrics.png` |
| Policy comparison | `results/plots/policy_comparison.png` |
| Irrigation-yield tradeoff | `results/plots/irrigation_yield_tradeoff.png` |
| Ablation study | `results/plots/ablation_study.png` |
| Ablation learning curves | `results/plots/ablation_learning_curves.png` |
| Hyperparameter sensitivity | `results/plots/hyperparameter_sensitivity.png` |
| Unseen environments | `results/plots/unseen_environment.png` |
| Robustness analysis | `results/plots/robustness_analysis.png` |

## Repository Structure

```text
RL_sac_metho/
|-- environment.py             # CropIrrigationEnv simulator
|-- sac_agent.py               # TSA-SAC, BiLSTM, attention, replay buffer
|-- ppo_agent.py               # PPO baseline
|-- ddpg_agent.py              # DDPG baseline
|-- baselines.py               # Rule-based policies
|-- train.py                   # Training, evaluation, baselines, ablations
|-- evaluate_extra.py          # Unseen environments and hyperparameter analysis
|-- visualize.py               # Plot generation
|-- weather_data.py            # NASA POWER and offline weather CSVs
|-- multi_zone_env.py          # Shared-reservoir multi-zone environment
|-- multi_zone_train.py        # Multi-zone training
|-- demo_app.py                # Streamlit dashboard
|-- system_architecture.png    # Paper-style architecture diagram
|-- requirements.txt
|-- checkpoints/
|-- data/weather/
|-- results/
|   |-- plots/
|   `-- ablation/
`-- README.md
```

## Notes

- Run scripts from inside `RL_sac_metho` so relative paths resolve correctly.
- The bundled checkpoint is `checkpoints/tsa_sac_improved_best.pt`.
- If using real weather, generate or download CSVs before training with `--weather-source real`.
- Long training runs can be memory intensive; reduce `--batch-size`, `--buffer-size`, or episode count for quick tests.

## Future Work

- Joint irrigation and nitrogen management.
- Higher-fidelity DSSAT/AquaCrop integration.
- Climate-change scenario training.
- Multi-zone canal scheduling and shared-water allocation.
- Stochastic market prices for risk-aware policies.
- Federated learning across farms while preserving local data privacy.
