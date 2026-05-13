"""
Streamlit Demo App for TSA-SAC Irrigation RL Project
Run: streamlit run demo_app.py
"""
import json
import os
import sys
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSA-SAC Irrigation RL",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size:3.2rem; font-weight:800; color:#2E7D32;
                  text-align:center; margin-bottom:0.5rem;}
    .sub-header  {font-size:1.1rem; color:#555; text-align:center;
                  margin-bottom:2rem;}
    .metric-card {background:linear-gradient(135deg,#e8f5e9,#c8e6c9);
                  padding:1.2rem; border-radius:12px; text-align:center;
                  box-shadow:0 2px 8px rgba(0,0,0,0.08);}
    .metric-value{font-size:2rem; font-weight:700; color:#1B5E20;}
    .metric-label{font-size:0.85rem; color:#555;}
    .stTabs [data-baseweb="tab-list"] {gap:8px;}
    .stTabs [data-baseweb="tab"] {padding:10px 20px; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ── Helper: load JSON safely ─────────────────────────────────────────
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        txt = f.read()
    txt = txt.replace("NaN", "null").replace("Infinity", "null")
    return json.loads(txt)

# ── Load data ────────────────────────────────────────────────────────
baseline = load_json("results/baseline_comparison.json")
ablation = load_json("results/ablation/ablation_results.json")
history  = load_json("results/training_history.json")
unseen   = load_json("results/unseen_environment_analysis.json")
hp_data  = load_json("results/hyperparameter_analysis.json")

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Training Results",
    "Policy Comparison",
    "Ablation Study",
    "Unseen Environments",
    "Hyperparameters",
    "Environment",
    "Live Simulation",
    "Architecture",
])

# ══════════════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown('<p class="main-header">🌾 Smart Irrigation with TSA-SAC</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Reinforcement Learning for Water-Efficient'
                ' Crop Irrigation Scheduling</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">$1,039</div>'
                    '<div class="metric-label">Season Profit</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-value">7,600</div>'
                    '<div class="metric-label">Yield (kg/ha)</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-value">1,152mm</div>'
                    '<div class="metric-label">Irrigation Used</div></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-value">6.60</div>'
                    '<div class="metric-label">IWUE (kg/ha/mm)</div></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Problem Statement")
    st.markdown("""
    Agriculture consumes **~70% of global freshwater**, yet traditional irrigation methods
    (fixed schedules, threshold-based) waste water and fail to adapt to dynamic weather.

    **Our solution:** A Temporal-State-Aware Soft Actor-Critic (TSA-SAC) agent that learns
    to schedule irrigation by observing weather, soil moisture, crop growth stage, and
    water budget — achieving **higher profit with less water** than expert heuristics.
    """)

    st.subheader("Key Contributions")
    st.markdown("""
    1. **Custom Physics-Based Environment** — FAO-56 soil water balance + RUE crop model
    2. **TSA Encoder** — BiLSTM + Temporal Attention over 7-day history
    3. **Stage-Aware Dynamic Reward** — Agronomic prior knowledge in reward shaping
    4. **Comprehensive Evaluation** — 4 baselines + 3 RL algorithms + ablation study
    """)

# ══════════════════════════════════════════════════════════════════════
#  TRAINING RESULTS
# ══════════════════════════════════════════════════════════════════════
elif page == "Training Results":
    st.header("📊 Training Results — 1000 Episodes")

    if history:
        tab1, tab2 = st.tabs(["Training Curves", "Agent Metrics"])

        with tab1:
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            fig.suptitle("TSA-SAC Training Curves", fontsize=16, fontweight="bold")

            def smooth(arr, w=20):
                if len(arr) < w: return arr
                return np.convolve(arr, np.ones(w)/w, mode="valid")

            keys = [("reward", "Episode Reward", "steelblue"),
                    ("profit", "Profit ($)", "seagreen"),
                    ("irrigation", "Irrigation (mm)", "mediumpurple"),
                    ("yield", "Yield (kg/ha)", "mediumseagreen"),
                    ("critic_loss", "Critic Loss", "indianred"),
                    ("alpha", "Entropy α", "deepskyblue")]

            for ax, (k, title, color) in zip(axes.flat, keys):
                if k in history and history[k]:
                    data = [x if x is not None else 0 for x in history[k]]
                    ax.plot(data, alpha=0.2, color=color, linewidth=0.5)
                    ax.plot(smooth(data), color=color, linewidth=2)
                    ax.set_title(title)
                    ax.set_xlabel("Episode")
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab2:
            fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
            fig2.suptitle("TSA-SAC Agent Metrics", fontsize=16, fontweight="bold")
            metrics = [("reward", "Reward", "steelblue"),
                       ("critic_loss", "Critic Loss", "indianred"),
                       ("actor_loss", "Actor Loss", "mediumpurple"),
                       ("alpha", "Entropy α", "deepskyblue")]
            for ax, (k, title, color) in zip(axes2.flat, metrics):
                if k in history and history[k]:
                    data = [x if x is not None else 0 for x in history[k]]
                    ax.plot(smooth(data), color=color, linewidth=2)
                    ax.set_title(title); ax.set_xlabel("Episode"); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
    else:
        st.warning("Training history not found. Run training first.")

    # Show saved plots if available
    plot_dir = "results/plots"
    if os.path.exists(plot_dir):
        st.subheader("📈 Saved Plots")
        cols = st.columns(2)
        for i, fname in enumerate(sorted(os.listdir(plot_dir))):
            if fname.endswith(".png"):
                with cols[i % 2]:
                    st.image(os.path.join(plot_dir, fname), caption=fname, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  POLICY COMPARISON
# ══════════════════════════════════════════════════════════════════════
elif page == "Policy Comparison":
    st.header("🏆 Policy Comparison — TSA-SAC vs Baselines")

    if baseline:
        policies = list(baseline.keys())
        profits = [baseline[p]["profit_mean"] for p in policies]
        irrig   = [baseline[p]["irrigation_mean"] for p in policies]
        yields  = [baseline[p]["yield_mean"] for p in policies]
        iwue    = [baseline[p]["iwue_mean"] for p in policies]
        stress  = [baseline[p]["stress_days_mean"] for p in policies]

        # Summary table
        st.subheader("Performance Summary")
        import pandas as pd
        df = pd.DataFrame({
            "Policy": policies,
            "Profit ($/ha)": [f"${p:.1f}" for p in profits],
            "Yield (kg/ha)": [f"{y:.0f}" for y in yields],
            "Irrigation (mm)": [f"{i:.0f}" for i in irrig],
            "IWUE": [f"{w:.2f}" for w in iwue],
            "Stress Days": [f"{s:.1f}" for s in stress],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Bar charts
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = ["#78909C", "#78909C", "#78909C", "#78909C", "#2E7D32"]

        for ax, data, title, ylabel in zip(axes.flat,
            [profits, irrig, iwue, stress],
            ["Profit ($/ha)", "Irrigation (mm)", "IWUE (kg/ha/mm)", "Stress Days"],
            ["USD", "mm", "kg/ha/mm", "days"]):
            bars = ax.bar(policies, data, color=colors)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_ylabel(ylabel)
            for bar, val in zip(bars, data):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{val:.1f}", ha="center", va="bottom", fontsize=9)
            ax.tick_params(axis="x", rotation=15)

        fig.suptitle("Policy Comparison", fontsize=16, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.success("**TSA-SAC achieves highest profit ($1,039) with lowest irrigation (1,152mm)** "
                   "— 7% less water than Farmer Expert while earning 5% more profit.")
    else:
        st.warning("Baseline comparison data not found.")

# ══════════════════════════════════════════════════════════════════════
#  ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════
elif page == "Ablation Study":
    st.header("🔬 Ablation Study — Component Contribution")

    st.markdown("""
    **Setup:** 500 episodes, 400mm water budget (moderate constraint), seed=42

    Each variant removes or changes one component to measure its contribution:
    """)

    if ablation:
        variants = list(ablation.keys())
        profits  = [ablation[v]["profit_mean"] for v in variants]
        yields_a = [ablation[v]["yield_mean"] for v in variants]
        iwue_a   = [ablation[v].get("iwue_mean", 0) for v in variants]
        stress_a = [ablation[v]["stress_days_mean"] for v in variants]

        import pandas as pd
        df = pd.DataFrame({
            "Variant": variants,
            "Profit ($)": [f"${p:.1f}" for p in profits],
            "Yield (kg/ha)": [f"{y:.0f}" for y in yields_a],
            "IWUE": [f"{w:.2f}" for w in iwue_a],
            "Stress Days": [f"{s:.1f}" for s in stress_a],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        palette = ["#78909C", "#42A5F5", "#FF9800", "#4CAF50", "#AB47BC", "#EF5350"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        bars1 = axes[0].bar(variants, profits, color=palette)
        axes[0].set_title("Mean Season Profit ($)", fontweight="bold")
        axes[0].tick_params(axis="x", rotation=20)
        for b, v in zip(bars1, profits):
            axes[0].text(b.get_x()+b.get_width()/2, b.get_height(), f"${v:.0f}",
                         ha="center", va="bottom", fontsize=9)

        bars2 = axes[1].bar(variants, yields_a, color=palette)
        axes[1].set_title("Mean Final Yield (kg/ha)", fontweight="bold")
        axes[1].tick_params(axis="x", rotation=20)
        for b, v in zip(bars2, yields_a):
            axes[1].text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.0f}",
                         ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Key Findings")
        st.markdown("""
        - **SAC >>> PPO** (+136%) — SAC is the right algorithm for continuous irrigation control
        - **SAC > DDPG** (+2.5%) — SAC's entropy regularization provides better exploration
        - **TSA encoder > MLP** (+1.7%) — Temporal attention helps capture weather patterns
        - **TSA-SAC (fixed) best under constraint** — with limited water, fixed reward avoids risky exploration
        """)
    else:
        st.warning("Ablation data not found.")

# ══════════════════════════════════════════════════════════════════════
#  LIVE SIMULATION
# ══════════════════════════════════════════════════════════════════════
elif page == "Live Simulation":
    st.header("🎮 Live Irrigation Simulation")
    st.markdown("Watch the trained TSA-SAC agent make irrigation decisions in real-time.")

    # Add environment path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    col1, col2 = st.columns([1, 1])
    with col1:
        climate = st.selectbox("Climate", ["arid", "semi_arid", "humid"], index=0)
        reservoir = st.slider("Reservoir (mm)", 500, 3000, 2000, 100)
    with col2:
        n_days = st.slider("Simulation Days", 30, 180, 180, 10)
        run_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        try:
            import torch
            from environment import CropIrrigationEnv
            from sac_agent import SACAgent

            # Load environment and agent
            env = CropIrrigationEnv(crop="cotton", climate=climate,
                                    water_budget_level="generous",
                                    reservoir_capacity=reservoir,
                                    weather_source="synthetic")

            ckpt_path = "checkpoints/tsa_sac_improved_best.pt"
            if not os.path.exists(ckpt_path):
                st.error("Checkpoint not found. Train the model first.")
                st.stop()

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            obs_dim = env.observation_space.shape[0]
            agent = SACAgent(obs_dim=obs_dim, act_dim=1, hidden=256,
                             seq_len=7, encoder_type="tsa", lr=3e-4)
            agent.actor.load_state_dict(ckpt["actor"])

            # Run episode
            obs, _ = env.reset(seed=42)
            seq_buf = np.zeros((7, obs_dim), dtype=np.float32)

            days, irrigations, soil_moistures, yields_t = [], [], [], []
            profits_t, stages, rains = [], [], []
            total_irrig = 0.0

            progress = st.progress(0, "Simulating...")
            for day in range(n_days):
                seq_buf = np.roll(seq_buf, -1, axis=0)
                seq_buf[-1] = obs
                seq_tensor = torch.FloatTensor(seq_buf).unsqueeze(0)

                with torch.no_grad():
                    action = agent.actor.sample(seq_tensor)[0].cpu().numpy().flatten()

                irrig_mm = float(np.clip(action[0], 0, 1)) * 60.0
                obs, reward, terminated, truncated, info = env.step(action)

                days.append(day + 1)
                irrigations.append(irrig_mm)
                soil_moistures.append(info.get("soil_water_content", obs[3]))
                yields_t.append(info.get("biomass", obs[1]) * 7600)
                rains.append(info.get("rain", obs[7] * 30))
                total_irrig += irrig_mm

                stage_idx = np.argmax(obs[10:15]) if max(obs[10:15]) > 0 else -1
                stage_names = ["Emergence", "Vegetative", "Flowering", "Boll", "Maturity"]
                stages.append(stage_names[stage_idx] if stage_idx >= 0 else "Pre-plant")

                progress.progress((day + 1) / n_days)
                if terminated or truncated:
                    break

            progress.empty()

            # Display results
            final_yield = info.get("yield", yields_t[-1] if yields_t else 0)
            final_profit = info.get("profit", 0)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Final Profit", f"${final_profit:.1f}")
            m2.metric("Final Yield", f"{final_yield:.0f} kg/ha")
            m3.metric("Total Irrigation", f"{total_irrig:.0f} mm")
            m4.metric("Days Simulated", f"{len(days)}")

            # Plot simulation
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            fig.suptitle("TSA-SAC Agent — Live Irrigation Decisions", fontsize=14, fontweight="bold")

            # Irrigation + Rainfall
            axes[0].bar(days, irrigations, color="#2196F3", alpha=0.7, label="Irrigation", width=1)
            axes[0].bar(days, rains, color="#90CAF9", alpha=0.5, label="Rainfall", width=1)
            axes[0].set_ylabel("Water (mm)")
            axes[0].legend()
            axes[0].set_title("Daily Water Application")

            # Soil moisture
            axes[1].plot(days, soil_moistures, color="#4CAF50", linewidth=2)
            axes[1].fill_between(days, 0, soil_moistures, alpha=0.2, color="#4CAF50")
            axes[1].set_ylabel("Soil Moisture")
            axes[1].set_title("Root Zone Soil Water Content")
            axes[1].axhline(y=0.45, color="red", linestyle="--", alpha=0.5, label="Stress threshold")
            axes[1].legend()

            # Crop growth
            axes[2].plot(days, yields_t, color="#FF9800", linewidth=2)
            axes[2].fill_between(days, 0, yields_t, alpha=0.2, color="#FF9800")
            axes[2].set_ylabel("Biomass")
            axes[2].set_xlabel("Day of Season")
            axes[2].set_title("Crop Biomass Accumulation")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.success(f"✅ Simulation complete! Agent earned **${final_profit:.1f}** "
                       f"using **{total_irrig:.0f}mm** irrigation.")

        except Exception as e:
            st.error(f"Error running simulation: {e}")
            import traceback
            st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════
#  ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════
elif page == "Architecture":
    st.header("🏗️ System Architecture")

    st.subheader("High-Level Architecture")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────┐
    │                  Weather Generator                       │
    │         (Markov Chain / NASA POWER API)                  │
    └──────────────────┬───────────────────────────────────────┘
                       │ temp, rain, ET₀
    ┌──────────────────▼───────────────────────────────────────┐
    │            CropIrrigationEnv (Custom Gym)                │
    │  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐  │
    │  │  FAO-56     │  │ RUE Crop     │  │ Water Budget    │  │
    │  │  Soil Model │  │ Growth Model │  │ Constraint      │  │
    │  └────────────┘  └──────────────┘  └─────────────────┘  │
    │       16-dim observation + stage-aware reward             │
    └──────────────────┬──────────────▲────────────────────────┘
                       │ obs          │ action [0-60mm]
    ┌──────────────────▼──────────────┴────────────────────────┐
    │                  TSA-SAC Agent                            │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │  BiLSTM Encoder (7-day history) → Attention         │ │
    │  │  → Feature Attention → Actor (Gaussian) + Critic    │ │
    │  │  Twin Q-Networks + Auto-tuned α                     │ │
    │  └─────────────────────────────────────────────────────┘ │
    └──────────────────────────────────────────────────────────┘
    ```
    """)

    st.subheader("RL Formulation")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**State Space (16-dim)**")
        st.markdown("""
        | Feature | Description |
        |---|---|
        | LAI, Biomass, Root depth | Crop status |
        | Soil water, Reservoir | Water status |
        | Water stress | Stress indicator |
        | ET₀, Rain, Forecasts | Weather |
        | Growth stage (one-hot) | Phenology |
        | Budget remaining | Constraint |
        """)
    with col2:
        st.markdown("**Action & Reward**")
        st.markdown("""
        - **Action**: Continuous irrigation ∈ [0, 60] mm/day
        - **Reward (daily)**: `w_y·yield_gain − w_w·water_cost·scarcity − w_s·stress`
        - **Reward (terminal)**: `(revenue − cost) / scale`
        - **Dynamic weights** adjust by growth stage (flowering gets higher yield weight)
        """)

    st.subheader("Experimental Setup")
    st.markdown("""
    | Parameter | Value |
    |---|---|
    | Crop | Cotton (180-day season) |
    | Climate | Arid (synthetic weather) |
    | Training Episodes | 1000 |
    | Reservoir | 2000mm (generous) |
    | Hidden Size | 256 |
    | Learning Rate | 3×10⁻⁴ |
    | Sequence Length | 7 days |
    | Replay Buffer | 1,000,000 |
    | Batch Size | 64 |
    | Discount (γ) | 0.99 |
    | Soft Update (τ) | 0.005 |
    """)

# ══════════════════════════════════════════════════════════════════════
#  UNSEEN ENVIRONMENTS
# ══════════════════════════════════════════════════════════════════════
elif page == "Unseen Environments":
    st.header("🌍 Generalization — Unseen Environments")
    st.markdown("Testing the trained agent (arid climate) on environments it was **never trained on**.")

    if unseen:
        import pandas as pd
        labels = list(unseen.keys())
        df = pd.DataFrame({
            "Scenario": labels,
            "Profit ($)": [f"${unseen[l]['profit_mean']:.1f} ± {unseen[l]['profit_std']:.1f}" for l in labels],
            "Yield (kg/ha)": [f"{unseen[l]['yield_mean']:.0f}" for l in labels],
            "Irrigation (mm)": [f"{unseen[l]['irrigation_mean']:.0f}" for l in labels],
            "IWUE": [f"{unseen[l]['iwue_mean']:.2f}" for l in labels],
            "Stress Days": [f"{unseen[l]['stress_days_mean']:.1f}" for l in labels],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Show plot if available
        plot_path = "results/plots/unseen_environment.png"
        if os.path.exists(plot_path):
            st.image(plot_path, caption="Unseen Environment Generalization", use_container_width=True)

        st.subheader("Interpretation")
        st.markdown("""
        - **Best case (humid)**: More natural rainfall → agent needs less irrigation → higher profit expected
        - **Average case (semi-arid, moderate budget)**: Moderate challenge → tests adaptability
        - **Worst case (arid, scarce budget)**: Extreme constraint → tests robustness under pressure
        """)
    else:
        st.warning("Run `python evaluate_extra.py` first to generate unseen environment results.")

# ══════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════
elif page == "Hyperparameters":
    st.header("⚙️ Hyperparameter Sensitivity Analysis")
    st.markdown("Testing how key hyperparameters affect agent performance (100-episode short runs).")

    if hp_data:
        for exp_name, data in hp_data.items():
            st.subheader(exp_name)
            import pandas as pd
            df = pd.DataFrame({
                "Value": data["labels"],
                "Profit ($)": [f"${p:.1f}" for p in data["profits"]],
                "Yield (kg/ha)": [f"{y:.0f}" for y in data["yields"]],
                "IWUE": [f"{w:.2f}" for w in data["iwues"]],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

        plot_path = "results/plots/hyperparameter_sensitivity.png"
        if os.path.exists(plot_path):
            st.image(plot_path, caption="Hyperparameter Sensitivity", use_container_width=True)

        st.subheader("Key Observations")
        st.markdown("""
        - **Learning Rate**: 3e-4 (default) provides the best balance of speed and stability
        - **Hidden Size**: 256 is optimal; 128 underfits, 512 overfits on limited data
        - **Sequence Length**: 7 days captures weekly weather patterns, improving scheduling
        """)
    else:
        st.warning("Run `python evaluate_extra.py` first to generate hyperparameter results.")
# ══════════════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════
elif page == "Environment":
    st.header("Environment — CropIrrigationEnv")
    st.markdown("A custom Gymnasium environment implementing physics-based crop irrigation simulation.")

    st.subheader("Overview")
    st.markdown("""
    The environment simulates a **170-day cotton growing season** with:
    - **FAO-56 Soil Water Balance** — tracks root-zone moisture, drainage, and stress
    - **RUE Crop Growth Model** — temperature & radiation-driven biomass accumulation
    - **Stochastic Weather** — Markov chain rainfall + sinusoidal temperature
    - **Seasonal Water Budget** — non-linear scarcity pricing as budget depletes
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("State Space (16-dim)")
        import pandas as pd
        state_df = pd.DataFrame({
            "Index": list(range(16)),
            "Variable": [
                "LAI (normalized)", "Biomass / max yield", "Root depth / max",
                "Available soil water", "Reservoir level", "Water stress (1-Ks)",
                "ET₀ / 12mm", "Today's rain / 30mm", "3-day rain forecast",
                "3-day ET₀ forecast", "Stage: Emergence", "Stage: Vegetative",
                "Stage: Flowering", "Stage: Boll Fill", "Stage: Maturity",
                "Budget remaining"
            ],
            "Range": ["[0,1]"]*16
        })
        st.dataframe(state_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Action & Reward")
        st.markdown("""
        **Action:** Continuous irrigation ∈ [0, 60] mm/day

        **Daily Reward:**
        ```
        r = w_y·yield_gain − w_w·water_cost·scarcity − w_s·(1−Ks)²
        ```

        **Terminal Reward (harvest):**
        ```
        r_T = (yield×price − irrigation×cost + IWUE_bonus) / scale
        ```

        **Dynamic Weights by Stage:**

        | Stage | Yield (w_y) | Water (w_w) | Stress (w_s) |
        |---|---|---|---|
        | Emergence | 0.8 | 0.4 | 0.8 |
        | Vegetative | 1.0 | 0.3 | 1.0 |
        | Flowering | 1.2 | 0.2 | 1.8 |
        | Boll Fill | 0.6 | 0.5 | 0.6 |
        | Maturity | 0.2 | 0.8 | 0.3 |
        """)

    st.subheader("Crop Profiles")
    crop_df = pd.DataFrame({
        "Parameter": ["Season (days)", "Max Yield (kg/ha)", "Kc_ini", "Kc_mid", "Kc_end",
                       "Field Capacity", "Wilting Point", "Root Depth (mm)", "Price ($/kg)", "Water Cost ($/mm)"],
        "Cotton": [170, 7600, 0.35, 1.15, 0.70, 0.33, 0.15, 1200, 0.22, 0.55],
        "Wheat": [120, 6000, 0.40, 1.15, 0.40, 0.35, 0.12, 600, 0.25, 0.45],
        "Maize": [100, 9000, 0.30, 1.20, 0.60, 0.32, 0.11, 700, 0.18, 0.45],
    })
    st.dataframe(crop_df, use_container_width=True, hide_index=True)

    st.subheader("Water Budget Presets (mm)")
    budget_df = pd.DataFrame({
        "Crop": ["Cotton", "Wheat", "Maize"],
        "Generous": [1500, 1000, 1000],
        "Moderate": [400, 280, 300],
        "Scarce": [250, 180, 200],
    })
    st.dataframe(budget_df, use_container_width=True, hide_index=True)

    st.subheader("Climate Presets")
    climate_df = pd.DataFrame({
        "Parameter": ["Rain Probability", "Rain Mean (mm)", "Temp Mean (°C)", "Solar Mean", "Wind Mean"],
        "Arid": [0.05, 3, 34, 25, 3.4],
        "Semi-Arid": [0.15, 6, 28, 22, 2.8],
        "Humid": [0.40, 10, 24, 18, 2.2],
    })
    st.dataframe(climate_df, use_container_width=True, hide_index=True)

# ── Footer ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**TSA-SAC Irrigation RL**")
st.sidebar.markdown("RL Course Project • 2026")
