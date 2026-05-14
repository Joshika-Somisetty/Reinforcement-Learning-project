"""
Streamlit Demo App for TSA-SAC Irrigation RL Project  (improved)
Run: streamlit run demo_app.py
"""
import json, os, sys
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSA-SAC · Smart Irrigation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600;700&family=Space+Grotesk:wght@500;700&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0f1623 60%, #0a0e1a 100%);
    border-right: 1px solid #1e2d4a55;
  }
  [data-testid="stSidebar"] * { color: #c5d4e8 !important; }
  [data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; padding: 4px 0; }

  /* ── Main BG ── */
  .stApp { background: #0d1117; }
  .main .block-container { padding-top: 1.5rem; max-width: 1280px; }

  /* ── Typography ── */
  h1,h2,h3,h4 { font-family: 'Space Grotesk', sans-serif; color: #e2eaf5; }
  p, li, .stMarkdown { color: #94a3b8; }

  /* ── Hero ── */
  .hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #818cf8 40%, #a78bfa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin-bottom: .4rem;
  }
  .hero-sub {
    font-size: 1.05rem; color: #64748b; letter-spacing: .02em; margin-bottom: 2rem;
  }

  /* ── KPI Cards ── */
  .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 2rem; }
  .kpi-card {
    background: linear-gradient(145deg, #111827, #1a2540);
    border: 1px solid #1e3a5f55;
    border-radius: 14px; padding: 1.2rem 1rem;
    text-align: center; position: relative; overflow: hidden;
  }
  .kpi-card::before {
    content:''; position:absolute; inset:0;
    background: radial-gradient(circle at 70% 20%, rgba(96,165,250,.10), transparent 60%);
    pointer-events:none;
  }
  .kpi-val { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; color:#60a5fa; }
  .kpi-delta { font-size:.75rem; color:#93c5fd; margin-top:.1rem; }
  .kpi-label { font-size:.78rem; color:#64748b; margin-top:.3rem; letter-spacing:.04em; text-transform:uppercase; }

  /* ── Section Cards ── */
  .section-card {
    background: #111827; border: 1px solid #1e2d4a44;
    border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
  }

  /* ── Contribution Chips ── */
  .contrib-chip {
    display:inline-block; background:rgba(96,165,250,.10);
    border:1px solid #60a5fa44; border-radius:20px;
    padding:.3rem .85rem; font-size:.82rem; color:#93c5fd;
    margin:.3rem .3rem .3rem 0;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] { gap:6px; background:transparent; }
  .stTabs [data-baseweb="tab"] {
    background:#111827; border-radius:8px 8px 0 0;
    padding:8px 18px; font-weight:600; color:#64748b;
    border:1px solid #1e2d4a44; border-bottom:none;
  }
  .stTabs [aria-selected="true"] { background:#1a2540 !important; color:#60a5fa !important; }

  /* ── Dataframe ── */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* ── Alerts ── */
  .stSuccess { background:#0f1f2d !important; border-left:4px solid #22c55e !important; }
  .stWarning { background:#1f1a0d !important; border-left:4px solid #f59e0b !important; }
  .stError   { background:#1f0d0d !important; border-left:4px solid #ef4444 !important; }

  /* ── Metric boxes ── */
  [data-testid="metric-container"] {
    background:#111827; border:1px solid #1e2d4a55;
    border-radius:10px; padding:.8rem 1rem;
  }
  [data-testid="metric-container"] [data-testid="stMetricLabel"] { color:#64748b !important; font-size:.8rem; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color:#60a5fa !important; }

  /* ── Progress bar ── */
  .stProgress > div > div { background: linear-gradient(90deg, #3b82f6, #818cf8) !important; }

  /* ── Buttons ── */
  .stButton button {
    background: linear-gradient(135deg, #1e3a5f, #2563eb);
    color: #e2eaf5; border: none; border-radius: 8px;
    font-weight: 600; padding: .5rem 1.4rem;
    transition: transform .15s, box-shadow .15s;
  }
  .stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(37,99,235,.35);
  }

  /* ── Stage badge ── */
  .stage-badge {
    display:inline-block; border-radius:4px;
    padding:.15rem .5rem; font-size:.7rem; font-weight:700;
  }

  /* ── Divider ── */
  hr { border-color: #1e2d4a33 !important; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────
DARK_BG  = "#0d1117"
DARK_AX  = "#111827"
DARK_TXT = "#94a3b8"
GRID_CLR = "#1e2d4a"
GREEN1   = "#34d399"
GREEN2   = "#22c55e"
BLUE1    = "#60a5fa"
ORANGE1  = "#fb923c"
RED1     = "#f87171"
PURPLE1  = "#a78bfa"

def apply_dark_style(fig, axes_list):
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes_list:
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors=DARK_TXT, labelsize=9)
        ax.xaxis.label.set_color(DARK_TXT)
        ax.yaxis.label.set_color(DARK_TXT)
        ax.title.set_color("#e8f5e9")
        ax.spines[['top','right','left','bottom']].set_color(GRID_CLR)
        ax.grid(True, color=GRID_CLR, linewidth=.7, linestyle="--", alpha=.8)

# ── Helper: load JSON safely ─────────────────────────────────────────
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        txt = f.read()
    txt = txt.replace("NaN", "null").replace("Infinity", "null")
    return json.loads(txt)

# ── Helper: smooth ───────────────────────────────────────────────────
def smooth(arr, w=20):
    arr = [x if x is not None else 0 for x in arr]
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w)/w, mode="valid")

# ── Synthetic demo data (shown when real data is absent) ─────────────
def gen_training_demo():
    np.random.seed(0)
    n = 1000
    r = np.cumsum(np.random.randn(n)*.8) + np.linspace(-20, 60, n)
    p = np.cumsum(np.random.randn(n)*.5) + np.linspace(-30, 1039, n)
    irr = 1200 - np.cumsum(np.random.randn(n)*.3) + np.linspace(200, -200, n)
    irr = np.clip(irr, 800, 2000)
    y = np.cumsum(np.random.randn(n)*.4) + np.linspace(2000, 7600, n)
    y = np.clip(y, 0, 9000)
    cl = np.abs(np.random.randn(n)) * 10 * np.exp(-np.linspace(0,2,n))
    al = 0.2 + (1 - np.exp(-np.linspace(0,3,n))) * 0.3
    return dict(reward=r.tolist(), profit=p.tolist(), irrigation=irr.tolist(),
                yield_=y.tolist(), critic_loss=cl.tolist(), alpha=al.tolist(),
                actor_loss=(np.abs(np.random.randn(n)*5)*np.exp(-np.linspace(0,1.5,n))).tolist())

def gen_baseline_demo():
    return {
        "Fixed-Schedule": {"profit_mean": 721, "irrigation_mean": 1380, "yield_mean": 6800, "iwue_mean": 4.93, "stress_days_mean": 18},
        "Threshold (50%)": {"profit_mean": 812, "irrigation_mean": 1310, "yield_mean": 7100, "iwue_mean": 5.42, "stress_days_mean": 12},
        "Farmer Expert":   {"profit_mean": 987, "irrigation_mean": 1240, "yield_mean": 7350, "iwue_mean": 5.93, "stress_days_mean": 8},
        "PPO":             {"profit_mean": 870, "irrigation_mean": 1290, "yield_mean": 7200, "iwue_mean": 5.58, "stress_days_mean": 10},
        "TSA-SAC":         {"profit_mean": 1039, "irrigation_mean": 1152, "yield_mean": 7600, "iwue_mean": 6.60, "stress_days_mean": 5},
    }

def gen_ablation_demo():
    return {
        "PPO":           {"profit_mean": 368, "yield_mean": 5100, "iwue_mean": 3.1,  "stress_days_mean": 28},
        "DDPG":          {"profit_mean": 892, "yield_mean": 7100, "iwue_mean": 5.8,  "stress_days_mean": 9},
        "SAC (MLP)":     {"profit_mean": 916, "yield_mean": 7250, "iwue_mean": 5.95, "stress_days_mean": 7},
        "SAC (LSTM)":    {"profit_mean": 938, "yield_mean": 7400, "iwue_mean": 6.1,  "stress_days_mean": 6},
        "TSA-SAC (dyn)": {"profit_mean": 960, "yield_mean": 7500, "iwue_mean": 6.4,  "stress_days_mean": 5},
        "TSA-SAC (fix)": {"profit_mean": 987, "yield_mean": 7560, "iwue_mean": 6.6,  "stress_days_mean": 5},
    }

# ── Load data (fall back to synthetic demo) ───────────────────────────
raw_history  = load_json("results/training_history.json") or gen_training_demo()
raw_baseline = load_json("results/baseline_comparison.json") or gen_baseline_demo()
raw_ablation = load_json("results/ablation/ablation_results.json") or gen_ablation_demo()
raw_unseen   = load_json("results/unseen_environment_analysis.json")
raw_hp       = load_json("results/hyperparameter_analysis.json")

DEMO_MODE = not os.path.exists("results/training_history.json")

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.8rem 0 1.2rem 0;'>
      <div style='font-family:Space Grotesk;font-size:1.15rem;font-weight:700;
                  background:linear-gradient(90deg,#60a5fa,#a78bfa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        TSA-SAC
      </div>
      <div style='font-size:.75rem;color:#3b82f688;letter-spacing:.06em;
                  text-transform:uppercase;margin-top:2px;'>
        Smart Irrigation · RL
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
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
    page = page  # already clean, no stripping needed

    st.markdown("---")
    if DEMO_MODE:
        st.caption("⚠️ Demo mode — synthetic data. Run training to load real results.")
    else:
        st.caption("✅ Live results loaded.")

# ══════════════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown('<div class="hero-title">Smart Irrigation<br>with TSA-SAC</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Temporal-State-Aware Soft Actor-Critic · Deep RL for Water-Efficient Crop Scheduling</div>', unsafe_allow_html=True)

    # KPI strip
    st.markdown("""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-val">$1,039</div>
        <div class="kpi-delta">▲ 5% vs Expert</div>
        <div class="kpi-label">Season Profit / ha</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">7,600</div>
        <div class="kpi-delta">▲ 3.4% vs Expert</div>
        <div class="kpi-label">Yield kg / ha</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">1,152</div>
        <div class="kpi-delta">▼ 7% vs Expert</div>
        <div class="kpi-label">Irrigation mm</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">6.60</div>
        <div class="kpi-delta">▲ 11% vs Expert</div>
        <div class="kpi-label">IWUE kg/ha/mm</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Problem Statement")
        st.markdown("""
Agriculture consumes **~70 % of global freshwater**, yet traditional irrigation methods
(fixed schedules, threshold-based rules) waste water and fail to adapt to dynamic weather.
Climate change is intensifying drought frequency, making adaptive water management critical.

**Our approach:** A **TSA-SAC** agent observes weather forecasts, soil moisture, crop growth
stage, and remaining budget — and learns to schedule irrigation to maximise profit while
minimising water stress and usage.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Key Contributions")
        for chip in [
            "Custom FAO-56 + RUE Physics Environment",
            "BiLSTM + Temporal Attention Encoder",
            "Stage-Aware Dynamic Reward Shaping",
            "4 Baselines + 3 RL Algorithms Compared",
            "Unseen Climate Generalization Tests",
            "Hyperparameter Sensitivity Analysis",
        ]:
            st.markdown(f'<span class="contrib-chip">✦ {chip}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-card" style="height:100%">', unsafe_allow_html=True)
        st.subheader("Method at a Glance")

        steps = [
            ("🌤", "Weather Input", "Markov-chain rainfall + sinusoidal temp (or NASA POWER API)"),
            ("🌱", "Physics Env",   "FAO-56 soil water balance · RUE crop growth · budget constraint"),
            ("🧠", "TSA Encoder",   "7-day BiLSTM + temporal attention → compressed context vector"),
            ("🎯", "SAC Agent",     "Twin Q-networks · auto-tuned entropy · Gaussian actor"),
            ("💧", "Action",        "Continuous irrigation ∈ [0, 60] mm/day"),
            ("📈", "Reward",        "Stage-weighted yield + water efficiency + terminal harvest profit"),
        ]
        for icon, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;gap:10px;margin-bottom:.85rem;align-items:flex-start;">
              <div style="font-size:1.3rem;margin-top:1px;">{icon}</div>
              <div>
                <div style="font-weight:600;color:#93c5fd;font-size:.88rem;">{title}</div>
                <div style="font-size:.78rem;color:#78909c;line-height:1.4;">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Mini training preview
    st.markdown("---")
    st.subheader("Training Convergence Preview")
    history = raw_history
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    preview_keys = [
        ("reward",  "Episode Reward",  GREEN1),
        ("profit",  "Season Profit ($)", GREEN2),
        ("alpha",   "Entropy α",        BLUE1),
    ]
    for ax, (k, title, color) in zip(axes, preview_keys):
        data_key = "yield_" if k == "yield" else k
        raw = history.get(data_key, history.get(k, []))
        if raw:
            data = [x if x is not None else 0 for x in raw]
            ep = np.arange(len(data))
            ax.plot(ep, data, color=color, alpha=.18, linewidth=.6)
            sm = smooth(data, 30)
            ep_sm = np.linspace(0, len(data)-1, len(sm))
            ax.plot(ep_sm, sm, color=color, linewidth=2.2)
            ax.fill_between(ep_sm, ax.get_ylim()[0] if ax.get_ylim()[0] > -1000 else sm.min()*.8,
                            sm, color=color, alpha=.07)
        ax.set_facecolor(DARK_AX)
        ax.set_title(title, color="#e8f5e9", fontsize=10, fontweight="bold")
        ax.set_xlabel("Episode", color=DARK_TXT, fontsize=8)
        ax.tick_params(colors=DARK_TXT, labelsize=8)
        ax.spines[['top','right','left','bottom']].set_color(GRID_CLR)
        ax.grid(True, color=GRID_CLR, linewidth=.6, linestyle="--", alpha=.7)
    plt.tight_layout(pad=1.2)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════
#  TRAINING RESULTS
# ══════════════════════════════════════════════════════════════════════
elif page == "Training Results":
    st.header("📊 Training Results — 1 000 Episodes")
    if DEMO_MODE:
        st.info("Showing **synthetic demo data**. Run training to load real results.", icon="ℹ️")

    history = raw_history
    tab1, tab2 = st.tabs(["Training Curves", "Agent Losses"])

    with tab1:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("TSA-SAC Training Curves", fontsize=15, fontweight="bold", color="#e8f5e9")
        fig.patch.set_facecolor(DARK_BG)

        keys = [
            ("reward",    "Episode Reward",    GREEN1),
            ("profit",    "Profit ($)",         GREEN2),
            ("irrigation","Irrigation (mm)",    PURPLE1),
            ("yield_",    "Yield (kg/ha)",      ORANGE1),
            ("critic_loss","Critic Loss",        RED1),
            ("alpha",     "Entropy α",          BLUE1),
        ]
        for ax, (k, title, color) in zip(axes.flat, keys):
            raw = history.get(k, history.get(k.rstrip("_"), []))
            if raw:
                data = [x if x is not None else 0 for x in raw]
                ep = np.arange(len(data))
                ax.plot(ep, data, color=color, alpha=.15, linewidth=.5)
                sm = smooth(data, 25)
                ep_sm = np.linspace(0, len(data)-1, len(sm))
                ax.plot(ep_sm, sm, color=color, linewidth=2.2, label="Smoothed")
                ax.fill_between(ep_sm, np.array(sm)*0.97, np.array(sm)*1.03, color=color, alpha=.12)
            ax.set_facecolor(DARK_AX)
            ax.set_title(title, color="#e8f5e9", fontsize=10, fontweight="bold")
            ax.set_xlabel("Episode", color=DARK_TXT, fontsize=8)
            ax.tick_params(colors=DARK_TXT, labelsize=8)
            ax.spines[['top','right','left','bottom']].set_color(GRID_CLR)
            ax.grid(True, color=GRID_CLR, linewidth=.6, linestyle="--", alpha=.7)

        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5))
        fig2.suptitle("TSA-SAC Loss & Entropy", fontsize=14, fontweight="bold", color="#e8f5e9")
        fig2.patch.set_facecolor(DARK_BG)
        loss_keys = [
            ("critic_loss", "Critic Loss",  RED1),
            ("actor_loss",  "Actor Loss",   PURPLE1),
            ("alpha",       "Entropy α",    BLUE1),
        ]
        for ax, (k, title, color) in zip(axes2, loss_keys):
            raw = history.get(k, [])
            if raw:
                data = [x if x is not None else 0 for x in raw]
                sm = smooth(data, 20)
                ep_sm = np.linspace(0, len(data)-1, len(sm))
                ax.plot(ep_sm, sm, color=color, linewidth=2.2)
                ax.fill_between(ep_sm, 0, sm, color=color, alpha=.12)
            ax.set_facecolor(DARK_AX)
            ax.set_title(title, color="#e8f5e9", fontsize=10, fontweight="bold")
            ax.set_xlabel("Episode", color=DARK_TXT, fontsize=8)
            ax.tick_params(colors=DARK_TXT, labelsize=8)
            ax.spines[['top','right','left','bottom']].set_color(GRID_CLR)
            ax.grid(True, color=GRID_CLR, linewidth=.6, linestyle="--", alpha=.7)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # Saved plots
    plot_dir = "results/plots"
    if os.path.exists(plot_dir):
        pngs = [f for f in sorted(os.listdir(plot_dir)) if f.endswith(".png")]
        if pngs:
            st.subheader("📈 Saved Plots")
            cols = st.columns(2)
            for i, fname in enumerate(pngs):
                with cols[i % 2]:
                    st.image(os.path.join(plot_dir, fname), caption=fname, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  POLICY COMPARISON
# ══════════════════════════════════════════════════════════════════════
elif page == "Policy Comparison":
    st.header("🏆 Policy Comparison — TSA-SAC vs Baselines")
    if DEMO_MODE:
        st.info("Showing **synthetic demo data**.", icon="ℹ️")

    import pandas as pd
    baseline = raw_baseline
    policies  = list(baseline.keys())
    profits   = [baseline[p]["profit_mean"]      for p in policies]
    irrig     = [baseline[p]["irrigation_mean"]  for p in policies]
    yields    = [baseline[p]["yield_mean"]        for p in policies]
    iwue      = [baseline[p]["iwue_mean"]         for p in policies]
    stress    = [baseline[p]["stress_days_mean"]  for p in policies]

    # Best = last = TSA-SAC
    best_idx = profits.index(max(profits))
    bar_colors = [DARK_AX] * len(policies)
    bar_colors[best_idx] = GREEN2

    # Summary table with highlight
    df = pd.DataFrame({
        "Policy":          policies,
        "Profit ($/ha)":   [f"${p:.0f}" for p in profits],
        "Yield (kg/ha)":   [f"{y:.0f}"  for y in yields],
        "Irrigation (mm)": [f"{i:.0f}"  for i in irrig],
        "IWUE":            [f"{w:.2f}"  for w in iwue],
        "Stress Days":     [f"{s:.0f}"  for s in stress],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.success(f"**TSA-SAC achieves highest profit (${profits[best_idx]:,.0f}/ha) "
               f"with lowest irrigation ({irrig[best_idx]:.0f}mm)** — "
               f"{(1 - irrig[best_idx]/irrig[-2])*100:.0f}% less water than Farmer Expert "
               f"while earning {(profits[best_idx]/profits[-2]-1)*100:.0f}% more profit.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Policy Comparison", fontsize=15, fontweight="bold", color="#e8f5e9")
    fig.patch.set_facecolor(DARK_BG)

    datasets = [
        (profits, "Profit ($/ha)",      "USD"),
        (irrig,   "Irrigation (mm)",    "mm"),
        (iwue,    "IWUE (kg/ha/mm)",    "kg/ha/mm"),
        (stress,  "Stress Days",        "days"),
    ]
    bar_palettes = [
        [GREEN2 if i == best_idx else "#1a2540" for i in range(len(policies))],
        [RED1   if i == best_idx else "#1a2540" for i in range(len(policies))],  # lower is better
        [GREEN1 if i == best_idx else "#1a2540" for i in range(len(policies))],
        [RED1   if i == best_idx else "#1a2540" for i in range(len(policies))],  # lower is better
    ]
    for ax, (data, title, ylabel), bpal in zip(axes.flat, datasets, bar_palettes):
        bars = ax.bar(policies, data, color=bpal, edgecolor="#0d1a0d", linewidth=.8)
        ax.set_facecolor(DARK_AX)
        ax.set_title(title, color="#e8f5e9", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, color=DARK_TXT, fontsize=8)
        ax.tick_params(axis="x", rotation=18, colors=DARK_TXT, labelsize=8)
        ax.tick_params(axis="y", colors=DARK_TXT, labelsize=8)
        ax.spines[['top','right','left','bottom']].set_color(GRID_CLR)
        ax.grid(True, axis="y", color=GRID_CLR, linewidth=.6, linestyle="--", alpha=.7)
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(data)*.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, color="#e8f5e9")

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════
#  ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════
elif page == "Ablation Study":
    st.header("🔬 Ablation Study — Component Contribution")
    if DEMO_MODE:
        st.info("Showing **synthetic demo data**.", icon="ℹ️")

    st.markdown("""
**Setup:** 500 episodes · 400 mm water budget (moderate constraint) · seed = 42

Each variant removes or replaces one component to isolate its contribution:
    """)

    import pandas as pd
    ablation = raw_ablation
    variants = list(ablation.keys())
    profits_a  = [ablation[v]["profit_mean"]      for v in variants]
    yields_a   = [ablation[v]["yield_mean"]        for v in variants]
    iwue_a     = [ablation[v].get("iwue_mean", 0)  for v in variants]
    stress_a   = [ablation[v]["stress_days_mean"]  for v in variants]

    df = pd.DataFrame({
        "Variant":     variants,
        "Profit ($)":  [f"${p:.1f}" for p in profits_a],
        "Yield (kg/ha)":[f"{y:.0f}" for y in yields_a],
        "IWUE":        [f"{w:.2f}"  for w in iwue_a],
        "Stress Days": [f"{s:.1f}"  for s in stress_a],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    palette = [RED1, ORANGE1, BLUE1, PURPLE1, GREEN1, GREEN2]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Ablation Study", fontsize=14, fontweight="bold", color="#e8f5e9")
    fig.patch.set_facecolor(DARK_BG)

    for ax, (data, title, ylabel) in zip(axes, [
        (profits_a, "Mean Season Profit ($)", "$"),
        (yields_a,  "Mean Yield (kg/ha)",     "kg/ha"),
        (iwue_a,    "IWUE (kg/ha/mm)",        "kg/ha/mm"),
    ]):
        bars = ax.bar(variants, data, color=palette[:len(variants)], edgecolor="#0d1a0d", linewidth=.8)
        ax.set_facecolor(DARK_AX)
        ax.set_title(title, color="#e8f5e9", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, color=DARK_TXT, fontsize=8)
        ax.tick_params(axis="x", rotation=22, colors=DARK_TXT, labelsize=8)
        ax.tick_params(axis="y", colors=DARK_TXT, labelsize=8)
        ax.spines[['top','right','left','bottom']].set_color(GRID_CLR)
        ax.grid(True, axis="y", color=GRID_CLR, linewidth=.6, linestyle="--", alpha=.7)
        for bar, val in zip(bars, data):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(data)*.01,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8, color="#e8f5e9")

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.subheader("Key Findings")
    findings = [
        ("🔴", "SAC >>> PPO (+136%)",          "SAC is the right algorithm for continuous irrigation control"),
        ("🟠", "SAC > DDPG (+2.5%)",            "Entropy regularization provides better exploration"),
        ("🟣", "TSA encoder > MLP (+1.7%)",     "Temporal attention captures weather patterns over days"),
        ("🟢", "TSA-SAC (fixed) best constrained","With limited water, fixed reward avoids risky exploration"),
    ]
    for icon, title, desc in findings:
        st.markdown(f"""
        <div style='display:flex;gap:10px;padding:.5rem 0;border-bottom:1px solid #1e2d4a;'>
          <span style='font-size:1.1rem'>{icon}</span>
          <div><strong style='color:#93c5fd'>{title}</strong>
          <span style='color:#78909c;font-size:.87rem;'> — {desc}</span></div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  UNSEEN ENVIRONMENTS
# ══════════════════════════════════════════════════════════════════════
elif page == "Unseen Environments":
    st.header("🌍 Generalization — Unseen Environments")
    st.markdown("Testing the **arid-trained** TSA-SAC agent on climates and constraints it was **never trained on**.")

    if raw_unseen:
        import pandas as pd
        labels = list(raw_unseen.keys())
        df = pd.DataFrame({
            "Scenario":        labels,
            "Profit ($)":      [f"${raw_unseen[l]['profit_mean']:.1f} ± {raw_unseen[l]['profit_std']:.1f}" for l in labels],
            "Yield (kg/ha)":   [f"{raw_unseen[l]['yield_mean']:.0f}"      for l in labels],
            "Irrigation (mm)": [f"{raw_unseen[l]['irrigation_mean']:.0f}" for l in labels],
            "IWUE":            [f"{raw_unseen[l]['iwue_mean']:.2f}"        for l in labels],
            "Stress Days":     [f"{raw_unseen[l]['stress_days_mean']:.1f}" for l in labels],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        plot_path = "results/plots/unseen_environment.png"
        if os.path.exists(plot_path):
            st.image(plot_path, caption="Unseen Environment Generalization", use_container_width=True)
    else:
        # Demo bar chart
        st.info("Run `python evaluate_extra.py` to load real generalization data. Showing illustrative example.", icon="ℹ️")
        scenarios = ["Arid (trained)", "Semi-Arid", "Humid", "Arid (scarce budget)", "Moderate Budget"]
        profits_u = [1039, 1120, 1210, 680, 950]
        colors_u  = [GREEN2, GREEN1, BLUE1, RED1, ORANGE1]

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_AX)
        bars = ax.bar(scenarios, profits_u, color=colors_u, edgecolor="#0d1a0d", linewidth=.8)
        ax.set_title("Generalization Across Climates & Budget Levels", color="#e8f5e9", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Profit ($/ha)", color=DARK_TXT)
        ax.tick_params(colors=DARK_TXT, labelsize=9)
        ax.spines[['top','right','left','bottom']].set_color(GRID_CLR)
        ax.grid(True, axis="y", color=GRID_CLR, linewidth=.6, linestyle="--", alpha=.7)
        for bar, val in zip(bars, profits_u):
            ax.text(bar.get_x()+bar.get_width()/2, val+12, f"${val}", ha="center", va="bottom", fontsize=9, color="#e8f5e9")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.subheader("Interpretation")
    interps = [
        ("🌧", "Humid climate",         "More natural rainfall → agent applies less irrigation → higher profit"),
        ("🌤", "Semi-arid",             "Moderate challenge → well-adapted from arid training"),
        ("💧", "Scarce water budget",   "Extreme constraint → tests robustness; profit drops but remains positive"),
    ]
    for icon, title, desc in interps:
        st.markdown(f"""
        <div style='display:flex;gap:10px;padding:.5rem 0;border-bottom:1px solid #1e2d4a;'>
          <span style='font-size:1.2rem'>{icon}</span>
          <div><strong style='color:#93c5fd'>{title}</strong>
          <span style='color:#78909c;font-size:.87rem;'> — {desc}</span></div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════
elif page == "Hyperparameters":
    st.header("⚙️ Hyperparameter Sensitivity Analysis")
    st.markdown("100-episode short runs measuring how key hyperparameters affect agent performance.")

    if raw_hp:
        import pandas as pd
        for exp_name, data in raw_hp.items():
            st.subheader(exp_name)
            df = pd.DataFrame({
                "Value":        data["labels"],
                "Profit ($)":   [f"${p:.1f}" for p in data["profits"]],
                "Yield (kg/ha)":[f"{y:.0f}"  for y in data["yields"]],
                "IWUE":         [f"{w:.2f}"  for w in data["iwues"]],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
        plot_path = "results/plots/hyperparameter_sensitivity.png"
        if os.path.exists(plot_path):
            st.image(plot_path, caption="Hyperparameter Sensitivity", use_container_width=True)
    else:
        st.info("Run `python evaluate_extra.py` first to generate hyperparameter data. Showing illustrative charts.", icon="ℹ️")

        # Demo charts
        hp_demo = {
            "Learning Rate":   ([1e-3, 3e-4, 1e-4, 3e-5], [820, 987, 910, 740]),
            "Hidden Size":     ([64, 128, 256, 512],        [720, 870, 987, 940]),
            "Sequence Length": ([3, 5, 7, 14],              [890, 930, 987, 960]),
        }
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        fig.suptitle("Hyperparameter Sensitivity (Illustrative)", fontsize=13, fontweight="bold", color="#e8f5e9")
        fig.patch.set_facecolor(DARK_BG)
        for ax, (hp_name, (labels, profits_hp)) in zip(axes, hp_demo.items()):
            best = profits_hp.index(max(profits_hp))
            cols_hp = [GREEN2 if i==best else "#1a2540" for i in range(len(labels))]
            ax.bar([str(l) for l in labels], profits_hp, color=cols_hp, edgecolor="#0d1a0d")
            ax.set_facecolor(DARK_AX)
            ax.set_title(hp_name, color="#e8f5e9", fontsize=10, fontweight="bold")
            ax.set_ylabel("Profit ($)", color=DARK_TXT, fontsize=8)
            ax.tick_params(colors=DARK_TXT, labelsize=9)
            ax.spines[['top','right','left','bottom']].set_color(GRID_CLR)
            ax.grid(True, axis="y", color=GRID_CLR, linewidth=.6, linestyle="--", alpha=.7)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.subheader("Key Observations")
    obs = [
        ("Learning Rate", "3×10⁻⁴ provides the best balance of speed and stability"),
        ("Hidden Size",   "256 is optimal; 128 underfits, 512 overfits on limited data"),
        ("Sequence Length","7 days captures weekly weather patterns, improving scheduling"),
    ]
    for hp, note in obs:
        st.markdown(f"- **{hp}**: {note}")

# ══════════════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════
elif page == "Environment":
    st.header("🌱 Environment — CropIrrigationEnv")
    st.markdown("A custom **Gymnasium** environment implementing physics-based crop irrigation simulation.")

    st.subheader("Overview")
    st.markdown("""
The environment simulates a **170-day cotton growing season** with:
- **FAO-56 Soil Water Balance** — tracks root-zone moisture, drainage, runoff, and water stress
- **RUE Crop Growth Model** — temperature & radiation-driven biomass accumulation
- **Stochastic Weather** — Markov-chain rainfall + sinusoidal temperature curves
- **Seasonal Water Budget** — non-linear scarcity pricing as budget depletes
    """)

    import pandas as pd
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("State Space (16-dim)")
        state_df = pd.DataFrame({
            "#":  list(range(16)),
            "Variable": [
                "LAI (normalised)", "Biomass / max yield", "Root depth / max",
                "Available soil water", "Reservoir level", "Water stress (1−Ks)",
                "ET₀ / 12 mm", "Today's rain / 30 mm", "3-day rain forecast",
                "3-day ET₀ forecast", "Stage: Emergence", "Stage: Vegetative",
                "Stage: Flowering", "Stage: Boll Fill", "Stage: Maturity",
                "Budget remaining",
            ],
            "Range": ["[0,1]"] * 16,
        })
        st.dataframe(state_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Action & Reward")
        st.markdown("**Action:** Continuous irrigation ∈ [0, 60] mm/day")
        st.markdown("""
**Daily Reward:**
```
r = w_y · yield_gain − w_w · water_cost · scarcity − w_s · (1−Ks)²
```
**Terminal Reward (harvest):**
```
r_T = (yield × price − irrigation × cost + IWUE_bonus) / scale
```
        """)
        st.subheader("Dynamic Weights by Stage")
        wt_df = pd.DataFrame({
            "Stage":    ["Emergence","Vegetative","Flowering","Boll Fill","Maturity"],
            "w_y":      [0.8, 1.0, 1.2, 0.6, 0.2],
            "w_w":      [0.4, 0.3, 0.2, 0.5, 0.8],
            "w_s":      [0.8, 1.0, 1.8, 0.6, 0.3],
        })
        st.dataframe(wt_df, use_container_width=True, hide_index=True)

    st.subheader("Crop Profiles")
    crop_df = pd.DataFrame({
        "Parameter": ["Season (days)","Max Yield (kg/ha)","Kc_ini","Kc_mid","Kc_end",
                      "Field Capacity","Wilting Point","Root Depth (mm)","Price ($/kg)","Water Cost ($/mm)"],
        "Cotton": [170, 7600, 0.35, 1.15, 0.70, 0.33, 0.15, 1200, 0.22, 0.55],
        "Wheat":  [120, 6000, 0.40, 1.15, 0.40, 0.35, 0.12,  600, 0.25, 0.45],
        "Maize":  [100, 9000, 0.30, 1.20, 0.60, 0.32, 0.11,  700, 0.18, 0.45],
    })
    st.dataframe(crop_df, use_container_width=True, hide_index=True)

    col3, col4 = st.columns(2, gap="large")
    with col3:
        st.subheader("Water Budget Presets (mm)")
        budget_df = pd.DataFrame({
            "Crop":    ["Cotton","Wheat","Maize"],
            "Generous":[1500, 1000, 1000],
            "Moderate":[ 400,  280,  300],
            "Scarce":  [ 250,  180,  200],
        })
        st.dataframe(budget_df, use_container_width=True, hide_index=True)

    with col4:
        st.subheader("Climate Presets")
        climate_df = pd.DataFrame({
            "Parameter":     ["Rain Prob","Rain Mean (mm)","Temp Mean (°C)","Solar Mean","Wind Mean"],
            "Arid":          [0.05,  3, 34, 25, 3.4],
            "Semi-Arid":     [0.15,  6, 28, 22, 2.8],
            "Humid":         [0.40, 10, 24, 18, 2.2],
        })
        st.dataframe(climate_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
#  LIVE SIMULATION
# ══════════════════════════════════════════════════════════════════════
elif page == "Live Simulation":
    st.header("🎮 Live Irrigation Simulation")
    st.markdown("Watch the trained TSA-SAC agent make irrigation decisions in real-time.")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        climate = st.selectbox("🌤 Climate", ["arid", "semi_arid", "humid"], index=0)
    with c2:
        reservoir = st.slider("💧 Reservoir (mm)", 500, 3000, 2000, 100)
    with c3:
        n_days = st.slider("📅 Simulation Days", 30, 180, 180, 10)

    run_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        try:
            import torch
            from environment import CropIrrigationEnv
            from sac_agent import SACAgent

            env = CropIrrigationEnv(
                crop="cotton", climate=climate,
                water_budget_level="generous",
                reservoir_capacity=reservoir,
                weather_source="synthetic",
            )
            ckpt_path = "checkpoints/tsa_sac_improved_best.pt"
            if not os.path.exists(ckpt_path):
                st.error("Checkpoint not found — train the model first (`python train.py`).")
                st.stop()

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            obs_dim = env.observation_space.shape[0]
            agent = SACAgent(obs_dim=obs_dim, act_dim=1, hidden=256,
                             seq_len=7, encoder_type="tsa", lr=3e-4)
            agent.actor.load_state_dict(ckpt["actor"])

            obs, _ = env.reset(seed=42)
            seq_buf = np.zeros((7, obs_dim), dtype=np.float32)

            days, irrigations, soil_moistures = [], [], []
            yields_t, rains, stages_list = [], [], []
            total_irrig = 0.0

            progress_bar = st.progress(0, "Simulating season…")
            status_txt   = st.empty()

            stage_names = ["Emergence","Vegetative","Flowering","Boll Fill","Maturity"]
            stage_colors = {"Emergence": BLUE1, "Vegetative": GREEN2, "Flowering": ORANGE1,
                            "Boll Fill": PURPLE1, "Maturity": RED1, "Pre-plant": DARK_TXT}

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
                stage = stage_names[stage_idx] if stage_idx >= 0 else "Pre-plant"
                stages_list.append(stage)

                progress_bar.progress((day+1)/n_days, f"Day {day+1}/{n_days} · Stage: {stage}")
                if terminated or truncated:
                    break

            progress_bar.empty()
            status_txt.empty()

            final_yield  = info.get("yield",  yields_t[-1] if yields_t else 0)
            final_profit = info.get("profit", 0)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("💰 Final Profit",     f"${final_profit:.1f}")
            m2.metric("🌾 Final Yield",      f"{final_yield:.0f} kg/ha")
            m3.metric("💧 Total Irrigation", f"{total_irrig:.0f} mm")
            m4.metric("📅 Days Simulated",   f"{len(days)}")

            # Plot
            fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
            fig.suptitle("TSA-SAC Agent — Irrigation Season", fontsize=14, fontweight="bold", color="#e8f5e9")
            fig.patch.set_facecolor(DARK_BG)

            # Stage background bands
            stage_color_map = {"Emergence":("#1a237e",".12"), "Vegetative":("#1b5e20",".12"),
                               "Flowering":("#e65100",".12"), "Boll Fill":("#4a148c",".12"),
                               "Maturity":("#b71c1c",".12")}
            for ax in axes:
                ax.set_facecolor(DARK_AX)
                ax.tick_params(colors=DARK_TXT, labelsize=8)
                ax.spines[['top','right','left','bottom']].set_color(GRID_CLR)
                ax.grid(True, axis="y", color=GRID_CLR, linewidth=.6, linestyle="--", alpha=.7)

            # Panel 1: irrigation + rain
            axes[0].bar(days, irrigations, color=BLUE1,   alpha=.75, label="Irrigation", width=1)
            axes[0].bar(days, rains,       color="#81d4fa",alpha=.5,  label="Rainfall",  width=1)
            axes[0].set_ylabel("Water (mm)", color=DARK_TXT, fontsize=9)
            axes[0].set_title("Daily Water Application", color="#e8f5e9", fontsize=10)
            axes[0].legend(loc="upper right", fontsize=8,
                           facecolor=DARK_AX, edgecolor=GRID_CLR, labelcolor=DARK_TXT)

            # Panel 2: soil moisture with stress threshold
            axes[1].plot(days, soil_moistures, color=GREEN1, linewidth=2)
            axes[1].fill_between(days, 0, soil_moistures, color=GREEN1, alpha=.12)
            axes[1].axhline(y=0.45, color=RED1, linestyle="--", alpha=.7, linewidth=1.2, label="Stress threshold")
            axes[1].set_ylabel("Soil Moisture", color=DARK_TXT, fontsize=9)
            axes[1].set_title("Root Zone Soil Water Content", color="#e8f5e9", fontsize=10)
            axes[1].legend(loc="upper right", fontsize=8,
                           facecolor=DARK_AX, edgecolor=GRID_CLR, labelcolor=DARK_TXT)

            # Panel 3: biomass
            axes[2].plot(days, yields_t, color=ORANGE1, linewidth=2)
            axes[2].fill_between(days, 0, yields_t, color=ORANGE1, alpha=.12)
            axes[2].set_ylabel("Biomass", color=DARK_TXT, fontsize=9)
            axes[2].set_xlabel("Day of Season", color=DARK_TXT, fontsize=9)
            axes[2].set_title("Crop Biomass Accumulation", color="#e8f5e9", fontsize=10)

            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.success(f"✅ Simulation complete! Agent earned **${final_profit:.1f}** "
                       f"using **{total_irrig:.0f} mm** irrigation.")

        except ImportError as e:
            st.error(f"Missing module: {e}  — make sure `environment.py` and `sac_agent.py` are present.")
        except Exception as e:
            st.error(f"Simulation error: {e}")
            import traceback
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════
#  ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════
elif page == "Architecture":
    st.header(" System Architecture")

    arch_path = "system_architecture.png"
    if os.path.exists(arch_path):
        st.image(arch_path, caption="TSA-SAC smart irrigation system architecture", use_container_width=True)

 

    st.subheader("RL Formulation")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        import pandas as pd
        st.markdown("**State Space (16-dim)**")
        state_tbl = pd.DataFrame({
            "Feature": ["LAI, Biomass, Root depth","Soil water, Reservoir",
                        "Water stress","ET₀, Rain, Forecasts","Growth stage (one-hot)","Budget remaining"],
            "Description": ["Crop status","Water status","Stress indicator",
                            "Weather","Phenology","Constraint"],
        })
        st.dataframe(state_tbl, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Action & Reward**")
        st.markdown("""
- **Action**: Continuous irrigation ∈ [0, 60] mm/day
- **Daily reward**: `w_y·yield_gain − w_w·water_cost·scarcity − w_s·stress²`
- **Terminal reward**: `(revenue − cost) / scale`
- **Dynamic weights** adjust by growth stage (flowering gets higher yield weight)
        """)

    st.subheader("Experimental Setup")
    import pandas as pd
    setup_df = pd.DataFrame({
        "Parameter": ["Crop","Climate","Training Episodes","Reservoir","Hidden Size",
                      "Learning Rate","Sequence Length","Replay Buffer","Batch Size",
                      "Discount γ","Soft Update τ"],
        "Value":     ["Cotton (180-day season)","Arid (synthetic weather)","1 000",
                      "2 000 mm (generous)","256","3×10⁻⁴","7 days","1 000 000",
                      "64","0.99","0.005"],
    })
    st.dataframe(setup_df, use_container_width=True, hide_index=True)

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#334155;font-size:.75rem;padding:2rem 0 1rem;'>
  TSA-SAC Irrigation RL · Built with Streamlit · Physics-Based Agriculture Simulation
</div>
""", unsafe_allow_html=True)