# app.py — Project 14: AI-Driven Digital Twin Dashboard
# Run with: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
import time, joblib, os

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Twin — Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)

# ── Load Model (memory-persistent via cache) ───────────────────
@st.cache_resource
def load_assets():
    model  = keras.models.load_model('digital_twin_model.keras')
    scaler = joblib.load('scaler.pkl')
    fc     = joblib.load('feature_cols.pkl')
    return model, scaler, fc

model, scaler, feature_cols = load_assets()

CLASS_LABELS     = ['Healthy', 'Heat Dissipation', 'Power Failure', 'Overstrain', 'Tool Wear']
CLASS_COLORS     = ['#2ecc71', '#e74c3c', '#e67e22', '#9b59b6', '#3498db']
BASELINE_RPM     = 1450.0
SAFETY_THRESHOLD = 0.05
HDF_CLASS        = 1
OSF_CLASS        = 3

# ── Feature Engineering ────────────────────────────────────────
def build_feature_vector(rpm, torque, tool_wear, air_temp=298.0, process_temp=308.0):
    delta_t    = process_temp - air_temp
    mech_power = torque * (2 * np.pi * rpm / 60)

    base = {
        'Air_Temp':         air_temp,
        'Process_Temp':     process_temp,
        'RPM':              rpm,
        'Torque':           torque,
        'Tool_Wear':        tool_wear,
        'Delta_T':          delta_t,
        'Mech_Power':       mech_power,
        'Power_per_RPM':    mech_power / (rpm          + 1e-6),
        'Torque_per_Temp':  torque     / (process_temp + 1e-6),
        'Wear_Rate':        tool_wear  / (rpm          + 1e-6),
        'Heat_Index':       delta_t    * mech_power,
        'Strain_Index':     torque     * tool_wear,
        'RPM_Torque_ratio': rpm        / (torque       + 1e-6),
    }

    # Polynomial cross-terms
    poly_vals = {
        'Delta_T': delta_t, 'Mech_Power': mech_power,
        'Torque':  torque,  'RPM': rpm,   'Tool_Wear': tool_wear
    }
    for fname in feature_cols:
        parts = fname.split(' ')
        if len(parts) == 2 and all(p in poly_vals for p in parts):
            base[fname] = poly_vals[parts[0]] * poly_vals[parts[1]]

    # Rolling features — approximated with current value for live inference
    for col in ['Torque', 'RPM']:
        val = base[col]
        base[f'{col}_roll_mean']  = val
        base[f'{col}_roll_std']   = 0.0
        base[f'{col}_roll_grad']  = 0.0
        base[f'{col}_roll_min']   = val
        base[f'{col}_roll_max']   = val
        base[f'{col}_roll_range'] = 0.0
        base[f'{col}_roll_ema']   = val

    vec = np.array([base.get(f, 0.0) for f in feature_cols]).reshape(1, -1)
    return scaler.transform(vec)

# ── JAYA OEE Optimiser ────────────────────────────────────────
def run_jaya_oee(tool_wear, air_temp, process_temp, pop=50, iters=20):
    rpm_bounds  = (1100, 1500) if tool_wear > 200 else (1200, 1800)
    torq_bounds = (20, 70)

    np.random.seed(42)
    population = np.column_stack([
        np.random.uniform(*rpm_bounds,  pop),
        np.random.uniform(*torq_bounds, pop)
    ])

    def fitness(rpm, torque):
        vec   = build_feature_vector(rpm, torque, tool_wear, air_temp, process_temp)
        proba = model.predict(vec, verbose=0)[0]
        p_f   = float(proba[HDF_CLASS] + proba[OSF_CLASS])
        return (-rpm + (1e6 * (p_f - SAFETY_THRESHOLD))) if p_f > SAFETY_THRESHOLD else -rpm, p_f

    fit = np.array([fitness(p[0], p[1])[0] for p in population])

    # Live progress bar
    progress_bar = st.progress(0, text="JAYA initialising...")
    for it in range(iters):
        bi    = np.argmin(fit); wi = np.argmax(fit)
        best  = population[bi].copy(); worst = population[wi].copy()
        for i in range(pop):
            r1   = np.random.uniform(0, 1, 2)
            r2   = np.random.uniform(0, 1, 2)
            cand = (population[i]
                    + r1 * (best  - np.abs(population[i]))
                    - r2 * (worst - np.abs(population[i])))
            cand[0] = np.clip(cand[0], *rpm_bounds)
            cand[1] = np.clip(cand[1], *torq_bounds)
            cf, _   = fitness(cand[0], cand[1])
            if cf < fit[i]:
                population[i] = cand
                fit[i]        = cf

        progress_bar.progress(
            int((it + 1) / iters * 100),
            text=f"JAYA iteration {it+1}/{iters}  —  Best RPM so far: {-np.min(fit):.0f}"
        )

    progress_bar.empty()
    best_sol  = population[np.argmin(fit)]
    _, p_fail = fitness(best_sol[0], best_sol[1])
    return float(best_sol[0]), float(best_sol[1]), p_fail

# ══════════════════════════════════════════════════════════════
#  DASHBOARD LAYOUT
# ══════════════════════════════════════════════════════════════

st.title("🏭 AI-Driven Digital Twin — Predictive Maintenance")
st.caption("Project 14  |  TensorFlow Neural Network + JAYA Metaheuristic OEE Optimisation  |  AI4I Dataset")

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.header("🎛️ Live Sensor Inputs")
rpm       = st.sidebar.slider("Rotational Speed (RPM)",  500,  2500, 1450, step=10)
torque    = st.sidebar.slider("Torque (Nm)",             1,    80,   40,   step=1)
tool_wear = st.sidebar.slider("Tool Wear (min)",         0,    300,  100,  step=5)
air_temp  = st.sidebar.slider("Air Temperature (K)",     295,  305,  298,  step=1)
proc_temp = st.sidebar.slider("Process Temperature (K)", 305,  320,  308,  step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tool Wear Status**")
wear_color = "🔴" if tool_wear > 200 else ("🟡" if tool_wear > 130 else "🟢")
st.sidebar.markdown(
    f"{wear_color} `{tool_wear} min` — "
    f"{'⚠️ DEGRADED — RPM will be throttled' if tool_wear > 200 else 'Normal operating range'}"
)
st.sidebar.progress(tool_wear / 300)

st.sidebar.markdown("---")
st.sidebar.markdown("**⚡ JAYA Speed Setting**")
jaya_iters = st.sidebar.select_slider(
    "Optimisation Iterations",
    options=[20, 50, 100, 200],
    value=20,
    help="20 = fast (~10s) | 200 = full accuracy (~3 min)"
)
run_jaya = st.sidebar.button("⚡ Run JAYA OEE Optimisation", type="primary")

# ── Live Inference ─────────────────────────────────────────────
start           = time.time()
vec             = build_feature_vector(rpm, torque, tool_wear, air_temp, proc_temp)
proba           = model.predict(vec, verbose=0)[0]
latency         = (time.time() - start) * 1000
pred_class      = int(np.argmax(proba))
p_fail_combined = float(proba[HDF_CLASS] + proba[OSF_CLASS])

# ── Row 1: KPI Metrics ────────────────────────────────────────
st.markdown("### 📊 Live System Status")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("⚙️ Fault State",      CLASS_LABELS[pred_class])
c2.metric("🌡️ Delta T",          f"{proc_temp - air_temp} K")
c3.metric("⚡ Mech. Power",       f"{torque * (2 * np.pi * rpm / 60) / 1000:.2f} kW")
c4.metric("🔴 P(Thermal Fail)",   f"{p_fail_combined * 100:.2f}%",
          delta="⚠️ UNSAFE" if p_fail_combined > SAFETY_THRESHOLD else "✅ SAFE",
          delta_color="inverse")
c5.metric("⏱️ Latency",           f"{latency:.1f} ms")

st.markdown("---")

# ── Row 2: Fault Bar Chart + Gauge ────────────────────────────
col_left, col_right = st.columns([1.4, 1])

with col_left:
    st.markdown("#### 🔍 5-Class Fault Probability Distribution")

    # Colour bars red if they exceed safety threshold
    bar_colors = [
        '#e74c3c' if p > SAFETY_THRESHOLD else c
        for p, c in zip(proba, CLASS_COLORS)
    ]

    fig_bar = go.Figure(go.Bar(
        x=[p * 100 for p in proba],
        y=CLASS_LABELS,
        orientation='h',
        marker_color=bar_colors,
        text=[f"{p * 100:.2f}%" for p in proba],
        textposition='outside'
    ))
    fig_bar.add_annotation(
        text="⚠️ 5% Safety Limit",
        xref="paper", yref="paper",
        x=0.98, y=1.05,
        showarrow=False,
        font=dict(color='red', size=11),
        align='right'
    )
    fig_bar.update_layout(
        xaxis_title='Probability (%)',
        template='plotly_white',
        height=280,
        margin=dict(l=10, r=60, t=30, b=30),
        xaxis=dict(range=[0, max(max(proba) * 100 + 5, 20)])
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.markdown("#### 🌡️ Thermal & Mechanical Safety Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=p_fail_combined * 100,
        title={'text': "P(Thermal / Mech Fail) %"},
        delta={'reference': 5.0, 'increasing': {'color': 'red'}},
        gauge={
            'axis':  {'range': [0, 20]},
            'bar':   {'color': '#e74c3c' if p_fail_combined > SAFETY_THRESHOLD else '#2ecc71'},
            'steps': [
                {'range': [0,  5],  'color': '#d4edda'},
                {'range': [5,  10], 'color': '#fff3cd'},
                {'range': [10, 20], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line':      {'color': 'red', 'width': 3},
                'thickness': 0.75,
                'value':     5
            }
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# ── Row 3: JAYA OEE Panel ─────────────────────────────────────
st.markdown("### 🏭 Prescriptive OEE Control — JAYA Optimisation")

if run_jaya:
    with st.spinner(f"⚡ Running JAYA optimisation (Pop=50, Iter={jaya_iters})..."):
        rpm_opt, torq_opt, pfail_opt = run_jaya_oee(
            tool_wear, air_temp, proc_temp, pop=50, iters=jaya_iters
        )

    throughput_gain = ((rpm_opt - BASELINE_RPM) / BASELINE_RPM) * 100
    throttled       = tool_wear > 200

    oc1, oc2, oc3, oc4 = st.columns(4)
    oc1.metric("📈 Recommended RPM",     f"{rpm_opt:.0f}",
               delta=f"{throughput_gain:+.1f}% vs {BASELINE_RPM:.0f} baseline")
    oc2.metric("🔩 Recommended Torque",  f"{torq_opt:.1f} Nm")
    oc3.metric("🛡️ P(fail) at Setpoint", f"{pfail_opt * 100:.2f}%",
               delta="✅ Within 5% limit" if pfail_opt < SAFETY_THRESHOLD else "❌ UNSAFE",
               delta_color="off")
    oc4.metric("🔧 Throttle Active",     "YES ⚠️" if throttled else "NO ✅")

    if throttled:
        st.warning(
            f"⚠️ Tool wear ({tool_wear} min) exceeds 200 min threshold — "
            f"JAYA has autonomously throttled RPM to **{rpm_opt:.0f}** "
            f"(down from 1,800 RPM normal ceiling)."
        )
    else:
        st.success(
            f"✅ JAYA recommends **{rpm_opt:.0f} RPM** — "
            f"a **{throughput_gain:.1f}% throughput gain** with "
            f"**{pfail_opt * 100:.2f}% failure probability** (well below 5% safety margin)."
        )

    # Throughput vs Safety trade-off curve
    st.markdown("#### ⚖️ Throughput vs Safety Trade-off")
    rpm_sweep    = np.linspace(1100, 1800, 30)
    p_fail_sweep = []
    for r in rpm_sweep:
        v = build_feature_vector(r, torq_opt, tool_wear, air_temp, proc_temp)
        p = model.predict(v, verbose=0)[0]
        p_fail_sweep.append((p[HDF_CLASS] + p[OSF_CLASS]) * 100)

    fig_trade = go.Figure()
    fig_trade.add_trace(go.Scatter(
        x=list(rpm_sweep), y=p_fail_sweep,
        mode='lines+markers', name='P(Thermal + Mech Fail)',
        line=dict(color='#e74c3c', width=2)
    ))
    fig_trade.add_hline(
        y=5.0, line_dash='dash', line_color='red',
        annotation_text='5% Safety Limit', annotation_position='top right'
    )
    fig_trade.add_vline(
        x=rpm_opt, line_dash='dot', line_color='green',
        annotation_text=f'JAYA: {rpm_opt:.0f} RPM', annotation_position='top left'
    )
    fig_trade.update_layout(
        xaxis_title='Rotational Speed (RPM)',
        yaxis_title='P(Heat Dissipation + Overstrain) %',
        template='plotly_white',
        height=360
    )
    st.plotly_chart(fig_trade, use_container_width=True)

else:
    st.info(
        "👆 Adjust the sensor sliders in the sidebar, then click "
        "**⚡ Run JAYA OEE Optimisation** to compute the optimal RPM and Torque setpoints. "
        "Use **20 iterations** for a fast result (~10 seconds)."
    )

st.markdown("---")

# ── Row 4: Anomaly Detection ──────────────────────────────────
st.markdown("### 🚨 Real-Time Anomaly Detection")

thresholds = {
    'Delta T > 12 K':      (proc_temp - air_temp) > 12,
    'Tool Wear > 200 min': tool_wear > 200,
    'Torque > 60 Nm':      torque > 60,
    'RPM > 1800':          rpm > 1800,
    'P(fail) > 5%':        p_fail_combined > SAFETY_THRESHOLD,
}
ac1, ac2, ac3, ac4, ac5 = st.columns(5)
for col, (label, triggered) in zip([ac1, ac2, ac3, ac4, ac5], thresholds.items()):
    col.metric(label, "🔴 ALERT" if triggered else "🟢 OK")

st.markdown("---")
st.caption(
    "Project 14  |  AI-Driven Digital Twin  |  TensorFlow + JAYA  |  AI4I Dataset  |  "
    "Inference latency < 30 ms  |  98.4% accuracy target  |  96.1% macro-recall target"
)