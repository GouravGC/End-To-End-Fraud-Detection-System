import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go

from src.constants.paths import FEATURE_NAMES_FILE
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils import load_object

# --- 1. INITIAL SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CACHED MODEL & FEATURE LOADERS ---
@st.cache_resource
def get_prediction_pipeline() -> PredictionPipeline:
    return PredictionPipeline()

@st.cache_resource
def get_feature_names() -> list[str]:
    return load_object(FEATURE_NAMES_FILE)

pipeline = get_prediction_pipeline()
feature_names = get_feature_names()

# --- 4. CUSTOM CSS ---
st.markdown("""
<style>
/* Remove top padding */
.block-container {
    padding-top: 2rem;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
}

/* Main title */
.main-title {
    font-size: 48px;
    font-weight: 800;
    color: white;
    margin-bottom: 0px;
}

.sub-title {
    color: #cbd5e1;
    font-size: 20px;
}

/* Glass Cards */
.glass {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(16px);
    padding: 28px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,.35);
    margin-bottom: 20px;
}

/* Prediction Result Cards */
.success-card {
    background: #052e16;
    border-left: 8px solid #22c55e;
    padding: 25px;
    border-radius: 15px;
    font-size: 24px;
    font-weight: 700;
    color: white;
}

.error-card {
    background: #450a0a;
    border-left: 8px solid #ef4444;
    padding: 25px;
    border-radius: 15px;
    font-size: 24px;
    font-weight: 700;
    color: white;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #111827;
}

/* Primary Button styling */
.stButton>button {
    background: #2563eb;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    height: 60px;
    width: 100%;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    background: #1d4ed8;
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)

# --- 5. HEADER BANNER ---
st.markdown("""
<div class="glass">
    <div class="main-title">💳 Credit Card Fraud Detection AI</div>
    <div class="sub-title">Real-Time Machine Learning Powered Transaction Risk Analysis</div>
    <br>
    Detect suspicious credit card transactions instantly using a trained Machine Learning model.
    The PCA features (V1-V28) are anonymized representations of the original transaction data.
</div>
""", unsafe_allow_html=True)

# --- 6. TOP METRIC DASHBOARD ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("🤖 Model", "Online")

with kpi2:
    st.metric("📊 Features Loaded", len(feature_names))

with kpi3:
    st.metric("⚡ Status", "Ready")

with kpi4:
    st.metric("🛡️ Security", "Active")

with st.expander("📘 Understanding the Input Features"):
    st.write("""
### Feature Information
- **V1 – V28** are anonymized numerical features generated using PCA to protect user privacy.
- **Time** represents the elapsed seconds from the first recorded transaction in the dataset.
- **Amount** is the total monetary value of the transaction.

These inputs are fed into the machine learning pipeline to classify the transaction as **Legitimate** or **Fraudulent**.
""")

# --- 7. SIDEBAR INPUTS ---
st.sidebar.markdown("""
# 💳 Transaction Inputs
---
Enter transaction features below and click **Predict Transaction**.
""")

input_values = {}

for feature_name in feature_names:
    display_name = feature_name
    if feature_name.startswith("V"):
        display_name = f"PCA Feature {feature_name}"
    elif feature_name == "Amount":
        display_name = "Transaction Amount"
    elif feature_name == "Time":
        display_name = "Transaction Time"

    with st.sidebar.expander(display_name, expanded=False):
        input_values[feature_name] = st.number_input(
            label=display_name,
            value=0.0,
            format="%f",
            key=f"input_{feature_name}"
        )

input_df = pd.DataFrame([input_values])

# --- 8. MAIN DISPLAY AREA ---
col1, col2 = st.columns([1.7, 1.3])

with col1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📋 Transaction Summary")
    st.caption("Review the active parameters before running evaluation.")
    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass">
        <h2>📊 Prediction Result</h2>
        Click below to run the AI classification pipeline.
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔍 Predict Transaction"):
        with st.spinner("🤖 AI is analyzing the transaction..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.003)
                progress.progress(i + 1)
            progress.empty()

        # Run Prediction
        result = pipeline.predict(input_df)
        prediction = int(result["prediction"][0])
        
        # Safely parse probability format
        fraud_probability = None
        if "probability" in result and result["probability"] is not None:
            raw_prob = result["probability"][0]
            if isinstance(raw_prob, (list, tuple, pd.Series)):
                fraud_probability = float(raw_prob[-1])
            else:
                fraud_probability = float(raw_prob)

        # Output Alert Card
        if prediction == 1:
            st.markdown("""
            <div class="error-card">
                🚨 FRAUD DETECTED
                <br><br>
                This transaction exhibits high anomaly markers.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-card">
                ✅ LEGITIMATE TRANSACTION
                <br><br>
                No fraudulent indicators detected.
            </div>
            """, unsafe_allow_html=True)

        # Output Probability Metrics and Gauge
        if fraud_probability is not None:
            st.metric(
                "🎯 Fraud Risk Score",
                f"{fraud_probability * 100:.2f}%"
            )

            # Update Session History
            st.session_state.history.append({
                "Prediction": "Fraud" if prediction == 1 else "Legitimate",
                "Probability": round(fraud_probability * 100, 2)
            })
            st.session_state.history = st.session_state.history[-5:]

            # Plot Gauge Chart
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=fraud_probability * 100,
                    number={"suffix": "%"},
                    title={"text": "Risk Level"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2563eb"},
                        "steps": [
                            {"range": [0, 30], "color": "#22c55e"},
                            {"range": [30, 70], "color": "#facc15"},
                            {"range": [70, 100], "color": "#ef4444"},
                        ],
                    },
                )
            )

            fig.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )

            st.plotly_chart(fig, use_container_width=True)

            if fraud_probability < 0.30:
                st.success("🟢 Low Risk Level")
            elif fraud_probability < 0.70:
                st.warning("🟡 Medium Risk Level")
            else:
                st.error("🔴 Critical Risk Level")

        # Download Report Generator
        report = pd.DataFrame({
            "Prediction": ["Fraud" if prediction == 1 else "Legitimate"],
            "Fraud Risk (%)": [
                round(fraud_probability * 100, 2) if fraud_probability is not None else "N/A"
            ]
        })

        st.download_button(
            label="⬇ Download Prediction Report",
            data=report.to_csv(index=False),
            file_name="fraud_prediction_report.csv",
            mime="text/csv"
        )

# --- 9. HISTORY TABLE ---
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Recent Predictions (Last 5)")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

# --- 10. FOOTER ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#94a3b8;padding:15px;">
        💳 <b>Credit Card Fraud Detection AI</b><br>
        Built with Streamlit, Scikit-learn & Plotly<br><br>
        © 2026 Gourav Chhatwani
    </div>
    """,
    unsafe_allow_html=True
)