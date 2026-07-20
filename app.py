import streamlit as st
import pandas as pd

from src.constants.paths import FEATURE_NAMES_FILE
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils import load_object


st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

/* Remove top padding */
.block-container{
    padding-top:2rem;
}

/* Background */
.stApp{
    background:linear-gradient(135deg,#0f172a,#111827,#1e293b);
}

/* Main title */
.main-title{
    font-size:48px;
    font-weight:800;
    color:white;
    margin-bottom:0px;
}

.sub-title{
    color:#cbd5e1;
    font-size:20px;
}

/* Glass Cards */

.glass{
background:rgba(255,255,255,0.06);
border:1px solid rgba(255,255,255,0.1);
backdrop-filter: blur(16px);
padding:28px;
border-radius:20px;
box-shadow:0 10px 30px rgba(0,0,0,.35);
margin-bottom:20px;
}

/* Prediction Cards */

.success-card{

background:#052e16;
border-left:8px solid #22c55e;
padding:25px;
border-radius:15px;
font-size:24px;
font-weight:700;
color:white;

}

.error-card{

background:#450a0a;
border-left:8px solid #ef4444;
padding:25px;
border-radius:15px;
font-size:24px;
font-weight:700;
color:white;

}

.metric-card{

background:#1e293b;
padding:18px;
border-radius:15px;
text-align:center;
border:1px solid #334155;

}

/* Sidebar */

section[data-testid="stSidebar"]{
background:#111827;
}

/* Button */

.stButton>button{

background:#2563eb;
color:white;
font-size:20px;
font-weight:bold;
border-radius:12px;
height:60px;
width:100%;
border:none;
transition:0.3s;

}

.stButton>button:hover{

background:#1d4ed8;
transform:scale(1.02);

}

</style>
""",unsafe_allow_html=True)
st.markdown("""

<div class="glass">

<div class="main-title">

💳 Credit Card Fraud Detection AI

</div>

<div class="sub-title">

Real-Time Machine Learning Powered Transaction Risk Analysis

</div>

<br>

Detect suspicious credit card transactions instantly using a trained Machine Learning model.
The PCA features (V1-V28) are anonymized representations of the original transaction data.

</div>

""",unsafe_allow_html=True)

with st.expander("📘 Understanding the Input Features"):

    st.write("""

### Feature Information

- **V1 – V28** are anonymized numerical features generated using PCA.

- Their original meaning is intentionally hidden to protect user privacy.

- **Time** represents the elapsed seconds from the first transaction.

- **Amount** is the monetary value of the transaction.

These features are passed to a trained Machine Learning model which predicts whether the transaction is **Legitimate** or **Fraudulent**.

""")

@st.cache_resource
def get_prediction_pipeline() -> PredictionPipeline:
    return PredictionPipeline()


@st.cache_resource
def get_feature_names() -> list[str]:
    return load_object(FEATURE_NAMES_FILE)


pipeline = get_prediction_pipeline()
feature_names = get_feature_names()

st.sidebar.header("📝 Transaction Details")
st.sidebar.info(
    "Enter the transaction details below. "
    "Features V1–V28 are anonymized PCA components."
)

input_values = {}
for feature_name in feature_names:

    display_name = feature_name

    if feature_name.startswith("V"):
        display_name = f"PCA Feature {feature_name}"

    elif feature_name == "Amount":
        display_name = "Transaction Amount"

    elif feature_name == "Time":
        display_name = "Transaction Time"

    input_values[feature_name] = st.sidebar.number_input(
        display_name,
        value=0.0,
        format="%f"
    )

input_df = pd.DataFrame([input_values])

col1, col2 = st.columns([1.7, 1.3])

with col1:
    st.subheader("📋 Transaction Summary")
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("📊 Prediction Result")
    if st.button("🔍 Predict Transaction"):
        result = pipeline.predict(input_df)
        prediction = int(result["prediction"][0])
        probability = result.get("probability")
        fraud_probability = float(probability[0]) if probability is not None else None

        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")

        if fraud_probability is not None:
            st.metric("Fraud Probability",f"{fraud_probability*100:.2f}%"
)
