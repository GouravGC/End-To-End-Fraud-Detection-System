import streamlit as st
import pandas as pd

from src.constants.paths import FEATURE_NAMES_FILE
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils import load_object


st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detection System")

with st.expander("ℹ️ About the Input Features"):
    st.write("""
The dataset has been anonymized using Principal Component Analysis (PCA).

- V1–V28 are transformed numerical features.
- Their original meanings are not publicly available.
- Time represents elapsed transaction time.
- Amount represents the transaction amount.
""")

st.markdown("""
Predict whether a credit card transaction is **Legitimate** or **Fraudulent**
using a trained Machine Learning model.
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

col1, col2 = st.columns([2, 1])

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
