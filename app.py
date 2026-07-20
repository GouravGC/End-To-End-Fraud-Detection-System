import streamlit as st
import pandas as pd

from src.constants.paths import FEATURE_NAMES_FILE
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils import load_object


st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="wide")
st.title("Fraud Detection Prediction App")
st.caption("This application uses a pre-trained machine learning model to predict fraudulent transactions in real time.")

@st.cache_resource
def get_prediction_pipeline() -> PredictionPipeline:
    return PredictionPipeline()


@st.cache_resource
def get_feature_names() -> list[str]:
    return load_object(FEATURE_NAMES_FILE)


pipeline = get_prediction_pipeline()
feature_names = get_feature_names()

st.sidebar.header("Transaction Input")
st.sidebar.write("Enter the transaction features below.")

input_values = {}
for feature_name in feature_names:
    input_values[feature_name] = st.sidebar.number_input(feature_name, value=0.0, format="%f")

input_df = pd.DataFrame([input_values])

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Summary")
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("Prediction")
    if st.button("Predict Fraud Risk"):
        result = pipeline.predict(input_df)
        prediction = int(result["prediction"][0])
        probability = result.get("probability")
        fraud_probability = float(probability[0]) if probability is not None else None

        if prediction == 1:
            st.error("Prediction: Fraudulent transaction")
        else:
            st.success("Prediction: Legitimate transaction")

        if fraud_probability is not None:
            st.metric("Fraud Probability", f"{fraud_probability:.4f}")
