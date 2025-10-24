import streamlit as st
import pandas as pd
import requests

BACKEND_URL = "http://localhost:8000"

st.title("Sentiment Analysis")

st.markdown("### Single Text Prediction")
text = st.text_area("Insert here a short text:")
if st.button("Analyze"):
    if text.strip():
        payload = {"text": text}
        with st.spinner("Analyzing..."):
            try:
                res = requests.post(f"{BACKEND_URL}/predict", json=payload)
                res.raise_for_status()
                data = res.json()
                st.dataframe(pd.DataFrame([data]))
                st.success("Prediction complete!")
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.header("Batch Prediction from CSV")
text_column = st.text_area("Insert the text column name:")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None and st.button("Predict file"):
    if text_column.strip():
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
        data = {"text_column": text_column}
        with st.spinner("Processing... Please wait"):
            try:
                response = requests.post(f"{BACKEND_URL}/predict_csv", files=files, data=data)
                response.raise_for_status()
                data = response.json()
                df = pd.DataFrame(data)
                if "text" in df.columns:
                    df["Text"] = df["text"]
                    df = df[["Text", "label", "score"]]
                    df = df.rename(columns={"label": "Label", "score": "Score"})
                st.dataframe(df)
                st.success("Batch prediction complete!")
            except Exception as e:
                st.error(f"Error during batch prediction: {e}")
    else:
        st.warning("Please specify the text column name.")