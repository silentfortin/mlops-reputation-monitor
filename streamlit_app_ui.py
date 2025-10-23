import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000" 

st.title("Sentiment Analysis")

text = st.text_area("Insert here a short text:")
if st.button("Analize"):
    if text.strip():
        payload = {"text": text}
        # Calling FastAPI
        res = requests.post(f"{BACKEND_URL}/predict", json=payload)
        data = res.json()

        # converting to df
        import pandas as pd
        df = pd.DataFrame([data])

        # show table
        st.table(df)

# Upload di CSV
st.header("Upload di file CSV")
uploaded_file = st.file_uploader("Choose a file", type=["csv"])
# Can be used wherever a "file-like" object is accepted:
if uploaded_file is not None and st.button("Predict file"):
    files = {'file': (uploaded_file.name, uploaded_file, 'text/csv')}
    response = requests.post(f"{BACKEND_URL}/predict_batch", files=files)