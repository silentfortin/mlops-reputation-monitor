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
        st.write(res.json())
