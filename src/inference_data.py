from transformers import pipeline
from src.test_data import get_preprocessed_df

df = get_preprocessed_df()
texts_to_infer = df['processed_text'].tolist() if not df.empty else []

classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

#results = classifier(texts_to_infer)
#print(results)