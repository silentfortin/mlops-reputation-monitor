from transformers import pipeline, AutoTokenizer
from src.test_data import get_preprocessed_df

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest", use_fast=True)

df = get_preprocessed_df()
texts_to_infer = df['processed_text'].tolist() if not df.empty else []

classifier = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
    tokenizer=tokenizer,
    )

#results = classifier(texts_to_infer)
#print(results)