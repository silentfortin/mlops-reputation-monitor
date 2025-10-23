import kagglehub
import re
import pandas as pd
import string
from kagglehub import KaggleDatasetAdapter

file_path = "test_data.csv"
# column_names = ["sentiment", "id", "date", "query", "user", "text"]

try:
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "krishbaisoya/tweets-sentiment-analysis",
        file_path,
        pandas_kwargs={"encoding": "latin1"},
    )
    if df.empty:
        print("Dataset [translate:empty], proceeding to download or handle it\n")
        # code to download or reload goes here
    else:
        print("Dataset loaded successfully\n")
except Exception as e:
    print(f"Dataset loading error: {e}")
    # code to download or handle the problem goes here

# print the first 5 records
# print("First 5 records:", df.head())

# Pre-compile regexes
emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\u2702-\u27B0"
    u"\u24C2-\U0001F251"
    "]+", flags=re.UNICODE)

url_pattern = re.compile(r'((www\.[^\s]+)|(https?://[^\s]+))')
user_pattern = re.compile(r'@[^\s]+')
hashtag_pattern = re.compile(r'#([^\s]+)')
short_words_pattern = re.compile(r'\W*\b\w{1,3}\b')

# Func for text preprocessing
def preprocess_text_series(text_series: pd.Series) -> pd.Series:
    # Convert to lowercase
    text_series = text_series.str.lower()
    # Remove emojis
    text_series = text_series.str.replace(emoji_pattern, '', regex=True)
    # Remove URLs
    text_series = text_series.str.replace(url_pattern, '', regex=True)
    # Remove usernames
    text_series = text_series.str.replace(user_pattern, '', regex=True)
    # Replace hashtags with the word only
    text_series = text_series.str.replace(hashtag_pattern, r'\1', regex=True)
    # Fix multiple whitespaces with single space
    text_series = text_series.str.replace(r'\s+', ' ', regex=True)
    # Remove words shorter than 4 letters
    text_series = text_series.str.replace(short_words_pattern, '', regex=True)
    # Strip leading/trailing spaces **and** punctuation
    text_series = text_series.str.strip(string.punctuation + " \t\n\r")
    # Fill NaNs with empty string
    text_series = text_series.fillna('')
    
    return text_series

df['processed_text'] = preprocess_text_series(df['sentence'])
print(df['processed_text'])