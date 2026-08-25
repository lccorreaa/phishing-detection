import os
import string
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re

# Display terminal output without truncation
pd.set_option("display.max_colwidth", None)

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

#--------------------------------------------------------------------

def strip_html(text):
    """Strip HTML tags from text using BeautifulSoup."""
    return BeautifulSoup(str(text), "html.parser").get_text()


def extract_structural_features(df, text_col='body'):
    """Extract signal from raw text BEFORE cleaning strips it away."""
    df['num_links'] = df[text_col].astype(str).apply(
        lambda x: len(re.findall(r'http[s]?://\S+', x))
    )
    df['has_html'] = df[text_col].astype(str).str.contains('<html|<body|<a href', case=False)
    df['num_exclamations'] = df[text_col].astype(str).str.count('!')
    return df


def clean_text(text):
    text = strip_html(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def tokenize_and_lemmatize(text):
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_dataframe(df, text_col='body'):
    """Full pipeline: structural features -> clean -> tokenize -> final text column."""
    df = extract_structural_features(df, text_col)
    df['body_clean'] = df[text_col].apply(clean_text)
    df['tokens'] = df['body_clean'].apply(tokenize_and_lemmatize)
    df['body_final'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))
    return df


def save_processed(df, path="data/processed/", filename="CEAS_08_processed.csv"):
    # Saves the processed dataframe to data/processed/, creating the folder if needed.
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    df.to_csv(full_path, index=False)
    print(f"Saved processed data to '{full_path}'")


if __name__ == "__main__":
    df = pd.read_csv("data/raw/CEAS_08.csv")
    df = preprocess_dataframe(df, text_col='body')  # adjust to your actual column name
    save_processed(df)