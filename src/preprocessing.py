# import re
# import string
# from bs4 import BeautifulSoup
# import nltk
# from nltk.stem import WordNetLemmatizer
import pandas as pd

df = pd.read_csv("data/raw/CEAS_08.csv")
pd.set_option("display.max_colwidth", None)
sample = df["body"].sample(1, random_state=42)
print(sample)
# nltk.download('punkt', quiet=True)
# nltk.download('wordnet', quiet=True)

# lemmatizer = WordNetLemmatizer()


# def extract_structural_features(df, text_col='body'):
#     """Extract signal from raw text BEFORE cleaning strips it away."""
#     df['num_links'] = df[text_col].astype(str).apply(
#         lambda x: len(re.findall(r'http[s]?://\S+', x))
#     )
#     df['has_html'] = df[text_col].astype(str).str.contains('<html|<body|<a href', case=False)
#     df['num_exclamations'] = df[text_col].astype(str).str.count('!')
#     return df


# def strip_html(text):
#     return BeautifulSoup(str(text), "html.parser").get_text()


# def clean_text(text):
#     text = strip_html(text)
#     text = text.lower()
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     return text


# def tokenize_and_lemmatize(text):
#     tokens = nltk.word_tokenize(text)
#     return [lemmatizer.lemmatize(t) for t in tokens]


# def preprocess_dataframe(df, text_col='body'):
#     """Full pipeline: structural features -> clean -> tokenize -> final text column."""
#     df = extract_structural_features(df, text_col)
#     df['body_clean'] = df[text_col].apply(clean_text)
#     df['tokens'] = df['body_clean'].apply(tokenize_and_lemmatize)
#     df['body_final'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))
#     return df

