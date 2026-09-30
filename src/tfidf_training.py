"""Train and save a TF-IDF + logistic regression phishing classifier."""

from pathlib import Path
import re

import joblib
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "CEAS_08.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "tfidf_logistic_regression.joblib"


def clean_email_body(body: object) -> str:
    """Remove HTML markup and normalize whitespace before vectorizing text."""
    text = "" if pd.isna(body) else str(body)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    print(f"Loading raw dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = {"body", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Prepare raw text and labels in this script so users need only run one file.
    df = df.dropna(subset=["label"]).copy()
    X = df["body"].apply(clean_email_body)
    y = df["label"].astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("The label column must contain only 0 (legitimate) and 1 (malicious).")

    print(f"Prepared {len(df):,} records from the raw dataset.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=100_000,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Training records: {len(X_train):,}")
    print(f"Held-out records: {len(X_test):,}")
    print("\nClassification report (class 1 = malicious):")
    print(classification_report(y_test, predictions, target_names=["legitimate", "malicious"]))
    print("Confusion matrix (rows=true, columns=predicted; labels 0, 1):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
