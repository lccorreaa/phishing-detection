# Phishing Email Detection

A college personal project exploring whether email body text can be used to distinguish malicious emails from legitimate ones. It trains a TF-IDF text representation with logistic regression and reports performance on a held-out portion of the dataset.

## Repository contents

```text
data/CEAS_08.csv                         Raw dataset used for training
models/tfidf_logistic_regression.joblib  Saved TF-IDF and classifier pipeline
notebooks/tests.ipynb                    Notebook for exploration
src/tfidf_training.py                    Data cleanup, training, evaluation, and saving
```

The training script uses only `body` as its input and `label` as its target. Labels are `0` for legitimate and `1` for malicious. The sender, receiver, date, and URL columns are not currently model inputs.

## Set up

From the repository root, create and activate a virtual environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train and evaluate

Run:

```powershell
python src/tfidf_training.py
```

The script reads `data/CEAS_08.csv`, removes HTML markup from email bodies, normalizes whitespace, and splits the labeled records into 80% training and 20% test data. The split is stratified and uses a fixed random seed (`42`) so the same split can be reproduced.

It prints a classification report (precision, recall, and F1) and a confusion matrix. In the matrix, rows are the true labels and columns are predicted labels, ordered as `0` (legitimate) and `1` (malicious).

With the current dataset and settings, the held-out set contains 7,831 emails. The confusion matrix is:

| Actual / Predicted | Legitimate (0) | Malicious (1) |
| --- | ---: | ---: |
| Legitimate (0) | 3,446 | 16 |
| Malicious (1) | 18 | 4,351 |

That is, the model correctly classified 3,446 legitimate emails and 4,351 malicious emails; it flagged 16 legitimate emails as malicious and missed 18 malicious emails. These numbers correspond to the fixed random split (`random_state=42`) and will change if the dataset or model settings change.

The trained pipeline is saved to:

```text
models/tfidf_logistic_regression.joblib
```

The pipeline includes both TF-IDF and logistic regression, so they can be loaded together for prediction. The text passed at prediction time should receive the same HTML and whitespace cleanup as training data.

## Model details

- **Text feature extraction:** TF-IDF with unigrams and bigrams, terms appearing in at least two documents, and a maximum of 100,000 features.
- **Classifier:** logistic regression with up to 1,000 iterations.
- **Evaluation:** stratified random 80/20 train/test split.

The current evaluation is a baseline. Emails from related senders or campaigns may occur in both sides of a random split, so results may not represent performance on entirely new campaigns or future emails.
