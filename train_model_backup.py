
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tickets.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 1. Load and validate the dataset
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["ticket_text", "category"])

df["ticket_text"] = df["ticket_text"].astype(str).str.strip()
df["category"] = df["category"].astype(str).str.strip()
df = df[(df["ticket_text"] != "") & (df["category"] != "")]

if df["category"].nunique() < 2:
    raise ValueError("The dataset must contain at least two categories.")

if df["category"].value_counts().min() < 2:
    raise ValueError("Each category needs at least two examples.")

X = df["ticket_text"]
y = df["category"]

# 2. Split data before training to test on unseen examples
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# 3. Build an NLP + machine learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            strip_accents="unicode",
        ),
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000, random_state=42),
    ),
])

# 4. Train the model
model.fit(X_train, y_train)

# 5. Evaluate it on held-out test examples
predictions = model.predict(X_test)

print("\n=== DATASET SUMMARY ===")
print(f"Total examples: {len(df)}")
print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")

print("\n=== CATEGORY CLASSIFICATION REPORT ===")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)

# 6. Save the trained pipeline
model_path = MODEL_DIR / "ticket_classifier.joblib"
joblib.dump(model, model_path)

print(f"\nModel saved successfully: {model_path}")
print("Training completed!")
