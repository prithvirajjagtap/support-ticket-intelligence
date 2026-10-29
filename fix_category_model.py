from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tickets_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "ticket_classifier.joblib"

df = pd.read_csv(DATA_PATH)

X = df["ticket_text"].astype(str)
y = df["category"].astype(str)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1,2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True
    )),
    ("classifier", LinearSVC(class_weight="balanced", random_state=42))
])

model.fit(X, y)

joblib.dump(model, MODEL_PATH)

print("Category model fixed!")
print("Labels:", model.classes_)