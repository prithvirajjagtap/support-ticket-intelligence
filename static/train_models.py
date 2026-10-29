
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tickets_clean.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["ticket_text", "category", "priority"])

X = df["ticket_text"].astype(str)
y_category = df["category"].astype(str)
y_priority = df["priority"].astype(str)

X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = (
    train_test_split(
        X,
        y_category,
        y_priority,
        test_size=0.20,
        random_state=42,
        stratify=y_category,
    )
)

algorithms = {
    "Logistic Regression": LogisticRegression(
        max_iter=1500, class_weight="balanced", random_state=42
    ),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(
        class_weight="balanced", random_state=42
    ),
}


def build_pipeline(classifier):
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_features=100000,
                sublinear_tf=True,
            ),
        ),
        ("classifier", classifier),
    ])


def evaluate_models(X_train, X_test, y_train, y_test, task_name):
    results = []
    trained_models = {}

    print(f"\n{'=' * 65}")
    print(f"{task_name.upper()} MODEL COMPARISON")
    print(f"{'=' * 65}")

    for name, classifier in algorithms.items():
        pipeline = build_pipeline(classifier)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )

        results.append({
            "algorithm": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        })
        trained_models[name] = pipeline

        print(
            f"{name}: Accuracy={accuracy:.4f}, "
            f"Macro F1={macro_f1:.4f}, "
            f"Weighted F1={weighted_f1:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        "macro_f1", ascending=False
    )
    print("\nDetailed results:")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["algorithm"]
    best_model = trained_models[best_name]

    print(f"\nSelected model: {best_name}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        best_model.predict(X_test),
        zero_division=0,
    ))

    return best_model, results_df


category_model, category_results = evaluate_models(
    X_train, X_test, y_cat_train, y_cat_test, "Category"
)

priority_model, priority_results = evaluate_models(
    X_train, X_test, y_pri_train, y_pri_test, "Priority"
)

joblib.dump(category_model, MODEL_DIR / "ticket_classifier.joblib")
joblib.dump(priority_model, MODEL_DIR / "priority_classifier.joblib")

category_results.to_csv(
    MODEL_DIR / "category_model_comparison.csv", index=False
)
priority_results.to_csv(
    MODEL_DIR / "priority_model_comparison.csv", index=False
)

print("\nTraining complete!")
print("Category model:", MODEL_DIR / "ticket_classifier.joblib")
print("Priority model:", MODEL_DIR / "priority_classifier.joblib")
print("Comparison reports saved in:", MODEL_DIR)
