
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tickets_clean.csv"
OUTPUT_DIR = BASE_DIR / "models" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH).dropna(
    subset=["ticket_text", "category", "priority"]
)

X = df["ticket_text"].astype(str)
y_category = df["category"].astype(str)
y_priority = df["priority"].astype(str)

# Use the same stratified split settings as the original comparison.
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


def build_model():
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
        (
            "classifier",
            LinearSVC(class_weight="balanced", random_state=42),
        ),
    ])


def evaluate_task(task_name, y_train, y_test):
    print(f"\n{'=' * 60}")
    print(f"{task_name.upper()} EVALUATION")
    print(f"{'=' * 60}")

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nClassification report:")
    report = classification_report(
        y_test, predictions, zero_division=0
    )
    print(report)

    safe_name = task_name.lower().replace(" ", "_")

    # Save a text report.
    report_path = OUTPUT_DIR / f"{safe_name}_report.txt"
    report_path.write_text(
        f"{task_name} evaluation\n"
        f"Test samples: {len(y_test)}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n\n"
        f"{report}",
        encoding="utf-8",
    )

    # Save a confusion matrix.
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=model.classes_,
        xticks_rotation=45,
        cmap="Blues",
        values_format="d",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"{task_name} — Held-Out Test Set")
    fig.tight_layout()
    chart_path = OUTPUT_DIR / f"{safe_name}_confusion_matrix.png"
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "task": task_name,
        "test_samples": len(y_test),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report_file": str(report_path),
        "confusion_matrix": str(chart_path),
    }


results = [
    evaluate_task("Category", y_cat_train, y_cat_test),
    evaluate_task("Priority", y_pri_train, y_pri_test),
]

summary = pd.DataFrame(results)
summary.to_csv(OUTPUT_DIR / "evaluation_summary.csv", index=False)

print(f"\nEvaluation complete. Files saved in:\n{OUTPUT_DIR}")
print(summary[
    ["task", "test_samples", "accuracy", "macro_f1", "weighted_f1"]
].to_string(index=False))
