from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def load_processed_data(processed_dir):
    X_train = np.load(processed_dir / "X_train_32.npy")
    X_test = np.load(processed_dir / "X_test_32.npy")
    y_train = np.load(processed_dir / "y_train_pattern.npy")
    y_test = np.load(processed_dir / "y_test_pattern.npy")
    label_info = pd.read_csv(processed_dir / "pattern_label_mapping.csv")

    return X_train, X_test, y_train, y_test, label_info


def flatten_images(X_train, X_test):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    return X_train_flat, X_test_flat


def evaluate_model(model_name, y_true, y_pred, target_names):
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def make_result_row(model_name, threshold, y_true, y_pred):
    return {
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "pattern_precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "pattern_recall": recall_score(y_true, y_pred, pos_label=1),
        "pattern_f1": f1_score(y_true, y_pred, pos_label=1),
        "predicted_pattern_count": np.sum(y_pred == 1),
    }


def compare_thresholds(y_true, y_proba, thresholds):
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "pattern_precision": precision_score(
                    y_true, y_pred, pos_label=1, zero_division=0
                ),
                "pattern_recall": recall_score(y_true, y_pred, pos_label=1),
                "pattern_f1": f1_score(y_true, y_pred, pos_label=1),
                "predicted_pattern_count": np.sum(y_pred == 1),
            }
        )

    return pd.DataFrame(results)


def plot_baseline_comparison(baseline_compare):
    baseline_compare.set_index("model")[
        ["pattern_precision", "pattern_recall", "pattern_f1"]
    ].plot(kind="bar", figsize=(8, 5))

    plt.title("Baseline Model Comparison for Pattern Detection")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    processed_dir = Path("../data/processed")

    X_train, X_test, y_train, y_test, label_info = load_processed_data(processed_dir)
    target_names = label_info["class_name"].tolist()

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("\nLabel mapping:")
    print(label_info)

    X_train_flat, X_test_flat = flatten_images(X_train, X_test)

    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    logistic_model.fit(X_train_flat, y_train)
    y_pred_logistic = logistic_model.predict(X_test_flat)

    evaluate_model(
        "Logistic Regression",
        y_test,
        y_pred_logistic,
        target_names,
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(X_train_flat, y_train)

    y_proba_rf = rf_model.predict_proba(X_test_flat)[:, 1]

    threshold_results = compare_thresholds(
        y_test,
        y_proba_rf,
        thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
    )

    print("\nRandom Forest threshold comparison:")
    print(threshold_results)

    best_f1_row = threshold_results.loc[threshold_results["pattern_f1"].idxmax()]
    best_threshold = best_f1_row["threshold"]

    y_pred_rf_best = (y_proba_rf >= best_threshold).astype(int)

    evaluate_model(
        f"Random Forest threshold={best_threshold}",
        y_test,
        y_pred_rf_best,
        target_names,
    )

    baseline_compare = pd.DataFrame(
        [
            make_result_row(
                "Logistic Regression",
                "default",
                y_test,
                y_pred_logistic,
            ),
            make_result_row(
                "Random Forest",
                best_threshold,
                y_test,
                y_pred_rf_best,
            ),
        ]
    )

    print("\nBaseline model comparison:")
    print(baseline_compare)

    plot_baseline_comparison(baseline_compare)


if __name__ == "__main__":
    main()
    
    
# 실행 시 
# cd baseline_modeling
#python3 baseline_modeling.py