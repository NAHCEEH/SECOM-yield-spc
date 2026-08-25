from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

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


def build_cnn_model(input_shape):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def evaluate_binary_model(model_name, y_true, y_pred, target_names):
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


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


def plot_threshold_performance(threshold_results):
    threshold_results.set_index("threshold")[
        ["pattern_precision", "pattern_recall", "pattern_f1"]
    ].plot(marker="o", figsize=(8, 5))

    plt.title("CNN Threshold Performance")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
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

    model = build_cnn_model(input_shape=(32, 32, 1))
    model.summary()

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=256,
        verbose=1,
    )

    y_proba = model.predict(X_test).ravel()

    y_pred_default = (y_proba >= 0.5).astype(int)

    evaluate_binary_model(
        "CNN threshold=0.5",
        y_test,
        y_pred_default,
        target_names,
    )

    threshold_results = compare_thresholds(
        y_test,
        y_proba,
        thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )

    print("\nCNN threshold comparison:")
    print(threshold_results)

    plot_threshold_performance(threshold_results)

    selected_threshold = 0.75
    y_pred_selected = (y_proba >= selected_threshold).astype(int)

    evaluate_binary_model(
        f"CNN threshold={selected_threshold}",
        y_test,
        y_pred_selected,
        target_names,
    )

    selected_result = pd.DataFrame(
        [
            {
                "model": "CNN",
                "threshold": selected_threshold,
                "accuracy": accuracy_score(y_test, y_pred_selected),
                "pattern_precision": precision_score(
                    y_test, y_pred_selected, pos_label=1, zero_division=0
                ),
                "pattern_recall": recall_score(y_test, y_pred_selected, pos_label=1),
                "pattern_f1": f1_score(y_test, y_pred_selected, pos_label=1),
                "predicted_pattern_count": np.sum(y_pred_selected == 1),
            }
        ]
    )

    print("\nSelected CNN result:")
    print(selected_result)


if __name__ == "__main__":
    main()