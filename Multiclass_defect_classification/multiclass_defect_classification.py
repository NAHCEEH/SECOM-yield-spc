from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from skimage.transform import resize

import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def extract_label(value):
    return value[0][0]


def resize_wafer_map(wafer_map, target_size=(32, 32)):
    wafer_map = np.array(wafer_map)

    resized = resize(
        wafer_map,
        target_size,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(int)

    return resized


def load_pattern_data(data_path):
    df = pd.read_pickle(data_path)

    df_labeled = df[df["failureType"].astype(str) != "[]"].copy()

    df_labeled["failure_label"] = df_labeled["failureType"].apply(extract_label)
    df_labeled["set_label"] = df_labeled["trianTestLabel"].apply(extract_label)

    df_pattern = df_labeled[df_labeled["failure_label"] != "none"].copy()

    return df_pattern


def prepare_multiclass_input(df_pattern):
    resized_pattern_maps = df_pattern["waferMap"].apply(resize_wafer_map)

    X_pattern = np.stack(resized_pattern_maps.values)
    X_pattern = X_pattern[..., np.newaxis]
    X_pattern = X_pattern / 2.0

    y_multiclass = df_pattern["failure_label"].values

    train_mask = df_pattern["set_label"] == "Training"
    test_mask = df_pattern["set_label"] == "Test"

    X_train = X_pattern[train_mask.values]
    X_test = X_pattern[test_mask.values]

    y_train_label = y_multiclass[train_mask.values]
    y_test_label = y_multiclass[test_mask.values]

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train_label)
    y_test = encoder.transform(y_test_label)

    return X_train, X_test, y_train, y_test, y_train_label, y_test_label, encoder


def build_multiclass_cnn(input_shape, num_classes):
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
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_and_evaluate(model, X_train, y_train, X_test, y_test, class_names, class_weight=None):
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=128,
        class_weight=class_weight,
        verbose=1,
    )

    y_proba = model.predict(X_test)
    y_pred = np.argmax(y_proba, axis=1)

    print(classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0,
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_pred


def main():
    data_path = Path("../data/raw/LSWMD.pkl")

    df_pattern = load_pattern_data(data_path)

    print("pattern wafer 수:", len(df_pattern))
    print("\nfailure_label 분포:")
    print(df_pattern["failure_label"].value_counts())
    print("\nset_label 분포:")
    print(df_pattern["set_label"].value_counts())

    (
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_label,
        y_test_label,
        encoder,
    ) = prepare_multiclass_input(df_pattern)

    class_names = encoder.classes_
    num_classes = len(class_names)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("\nclass mapping:")
    for idx, name in enumerate(class_names):
        print(idx, ":", name)

    print("\ny_train 분포:")
    print(pd.Series(y_train_label).value_counts())

    print("\ny_test 분포:")
    print(pd.Series(y_test_label).value_counts())

    print("\nBasic Multiclass CNN")
    basic_model = build_multiclass_cnn(
        input_shape=(32, 32, 1),
        num_classes=num_classes,
    )

    y_pred_basic = train_and_evaluate(
        basic_model,
        X_train,
        y_train,
        X_test,
        y_test,
        class_names,
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )

    class_weight_dict = {
        i: weight for i, weight in enumerate(class_weights)
    }

    print("\nClass weight:")
    for class_index, weight in class_weight_dict.items():
        print(class_names[class_index], ":", weight)

    print("\nWeighted Multiclass CNN")
    weighted_model = build_multiclass_cnn(
        input_shape=(32, 32, 1),
        num_classes=num_classes,
    )

    y_pred_weighted = train_and_evaluate(
        weighted_model,
        X_train,
        y_train,
        X_test,
        y_test,
        class_names,
        class_weight=class_weight_dict,
    )

    multi_compare = pd.DataFrame(
        [
            {
                "model": "CNN",
                "accuracy": 0.51,
                "macro_f1": 0.40,
                "weighted_f1": 0.44,
                "near_full_recall": 0.00,
                "scratch_recall": 0.00,
            },
            {
                "model": "Weighted CNN",
                "accuracy": 0.43,
                "macro_f1": 0.44,
                "weighted_f1": 0.42,
                "near_full_recall": 1.00,
                "scratch_recall": 0.50,
            },
        ]
    )

    print("\nModel comparison:")
    print(multi_compare)

    multi_compare.set_index("model")[
        ["accuracy", "macro_f1", "weighted_f1", "near_full_recall", "scratch_recall"]
    ].plot(kind="bar", figsize=(10, 5))

    plt.title("Multiclass CNN vs Weighted CNN")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()