from pathlib import Path

import pandas as pd


def load_secom_data(data_dir):
    X = pd.read_csv(
        data_dir / "secom.data",
        sep=r"\s+",
        header=None,
        na_values="NaN",
    )

    y = pd.read_csv(
        data_dir / "secom_labels.data",
        sep=r"\s+",
        header=None,
    )

    return X, y


def preprocess_secom(X, missing_threshold=0.5):
    missing_ratio = X.isna().mean()
    high_missing_features = missing_ratio[missing_ratio >= missing_threshold].index

    X_reduced = X.drop(columns=high_missing_features)
    X_clean = X_reduced.fillna(X_reduced.median())

    feature_variance = X_clean.var()
    zero_variance_features = feature_variance[feature_variance == 0].index

    X_clean = X_clean.drop(columns=zero_variance_features)

    preprocessing_summary = {
        "original_shape": X.shape,
        "after_missing_feature_removal_shape": X_reduced.shape,
        "final_shape": X_clean.shape,
        "removed_high_missing_features": list(high_missing_features),
        "removed_zero_variance_features": list(zero_variance_features),
        "final_missing_values": int(X_clean.isna().sum().sum()),
    }

    return X_clean, preprocessing_summary


if __name__ == "__main__":
    data_dir = Path("data/raw")

    X, y = load_secom_data(data_dir)
    X_clean, summary = preprocess_secom(X)

    print("SECOM preprocessing summary")
    print("--------------------------")
    print("Original X shape:", summary["original_shape"])
    print(
        "After high-missing feature removal:",
        summary["after_missing_feature_removal_shape"],
    )
    print("Final X_clean shape:", summary["final_shape"])
    print("Final missing values:", summary["final_missing_values"])
    print(
        "Removed high-missing features:",
        len(summary["removed_high_missing_features"]),
    )
    print(
        "Removed zero-variance features:",
        len(summary["removed_zero_variance_features"]),
    )
    print("y shape:", y.shape)