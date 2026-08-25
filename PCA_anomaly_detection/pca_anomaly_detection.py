from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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

    return X_clean


def scale_features(X_clean):
    scaler = StandardScaler()

    X_scaled_array = scaler.fit_transform(X_clean)

    X_scaled = pd.DataFrame(
        X_scaled_array,
        columns=X_clean.columns,
        index=X_clean.index,
    )

    return X_scaled, X_scaled_array


def run_pca_reconstruction(X_scaled_array, explained_variance=0.90):
    pca = PCA(n_components=explained_variance, random_state=42)

    X_pca = pca.fit_transform(X_scaled_array)
    X_reconstructed = pca.inverse_transform(X_pca)

    reconstruction_error = ((X_scaled_array - X_reconstructed) ** 2).mean(axis=1)

    pca_info = {
        "original_shape": X_scaled_array.shape,
        "compressed_shape": X_pca.shape,
        "reconstructed_shape": X_reconstructed.shape,
        "n_components": pca.n_components_,
        "explained_variance_ratio": pca.explained_variance_ratio_.sum(),
        "scaled_nan_count": int(np.isnan(X_scaled_array).sum()),
        "scaled_inf_count": int(np.isinf(X_scaled_array).sum()),
        "reconstructed_nan_count": int(np.isnan(X_reconstructed).sum()),
        "reconstructed_inf_count": int(np.isinf(X_reconstructed).sum()),
    }

    return reconstruction_error, pca_info


def create_pca_result(y, reconstruction_error):
    pca_result = pd.DataFrame({
        "label": y[0],
        "timestamp": y[1],
        "reconstruction_error": reconstruction_error,
    })

    return pca_result


def evaluate_pca_thresholds(pca_result, alert_ratios):
    pca_threshold_results = []

    total_fail = (pca_result["label"] == 1).sum()
    total_pass = (pca_result["label"] == -1).sum()

    for ratio in alert_ratios:
        cutoff = pca_result["reconstruction_error"].quantile(1 - ratio)

        predicted_alert = pca_result["reconstruction_error"] >= cutoff

        total_alerts = predicted_alert.sum()
        fail_alerts = ((pca_result["label"] == 1) & predicted_alert).sum()
        pass_alerts = ((pca_result["label"] == -1) & predicted_alert).sum()

        fail_capture_rate = fail_alerts / total_fail
        false_alert_rate = pass_alerts / total_pass
        alert_fail_ratio = fail_alerts / total_alerts if total_alerts > 0 else 0

        pca_threshold_results.append({
            "alert_ratio": ratio,
            "error_cutoff": cutoff,
            "total_alerts": total_alerts,
            "fail_alerts": fail_alerts,
            "pass_alerts": pass_alerts,
            "fail_capture_rate": fail_capture_rate,
            "false_alert_rate": false_alert_rate,
            "alert_fail_ratio": alert_fail_ratio,
        })

    return pd.DataFrame(pca_threshold_results)


def create_model_comparison():
    model_comparison_df = pd.DataFrame([
        {
            "method": "SPC baseline",
            "threshold": "alert_count >= 1",
            "total_alerts": 92,
            "fail_alerts": 19,
            "pass_alerts": 73,
            "fail_capture_rate": 0.182692,
            "false_alert_rate": 0.049897,
            "alert_fail_ratio": 0.206522,
        },
        {
            "method": "Isolation Forest",
            "threshold": "score bottom 5%",
            "total_alerts": 79,
            "fail_alerts": 11,
            "pass_alerts": 68,
            "fail_capture_rate": 0.105769,
            "false_alert_rate": 0.046480,
            "alert_fail_ratio": 0.139241,
        },
        {
            "method": "PCA reconstruction",
            "threshold": "error top 5%",
            "total_alerts": 79,
            "fail_alerts": 10,
            "pass_alerts": 69,
            "fail_capture_rate": 0.096154,
            "false_alert_rate": 0.047163,
            "alert_fail_ratio": 0.126582,
        },
    ])

    return model_comparison_df


if __name__ == "__main__":
    data_dir = Path("data/raw")

    X, y = load_secom_data(data_dir)

    X_clean = preprocess_secom(X)
    X_scaled, X_scaled_array = scale_features(X_clean)

    reconstruction_error, pca_info = run_pca_reconstruction(
        X_scaled_array,
        explained_variance=0.90,
    )

    pca_result = create_pca_result(y, reconstruction_error)

    print("Data shape")
    print("X_clean:", X_clean.shape)
    print("X_scaled:", X_scaled.shape)
    print("y:", y.shape)

    print("\nPCA info")
    for key, value in pca_info.items():
        print(f"{key}: {value}")

    print("\nReconstruction error summary")
    print(pca_result["reconstruction_error"].describe())

    print("\nPass mean reconstruction error")
    print(pca_result.loc[pca_result["label"] == -1, "reconstruction_error"].mean())

    print("\nFail mean reconstruction error")
    print(pca_result.loc[pca_result["label"] == 1, "reconstruction_error"].mean())

    pca_threshold_df = evaluate_pca_thresholds(
        pca_result,
        alert_ratios=[0.05, 0.10, 0.15],
    )

    print("\nPCA threshold results")
    print(pca_threshold_df)

    model_comparison_df = create_model_comparison()

    print("\nSPC vs Isolation Forest vs PCA")
    print(model_comparison_df)