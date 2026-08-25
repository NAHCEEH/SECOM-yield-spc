from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
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

    return X_scaled


def run_isolation_forest(X_scaled, y):
    fail_ratio = (y[0] == 1).mean()

    iso_forest = IsolationForest(
        contamination=fail_ratio,
        random_state=42,
    )

    iso_forest.fit(X_scaled)

    anomaly_prediction = iso_forest.predict(X_scaled)
    anomaly_score = iso_forest.decision_function(X_scaled)

    ml_result = pd.DataFrame({
        "label": y[0],
        "timestamp": y[1],
        "anomaly_prediction": anomaly_prediction,
        "anomaly_score": anomaly_score,
    })

    return ml_result, fail_ratio


def evaluate_prediction(ml_result):
    ml_anomaly_mask = ml_result["anomaly_prediction"] == -1

    total_anomalies = ml_anomaly_mask.sum()
    fail_anomalies = ((ml_result["label"] == 1) & ml_anomaly_mask).sum()
    pass_anomalies = ((ml_result["label"] == -1) & ml_anomaly_mask).sum()

    total_fail = (ml_result["label"] == 1).sum()
    total_pass = (ml_result["label"] == -1).sum()

    fail_capture_rate = fail_anomalies / total_fail
    false_alert_rate = pass_anomalies / total_pass
    alert_fail_ratio = fail_anomalies / total_anomalies

    result = {
        "total_anomalies": total_anomalies,
        "fail_anomalies": fail_anomalies,
        "pass_anomalies": pass_anomalies,
        "fail_capture_rate": fail_capture_rate,
        "false_alert_rate": false_alert_rate,
        "alert_fail_ratio": alert_fail_ratio,
    }

    return result


def evaluate_score_thresholds(ml_result, score_thresholds):
    score_based_results = []

    total_fail = (ml_result["label"] == 1).sum()
    total_pass = (ml_result["label"] == -1).sum()

    for ratio in score_thresholds:
        cutoff = ml_result["anomaly_score"].quantile(ratio)

        predicted_alert = ml_result["anomaly_score"] <= cutoff

        total_alerts = predicted_alert.sum()
        fail_alerts = ((ml_result["label"] == 1) & predicted_alert).sum()
        pass_alerts = ((ml_result["label"] == -1) & predicted_alert).sum()

        fail_capture_rate = fail_alerts / total_fail
        false_alert_rate = pass_alerts / total_pass
        alert_fail_ratio = fail_alerts / total_alerts if total_alerts > 0 else 0

        score_based_results.append({
            "alert_ratio": ratio,
            "score_cutoff": cutoff,
            "total_alerts": total_alerts,
            "fail_alerts": fail_alerts,
            "pass_alerts": pass_alerts,
            "fail_capture_rate": fail_capture_rate,
            "false_alert_rate": false_alert_rate,
            "alert_fail_ratio": alert_fail_ratio,
        })

    return pd.DataFrame(score_based_results)


def compare_with_spc_baseline():
    comparison_df = pd.DataFrame([
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
    ])

    return comparison_df


if __name__ == "__main__":
    data_dir = Path("data/raw")

    X, y = load_secom_data(data_dir)

    X_clean = preprocess_secom(X)
    X_scaled = scale_features(X_clean)

    print("X_clean shape:", X_clean.shape)
    print("X_scaled shape:", X_scaled.shape)
    print("y shape:", y.shape)

    ml_result, fail_ratio = run_isolation_forest(X_scaled, y)

    print("\nFail ratio:", fail_ratio)

    print("\nIsolation Forest prediction distribution:")
    print(ml_result["anomaly_prediction"].value_counts())

    base_result = evaluate_prediction(ml_result)

    print("\nBasic Isolation Forest result:")
    for key, value in base_result.items():
        print(f"{key}: {value}")

    score_based_df = evaluate_score_thresholds(
        ml_result,
        score_thresholds=[0.05, 0.10, 0.15],
    )

    print("\nScore-based threshold results:")
    print(score_based_df)

    comparison_df = compare_with_spc_baseline()

    print("\nSPC baseline vs Isolation Forest:")
    print(comparison_df)