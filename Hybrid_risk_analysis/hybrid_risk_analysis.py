
## 2. `Hybrid_risk_analysis/hybrid_risk_analysis.py`
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


def create_spc_alert(X_clean):
    spc_features = [59, 103, 510, 348, 431, 434, 430, 21, 435, 28]

    spc_signal_table = pd.DataFrame(index=X_clean.index)

    for feature in spc_features:
        values = X_clean[feature]

        cl = values.mean()
        std = values.std()
        ucl = cl + 3 * std
        lcl = cl - 3 * std

        spc_signal_table[feature] = (values > ucl) | (values < lcl)

    spc_alert_count = spc_signal_table.sum(axis=1)
    spc_alert = spc_alert_count >= 1

    return spc_alert, spc_alert_count


def create_ml_alert(X_scaled, y):
    fail_ratio = (y[0] == 1).mean()

    iso_forest = IsolationForest(
        contamination=fail_ratio,
        random_state=42,
    )

    iso_forest.fit(X_scaled)

    anomaly_score = iso_forest.decision_function(X_scaled)

    ml_result = pd.DataFrame({
        "label": y[0],
        "timestamp": y[1],
        "anomaly_score": anomaly_score,
    })

    ml_cutoff = ml_result["anomaly_score"].quantile(0.05)
    ml_alert = ml_result["anomaly_score"] <= ml_cutoff

    return ml_alert, ml_result, ml_cutoff


def assign_overlap_group(row):
    if row["spc_alert"] and row["ml_alert"]:
        return "Both"
    if row["spc_alert"] and not row["ml_alert"]:
        return "SPC only"
    if not row["spc_alert"] and row["ml_alert"]:
        return "ML only"
    return "No alert"


def create_overlap_summary(y, spc_alert, ml_alert):
    overlap_summary = pd.DataFrame({
        "label": y[0],
        "timestamp": y[1],
        "spc_alert": spc_alert,
        "ml_alert": ml_alert,
    })

    overlap_summary["group"] = overlap_summary.apply(assign_overlap_group, axis=1)

    return overlap_summary


def summarize_overlap(overlap_summary, y):
    overlap_group_summary = (
        overlap_summary
        .groupby("group")
        .agg(
            total_samples=("label", "count"),
            fail_samples=("label", lambda s: (s == 1).sum()),
            pass_samples=("label", lambda s: (s == -1).sum()),
            fail_ratio=("label", lambda s: (s == 1).mean()),
        )
        .reset_index()
    )

    overlap_group_summary["fail_capture_rate"] = (
        overlap_group_summary["fail_samples"] / (y[0] == 1).sum()
    )

    overlap_group_summary["pass_share"] = (
        overlap_group_summary["pass_samples"] / (y[0] == -1).sum()
    )

    return overlap_group_summary


def create_risk_summary(overlap_summary):
    risk_level_map = {
        "No alert": "Normal",
        "ML only": "Watch",
        "SPC only": "Warning",
        "Both": "High risk",
    }

    overlap_summary["risk_level"] = overlap_summary["group"].map(risk_level_map)

    risk_summary = (
        overlap_summary
        .groupby("risk_level")
        .agg(
            total_samples=("label", "count"),
            fail_samples=("label", lambda s: (s == 1).sum()),
            pass_samples=("label", lambda s: (s == -1).sum()),
            fail_ratio=("label", lambda s: (s == 1).mean()),
        )
        .reset_index()
    )

    risk_order = ["Normal", "Watch", "Warning", "High risk"]
    risk_summary["risk_level"] = pd.Categorical(
        risk_summary["risk_level"],
        categories=risk_order,
        ordered=True,
    )

    risk_summary = risk_summary.sort_values("risk_level")

    return risk_summary


if __name__ == "__main__":
    data_dir = Path("data/raw")

    X, y = load_secom_data(data_dir)

    X_clean = preprocess_secom(X)
    X_scaled = scale_features(X_clean)

    spc_alert, spc_alert_count = create_spc_alert(X_clean)
    ml_alert, ml_result, ml_cutoff = create_ml_alert(X_scaled, y)

    overlap_summary = create_overlap_summary(y, spc_alert, ml_alert)
    overlap_group_summary = summarize_overlap(overlap_summary, y)
    risk_summary = create_risk_summary(overlap_summary)

    print("Data shape")
    print("X_clean:", X_clean.shape)
    print("X_scaled:", X_scaled.shape)
    print("y:", y.shape)

    print("\nSPC alert count")
    print(spc_alert.value_counts())

    print("\nML alert cutoff")
    print(ml_cutoff)

    print("\nML alert count")
    print(ml_alert.value_counts())

    print("\nOverlap group summary")
    print(overlap_group_summary)

    print("\nHybrid risk summary")
    print(risk_summary)