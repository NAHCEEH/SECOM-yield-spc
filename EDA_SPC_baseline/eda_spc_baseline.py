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

    return X_clean


def calculate_mean_difference(X_clean, y, top_n=10):
    pass_mask = y[0] == -1
    fail_mask = y[0] == 1

    pass_mean = X_clean[pass_mask].mean()
    fail_mean = X_clean[fail_mask].mean()
    overall_std = X_clean.std()

    mean_diff_score = ((fail_mean - pass_mean) / overall_std).abs()
    top_features = mean_diff_score.sort_values(ascending=False).head(top_n)

    comparison_table = pd.DataFrame({
        "pass_mean": pass_mean[top_features.index],
        "fail_mean": fail_mean[top_features.index],
        "difference": fail_mean[top_features.index] - pass_mean[top_features.index],
        "abs_standardized_difference": top_features,
    })

    return comparison_table


def calculate_spc_summary(X_clean, y, spc_features):
    spc_summary = []

    for feature in spc_features:
        values = X_clean[feature]

        cl = values.mean()
        std = values.std()
        ucl = cl + 3 * std
        lcl = cl - 3 * std

        out_signal = (values > ucl) | (values < lcl)

        total_alerts = out_signal.sum()
        fail_alerts = ((y[0] == 1) & out_signal).sum()
        pass_alerts = ((y[0] == -1) & out_signal).sum()

        fail_capture_rate = fail_alerts / (y[0] == 1).sum()

        spc_summary.append({
            "feature": feature,
            "CL": cl,
            "UCL": ucl,
            "LCL": lcl,
            "total_alerts": total_alerts,
            "fail_alerts": fail_alerts,
            "pass_alerts": pass_alerts,
            "fail_capture_rate": fail_capture_rate,
        })

    return pd.DataFrame(spc_summary)


def calculate_sample_alert_counts(X_clean, y, spc_features):
    spc_signal_table = pd.DataFrame(index=X_clean.index)

    for feature in spc_features:
        values = X_clean[feature]

        cl = values.mean()
        std = values.std()
        ucl = cl + 3 * std
        lcl = cl - 3 * std

        spc_signal_table[feature] = (values > ucl) | (values < lcl)

    spc_alert_count = spc_signal_table.sum(axis=1)

    sample_spc_summary = pd.DataFrame({
        "spc_alert_count": spc_alert_count,
        "label": y[0],
        "timestamp": y[1],
    })

    return sample_spc_summary


def evaluate_spc_thresholds(sample_spc_summary, y, thresholds):
    threshold_results = []

    total_fail = (y[0] == 1).sum()
    total_pass = (y[0] == -1).sum()

    for threshold in thresholds:
        predicted_alert = sample_spc_summary["spc_alert_count"] >= threshold

        total_alerts = predicted_alert.sum()
        fail_alerts = ((y[0] == 1) & predicted_alert).sum()
        pass_alerts = ((y[0] == -1) & predicted_alert).sum()

        fail_capture_rate = fail_alerts / total_fail
        false_alert_rate = pass_alerts / total_pass
        alert_fail_ratio = fail_alerts / total_alerts if total_alerts > 0 else 0

        threshold_results.append({
            "threshold": threshold,
            "total_alerts": total_alerts,
            "fail_alerts": fail_alerts,
            "pass_alerts": pass_alerts,
            "fail_capture_rate": fail_capture_rate,
            "false_alert_rate": false_alert_rate,
            "alert_fail_ratio": alert_fail_ratio,
        })

    return pd.DataFrame(threshold_results)


if __name__ == "__main__":
    data_dir = Path("data/raw")

    X, y = load_secom_data(data_dir)
    X_clean = preprocess_secom(X)

    print("Cleaned data shape:", X_clean.shape)
    print("Label data shape:", y.shape)

    print("\nPass/Fail distribution:")
    print(y[0].value_counts())

    mean_diff_table = calculate_mean_difference(X_clean, y, top_n=10)

    print("\nTop 10 features by pass/fail mean difference:")
    print(mean_diff_table)

    spc_features = list(mean_diff_table.index)

    spc_summary_df = calculate_spc_summary(X_clean, y, spc_features)

    print("\nFeature-level SPC summary:")
    print(spc_summary_df)

    sample_spc_summary = calculate_sample_alert_counts(X_clean, y, spc_features)

    print("\nSample-level SPC alert count distribution:")
    print(sample_spc_summary["spc_alert_count"].value_counts().sort_index())

    print("\nFail ratio by SPC alert count:")
    print(
        sample_spc_summary
        .groupby("spc_alert_count")["label"]
        .apply(lambda s: (s == 1).mean())
    )

    threshold_results = evaluate_spc_thresholds(
        sample_spc_summary,
        y,
        thresholds=[1, 2, 5],
    )

    print("\nSPC threshold performance:")
    print(threshold_results)