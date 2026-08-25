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


def calculate_pass_fail_mean_diff(X_clean, y, top_n=15):
    pass_mask = y[0] == -1
    fail_mask = y[0] == 1

    pass_mean = X_clean[pass_mask].mean()
    fail_mean = X_clean[fail_mask].mean()
    overall_std = X_clean.std()

    standardized_diff = (fail_mean - pass_mean) / overall_std
    abs_standardized_diff = standardized_diff.abs()

    mean_diff_ranking = pd.DataFrame({
        "feature": abs_standardized_diff.index,
        "pass_mean": pass_mean,
        "fail_mean": fail_mean,
        "difference": fail_mean - pass_mean,
        "standardized_difference": standardized_diff,
        "abs_standardized_difference": abs_standardized_diff,
    })

    mean_diff_ranking = (
        mean_diff_ranking
        .sort_values("abs_standardized_difference", ascending=False)
        .reset_index(drop=True)
    )

    return mean_diff_ranking.head(top_n), mean_diff_ranking


def calculate_spc_feature_impact(X_clean, y, spc_features):
    spc_feature_impact = []

    for feature in spc_features:
        values = X_clean[feature]

        cl = values.mean()
        std = values.std()
        ucl = cl + 3 * std
        lcl = cl - 3 * std

        spc_alert = (values > ucl) | (values < lcl)

        total_alerts = spc_alert.sum()
        fail_alerts = ((y[0] == 1) & spc_alert).sum()
        pass_alerts = ((y[0] == -1) & spc_alert).sum()

        alert_fail_ratio = fail_alerts / total_alerts if total_alerts > 0 else 0
        fail_capture_rate = fail_alerts / (y[0] == 1).sum()

        spc_feature_impact.append({
            "feature": feature,
            "total_alerts": total_alerts,
            "fail_alerts": fail_alerts,
            "pass_alerts": pass_alerts,
            "alert_fail_ratio": alert_fail_ratio,
            "fail_capture_rate": fail_capture_rate,
        })

    spc_feature_impact_df = (
        pd.DataFrame(spc_feature_impact)
        .sort_values(["fail_alerts", "alert_fail_ratio"], ascending=False)
        .reset_index(drop=True)
    )

    return spc_feature_impact_df


def calculate_ml_feature_impact(X_clean, X_scaled, y, top_n=15):
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

    ml_alert_mean = X_clean[ml_alert].mean()
    ml_normal_mean = X_clean[~ml_alert].mean()
    ml_overall_std = X_clean.std()

    ml_standardized_diff = (ml_alert_mean - ml_normal_mean) / ml_overall_std
    ml_abs_standardized_diff = ml_standardized_diff.abs()

    ml_feature_impact = pd.DataFrame({
        "feature": ml_abs_standardized_diff.index,
        "ml_alert_mean": ml_alert_mean,
        "ml_normal_mean": ml_normal_mean,
        "difference": ml_alert_mean - ml_normal_mean,
        "standardized_difference": ml_standardized_diff,
        "abs_standardized_difference": ml_abs_standardized_diff,
    })

    ml_feature_impact = (
        ml_feature_impact
        .sort_values("abs_standardized_difference", ascending=False)
        .reset_index(drop=True)
    )

    return ml_feature_impact.head(top_n), ml_feature_impact


def create_combined_feature_summary(
    mean_diff_top,
    spc_feature_impact_df,
    ml_feature_impact_top,
    mean_diff_ranking,
    ml_feature_impact_full,
):
    mean_diff_top_set = set(mean_diff_top["feature"])
    spc_top_set = set(spc_feature_impact_df["feature"].head(15))
    ml_top_set = set(ml_feature_impact_top["feature"])

    all_candidate_features = sorted(mean_diff_top_set | spc_top_set | ml_top_set)

    feature_summary = []

    for feature in all_candidate_features:
        in_mean_diff = feature in mean_diff_top_set
        in_spc = feature in spc_top_set
        in_ml = feature in ml_top_set

        score = int(in_mean_diff) + int(in_spc) + int(in_ml)

        feature_summary.append({
            "feature": feature,
            "in_pass_fail_mean_diff_top15": in_mean_diff,
            "in_spc_impact_top15": in_spc,
            "in_ml_anomaly_top15": in_ml,
            "combined_score": score,
        })

    feature_summary_df = (
        pd.DataFrame(feature_summary)
        .sort_values(["combined_score", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )

    top_feature_candidates = feature_summary_df[
        feature_summary_df["combined_score"] >= 2
    ]["feature"].tolist()

    final_feature_impact = feature_summary_df[
        feature_summary_df["feature"].isin(top_feature_candidates)
    ].copy()

    final_feature_impact = final_feature_impact.merge(
        mean_diff_ranking[[
            "feature",
            "standardized_difference",
            "abs_standardized_difference",
        ]],
        on="feature",
        how="left",
    )

    final_feature_impact = final_feature_impact.merge(
        spc_feature_impact_df[[
            "feature",
            "total_alerts",
            "fail_alerts",
            "alert_fail_ratio",
        ]],
        on="feature",
        how="left",
    )

    final_feature_impact = final_feature_impact.merge(
        ml_feature_impact_full[[
            "feature",
            "standardized_difference",
            "abs_standardized_difference",
        ]].rename(columns={
            "standardized_difference": "ml_standardized_difference",
            "abs_standardized_difference": "ml_abs_standardized_difference",
        }),
        on="feature",
        how="left",
    )

    final_feature_impact = (
        final_feature_impact
        .sort_values(
            ["combined_score", "fail_alerts", "ml_abs_standardized_difference"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return feature_summary_df, final_feature_impact


if __name__ == "__main__":
    data_dir = Path("data/raw")

    X, y = load_secom_data(data_dir)

    X_clean = preprocess_secom(X)
    X_scaled = scale_features(X_clean)

    print("Data shape")
    print("X_clean:", X_clean.shape)
    print("X_scaled:", X_scaled.shape)
    print("y:", y.shape)

    mean_diff_top, mean_diff_ranking = calculate_pass_fail_mean_diff(
        X_clean,
        y,
        top_n=15,
    )

    print("\nPass/fail mean difference top 15")
    print(mean_diff_top)

    spc_features = mean_diff_top["feature"].tolist()

    spc_feature_impact_df = calculate_spc_feature_impact(
        X_clean,
        y,
        spc_features,
    )

    print("\nSPC feature impact")
    print(spc_feature_impact_df)

    ml_feature_impact_top, ml_feature_impact_full = calculate_ml_feature_impact(
        X_clean,
        X_scaled,
        y,
        top_n=15,
    )

    print("\nML anomaly feature impact top 15")
    print(ml_feature_impact_top)

    feature_summary_df, final_feature_impact = create_combined_feature_summary(
        mean_diff_top,
        spc_feature_impact_df,
        ml_feature_impact_top,
        mean_diff_ranking,
        ml_feature_impact_full,
    )

    print("\nCombined feature summary")
    print(feature_summary_df)

    print("\nFinal feature impact table")
    print(final_feature_impact)

    print("\nTier 1 feature candidates")
    print(
        final_feature_impact.loc[
            final_feature_impact["combined_score"] == 3,
            "feature",
        ].tolist()
    )

    print("\nTier 2 feature candidates")
    print(
        final_feature_impact.loc[
            final_feature_impact["combined_score"] == 2,
            "feature",
        ].tolist()
    )