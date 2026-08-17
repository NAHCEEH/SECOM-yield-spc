# SECOM Yield SPC and Anomaly Detection

## Project Overview

This project uses the UCI SECOM semiconductor manufacturing dataset to analyze process sensor data and detect abnormal process conditions.

The main goal is to apply Statistical Process Control (SPC) and machine learning-based anomaly detection to identify samples that may be related to yield failure.

## Dataset

- Dataset: UCI SECOM
- Domain: Semiconductor manufacturing
- Samples: 1,567
- Features: 590 anonymized process/sensor variables
- Label:
  - `-1`: pass
  - `1`: fail

## Main Questions

- Can process sensor data be used to detect abnormal manufacturing conditions?
- How imbalanced are pass and fail samples?
- Which sensor features show abnormal behavior?
- How do SPC-based alerts compare with machine learning-based anomaly detection?

## Planned Workflow

1. Load and inspect the SECOM dataset
2. Check pass/fail distribution
3. Analyze missing values
4. Explore sensor feature distributions
5. Apply SPC control charts
6. Apply machine learning anomaly detection
7. Analyze important variables
8. Summarize findings

## Notes

The SECOM feature names are anonymized, so each feature is treated as a process or sensor measurement rather than a directly named equipment variable.