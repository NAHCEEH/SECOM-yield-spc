# SECOM 수율 SPC 및 이상탐지 프로젝트

## 프로젝트 개요

이 프로젝트는 UCI SECOM 반도체 제조 데이터셋을 활용하여 SPC 관리도와 머신러닝 기반 이상탐지 방법으로 fail-risk sample을 선별하는 것을 목표로 합니다.

주요 목적은 통계적 공정 관리 방식과 머신러닝 이상탐지 방식을 비교하고, fail 및 anomaly와 관련 가능성이 높은 feature 후보를 도출하는 것입니다.

## 데이터셋

- 데이터셋: UCI SECOM
- 도메인: 반도체 제조 공정
- sample 수: 1,567개
- 원본 feature 수: 590개
- label:
  - `-1`: pass
  - `1`: fail

SECOM 데이터셋의 feature 이름은 익명화되어 있습니다.  
따라서 각 feature 번호는 실제 센서명이나 공정 변수명이 아니라 익명화된 공정/센서 측정값으로 해석합니다.

## 프로젝트 진행 흐름

1. 데이터 구조 확인
2. 데이터 전처리
3. EDA 및 SPC baseline 분석
4. 머신러닝 기반 이상탐지
5. SPC와 ML overlap 분석
6. PCA reconstruction error 기반 이상탐지
7. 주요 feature impact 분석

## 데이터 전처리

원본 feature 데이터에는 결측치가 많이 포함되어 있었습니다.

전처리 과정은 다음과 같습니다.

1. 결측치 비율이 50% 이상인 feature 제거
2. 남은 결측치는 각 feature의 중앙값으로 대체
3. 분산이 0인 feature 제거

전처리 결과는 다음과 같습니다.

| 단계 | 데이터 크기 |
|---|---:|
| 원본 feature 데이터 | 1,567 × 590 |
| 고결측 feature 제거 후 | 1,567 × 562 |
| 최종 정제 feature 데이터 | 1,567 × 446 |

추가 결과:

- 제거된 고결측 feature 수: 28개
- 제거된 분산 0 feature 수: 116개
- 최종 결측치 수: 0개

## Label 분포

| Label | 의미 | 개수 | 비율 |
|---:|---|---:|---:|
| `-1` | pass | 1,463 | 93.36% |
| `1` | fail | 104 | 6.64% |

SECOM 데이터는 pass sample이 대부분이고 fail sample이 적은 클래스 불균형 데이터입니다.

따라서 단순 정확도는 적절한 평가 지표가 아니며, fail sample을 얼마나 잘 선별하는지가 중요합니다.

## 평가 지표

이상탐지 방법은 다음 세 가지 지표로 평가했습니다.

| 지표 | 의미 |
|---|---|
| Fail capture rate | 전체 fail sample 중 alert가 잡아낸 비율 |
| False alert rate | 전체 pass sample 중 alert로 잘못 잡힌 비율 |
| Alert fail ratio | alert sample 중 실제 fail sample의 비율 |

## SPC Baseline

SPC는 Statistical Process Control의 약자로, 통계적 공정 관리를 의미합니다.

이번 분석에서는 각 feature의 관리한계를 다음과 같이 계산했습니다.


CL = 평균
UCL = 평균 + 3 × 표준편차
LCL = 평균 - 3 × 표준편차

SPC baseline은 pass/fail 평균 차이가 컸던 상위 10개 feature를 사용했습니다.
sample이 선택된 feature 중 하나 이상에서 관리한계를 벗어나면 SPC alert로 판단했습니다.

SPC threshold = alert count >= 1

SPC baseline 결과는 다음과 같습니다.
| Method | Threshold | Total Alerts | Fail Alerts | Pass Alerts | Fail Capture Rate | False Alert Rate | Alert Fail Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| SPC baseline | alert count >= 1 | 92 | 19 | 73 | 18.27% | 4.99% | 20.65% |

## Isolation Forest

전처리된 feature를 표준화한 뒤 Isolation Forest를 적용했습니다.
Isolation Forest는 여러 feature 조합에서 다른 sample들과 동떨어진 sample을 anomaly로 판단하는 머신러닝 이상탐지 기법입니다.
비교 기준으로 anomaly score 하위 5%를 ML alert로 설정했습니다.

| Method | Threshold | Total Alerts | Fail Alerts | Pass Alerts | Fail Capture Rate | False Alert Rate | Alert Fail Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | score bottom 5% | 79 | 11 | 68 | 10.58% | 4.65% | 13.92% |

## PCA Reconstruction Error

PCA는 Principal Component Analysis, 즉 주성분 분석입니다.
이번 분석에서는 PCA를 차원축소뿐 아니라 reconstruction error 기반 이상탐지에 활용했습니다.
PCA는 전체 분산의 90%를 설명하도록 설정했습니다.
- 전처리 후 feature 수: 446개
- 선택된 PCA component 수: 129개
- 누적 설명 분산 비율: 90.04%

PCA로 압축한 뒤 다시 원래 feature 공간으로 복원하고, 원본과 복원값의 차이를 reconstruction error로 계산했습니다.
Reconstruction error 상위 5% sample을 PCA anomaly alert로 설정했습니다.

| Method | Threshold | Total Alerts | Fail Alerts | Pass Alerts | Fail Capture Rate | False Alert Rate | Alert Fail Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| PCA reconstruction | error top 5% | 79 | 10 | 69 | 9.62% | 4.72% | 12.66% |

## Model Compare

비슷한 false alert rate 조건에서 SPC, Isolation Forest, PCA를 비교했습니다.

| Method | Threshold | Total Alerts | Fail Alerts | Pass Alerts | Fail Capture Rate | False Alert Rate | Alert Fail Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| SPC baseline | alert count >= 1 | 92 | 19 | 73 | 18.27% | 4.99% | 20.65% |
| Isolation Forest | score bottom 5% | 79 | 11 | 68 | 10.58% | 4.65% | 13.92% |
| PCA reconstruction | error top 5% | 79 | 10 | 69 | 9.62% | 4.72% | 12.66% |

현재 설정 기준 성능 순서는 다음과 같습니다.
SPC baseline > Isolation Forest > PCA reconstruction
이는 머신러닝이 항상 SPC보다 낮다는 뜻은 아닙니다.
현재 사용한 기본 설정에서는 SPC baseline이 가장 강한 결과를 보였다는 의미입니다.

## SPC와 ML Hybrid Risk Rule

SPC alert와 Isolation Forest alert의 overlap을 분석했습니다.

| Group | Total Samples | Fail Samples | Pass Samples | Fail Ratio |
|---|---:|---:|---:|---:|
| No alert | 1,429 | 82 | 1,347 | 5.74% |
| ML only | 46 | 3 | 43 | 6.52% |
| SPC only | 59 | 11 | 48 | 18.64% |
| Both | 33 | 8 | 25 | 24.24% |

이를 바탕으로 간단한 hybrid risk level을 정의했습니다.

| Risk Level | 정의 | Fail Ratio |
|---|---|---:|
| Normal | SPC도 ML도 alert 아님 | 5.74% |
| Watch | ML만 alert | 6.52% |
| Warning | SPC만 alert | 18.64% |
| High risk | SPC와 ML 모두 alert | 24.24% |

Hybrid 결과는 SPC alert가 fail risk를 구분하는 핵심 신호였고, ML alert는 SPC와 겹칠 때 high-risk sample을 강화하는 보조 신호로 활용될 수 있음을 보여줍니다.

## Feature Impact Analysis

주요 feature 후보는 세 가지 기준으로 분석했습니다.
1. Pass/fail 표준화 평균 차이
2. SPC 기반 feature impact
3. ML anomaly 그룹 feature 차이
각 기준에 등장할 때마다 1점을 부여하여 combined score를 정의했습니다.

combined_score = 해당 feature가 등장한 기준 수

### Tier 1 주요 feature 후보
세 기준에 모두 등장한 feature입니다.
298, 163, 434, 435, 436, 430, 431

### Tier 2 보조 feature 후보
두 기준에 등장한 feature입니다.

510, 21, 348, 210, 103, 59, 28, 129
이 feature들은 fail 및 anomaly와 관련 가능성이 높은 후보입니다.
하지만 SECOM feature는 익명화되어 있으므로 실제 공정 원인으로 단정할 수는 없습니다.

## 주요 결과 요약
- SECOM 데이터는 fail 비율이 6.64%인 클래스 불균형 데이터입니다.
- 현재 설정에서는 SPC baseline이 가장 높은 fail 선별 성능을 보였습니다.
- Isolation Forest와 PCA도 전체 평균보다 높은 fail 비율의 anomaly 후보군을 만들었지만 SPC를 넘어서지는 못했습니다.
- SPC와 ML overlap 분석을 통해 hybrid risk level을 설계했습니다.
- High risk 그룹은 fail 비율이 24.24%로 전체 평균보다 크게 높았습니다.
- Feature impact 분석 결과 298, 163, 434, 435, 436, 430, 431이 가장 강한 주요 feature 후보로 나타났습니다.

## 한계 
-SECOM feature 이름이 익명화되어 실제 센서명이나 공정 변수명을 알 수 없습니다.
-본 분석은 fail과 feature 사이의 관련성을 찾은 것이며, 실제 원인을 증명한 것은 아닙니다.
-ML 모델은 기본 설정 위주로 비교했습니다.
-Chamber, recipe, maintenance 이력, metrology 결과가 있다면 더 정밀한 원인 분석이 가능합니다.

## 향후 개선 방향

-Isolation Forest 파라미터 튜닝
-PCA component 수 및 threshold 조정
-Local Outlier Factor 또는 One-Class SVM 추가 비교
-SPC, Isolation Forest, PCA를 결합한 고도화된 hybrid risk score 설계
-실제 공정 metadata와 연결한 feature 해석