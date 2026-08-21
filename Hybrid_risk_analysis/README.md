# Hybrid Risk Analysis

이 문서는 SPC baseline alert와 Isolation Forest ML alert를 조합하여 sample별 risk level을 정의한 분석 과정을 정리합니다.

## 목적

이 단계의 목적은 SPC와 ML 이상탐지 결과가 서로 같은 sample을 잡는지 확인하고, 두 방법을 조합했을 때 fail risk가 높은 sample을 더 잘 선별할 수 있는지 확인하는 것입니다.

단일 SPC 또는 단일 ML 결과만 보는 것이 아니라, 두 alert의 overlap을 분석하여 hybrid risk rule을 설계했습니다.

## 사용한 Alert 기준

### SPC Alert

SPC alert는 이전 단계에서 선택한 baseline 기준을 사용했습니다.

- pass/fail 평균 차이가 컸던 상위 10개 feature 사용
- 각 feature에 대해 CL, UCL, LCL 계산
- sample이 하나 이상의 feature에서 관리한계를 벗어나면 SPC alert로 판단

기준:

SPC alert count >= 1

## ML Alert

ML alert는 Isolation Forest 결과를 사용했습니다.
- 전처리된 feature를 StandardScaler로 표준화
- Isolation Forest 적용
- anomaly score 하위 5% sample을 ML alert로 판단

기준:
Isolation Forest anomaly score bottom 5%


Overlap Group 정의

각 sample을 SPC alert 여부와 ML alert 여부에 따라 네 그룹으로 나누었습니다.

| Group | 의미 |
|---|---|
| No alert | SPC도 ML도 alert 아님 |
| ML only | ML만 alert |
| SPC only | SPC만 alert |
| Both | SPC와 ML 둘 다 alert |


Overlap 분석 결과

| Group | Total Samples | Fail Samples | Pass Samples | Fail Ratio |
|---|---:|---:|---:|---:|
| No alert | 1,429 | 82 | 1,347 | 5.74% |
| ML only | 46 | 3 | 43 | 6.52% |
| SPC only | 59 | 11 | 48 | 18.64% |
| Both | 33 | 8 | 25 | 24.24% |


전체 데이터의 fail 비율은 6.64%였습니다.    

### 해석

Both 그룹은 fail 비율이 24.24%로 가장 높았습니다.
이는 전체 fail 비율 6.64%보다 약 3.6배 높은 값입니다.
SPC only 그룹도 fail 비율이 18.64%로 전체 평균보다 높았습니다.
반면 ML only 그룹은 fail 비율이 6.52%로 전체 평균과 거의 비슷했습니다.
따라서 현재 설정에서는 SPC alert가 fail risk를 구분하는 핵심 신호였고, ML alert는 SPC와 겹칠 때 high-risk sample을 강화하는 보조 신호로 해석할 수 있습니다.


## Hybrid Risk Rule

Overlap 결과를 바탕으로 sample별 risk level을 다음과 같이 정의했습니다.

| Risk Level | 조건 | 해석 |
|---|---|---|
| Normal | SPC alert 없음, ML alert 없음 | 일반 sample |
| Watch | ML alert만 있음 | 관찰 대상 |
| Warning | SPC alert만 있음 | fail risk 상승 |
| High risk | SPC와 ML 모두 alert | 가장 높은 fail risk 후보 |

## Hybrid Risk 결과

| Risk Level | Total Samples | Fail Samples | Pass Samples | Fail Ratio |
|---|---:|---:|---:|---:|
| Normal | 1,429 | 82 | 1,347 | 5.74% |
| Watch | 46 | 3 | 43 | 6.52% |
| Warning | 59 | 11 | 48 | 18.64% |
| High risk | 33 | 8 | 25 | 24.24% |

## 결론

Hybrid risk rule을 적용했을 때 risk level이 높아질수록 fail 비율이 높아지는 경향을 확인했습니다.
특히 High risk 그룹은 전체 평균보다 훨씬 높은 fail 비율을 보였습니다.
현재 결과는 다음과 같이 정리할 수 있습니다.
- SPC alert는 fail risk sample 선별에 강한 신호로 작용했습니다.
- Isolation Forest 단독 alert는 현재 설정에서는 강한 fail risk 신호로 보기 어려웠습니다.
- 하지만 ML alert가 SPC alert와 겹칠 경우, high-risk sample을 구분하는 데 도움이 되었습니다.
- 따라서 현재 설정에서는 SPC 중심의 hybrid rule이 합리적입니다.

## 한계

현재 ML 모델은 기본 Isolation Forest 설정만 사용했습니다.
따라서 이 결과는 ML 전체의 한계를 의미하지 않습니다.
향후에는 PCA reconstruction error, Local Outlier Factor, One-Class SVM 등 다른 이상탐지 기법을 추가로 비교할 필요가 있습니다.

## 다음 단계 

다음 단계에서는 다른 ML 이상탐지 모델을 적용하거나, PCA 기반 reconstruction error를 이용해 SPC baseline 및 hybrid rule과 비교합니다.

