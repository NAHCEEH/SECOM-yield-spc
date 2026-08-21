# PCA 기반 이상탐지

이 문서는 UCI SECOM 반도체 제조 데이터셋을 대상으로 PCA reconstruction error를 이용해 이상탐지를 수행한 과정을 정리합니다.

## 목적

이번 단계의 목적은 Isolation Forest 외에 다른 머신러닝 기반 이상탐지 방법을 적용하고, SPC baseline 및 Isolation Forest 결과와 비교하는 것입니다.

PCA reconstruction error는 고차원 feature 데이터를 낮은 차원으로 압축한 뒤 다시 복원했을 때 발생하는 오차를 이용해 anomaly sample을 찾는 방법입니다.

## PCA 개념

PCA는 Principal Component Analysis의 약자로, 한국어로는 주성분 분석입니다.

PCA는 원래 feature들의 정보를 최대한 보존하면서 더 적은 수의 주성분으로 데이터를 압축하는 차원축소 기법입니다.

이번 프로젝트에서는 PCA를 단순 시각화 목적이 아니라 이상탐지에 활용했습니다.

흐름은 다음과 같습니다.

표준화된 feature 데이터
→ PCA로 낮은 차원으로 압축
→ 다시 원래 feature 공간으로 복원
→ 원본과 복원값의 차이 계산
→ 복원 오차가 큰 sample을 anomaly 후보로 판단

## 사용 데이터
전처리 후 사용한 데이터 크기는 다음과 같습니다.

| 데이터 | 크기 |
|---|---:|
| 원본 feature 데이터 | 1,567 × 590 |
| 전처리 후 feature 데이터 | 1,567 × 446 |
| 표준화 후 feature 데이터 | 1,567 × 446 |
| label 데이터 | 1,567 × 2 |

Label은 다음과 같이 해석합니다
-1: pass
1: fail

## 전처리 및 표준화 

PCA 적용 전 이전 단계와 동일한 전처리를 수행했습니다.
1. 결측치 비율이 50% 이상인 feature 제거
2. 남은 결측치를 각 feature의 중앙값으로 대체
3. 분산이 0인 feature 제거
4. StandardScaler를 이용해 feature 표준화
표준화를 적용한 이유는 PCA가 feature의 분산 구조를 사용하기 때문입니다.
feature마다 스케일이 다르면 값의 크기가 큰 feature가 PCA 결과에 과도하게 영향을 줄 수 있습니다.

## PCA 설정

PCA는 전체 분산의 90%를 설명하도록 설정했습니다.
결과는 다음과 같습니다

| 항목 | 값 |
|---|---:|
| 원본 feature 수 | 446 |
| 선택된 PCA component 수 | 129 |
| 누적 설명 분산 비율 | 90.04% |

즉 446개 feature를 129개 주성분으로 압축해도 원래 데이터 변동의 약 90%를 설명할 수 있었습니다.

## Reconstruction Error
PCA로 압축한 뒤 다시 원래 feature 공간으로 복원하고, 원본 데이터와 복원 데이터의 차이를 계산했습니다.
이번 분석에서는 sample별 평균제곱오차를 reconstruction error로 사용했습니다.

reconstruction error = feature별 복원 오차 제곱의 평균

오차를 제곱하는 이유는 양수/음수 오차가 상쇄되는 것을 막고, 큰 오차에 더 큰 가중을 주기 위해서입니다.
feature별 오차를 평균내는 이유는 sample 하나당 하나의 anomaly score를 만들기 위해서입니다.

## Pass/Fail Reconstruction Error 비교

Pass와 fail sample의 평균 reconstruction error는 다음과 같았습니다.

| Group | Mean Reconstruction Error |
|---|---:|
| Pass | 0.0988 |
| Fail | 0.1115 |

Fail sample의 평균 reconstruction error가 pass보다 높았습니다.
이는 fail sample이 PCA가 학습한 주요 데이터 패턴에서 평균적으로 조금 더 벗어나 있음을 의미합니다.
다만 차이가 크지는 않기 때문에 PCA reconstruction error 하나만으로 fail을 완벽히 구분하기는 어렵습니다

## Threshold 비교
Reconstruction error가 큰 sample을 anomaly 후보로 보고, error 상위 5%, 10%, 15% 기준을 비교했습니다.

| Alert Ratio | Error Cutoff | Total Alerts | Fail Alerts | Pass Alerts | Fail Capture Rate | False Alert Rate | Alert Fail Ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 0.1770 | 79 | 10 | 69 | 9.62% | 4.72% | 12.66% |
| 10% | 0.1365 | 157 | 18 | 139 | 17.31% | 9.50% | 11.46% |
| 15% | 0.1194 | 235 | 25 | 210 | 24.04% | 14.35% | 10.64% |

Alert ratio를 높이면 더 많은 fail sample을 잡을 수 있었지만, pass sample 오탐도 함께 증가했습니다.
또한 alert fail ratio는 alert 범위가 넓어질수록 낮아졌습니다.

## SPC 및 Isolation Forest와 비교

비슷한 false alert rate 조건에서 비교하기 위해 PCA는 error 상위 5% 기준을 사용했습니다.

| Method | Threshold | Total Alerts | Fail Alerts | Pass Alerts | Fail Capture Rate | False Alert Rate | Alert Fail Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| SPC baseline | alert count >= 1 | 92 | 19 | 73 | 18.27% | 4.99% | 20.65% |
| Isolation Forest | score bottom 5% | 79 | 11 | 68 | 10.58% | 4.65% | 13.92% |
| PCA reconstruction | error top 5% | 79 | 10 | 69 | 9.62% | 4.72% | 12.66% |

세 방법 모두 false alert rate는 약 5% 수준으로 비슷했습니다.
그러나 fail capture rate와 alert fail ratio는 SPC baseline이 가장 높았습니다.
현재 설정 기준 성능 순서는 다음과 같습니다.

#### SPC baseline > Isolation Forest > PCA reconstruction

## 해석
PCA reconstruction error는 전체 fail 비율보다 높은 fail 비율을 가진 anomaly 후보군을 만들었습니다.
전체 fail 비율은 6.64%였고, PCA error 상위 5% alert fail ratio는 12.66%였습니다.
따라서 PCA는 fail risk가 상대적으로 높은 sample을 어느 정도 선별했습니다.
하지만 현재 설정에서는 SPC baseline보다 성능이 낮았습니다.

## 결론
PCA reconstruction error는 SECOM 데이터에서 이상탐지 후보 방법으로 의미가 있었지만, 현재 설정에서는 SPC baseline을 넘어서지는 못했습니다.
다만 이는 PCA 방식 전체의 한계를 의미하지 않습니다.
PCA component 수, explained variance 기준, threshold 설정을 바꾸면 결과가 달라질 수 있습니다.