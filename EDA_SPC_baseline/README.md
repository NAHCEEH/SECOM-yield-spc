# EDA 및 SPC Baseline 분석

이 문서는 UCI SECOM 반도체 제조 데이터셋을 대상으로 수행한 초기 탐색 분석과 SPC 관리도 기반 baseline 분석 과정을 정리합니다.

## 목적

이번 단계의 목적은 전처리된 SECOM 데이터를 이용해 pass/fail 샘플의 차이를 탐색하고, SPC 관리도를 통해 공정 이상 신호를 탐지할 수 있는지 확인하는 것입니다.

SPC 분석은 이후 머신러닝 이상탐지 결과와 비교하기 위한 baseline 역할을 합니다.

## 사용 데이터

전처리 후 사용한 데이터 크기는 다음과 같습니다.

| 데이터 | 크기 |
|---|---:|
| 원본 feature 데이터 | 1,567 × 590 |
| 전처리 후 feature 데이터 | 1,567 × 446 |
| label 데이터 | 1,567 × 2 |

Label은 다음과 같이 해석합니다.

- `-1`: pass
- `1`: fail

## Pass/Fail 분포

SECOM 데이터는 pass 샘플이 많고 fail 샘플이 적은 클래스 불균형 데이터입니다.

| Label | 의미 | 개수 | 비율 |
|---|---|---:|---:|
| `-1` | pass | 1,463 | 93.36% |
| `1` | fail | 104 | 6.64% |

따라서 단순 정확도보다 fail 샘플을 얼마나 잘 탐지하는지가 중요합니다.

## Pass/Fail 평균 차이 분석

전처리된 feature를 기준으로 pass 그룹과 fail 그룹의 평균 차이를 확인했습니다.

feature마다 단위와 스케일이 다르기 때문에, 단순 평균 차이뿐 아니라 표준편차로 나눈 표준화된 평균 차이를 사용했습니다.

pass/fail 평균 차이가 큰 상위 feature는 다음과 같습니다.

| 순위 | Feature |
|---:|---:|
| 1 | 59 |
| 2 | 103 |
| 3 | 510 |
| 4 | 348 |
| 5 | 431 |
| 6 | 434 |
| 7 | 430 |
| 8 | 21 |
| 9 | 435 |
| 10 | 28 |

이 feature들은 fail 샘플과 pass 샘플 사이에서 값의 차이가 상대적으로 큰 후보 변수입니다.

## SPC 관리도 개념

SPC는 Statistical Process Control의 약자로, 통계적 공정 관리를 의미합니다.

이번 분석에서는 각 feature의 관리한계를 다음과 같이 계산했습니다.

- CL = 평균
- UCL = 평균 + 3 × 표준편차
- LCL = 평균 - 3 × 표준편차

feature 값이 UCL보다 크거나 LCL보다 작으면 SPC 이상 신호로 판단했습니다.

## Feature 59 SPC 결과

첫 번째 SPC 관리도는 feature 59를 대상으로 작성했습니다.

| 항목 | 값 |
|---|---:|
| CL | 2.95 |
| UCL | 31.49 |
| LCL | -25.58 |
| 관리한계 이탈 샘플 수 | 5 |
| 전체 샘플 대비 비율 | 0.32% |

관리한계 이탈 샘플의 label 분포는 다음과 같았습니다.

| Label | 의미 | 개수 |
|---|---|---:|
| `-1` | pass | 4 |
| `1` | fail | 1 |

feature 59 하나만으로는 대부분의 fail 샘플을 탐지하기 어렵다는 것을 확인했습니다.

## 다중 Feature SPC 분석

단일 feature만으로는 탐지 성능이 제한적이므로, pass/fail 평균 차이가 컸던 상위 10개 feature에 대해 SPC 이상 신호를 계산했습니다.

사용한 feature는 다음과 같습니다.

- 59
- 103
- 510
- 348
- 431
- 434
- 430
- 21
- 435
- 28

각 sample에 대해 몇 개 feature에서 관리한계를 벗어났는지 계산했습니다. 이를 SPC alert count라고 정의했습니다.

## SPC Alert Count 의미

SPC alert count는 한 sample이 여러 feature 중 몇 개에서 이상 신호를 보였는지를 나타냅니다.

예를 들어 SPC alert count가 0이면 선택한 feature 중 관리한계를 벗어난 feature가 없다는 뜻입니다.

SPC alert count가 5이면 선택한 feature 중 5개 feature에서 동시에 관리한계를 벗어났다는 뜻입니다.

따라서 alert count가 높을수록 여러 feature에서 동시에 비정상적인 움직임이 나타난 sample로 볼 수 있습니다.

## SPC Alert Count 결과

sample별 SPC alert count 분포는 다음과 같았습니다.

| SPC alert count | Sample 수 |
|---:|---:|
| 0 | 1,475 |
| 1 | 55 |
| 2 | 5 |
| 4 | 8 |
| 5 | 21 |
| 6 | 3 |

SPC alert count별 fail 비율은 다음과 같았습니다.

| SPC alert count | Fail 비율 |
|---:|---:|
| 0 | 5.76% |
| 1 | 18.18% |
| 2 | 40.00% |
| 4 | 12.50% |
| 5 | 28.57% |
| 6 | 0.00% |

전체 데이터의 fail 비율이 6.64%였다는 점을 고려하면, SPC alert count가 있는 sample은 fail 비율이 더 높아지는 경향이 있었습니다.

## Threshold별 SPC 성능 비교

SPC alert count 기준을 다르게 설정하여 fail 탐지 성능을 비교했습니다.

| Threshold | Total Alerts | Fail Alerts | Pass Alerts | Fail Capture Rate | False Alert Rate | Alert Fail Ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 92 | 19 | 73 | 18.27% | 4.99% | 20.65% |
| 2 | 37 | 9 | 28 | 8.65% | 1.91% | 24.32% |
| 5 | 24 | 6 | 18 | 5.77% | 1.23% | 25.00% |

## 해석

Threshold가 낮을수록 더 많은 fail 샘플을 잡을 수 있지만, pass 샘플도 더 많이 alert로 잡게 됩니다.

Threshold가 높을수록 오탐은 줄어들지만, 탐지되는 fail 샘플 수도 줄어듭니다.

이번 결과에서는 `SPC alert count >= 1`을 baseline 기준으로 선택할 수 있습니다.

이 기준은 전체 fail 중 18.27%를 탐지했고, false alert rate는 4.99%였습니다.

## 결론

SPC alert는 전체 데이터 평균보다 fail 비율이 높은 sample을 선별하는 데 도움이 되었습니다.

전체 fail 비율은 6.64%였지만, SPC alert로 잡힌 sample의 fail 비율은 약 20~25% 수준이었습니다.

따라서 SPC는 fail 가능성이 높은 sample을 우선적으로 확인하는 baseline 방법으로 사용할 수 있습니다.

다만 SPC만으로 전체 fail을 충분히 탐지하기에는 한계가 있으므로, 다음 단계에서는 머신러닝 기반 이상탐지를 적용하고 SPC 결과와 비교할 필요가 있습니다.