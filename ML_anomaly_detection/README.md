# ML 기반 이상탐지

이 문서는 UCI SECOM 반도체 제조 데이터셋을 대상으로 수행한 머신러닝 기반 이상탐지 분석 과정을 정리합니다.

## 목적

이번 단계의 목적은 전처리된 SECOM 데이터를 이용해 머신러닝 모델이 공정 이상 가능성이 있는 sample을 얼마나 잘 선별하는지 확인하는 것입니다.

이전 단계에서 만든 SPC baseline 결과와 비교하여, 머신러닝 기반 이상탐지가 fail sample 탐지에 더 효과적인지 평가합니다.

## 사용 데이터

전처리 후 사용한 feature 데이터 크기는 다음과 같습니다.

| 데이터 | 크기 |
|---|---:|
| 원본 feature 데이터 | 1,567 × 590 |
| 전처리 후 feature 데이터 | 1,567 × 446 |
| label 데이터 | 1,567 × 2 |

Label은 다음과 같이 해석합니다.

- `-1`: pass
- `1`: fail

## 데이터 전처리

머신러닝 모델에는 이전 단계와 동일하게 전처리된 데이터를 사용했습니다.

전처리 과정은 다음과 같습니다.

1. 결측치 비율이 50% 이상인 feature 제거
2. 남은 결측치를 각 feature의 중앙값으로 대체
3. 분산이 0인 feature 제거

최종적으로 446개의 feature를 사용했습니다.

## 표준화

머신러닝 모델 적용 전 `StandardScaler`를 이용해 feature를 표준화했습니다.

표준화는 각 feature를 평균 0, 표준편차 1에 가깝게 변환하는 과정입니다.

SECOM 데이터는 feature마다 스케일이 다르기 때문에, 표준화를 통해 특정 feature가 값의 크기 때문에 과도하게 영향을 주는 것을 줄이고자 했습니다.

## 사용 모델: Isolation Forest

이번 분석에서는 첫 번째 머신러닝 이상탐지 모델로 Isolation Forest를 사용했습니다.

Isolation Forest는 다른 sample들과 동떨어진 sample을 더 쉽게 고립되는 sample로 보고 이상치로 판단하는 알고리즘입니다.

쉽게 말해, 여러 feature 조합이 일반적인 sample들과 다르게 보이는 sample을 anomaly로 분류합니다.

## 기본 Isolation Forest 결과

처음에는 실제 fail 비율과 동일하게 contamination 값을 설정했습니다.

SECOM 데이터의 fail 비율은 다음과 같습니다.

104 / 1567 = 6.64%


따라서 Isolation Forest는 전체 sample 중 약 6.64%를 anomaly로 판단했습니다.
예측 결과는 다음과 같습니다.

| 예측 결과 | 의미 | 개수 |
|---|---|---:|
| `1` | normal | 1,463 |
| `-1` | anomaly | 104 |

주의할 점은 anomaly 개수가 fail 개수와 같다고 해서 같은 sample을 정확히 찾았다는 의미는 아니라는 것입니다.

## 기본 모델 성능

Isolation Forest가 anomaly로 판단한 104개 sample 중 실제 fail은 14개였습니다.

| 항목 | 값 |
|---|---:|
| ML anomaly total | 104 |
| ML anomaly 중 fail | 14 |
| ML anomaly 중 pass | 90 |
| Fail capture rate | 13.46% |
| False alert rate | 6.15% |
| Alert fail ratio | 13.46% |

평가 지표 의미
Fail Capture Rate
전체 fail sample 중 모델이 anomaly로 잡은 비율입니다.

fail_alerts / total_fail

False Alert Rate
전체 pass sample 중 모델이 anomaly로 잘못 잡은 비율입니다.

pass_alerts / total_pass

Alert Fail Ratio
모델이 anomaly로 잡은 sample 중 실제 fail 비율입니다.

fail_alerts / total_alerts

### Anomaly Score 기반 Threshold 비교
Isolation Forest는 각 sample에 anomaly score를 부여합니다.
이번 분석에서는 anomaly score가 낮을수록 더 이상한 sample로 해석했습니다.
따라서 anomaly score 하위 5%, 10%, 15%를 각각 alert로 설정하여 성능을 비교했습니다

| Alert Ratio | Total Alerts | Fail Alerts | Pass Alerts | Fail Capture Rate | False Alert Rate | Alert Fail Ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | 79 | 11 | 68 | 10.58% | 4.65% | 13.92% |
| 10% | 157 | 20 | 137 | 19.23% | 9.36% | 12.74% |
| 15% | 235 | 28 | 207 | 26.92% | 14.15% | 11.91% |

Alert ratio를 높이면 더 많은 fail sample을 잡을 수 있었지만, 동시에 pass sample 오탐도 증가했습니다.
또한 alert로 잡힌 sample 중 fail 비율은 오히려 낮아졌습니다.

## SPC Baseline과 비교

공정한 비교를 위해 Isolation Forest의 5% threshold를 SPC baseline과 비교했습니다.
이유는 두 방법의 false alert rate가 비슷했기 때문입니다.

| Method | Threshold | Total Alerts | Fail Alerts | Pass Alerts | Fail Capture Rate | False Alert Rate | Alert Fail Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| SPC baseline | alert count >= 1 | 92 | 19 | 73 | 18.27% | 4.99% | 20.65% |
| Isolation Forest | score bottom 5% | 79 | 11 | 68 | 10.58% | 4.65% | 13.92% |

비슷한 오탐율 조건에서 SPC baseline이 Isolation Forest보다 fail sample을 더 많이 탐지했고, alert sample 중 fail 비율도 더 높았습니다.

## 해석
기본 설정의 Isolation Forest는 전체 데이터 평균보다 fail 비율이 높은 anomaly sample을 선별했습니다.
전체 데이터의 fail 비율은 6.64%였고, Isolation Forest 5% threshold 기준 alert fail ratio는 13.92%였습니다.
즉, 무작위 sample보다 anomaly sample에서 fail 가능성이 더 높게 나타났습니다.
하지만 SPC baseline과 비교하면 현재 설정에서는 SPC가 더 좋은 성능을 보였습니다.

## 결론
기본 Isolation Forest 모델은 fail 가능성이 상대적으로 높은 sample을 선별하는 데 어느 정도 의미가 있었습니다.
그러나 현재 설정에서는 SPC baseline보다 fail 탐지 성능이 낮았습니다.
따라서 Isolation Forest 기본 모델만으로는 충분하지 않으며, 이후에는 다음 방향을 검토할 수 있습니다.
- Isolation Forest 파라미터 튜닝
- PCA 기반 이상탐지
- Local Outlier Factor
- One-Class SVM
- SPC alert와 ML anomaly 결과를 조합한 hybrid 방식
