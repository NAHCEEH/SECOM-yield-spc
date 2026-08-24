# Feature Impact Analysis

이 문서는 UCI SECOM 반도체 제조 데이터셋에서 fail 및 anomaly와 관련 가능성이 높은 주요 feature 후보를 분석한 과정을 정리합니다.

## 목적

이번 단계의 목적은 단순히 anomaly sample을 탐지하는 것을 넘어, 어떤 feature들이 fail 또는 anomaly 판단에 반복적으로 관련되어 나타나는지 확인하는 것입니다.

SECOM 데이터의 feature 이름은 익명화되어 있으므로 실제 센서명이나 공정 변수명은 알 수 없습니다.

따라서 본 분석에서는 feature를 실제 원인 변수로 단정하지 않고, fail 및 anomaly와 관련 가능성이 높은 주요 feature 후보로 해석합니다.

## 사용 데이터

이전 단계와 동일하게 전처리된 SECOM 데이터를 사용했습니다.

| 데이터 | 크기 |
|---|---:|
| 원본 feature 데이터 | 1,567 × 590 |
| 전처리 후 feature 데이터 | 1,567 × 446 |
| 표준화 후 feature 데이터 | 1,567 × 446 |
| label 데이터 | 1,567 × 2 |

Label은 다음과 같이 해석합니다.

- `-1`: pass
- `1`: fail

## Feature Impact 분석 기준

Feature impact는 세 가지 기준으로 분석했습니다.

1. Pass/fail 평균 차이
2. SPC 기반 feature impact
3. ML anomaly 그룹 차이

세 기준에서 반복적으로 등장한 feature를 더 중요한 후보로 보았습니다.

## 1. Pass/Fail 평균 차이

각 feature에 대해 pass sample의 평균값과 fail sample의 평균값을 비교했습니다.

feature마다 단위와 스케일이 다르기 때문에, 단순 평균 차이 대신 표준화된 평균 차이를 사용했습니다.

standardized difference = (fail mean - pass mean) / overall standard deviation

절댓값이 클수록 pass와 fail 사이에서 평균 차이가 큰 feature입니다.
Pass/fail 평균 차이 상위 feature는 다음과 같습니다.

<59, 103, 510, 348, 431, 434, 430, 21, 435, 28, 436, 210, 129, 298, 163>

## 2. SPC 기반 Feature Impact

SPC 분석에서는 pass/fail 평균 차이 상위 feature에 대해 관리한계를 계산했습니다.
CL = mean
UCL = mean + 3 × standard deviation
LCL = mean - 3 × standard deviation

각 feature에서 관리한계를 벗어난 sample 수와 그중 실제 fail sample 수를 확인했습니다.
SPC 관점에서 의미 있게 나타난 feature 후보는 다음과 같습니다.

510, 431, 434, 430, 435, 436, 298, 163, 21, 348, 210, 103, 59

Feature 28과 129는 pass/fail 평균 차이 기준에는 포함되었지만, SPC 관리한계 이탈 sample은 없었습니다.

## 3. ML Anomaly 그룹 차이
Isolation Forest anomaly score 하위 5%를 ML alert sample로 정의했습니다.
ML alert sample과 ML non-alert sample 사이에서 feature 평균 차이를 비교했습니다.
ML anomaly 그룹 차이 상위 feature는 다음과 같습니다.
300, 164, 165, 299, 298, 163, 434, 435, 436, 430, 431, 294, 26, 159, 295
이 feature들은 Isolation Forest가 anomaly로 판단한 sample에서 일반 sample과 값 차이가 크게 나타난 변수들입니다.

#### Combined Score

세 기준에서 반복적으로 등장하는 feature를 찾기 위해 combined score를 정의했습니다.
각 feature는 다음 기준에 등장할 때마다 1점을 받습니다.
- Pass/fail 평균 차이 top 15에 등장
- SPC impact top 15에 등장
- ML anomaly 차이 top 15에 등장
최대 점수는 3점입니다.

combined_score = 3 → 세 기준 모두에 등장
combined_score = 2 → 세 기준 중 두 기준에 등장
combined_score = 1 → 한 기준에만 등장

### Tier 1 주요 Feature 후보

세 기준에 모두 등장하여 combined score 3점을 받은 feature는 다음과 같습니다.
298, 163, 434, 435, 436, 430, 431
이 feature들은 pass/fail 평균 차이, SPC impact, ML anomaly 그룹 차이에서 모두 반복적으로 등장했습니다.
따라서 Tier 1 주요 feature 후보로 선정했습니다.

### Tier 2 보조 Feature 후보
combined score 2점을 받은 feature는 다음과 같습니다.
510, 21, 348, 210, 103, 59, 28, 129
이 feature들은 일부 기준에서는 의미 있게 나타났지만, 세 기준에 모두 등장하지는 않았습니다.
따라서 Tier 2 보조 feature 후보로 분류했습니다.

## 해석
Combined score가 높은 feature는 여러 분석 관점에서 반복적으로 관찰된 feature입니다.
이는 해당 feature가 fail 또는 anomaly와 관련 있을 가능성이 높다는 뜻입니다.
하지만 SECOM feature는 익명화되어 있으므로 실제 공정 원인으로 단정할 수 없습니다.
따라서 본 분석의 결론은 다음과 같이 표현하는 것이 적절합니다.
해당 feature들은 fail 및 anomaly와 관련 가능성이 높은 주요 feature 후보이다.

## 결론

Feature impact 분석 결과, 다음 feature들이 가장 강한 후보로 나타났습니다.
298, 163, 434, 435, 436, 430, 431
이 feature들은 세 가지 기준에서 모두 반복적으로 등장했습니다.
최종 보고서에서는 이들을 주요 영향 변수 후보로 우선 정리할 수 있습니다.

## 한계 
- SECOM feature 이름이 익명화되어 실제 센서명이나 공정 변수명을 알 수 없습니다.
- Combined score는 탐색적 점수이며 통계적으로 엄밀한 feature importance는 아닙니다.
- 실제 원인 분석을 위해서는 공정 지식, 장비 센서명, recipe 정보, metrology 결과가 추가로 필요합니다.