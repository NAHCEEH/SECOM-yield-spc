# Baseline Modeling

## 목적

이 단계에서는 LSWMD wafer map 데이터를 사용하여 `non_pattern`과 `pattern`을 분류하는 baseline 모델을 학습했다.

Baseline 모델의 목적은 CNN과 같은 이미지 기반 딥러닝 모델을 적용하기 전에, 단순한 머신러닝 모델이 어느 정도의 기준 성능을 보이는지 확인하는 것이다.

## 데이터 입력

전처리 단계에서 다양한 크기의 wafer map을 32x32 크기로 resize하고, 모델 입력을 위해 다음 형태로 변환했다.

- `X_train`: 54,355 x 32 x 32 x 1
- `X_test`: 118,595 x 32 x 32 x 1

이진 분류 label은 다음과 같이 정의했다.

- `non_pattern`: 특정 공간 불량 패턴이 없는 wafer
- `pattern`: Center, Edge-Ring, Edge-Loc, Loc, Scratch, Random, Donut, Near-full 등 특정 공간 불량 패턴이 있는 wafer

## 평가 지표

Test set에서 `pattern` class의 비율이 낮기 때문에 accuracy만으로는 모델을 평가하기 어렵다.

따라서 다음 지표를 함께 사용했다.

- `pattern precision`: pattern이라고 예측한 wafer 중 실제 pattern wafer의 비율
- `pattern recall`: 실제 pattern wafer 중 모델이 탐지한 비율
- `pattern f1-score`: precision과 recall의 균형
- `confusion matrix`: 오탐과 미탐 방향 확인

## Baseline 모델

### Logistic Regression

32x32 wafer map을 1차원 feature vector로 펼친 뒤 Logistic Regression을 학습했다.

주요 결과:

- accuracy: 0.57
- pattern precision: 0.10
- pattern recall: 0.72
- pattern f1-score: 0.18

해석:

Logistic Regression은 실제 pattern wafer를 비교적 많이 탐지했지만, pattern으로 잘못 경고한 non_pattern wafer도 많았다. 따라서 recall은 높지만 precision이 낮아 실제 운영 모델로 사용하기에는 오탐 부담이 크다.

### Random Forest

Random Forest 모델을 학습한 뒤, pattern 예측 확률 threshold를 조정했다.

기본 threshold 0.5에서는 pattern recall은 높았지만 오탐이 과도했다. 이후 threshold별 precision, recall, f1-score를 비교했고, F1-score가 가장 높은 threshold 0.7을 baseline 운영점으로 선택했다.

주요 결과:

- threshold: 0.7
- accuracy: 0.70
- pattern precision: 0.11
- pattern recall: 0.49
- pattern f1-score: 0.18

해석:

Threshold 0.7은 기본 threshold보다 오탐을 줄이고 accuracy를 개선했지만, pattern recall은 감소했다. 또한 Logistic Regression 대비 뚜렷한 f1-score 개선은 없었다.

## 주요 인사이트

Threshold 조정은 모델 성능 자체를 바꾸는 것이 아니라, `놓침`과 `오탐` 사이의 운영 기준을 조정하는 과정이다.

이번 데이터에서는 실제 현장의 missed alarm 비용과 false alarm 비용을 알 수 없기 때문에, precision과 recall의 균형을 보는 f1-score를 기준으로 Random Forest threshold를 선택했다.

그러나 Logistic Regression과 Random Forest 모두 pattern precision이 낮아, 실제 운영 환경에서는 과도한 false alarm이 발생할 수 있다.

## 다음 단계

Baseline 모델은 wafer map을 1차원 feature vector로 펼쳐 사용하기 때문에, 공간 패턴을 충분히 반영하지 못할 가능성이 있다.

따라서 다음 단계에서는 CNN 기반 모델을 사용하여 wafer map의 공간 구조를 직접 학습하고, baseline 모델 대비 성능 개선 여부를 확인한다.