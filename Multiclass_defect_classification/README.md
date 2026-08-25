# Multiclass Defect Pattern Classification

## 목적

이 단계에서는 `pattern` wafer만 대상으로 세부 불량 패턴을 분류하는 CNN 모델을 학습했다.

이전 단계에서는 wafer map이 `non_pattern`인지 `pattern`인지 구분하는 이진 분류를 수행했다. 이번 단계에서는 pattern wafer 내부에서 어떤 공간 불량 패턴이 발생했는지 분류하는 것을 목표로 한다.

분류 대상 class는 다음과 같다.

- Center
- Donut
- Edge-Loc
- Edge-Ring
- Loc
- Near-full
- Random
- Scratch

## 데이터 구성

전체 LSWMD 데이터 중 `failureType` label이 존재하고, `failure_label`이 `none`이 아닌 wafer만 사용했다.

- pattern wafer 수: 25,519개
- Training set: 17,625개
- Test set: 7,894개
- 입력 이미지 크기: 32x32x1

Class mapping은 다음과 같다.

| Index | Class |
|---:|---|
| 0 | Center |
| 1 | Donut |
| 2 | Edge-Loc |
| 3 | Edge-Ring |
| 4 | Loc |
| 5 | Near-full |
| 6 | Random |
| 7 | Scratch |

## Class Imbalance

다중분류 데이터는 class 불균형이 크다.

Training set 기준 class 분포는 다음과 같다.

| Class | Count |
|---|---:|
| Edge-Ring | 8,554 |
| Center | 3,462 |
| Edge-Loc | 2,417 |
| Loc | 1,620 |
| Random | 609 |
| Scratch | 500 |
| Donut | 409 |
| Near-full | 54 |

특히 `Near-full`은 training sample이 54개뿐이므로, 일반 CNN 모델이 충분히 학습하기 어렵다.

## 기본 CNN 결과

기본 CNN은 class weight 없이 학습했다.

주요 결과:

- accuracy: 0.51
- macro f1-score: 0.40
- weighted f1-score: 0.44

Class별 특징:

- Edge-Ring recall: 0.87
- Random recall: 0.91
- Donut recall: 0.77
- Center recall: 0.58
- Near-full recall: 0.00
- Scratch recall: 0.00

기본 CNN은 Edge-Ring, Random, Donut처럼 공간 구조가 뚜렷한 class는 비교적 잘 탐지했지만, Near-full과 Scratch 같은 일부 class는 거의 탐지하지 못했다.

## Weighted CNN 결과

Class imbalance를 보정하기 위해 `class_weight`를 적용한 CNN을 학습했다.

주요 결과:

- accuracy: 0.43
- macro f1-score: 0.44
- weighted f1-score: 0.42

Class별 특징:

- Near-full recall: 1.00
- Scratch recall: 0.50
- Donut recall: 0.74
- Edge-Ring recall: 0.62

Weighted CNN은 전체 accuracy는 낮아졌지만, Near-full과 Scratch 같은 희귀 class의 recall을 크게 개선했다. 또한 macro f1-score가 0.40에서 0.44로 상승하여 class별 균형성이 개선되었다.

## 모델 비교

| Model | Accuracy | Macro F1 | Weighted F1 | Near-full Recall | Scratch Recall |
|---|---:|---:|---:|---:|---:|
| CNN | 0.51 | 0.40 | 0.44 | 0.00 | 0.00 |
| Weighted CNN | 0.43 | 0.44 | 0.42 | 1.00 | 0.50 |

## 양산관리 관점의 해석

기본 CNN은 전체 accuracy가 높지만, 발생 빈도는 낮아도 수율 리스크가 큰 `Near-full` class를 탐지하지 못했다.

반면 Weighted CNN은 전체 accuracy를 일부 희생하는 대신, 희귀하고 고위험일 수 있는 불량 패턴을 더 잘 탐지했다.

따라서 모델 선택은 단순히 accuracy가 높은 모델을 고르는 문제가 아니라, 운영 목적에 따라 달라진다.

- 전체 자동 분류 정확도 우선: 기본 CNN
- 희귀 고위험 패턴 탐지 우선: Weighted CNN

양산관리 관점에서는 생산/품질 리스크가 큰 wafer를 놓치지 않는 것이 중요할 수 있으므로, Weighted CNN은 고위험 패턴 선별 목적의 모델 후보로 해석할 수 있다.

## 주요 인사이트

이번 실험에서는 단순히 accuracy가 높은 모델을 선택하지 않았다.

Class imbalance로 인해 소수 class가 무시되는 문제를 확인했고, `class_weight`를 적용하여 소수 class 탐지 성능을 개선했다.

이를 통해 모델 성능 평가는 단일 지표가 아니라, 현장에서 어떤 불량 패턴을 놓치면 안 되는지에 따라 달라져야 함을 확인했다.

## 한계 및 다음 단계

현재 모델은 32x32로 resize된 wafer map을 사용했다. Scratch처럼 얇은 선형 패턴은 resize 과정에서 일부 정보가 손실되었을 가능성이 있다.

다음 단계에서는 다음 개선을 검토할 수 있다.

- 64x64 resize와 성능 비교
- data augmentation 적용
- 더 깊은 CNN 구조 적용
- class별 confusion matrix 시각화
- unlabeled wafer에 대한 pseudo-labeling 또는 이상탐지 확장