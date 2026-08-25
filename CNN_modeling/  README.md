# CNN Modeling

## 목적

이 단계에서는 32x32로 resize된 LSWMD wafer map 데이터를 사용하여 `non_pattern`과 `pattern`을 분류하는 CNN 모델을 학습했다.

이전 baseline 모델인 Logistic Regression과 Random Forest는 wafer map을 1차원 feature vector로 펼쳐 사용했다. 이 방식은 단순하고 빠르지만, wafer map의 2차원 공간 구조를 충분히 반영하지 못할 수 있다.

CNN은 wafer map의 공간 정보를 직접 학습하여 Center, Edge-Ring, Scratch, Loc과 같은 불량 패턴의 위치적 특징을 더 잘 반영할 수 있는지 확인하기 위해 사용했다.

## 데이터 입력

전처리된 데이터는 `data/processed` 폴더에 저장된 `.npy` 파일을 사용했다.

- `X_train_32.npy`
- `X_test_32.npy`
- `y_train_pattern.npy`
- `y_test_pattern.npy`
- `pattern_label_mapping.csv`

입력 데이터 형태는 다음과 같다.

- `X_train`: 54,355 x 32 x 32 x 1
- `X_test`: 118,595 x 32 x 32 x 1

Label은 다음과 같이 정의했다.

- `0`: non_pattern
- `1`: pattern

## CNN 모델 구조

사용한 CNN 모델은 다음과 같은 구조이다.

1. Conv2D layer
2. MaxPooling2D layer
3. Conv2D layer
4. MaxPooling2D layer
5. Flatten layer
6. Dense layer
7. Dropout layer
8. Sigmoid output layer

출력층에서는 sigmoid 함수를 사용하여 wafer map이 `pattern`일 확률을 계산했다.

## Threshold 조정

CNN의 기본 threshold는 0.5이다.

- 예측 확률 >= 0.5: pattern
- 예측 확률 < 0.5: non_pattern

그러나 threshold는 단순한 모델 설정값이 아니라, 양산관리 관점에서는 `놓침`과 `오탐` 사이의 운영 기준으로 해석할 수 있다.

따라서 threshold를 0.3부터 0.9까지 조정하며 다음 지표를 비교했다.

- pattern precision
- pattern recall
- pattern f1-score
- predicted pattern count

## 최종 선택 threshold

최종 threshold는 0.75로 선택했다.

Threshold 0.75에서의 주요 결과는 다음과 같다.

- accuracy: 0.9117
- pattern precision: 0.3406
- pattern recall: 0.3486
- pattern f1-score: 0.3446
- predicted pattern count: 8,079

이 threshold에서는 pattern precision과 recall이 한쪽으로 크게 치우치지 않고 균형을 이루었다.

## Baseline 대비 해석

Baseline 모델과 비교하면 다음과 같다.

| Model | Threshold | Accuracy | Pattern Precision | Pattern Recall | Pattern F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | default | 0.57 | 0.10 | 0.72 | 0.18 |
| Random Forest | 0.70 | 0.70 | 0.11 | 0.49 | 0.18 |
| CNN | 0.75 | 0.9117 | 0.3406 | 0.3486 | 0.3446 |

Logistic Regression은 pattern recall은 높았지만 precision이 낮아 오탐 부담이 컸다. Random Forest는 threshold 조정 후에도 Logistic Regression 대비 뚜렷한 f1-score 개선을 보이지 못했다.

반면 CNN은 wafer map의 2차원 공간 구조를 반영하여 pattern precision과 f1-score를 개선했다.

## 양산관리 관점의 의미

양산관리 관점에서는 실제 pattern wafer를 많이 탐지하는 것뿐 아니라, 과도한 오탐으로 현장 검토 부담을 키우지 않는 것도 중요하다.

CNN threshold 0.75는 pattern wafer를 무리하게 많이 경고하지 않으면서도, precision과 recall의 균형을 확보한 운영 기준으로 해석할 수 있다.

즉, CNN은 wafer map 공간 패턴을 활용하여 관리 대상 wafer를 더 현실적인 규모와 신뢰도로 선별하는 모델 후보로 볼 수 있다.

## 한계 및 다음 단계

현재 모델은 `non_pattern`과 `pattern`을 구분하는 이진 분류 모델이다.

다음 단계에서는 pattern wafer만 대상으로 다음과 같은 세부 불량 유형을 분류하는 다중분류 모델을 학습할 수 있다.

- Center
- Edge-Ring
- Edge-Loc
- Loc
- Scratch
- Random
- Donut
- Near-full

이를 통해 단순히 관리 대상 여부를 판단하는 것을 넘어, 어떤 유형의 공간 불량 패턴이 발생했는지까지 분석할 수 있다.