# SECOM 데이터 전처리

이 문서는 UCI SECOM 반도체 제조 데이터셋의 전처리 과정과 현재까지의 전처리 결과를 정리합니다.

## 필요한 파일

전처리에 필요한 원본 파일은 다음과 같습니다.

- `secom.data`
- `secom_labels.data`

## 전처리 단계

1. `secom.data`에서 원본 SECOM feature 데이터를 불러옵니다.
2. `secom_labels.data`에서 pass/fail label 데이터를 불러옵니다.
3. 각 feature의 결측치 비율을 계산합니다.
4. 결측치 비율이 50% 이상인 feature를 제거합니다.
5. 남아 있는 결측치는 각 feature의 중앙값으로 대체합니다.
6. 분산이 0인 feature를 제거합니다.
7. 최종 정제된 feature table을 반환합니다.

## 현재 전처리 결과

| 단계 | 데이터 크기 |
|---|---:|
| 원본 feature table | 1,567 × 590 |
| 고결측 feature 제거 후 | 1,567 × 562 |
| 최종 정제 feature table | 1,567 × 446 |

## 추가 결과

- 제거된 고결측 feature 수: 28개
- 제거된 분산 0 feature 수: 116개
- 최종 결측치 수: 0개

## 참고 사항

SECOM 데이터셋의 feature 이름은 익명화되어 있습니다.

따라서 각 column은 실제 센서명이 아니라 익명화된 공정 또는 센서 feature로 해석합니다.