# sEMG 손동작·팔꿈치 각도 분류 재현 실험

교수님이 제공한 공개 sEMG 데이터로 4가지 손동작과 2가지 팔꿈치 각도 조합, 총 8개 클래스를 분류한다. 논문에서 사용한 시간영역 특징과 RF·SVM·XGBoost 계열 모델을 참고해 파일 단위 그룹 5-fold 평가를 수행한다. 논문 초록은 5-fold를 명시하지만 fold 구성 단위는 공개 초록에서 확인되지 않아, 이 평가를 논문과 완전히 동일한 재현으로 주장하지 않는다.

## 참고 논문 및 데이터

- 논문: Lee, S., Kim, J. & Choi, S.-I. “Simultaneous classification of hand gestures and elbow angles using sEMG signals: a comparison of machine learning performances.” *Biomedical Engineering Letters* (2026). [DOI: 10.1007/s13534-026-00607-7](https://doi.org/10.1007/s13534-026-00607-7)
- 공개 데이터: [sEMG-Hand-Gesture-Elbow-Angle-Dataset](https://github.com/TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset)
- 데이터는 참가자 8명(a–h), 손동작 4종(extor, fist, open, pro), 팔꿈치 각도 2종(30°, 60°)의 64개 CSV로 구성된다. 각 CSV의 5개 sEMG 채널은 1 kHz로 기록되며, 파일별 30회 시행을 포함한다.

> 원본 CSV는 약 932 MB이므로 이 저장소에 복제하지 않는다. 위 공개 데이터 저장소에서 CSV 64개를 받아 `data/H/` 폴더에 둔다. 이 저장소의 `.gitignore`는 해당 원본 CSV를 Git 추적에서 제외한다. 전처리 결과 `week2_results/dataset.npz`와 재현 코드는 저장소에 포함한다.

## 실행 환경 및 방법

필요 패키지는 `requirements.txt`에 적혀 있다. 프로젝트 폴더에서 가상환경을 활성화한 뒤 아래 순서대로 실행한다.

```powershell
python -m pip install -r requirements.txt
python .\week2_h_pipeline.py
python .\evaluate_5fold.py
```

현재 Windows 환경에서는 가상환경 Python 3.12의 scikit-learn 네이티브 DLL이 애플리케이션 제어 정책에 차단됐다. 이 PC에서는 설치된 Python 3.14 실행 파일로 성공적으로 평가했다. 같은 문제가 나면 아래처럼 실행한다.

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe" .\week2_h_pipeline.py
& "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe" .\evaluate_5fold.py
```

## 코드 파일 설명

| 파일 | 역할 |
|---|---|
| `week2_h_pipeline.py` | CSV 검사, 필터링, 시행 추출, 2초 구간 자르기, 슬라이딩 윈도우, 25차원 특징 계산, 파일 단위 분할, 표준화, `dataset.npz`와 확인용 그림·요약 저장 |
| `train_models.py` | 고정 80:20 홀드아웃 기준으로 세 모델을 학습·평가하는 참고용 코드 |
| `evaluate_5fold.py` | 같은 CSV에서 나온 윈도우가 fold를 넘지 않도록 그룹 5-fold 교차검증; fold별 표준화 및 성능표·혼동행렬 저장 |
| `week2_results/dataset.npz` | 표준화된 학습/테스트 특징 배열, 라벨, 클래스명, 분할 파일명 |
| `week2_results/week2_summary.json` | 파일·시행·윈도우 수, 파라미터와 데이터 구조 요약 |
| `week2_results/week2_filter_spectrum.png` | 필터 전후 스펙트럼 비교 그림 |
| `week2_results/week2_segment_5channels.png` | 추출한 2초 5채널 신호 그림 |
| `week2_results/models/model_results.json` | 모델 성능 및 혼동행렬의 수치 결과 |
| `week2_results/models/model_results.md` | 모델 성능표와 혼동행렬 이미지 링크 |
| `week2_results/cv5_grouped/cv5_results.json` | 파일 그룹 5-fold 결과의 수치 데이터 |
| `week2_results/cv5_grouped/cv5_results.md` | 파일 그룹 5-fold 성능표와 혼동행렬 링크 |

## 데이터 처리 및 평가 설정

1. CSV의 `mark`로 각 시행을 찾는다. 시행 간격 중앙값은 5,000 samples(5초)이며 파일마다 30회 시행이 확인되었다.
2. 각 시행에 60 Hz 노치 필터(Q=30)와 20–450 Hz 4차 Butterworth 대역통과 필터를 적용한다. 제공 CSV의 채널명은 이미 `bandpass 1`–`bandpass 5`로 되어 있으므로, 이번 코드는 강의 실습 절차에 따라 필터를 한 번 더 적용했다. 이 점은 논문 원본 처리와 차이가 날 수 있다.
3. `mark` 기준 1,500 samples 뒤에서 2,000 samples(2초)를 자른다. 이는 강의자료 예시의 시작 위치를 따른 실습 설정이다.
4. 200 samples(200 ms) 윈도우를 100 samples(100 ms) 간격으로 이동한다. 따라서 겹침은 50%이고, 2초 구간마다 19개 윈도우가 생긴다.
5. 채널별 RMS, MAV, WL, ZC, SSC를 계산하여 윈도우 하나를 25개 값으로 표현한다. ZC·SSC 임계값은 실습 코드의 `0.01`을 사용한다.
6. CSV 단위 stratified 80:20 분할(random seed 42) 배열도 함께 저장한다. 학습 51개 CSV, 테스트 13개 CSV이며 윈도우는 각각 29,070개와 7,410개다. 이는 `train_models.py`의 단일 홀드아웃 참고 평가에 사용된다.
7. 주 성능표는 전체 윈도우에 대해 `StratifiedGroupKFold(n_splits=5)`를 사용한다. 원본 CSV를 그룹으로 지정해 같은 파일의 윈도우가 학습과 검증에 갈라지지 않게 하고, StandardScaler는 매 fold 학습 자료에만 맞춘다. 같은 참가자의 서로 다른 조건 파일은 fold 간에 나뉠 수 있다.

## 모델별 성능 비교: 파일 단위 그룹 5-fold

각 원본 CSV의 윈도우는 한 fold에만 배정했다. StandardScaler는 fold마다 학습 자료에만 맞췄다. 아래 값은 5개 fold 점수의 평균 ± 표준편차이며 Precision·Recall·F1은 macro 평균이다.

| 모델 | Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) |
|---|---:|---:|---:|---:|
| Random Forest | 10.33 ± 1.50% | 10.46 ± 1.41% | **10.91 ± 0.72%** | **10.30 ± 1.03%** |
| SVM (RBF) | **10.58 ± 3.20%** | **12.00 ± 3.02%** | 11.18 ± 2.64% | 10.27 ± 2.47% |
| XGBoost | 10.31 ± 1.85% | 10.49 ± 2.29% | 10.64 ± 1.46% | 10.11 ± 1.75% |

이번 평가에서는 SVM의 평균 Accuracy와 macro Precision이 가장 높았고, RF의 macro Recall과 macro F1이 가장 높았다. 세 모델의 차이는 작고 모두 낮다. 논문은 RF 5-fold Accuracy 96.72 ± 0.58%를 보고하지만, 공개 초록에서 fold 단위가 명시되지 않아 이 결과와 직접 같은 절차라고 확인할 수 없다. 따라서 논문 성능을 재현했다고 주장하지 않는다. 이전 단일 80:20 홀드아웃 결과는 [별도 요약](week2_results/models/model_results.md)에 남겨 두었다.

## Confusion Matrix 분석

행은 실제 클래스, 열은 예측 클래스 순서다: `30_extor`, `30_fist`, `30_open`, `30_pro`, `60_extor`, `60_fist`, `60_open`, `60_pro`.

### Random Forest — pooled out-of-fold 결과

![Random Forest confusion matrix](week2_results/cv5_grouped/random_forest_confusion_matrix.png)

- 가장 잘 분류된 클래스는 `60_fist`(재현율 20.9%)이고, 가장 낮은 클래스는 `60_pro`(3.3%)다.
- 주요 오분류는 `60_pro → 30_pro` 1,611개, `30_pro → 60_pro` 1,580개, `30_open → 60_open` 1,570개다.
- 같은 손동작에서 각도 조건이 다른 클래스 간 혼동이 보인다. 낮은 전체 성능을 고려해 제한적으로 해석한다.

### SVM (RBF)

![SVM confusion matrix](week2_results/cv5_grouped/svm_rbf_confusion_matrix.png)

- 가장 잘 분류된 클래스는 `30_extor`(18.1%)이고, 가장 낮은 클래스는 `60_pro`(4.0%)다.
- 주요 오분류는 `60_extor → 30_extor` 2,072개, `30_pro → 60_pro` 1,823개, `60_open → 30_extor` 1,438개다.
- 일부 예측이 `30_extor`에 쏠리고 fold 간 점수 편차도 비교적 크다.

### XGBoost

![XGBoost confusion matrix](week2_results/cv5_grouped/xgboost_confusion_matrix.png)

- 가장 잘 분류된 클래스는 `60_fist`(18.4%)이고, 가장 낮은 클래스는 `60_pro`(3.4%)다.
- 주요 오분류는 `60_pro → 30_pro` 1,615개, `30_pro → 60_pro` 1,586개, `30_open → 60_open` 1,408개다.
- RF와 마찬가지로 각도 조건이 다른 같은 동작 간 혼동이 반복되었다.

## 최종 결과 및 한계

- 파일 단위 5-fold에서 SVM Accuracy는 10.58%, RF macro F1은 10.30%로 가장 높았고, 모델 간 차이는 작았다.
- 30°/60° 같은 동작 조합 사이에서 큰 오분류가 반복되었다.
- 전체 Accuracy 약 10%는 8-class 균등 무작위 기준 12.5%보다 낮다. 레이블 정렬·구간 선택·특징 구현을 다시 확인해야 하며, 현재 결과를 성공적인 논문 재현으로 제출하면 안 된다.
- 이 평가는 동일 CSV의 윈도우 누수를 막지만, 참가자의 다른 조건 CSV가 서로 다른 fold에 갈 수 있으므로 참가자 독립 일반화 성능은 측정하지 않는다.
- 논문 절차와 수치를 직접 맞추려면 원문에서 fold 구성 단위와 모델 설정을 확인해야 한다. 공개된 초록은 5-fold 결과만 제시한다.

## 결과 그림 및 파일

- [필터 전후 스펙트럼](week2_results/week2_filter_spectrum.png)
- [5채널 2초 신호](week2_results/week2_segment_5channels.png)
- [모델 결과와 혼동행렬 모음](week2_results/models/model_results.md)
- [5-fold 결과와 혼동행렬 모음](week2_results/cv5_grouped/cv5_results.md)
- [전처리 요약 JSON](week2_results/week2_summary.json)
