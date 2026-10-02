\# sEMG Classification using CWT and Deep Learning



\## 1. 프로젝트 개요



본 프로젝트는 sEMG(surface Electromyography, 표면 근전도) 신호를 이용하여 5개의 클래스(A\~E)를 분류하는 딥러닝 모델을 구현한 프로젝트이다.



수집된 sEMG 데이터를 전처리한 후 CWT(Continuous Wavelet Transform)를 이용하여 시간-주파수 특징으로 변환하고, DenseNet161과 ResNet18 모델을 이용하여 각 데이터를 분류하였다.



최종적으로 두 모델의 Accuracy, Precision, Recall, F1 Score를 비교하고 Confusion Matrix를 분석하여 모델의 성능과 오분류 특성을 확인하였다.



\---



\## 2. 데이터셋



\### 2.1 데이터 구성



데이터는 A\~E의 5개 클래스로 구성되어 있으며, 각 클래스마다 50개의 CSV 파일이 존재한다.



| 클래스    |    파일 수 |

| ------ | ------: |

| A      |      50 |

| B      |      50 |

| C      |      50 |

| D      |      50 |

| E      |      50 |

| \*\*전체\*\* | \*\*250\*\* |



각 CSV 파일은 2개의 sEMG 채널로 구성되어 있으며, 하나의 파일에는 3000개의 데이터가 저장되어 있다.



\* Sampling Frequency: 1000 Hz

\* 채널 수: 2

\* 파일당 데이터: 3000 samples

\* 클래스 수: 5



\### 2.2 Train / Test 데이터



전체 데이터를 학습 데이터와 테스트 데이터로 분리하여 사용하였다.



| 구분     |    파일 수 | Window 수 |

| ------ | ------: | -------: |

| Train  |     200 |     3800 |

| Test   |      50 |      950 |

| \*\*전체\*\* | \*\*250\*\* | \*\*4750\*\* |



각 클래스에서 40개의 파일을 학습에 사용하고 10개의 파일을 테스트에 사용하여 총 200개의 Train 파일과 50개의 Test 파일을 구성하였다.



\---



\## 3. 데이터 전처리



딥러닝 모델에 입력하기 전에 다음과 같은 전처리를 수행하였다.



\### 3.1 60 Hz Notch Filter



전원 노이즈와 같이 특정 주파수에서 발생하는 60 Hz 성분을 제거하기 위해 Notch Filter를 적용하였다.



\* Notch Frequency: 60 Hz

\* Q Factor: 30



\### 3.2 Band-pass Filter



sEMG 신호에서 필요한 주파수 영역을 사용하기 위해 20\~499 Hz 범위의 Band-pass Filter를 적용하였다.



\* Filter: 4차 Butterworth

\* 통과 대역: 20\~499 Hz



\### 3.3 Windowing



전체 신호를 일정한 길이의 Window로 분할하였다.



\* Window 크기: 300 samples

\* Sampling Frequency: 1000 Hz

\* Window 길이: 300 ms

\* Hop: 150 samples

\* 이동 간격: 150 ms



3000 samples의 하나의 파일에서 총 19개의 Window가 생성된다.



\### 3.4 Min-Max Normalization



각 Window에 대해 Min-Max Normalization을 적용하여 데이터를 0\~1 범위로 정규화하였다.



\### 3.5 CWT 변환



전처리된 sEMG 신호를 CWT(Continuous Wavelet Transform)를 사용하여 시간-주파수 영역으로 변환하였다.



\* Wavelet: Morlet (`morl`)

\* Scale: 1\~32

\* 기존 2개 채널을 각각 CWT로 변환

\* 두 채널의 평균값을 추가하여 총 3개 채널로 구성



따라서 최종적으로 하나의 입력 데이터는 다음과 같은 형태를 갖는다.



```text

(3, 32, 300)

```



\---



\## 4. 사용 모델



본 프로젝트에서는 동일한 데이터와 전처리 조건에서 두 가지 CNN 기반 모델을 학습하여 성능을 비교하였다.



\### 4.1 DenseNet161



DenseNet161을 사용하여 sEMG의 시간-주파수 특징을 분류하였다.



\* Model: DenseNet161

\* Pretrained Weight: 사용하지 않음

\* Classifier: 2208 → 5

\* Optimizer: Adam

\* Learning Rate: 0.001

\* Loss Function: Cross Entropy Loss

\* Batch Size: 16

\* Epoch: 5



최종 출력 클래스는 A, B, C, D, E의 5개이다.



\### 4.2 ResNet18



두 번째 모델로 ResNet18을 사용하였다.



\* Model: ResNet18

\* Pretrained Weight: 사용하지 않음

\* Fully Connected Layer: 512 → 5

\* Optimizer: Adam

\* Learning Rate: 0.001

\* Loss Function: Cross Entropy Loss

\* Batch Size: 16

\* Epoch: 5



DenseNet161과 동일한 데이터 및 전처리 조건에서 학습하여 두 모델의 성능을 비교하였다.



\---



\## 5. 학습 및 평가 방법



학습 과정에서는 Train 데이터 3800개 Window를 사용하였다.



각 Epoch마다 전체 Train 데이터를 학습하고 Loss를 기록하였다.



학습이 완료된 후 Test 데이터 950개 Window를 이용하여 모델의 성능을 평가하였다.



평가 지표로 다음 네 가지를 사용하였다.



\* Accuracy

\* Precision

\* Recall

\* F1 Score



Precision, Recall, F1 Score는 각 클래스별 값을 계산한 후 Macro Average를 사용하였다.



Confusion Matrix는 실제 클래스와 모델이 예측한 클래스를 비교하여 클래스별 분류 특성을 확인하였다.



\---



\## 6. 실행 방법



프로젝트 폴더에서 다음 명령어를 실행한다.



\### DenseNet161 학습 및 평가



```powershell

python train\_densenet.py

```



실행 후 다음 파일이 생성된다.



```text

densenet161\_semg.pth

loss\_curve.png

confusion\_matrix.png

densenet161\_results.txt

```



\### ResNet18 학습 및 평가



```powershell

python train\_resnet.py

```



실행 후 다음 파일이 생성된다.



```text

resnet18\_semg.pth

resnet18\_loss\_curve.png

resnet18\_confusion\_matrix.png

resnet18\_results.txt

```



학습된 `.pth` 모델 파일은 용량이 크기 때문에 GitHub 저장소에는 포함하지 않고 `.gitignore`로 관리하였다.



\---



\## 7. 주요 파일 설명



| 파일 / 폴더                         | 설명                           |

| ------------------------------- | ---------------------------- |

| `data/`                         | 원본 sEMG CSV 데이터              |

| `dataset\_split/`                | Train / Test로 분리된 데이터        |

| `train\_densenet.py`             | DenseNet161 학습 및 평가          |

| `train\_resnet.py`               | ResNet18 학습 및 평가             |

| `evaluate.py`                   | 학습된 DenseNet161 평가 코드        |

| `dataset\_summary.py`            | 데이터셋 구성 확인                   |

| `plot\_signal.py`                | sEMG 신호 파형 확인                |

| `cwt\_example.py`                | CWT 변환 예시 확인                 |

| `make\_window.py`                | Windowing 과정 확인용 코드          |

| `semg\_split.py`                 | 데이터 분할 확인용 코드                |

| `split\_dataset.py`              | 데이터 분할 관련 코드                 |

| `check\_cuda.py`                 | PyTorch 및 CUDA 환경 확인         |

| `loss\_curve.py`                 | Loss 그래프 확인 관련 코드            |

| `confusion\_matrix.png`          | DenseNet161 Confusion Matrix |

| `resnet18\_confusion\_matrix.png` | ResNet18 Confusion Matrix    |

| `loss\_curve.png`                | DenseNet161 학습 Loss          |

| `resnet18\_loss\_curve.png`       | ResNet18 학습 Loss             |

| `densenet161\_results.txt`       | DenseNet161 평가 결과            |

| `resnet18\_results.txt`          | ResNet18 평가 결과               |

| `densenet161\_semg.pth`          | DenseNet161 학습 모델            |

| `resnet18\_semg.pth`             | ResNet18 학습 모델               |



\---



\## 8. 모델 성능 비교



두 모델을 동일한 Train/Test 데이터와 동일한 전처리 조건에서 학습하여 성능을 비교하였다.



| Metric    | DenseNet161 | ResNet18 |

| --------- | ----------: | -------: |

| Accuracy  |  \*\*84.63%\*\* |   76.21% |

| Precision |  \*\*84.84%\*\* |   78.89% |

| Recall    |  \*\*84.63%\*\* |   76.21% |

| F1 Score  |  \*\*84.68%\*\* |   76.06% |



DenseNet161은 Accuracy 84.63%를 기록하였으며, ResNet18은 76.21%를 기록하였다.



Accuracy 기준으로 두 모델 사이에는 약 8.42%p의 차이가 있었다.



F1 Score 역시 DenseNet161이 84.68%, ResNet18이 76.06%로 DenseNet161이 더 높은 결과를 보였다.



동일한 데이터와 전처리 조건에서 비교했을 때 DenseNet161이 이번 실험에서 더 높은 분류 성능을 나타냈다.



\---



\## 9. Confusion Matrix 분석



\### 9.1 DenseNet161



DenseNet161의 Confusion Matrix는 다음과 같다.



```text

\[\[185   2   1   0   2]

&#x20;\[  1 150   9  11  19]

&#x20;\[  0  22 162   6   0]

&#x20;\[  2   7  15 160   6]

&#x20;\[  1  25   5  12 147]]

```



클래스별 정확도는 다음과 같다.



| Class | Accuracy |

| ----- | -------: |

| A     |   97.37% |

| B     |   78.95% |

| C     |   85.26% |

| D     |   84.21% |

| E     |   77.37% |



A 클래스가 97.37%로 가장 높은 분류 정확도를 보였다.



반면 E 클래스가 77.37%로 가장 낮은 정확도를 보였으며, B 클래스도 78.95%로 상대적으로 낮았다.



가장 크게 나타난 오분류는 다음과 같다.



\* E → B: 25개

\* C → B: 22개

\* B → E: 19개

\* D → C: 15개



특히 B와 E 사이에서 서로 오분류되는 경우가 많이 나타났다.



이는 두 클래스의 sEMG 신호가 전처리 및 CWT 변환 이후 유사한 특징을 가지고 있을 가능성이 있다. 또한 현재 데이터의 파일 수가 클래스당 50개로 제한되어 있기 때문에 일부 클래스의 특징을 충분히 학습하지 못했을 가능성도 있다.



다만 이러한 원인은 Confusion Matrix만으로 직접 확인할 수 있는 사실이 아니라, 오분류 결과를 바탕으로 추정한 원인이다.



\---



\### 9.2 ResNet18



ResNet18의 Confusion Matrix는 다음과 같다.



```text

\[\[172   1  16   1   0]

&#x20;\[  3 113  28  33  13]

&#x20;\[  0   3 144  41   2]

&#x20;\[  3   0   8 176   3]

&#x20;\[ 11  15  14  31 119]]

```



클래스별 정확도는 다음과 같다.



| Class | Accuracy |

| ----- | -------: |

| A     |   90.53% |

| B     |   59.47% |

| C     |   75.79% |

| D     |   92.63% |

| E     |   62.63% |



D 클래스가 92.63%로 가장 높은 분류 정확도를 보였고, A 클래스도 90.53%로 높은 결과를 보였다.



반면 B 클래스는 59.47%로 가장 낮은 정확도를 보였다.



가장 크게 나타난 오분류는 다음과 같다.



\* C → D: 41개

\* B → D: 33개

\* E → D: 31개

\* B → C: 28개



ResNet18에서는 여러 클래스가 D 클래스로 오분류되는 경향이 나타났다.



이러한 결과는 ResNet18이 현재의 CWT 입력 특징을 학습하는 과정에서 일부 클래스의 특징을 명확하게 구분하지 못했을 가능성을 보여준다.



특히 C, B, E 클래스에서 D 클래스로의 오분류가 상대적으로 많이 나타났으며, 이는 클래스 간 특징이 일부 유사하거나 현재 전처리 방식에서 클래스 구분에 필요한 특징이 충분히 표현되지 않았을 가능성이 있다.



이 역시 Confusion Matrix 결과를 바탕으로 한 추정이며, 실제 원인을 확인하기 위해서는 추가적인 특징 분석이나 데이터 증가 실험이 필요하다.



\---



\## 10. 최종 결과 및 고찰



이번 실험에서는 CWT를 이용하여 sEMG 신호를 시간-주파수 특징으로 변환한 후 DenseNet161과 ResNet18을 이용하여 5개 클래스를 분류하였다.



두 모델 모두 동일한 Train/Test 데이터와 전처리 과정을 사용하였다.



실험 결과 DenseNet161은 Accuracy 84.63%, F1 Score 84.68%를 기록하였으며 ResNet18은 Accuracy 76.21%, F1 Score 76.06%를 기록하였다.



DenseNet161에서는 A 클래스가 가장 높은 정확도를 보였고, B와 E 클래스 사이의 오분류가 상대적으로 많이 나타났다.



ResNet18에서는 D 클래스의 정확도가 높게 나타났지만, B와 E 클래스의 정확도가 낮았으며 C → D, B → D와 같은 오분류가 많이 발생하였다.



따라서 이번 실험에서는 DenseNet161이 전체적인 분류 성능 측면에서 더 높은 결과를 보였다.



다만 현재 실험은 클래스별 데이터가 50개로 제한되어 있고 Epoch도 5회로 설정되어 있기 때문에 추가적인 데이터 확보, 학습 횟수 조정, 전처리 방법 변경 등을 통해 성능이 달라질 수 있다.



향후에는 클래스별 데이터의 특징을 추가로 분석하고, 오분류가 많이 발생하는 B, C, E 클래스에 대한 데이터를 확인하여 모델의 분류 성능을 개선할 필요가 있다.



