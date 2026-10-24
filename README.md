\# sEMG Classification using CWT and Deep Learning



\## 1. 프로젝트 개요



이 프로젝트는 sEMG(표면 근전도) 데이터를 이용하여 A\~E 5개의 클래스를 분류하는 딥러닝 모델을 만들어 본 프로젝트이다.



sEMG 데이터를 전처리한 뒤 CWT를 적용하여 시간-주파수 데이터로 변환하고, DenseNet161과 ResNet18을 사용하여 분류하였다.



두 모델을 같은 조건에서 학습시킨 후 Accuracy, Precision, Recall, F1 Score를 비교하고 Confusion Matrix를 확인하였다.



\---



\## 2. 데이터



\### 데이터 구성



데이터는 A\~E 총 5개의 클래스로 구성되어 있고, 각 클래스마다 50개의 CSV 파일이 있다.



| 클래스    |    파일 수 |

| ------ | ------: |

| A      |      50 |

| B      |      50 |

| C      |      50 |

| D      |      50 |

| E      |      50 |

| \*\*전체\*\* | \*\*250\*\* |



각 CSV 파일은 2개의 sEMG 채널로 되어 있으며, 파일 하나당 3000개의 데이터가 들어 있다.



\* Sampling Frequency: 1000 Hz

\* 채널 수: 2

\* 파일당 데이터: 3000 samples

\* 클래스 수: 5



\### Train / Test



각 클래스에서 40개는 Train, 10개는 Test에 사용하였다.



| 구분     |    파일 수 | Window 수 |

| ------ | ------: | -------: |

| Train  |     200 |     3800 |

| Test   |      50 |      950 |

| \*\*전체\*\* | \*\*250\*\* | \*\*4750\*\* |



\---



\## 3. 데이터 전처리



모델에 넣기 전에 sEMG 데이터에 다음 전처리를 적용하였다.



\### 60 Hz Notch Filter



60 Hz 부근의 전원 노이즈를 제거하기 위해 Notch Filter를 적용하였다.



\* Notch Frequency: 60 Hz

\* Q Factor: 30



\### Band-pass Filter



20\~499 Hz 범위의 신호를 사용하기 위해 4차 Butterworth Band-pass Filter를 적용하였다.



\### Windowing



3000개의 데이터를 일정한 크기로 나누었다.



\* Window: 300 samples

\* Window 길이: 300 ms

\* Hop: 150 samples

\* 이동 간격: 150 ms



하나의 CSV 파일에서 19개의 Window가 만들어진다.



\### Min-Max Normalization



각 Window의 값을 0\~1 범위로 정규화하였다.



\### CWT



전처리한 신호에 CWT(Continuous Wavelet Transform)를 적용하였다.



\* Wavelet: Morlet (`morl`)

\* Scale: 1\~32

\* 기존 2개 채널을 각각 CWT 변환

\* 두 채널의 평균값을 추가



최종 입력 데이터의 형태는 다음과 같다.



```text

(3, 32, 300)

```



\---



\## 4. 사용 모델



이번 프로젝트에서는 DenseNet161과 ResNet18 두 가지 모델을 사용하였다.



\### DenseNet161



\* Model: DenseNet161

\* Pretrained Weight: 사용하지 않음

\* Classifier: 2208 → 5

\* Optimizer: Adam

\* Learning Rate: 0.001

\* Loss: Cross Entropy Loss

\* Batch Size: 16

\* Epoch: 5



\### ResNet18



\* Model: ResNet18

\* Pretrained Weight: 사용하지 않음

\* Fully Connected Layer: 512 → 5

\* Optimizer: Adam

\* Learning Rate: 0.001

\* Loss: Cross Entropy Loss

\* Batch Size: 16

\* Epoch: 5



두 모델 모두 같은 Train/Test 데이터와 전처리 방법을 사용하였다.



\---



\## 5. 학습 및 평가



Train 데이터 3800개의 Window를 이용하여 모델을 학습하였다.



학습이 끝난 후 Test 데이터 950개를 이용하여 성능을 확인하였다.



평가에는 다음 지표를 사용하였다.



\* Accuracy

\* Precision

\* Recall

\* F1 Score



Precision, Recall, F1 Score는 Macro Average를 사용하였다.



또한 Confusion Matrix를 통해 어떤 클래스에서 오분류가 많이 발생하는지 확인하였다.



\---



\## 6. 실행 방법



프로젝트 폴더에서 아래 명령어를 실행하면 된다.



\### DenseNet161



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



\### ResNet18



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



`.pth` 파일은 용량이 크기 때문에 GitHub에는 올리지 않고 `.gitignore`에 추가하였다.



\---



\## 7. 주요 파일



| 파일 / 폴더                         | 설명                           |

| ------------------------------- | ---------------------------- |

| `data/`                         | 원본 sEMG CSV 데이터              |

| `dataset\_split/`                | Train / Test 데이터             |

| `train\_densenet.py`             | DenseNet161 학습 및 평가          |

| `train\_resnet.py`               | ResNet18 학습 및 평가             |

| `evaluate.py`                   | DenseNet161 평가               |

| `dataset\_summary.py`            | 데이터셋 확인                      |

| `plot\_signal.py`                | sEMG 신호 확인                   |

| `cwt\_example.py`                | CWT 변환 확인                    |

| `make\_window.py`                | Windowing 관련 코드              |

| `semg\_split.py`                 | 데이터 분할 확인                    |

| `split\_dataset.py`              | 데이터 분할 관련 코드                 |

| `check\_cuda.py`                 | PyTorch / CUDA 확인            |

| `loss\_curve.py`                 | Loss 그래프 관련 코드               |

| `confusion\_matrix.png`          | DenseNet161 Confusion Matrix |

| `resnet18\_confusion\_matrix.png` | ResNet18 Confusion Matrix    |

| `loss\_curve.png`                | DenseNet161 Loss 그래프         |

| `resnet18\_loss\_curve.png`       | ResNet18 Loss 그래프            |

| `densenet161\_results.txt`       | DenseNet161 결과               |

| `resnet18\_results.txt`          | ResNet18 결과                  |



\---



\## 8. 모델 성능 비교



두 모델을 같은 데이터와 전처리 조건에서 학습하였다.



| Metric    | DenseNet161 | ResNet18 |

| --------- | ----------: | -------: |

| Accuracy  |  \*\*84.63%\*\* |   76.21% |

| Precision |  \*\*84.84%\*\* |   78.89% |

| Recall    |  \*\*84.63%\*\* |   76.21% |

| F1 Score  |  \*\*84.68%\*\* |   76.06% |



DenseNet161의 Accuracy는 84.63%, ResNet18은 76.21%로 나타났다.



Accuracy 차이는 약 8.42%p였다.



F1 Score도 DenseNet161 84.68%, ResNet18 76.06%로 나타났다.



이번 실험에서는 DenseNet161이 ResNet18보다 높은 성능을 보였다.



\---



\## 9. Confusion Matrix 분석



\### 9.1 DenseNet161



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



A 클래스의 정확도가 97.37%로 가장 높았다.



E 클래스는 77.37%로 가장 낮았고 B 클래스도 78.95%로 낮은 편이었다.



주요 오분류는 다음과 같다.



\* E → B: 25개

\* C → B: 22개

\* B → E: 19개

\* D → C: 15개



특히 B와 E 사이에서 오분류가 많이 발생하였다.



B와 E의 신호 특징이 일부 비슷해서 구분하기 어려웠을 가능성이 있다. 또한 클래스당 파일 수가 50개이기 때문에 데이터가 충분하지 않았을 가능성도 있다.



다만 정확한 원인은 추가적인 데이터와 특징 분석이 필요하다.



\### 9.2 ResNet18



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



D 클래스의 정확도가 92.63%로 가장 높았다.



B 클래스는 59.47%로 가장 낮았다.



주요 오분류는 다음과 같다.



\* C → D: 41개

\* B → D: 33개

\* E → D: 31개

\* B → C: 28개



ResNet18에서는 C, B, E 클래스가 D 클래스로 오분류되는 경우가 많이 나타났다.



현재 결과만 보면 일부 클래스의 특징을 충분히 구분하지 못한 것으로 볼 수 있다. 정확한 원인을 확인하려면 추가적인 데이터 분석이 필요하다.



\---



\## 10. 최종 결과



이번 프로젝트에서는 sEMG 데이터를 전처리하고 CWT를 적용한 뒤 DenseNet161과 ResNet18을 이용하여 A\~E 클래스를 분류하였다.



결과는 다음과 같다.



\* DenseNet161 Accuracy: \*\*84.63%\*\*

\* DenseNet161 F1 Score: \*\*84.68%\*\*

\* ResNet18 Accuracy: \*\*76.21%\*\*

\* ResNet18 F1 Score: \*\*76.06%\*\*



이번 실험에서는 DenseNet161이 ResNet18보다 높은 성능을 보였다.



DenseNet161에서는 B와 E 사이의 오분류가 많이 나타났고, ResNet18에서는 C, B, E에서 D로 오분류되는 경우가 많이 나타났다.



현재 데이터는 클래스당 50개의 파일로 구성되어 있고 학습도 5 Epoch로 진행했기 때문에, 데이터 수나 학습 조건을 변경하면 결과가 달라질 수 있다.



추가로 데이터를 확보하거나 전처리 방법 및 학습 조건을 변경하면 오분류가 많은 클래스의 성능을 개선할 수 있을 것으로 생각된다.



