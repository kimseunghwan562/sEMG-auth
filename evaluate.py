import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet161
from pathlib import Path
from scipy.signal import butter, iirnotch, filtfilt
import pywt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


FS = 1000
WIN = 300
HOP = 150
SCALES = np.arange(1, 33)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("사용 장치:", device)


# =========================
# 전처리
# =========================

def preprocess(x):
    # 60Hz 노치 필터
    bn, an = iirnotch(60, 30, FS)
    x = filtfilt(bn, an, x, axis=0)

    # 20~499Hz 밴드패스 필터
    b, a = butter(
        4,
        [20 / (FS / 2), 499 / (FS / 2)],
        btype="band"
    )

    return filtfilt(b, a, x, axis=0)


def load_csv(file):
    return np.loadtxt(
        file,
        delimiter=",",
        skiprows=1,
        dtype=np.float32
    )


def make_windows(x):
    windows = []

    for start in range(0, len(x) - WIN + 1, HOP):
        window = x[start:start + WIN]
        windows.append(window)

    return windows


def minmax(window):
    mn = window.min()
    mx = window.max()

    return (window - mn) / (mx - mn + 1e-8)


def to_cwt(window):
    maps = []

    for ch in range(2):
        coef, _ = pywt.cwt(
            window[:, ch],
            SCALES,
            "morl"
        )

        maps.append(np.abs(coef))

    # 두 채널 평균
    maps.append((maps[0] + maps[1]) / 2)

    return np.stack(maps).astype(np.float32)


label_map = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4
}


# =========================
# 테스트 데이터 만들기
# =========================

def make_dataset(folder):

    X = []
    y = []

    files = list(Path(folder).glob("*/*.csv"))

    print()
    print(folder, "파일 수:", len(files))

    for i, file in enumerate(files):

        signal = load_csv(file)

        signal = preprocess(signal)

        windows = make_windows(signal)

        label = label_map[file.parent.name]

        for window in windows:

            window = minmax(window)

            cwt = to_cwt(window)

            X.append(cwt)
            y.append(label)

        print(
            f"{i + 1}/{len(files)} 완료",
            end="\r"
        )

    print()

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)

    print("데이터 모양:", X.shape)
    print("라벨 모양:", y.shape)

    return X, y


X_test, y_test = make_dataset(
    "dataset_split/test"
)


# =========================
# Dataset
# =========================

class SEMGDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


test_dataset = SEMGDataset(
    X_test,
    y_test
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# =========================
# 모델 불러오기
# =========================

model = densenet161(weights=None)

model.classifier = nn.Linear(
    2208,
    5
)

model = model.to(device)

model.load_state_dict(
    torch.load(
        "densenet161_semg.pth",
        map_location=device
    )
)

print()
print("모델 불러오기 완료")


# =========================
# 테스트
# =========================

model.eval()

correct = 0
total = 0

all_labels = []
all_predictions = []


with torch.no_grad():

    for X, y in test_loader:

        X = X.to(device)
        y = y.to(device)

        output = model(X)

        _, predicted = torch.max(
            output,
            1
        )

        total += y.size(0)

        correct += (
            predicted == y
        ).sum().item()

        all_labels.extend(
            y.cpu().numpy()
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )


accuracy = correct / total * 100


print()
print("===== 테스트 결과 =====")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"정답: {correct}/{total}")


# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print()
print("===== Confusion Matrix =====")
print(cm)


# =========================
# 클래스별 정확도
# =========================

class_names = [
    "A",
    "B",
    "C",
    "D",
    "E"
]

print()
print("===== 클래스별 정확도 =====")

for i, class_name in enumerate(class_names):

    class_total = cm[i].sum()

    class_correct = cm[i, i]

    class_accuracy = (
        class_correct /
        class_total *
        100
    )

    print(
        f"{class_name}: "
        f"{class_accuracy:.2f}% "
        f"({class_correct}/{class_total})"
    )


# =========================
# Confusion Matrix 이미지 저장
# =========================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()

plt.title(
    "sEMG DenseNet161 Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix_eval.png",
    dpi=150
)

plt.show()

print()
print(
    "Confusion Matrix 저장 완료: "
    "confusion_matrix_eval.png"
)