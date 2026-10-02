import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from pathlib import Path
from scipy.signal import butter, iirnotch, filtfilt
import pywt


# ==========================================
# 1. 기본 설정
# ==========================================

FS = 1000
WIN = 300
HOP = 150
SCALES = np.arange(1, 33)

CLASS_NAMES = ["A", "B", "C", "D", "E"]

device = "cuda" if torch.cuda.is_available() else "cpu"

print("사용 장치:", device)


# ==========================================
# 2. 전처리
# ==========================================

def preprocess(x):

    # 60Hz 노치 필터
    bn, an = iirnotch(60, 30, FS)

    x = filtfilt(
        bn,
        an,
        x,
        axis=0
    )

    # 20~499Hz 대역통과 필터
    b, a = butter(
        4,
        [
            20 / (FS / 2),
            499 / (FS / 2)
        ],
        btype="band"
    )

    return filtfilt(
        b,
        a,
        x,
        axis=0
    )


# ==========================================
# 3. CSV 불러오기
# ==========================================

def load_csv(file):

    return np.loadtxt(
        file,
        delimiter=",",
        skiprows=1,
        dtype=np.float32
    )


# ==========================================
# 4. 300ms 윈도우 분할
# ==========================================

def make_windows(x):

    windows = []

    for start in range(
        0,
        len(x) - WIN + 1,
        HOP
    ):

        window = x[
            start:start + WIN
        ]

        windows.append(window)

    return windows


# ==========================================
# 5. Min-Max 정규화
# ==========================================

def minmax(window):

    mn = window.min()
    mx = window.max()

    return (
        window - mn
    ) / (
        mx - mn + 1e-8
    )


# ==========================================
# 6. CWT
# ==========================================

def to_cwt(window):

    maps = []

    for ch in range(2):

        coef, _ = pywt.cwt(
            window[:, ch],
            SCALES,
            "morl"
        )

        maps.append(
            np.abs(coef)
        )

    # 3번째 채널
    # 두 채널의 평균
    maps.append(
        (maps[0] + maps[1]) / 2
    )

    return np.stack(
        maps
    ).astype(np.float32)


# ==========================================
# 7. 데이터셋 만들기
# ==========================================

label_map = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4
}


def make_dataset(folder):

    X = []
    y = []

    files = list(
        Path(folder).glob("*/*.csv")
    )

    print()
    print(
        folder,
        "파일 수:",
        len(files)
    )

    for i, file in enumerate(files):

        signal = load_csv(file)

        signal = preprocess(signal)

        windows = make_windows(signal)

        label = label_map[
            file.parent.name
        ]

        for window in windows:

            window = minmax(
                window
            )

            cwt = to_cwt(
                window
            )

            X.append(cwt)
            y.append(label)

        print(
            f"{i + 1}/{len(files)} 완료",
            end="\r"
        )

    print()

    X = np.stack(X)

    y = np.array(
        y,
        dtype=np.int64
    )

    print(
        "데이터 모양:",
        X.shape
    )

    print(
        "라벨 모양:",
        y.shape
    )

    return X, y


# ==========================================
# 8. Train / Test 데이터 생성
# ==========================================

print()
print("===== Train 데이터 생성 =====")

X_train, y_train = make_dataset(
    "dataset_split/train"
)

print()
print("===== Test 데이터 생성 =====")

X_test, y_test = make_dataset(
    "dataset_split/test"
)


# ==========================================
# 9. PyTorch Dataset
# ==========================================

class SEMGDataset(Dataset):

    def __init__(
        self,
        X,
        y
    ):

        self.X = torch.tensor(X)

        self.y = torch.tensor(y)

    def __len__(self):

        return len(self.y)

    def __getitem__(
        self,
        index
    ):

        return (
            self.X[index],
            self.y[index]
        )


train_dataset = SEMGDataset(
    X_train,
    y_train
)

test_dataset = SEMGDataset(
    X_test,
    y_test
)


# ==========================================
# 10. DataLoader
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# ==========================================
# 11. ResNet18
# ==========================================

model = resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    5
)

model = model.to(device)


# ==========================================
# 12. Loss / Optimizer
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 13. 학습
# ==========================================

EPOCHS = 5

loss_history = []

print()
print("===== ResNet18 학습 시작 =====")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for X, y in train_loader:

        X = X.to(device)

        y = y.to(device)

        optimizer.zero_grad()

        output = model(X)

        loss = criterion(
            output,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
        )

    average_loss = (
        total_loss /
        len(train_loader)
    )

    loss_history.append(
        average_loss
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )


# ==========================================
# 14. 모델 저장
# ==========================================

torch.save(
    model.state_dict(),
    "resnet18_semg.pth"
)

print()
print(
    "모델 저장 완료: "
    "resnet18_semg.pth"
)


# ==========================================
# 15. Loss 그래프
# ==========================================

plt.figure()

plt.plot(
    range(
        1,
        EPOCHS + 1
    ),
    loss_history,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "ResNet18 Training Loss"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "resnet18_loss_curve.png",
    dpi=150
)

plt.close()

print(
    "Loss 그래프 저장 완료: "
    "resnet18_loss_curve.png"
)


# ==========================================
# 16. 테스트
# ==========================================

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


accuracy = (
    correct /
    total *
    100
)

print()
print("===== 테스트 결과 =====")

print(
    f"Test Accuracy: "
    f"{accuracy:.2f}%"
)

print(
    f"정답: {correct}/{total}"
)


# ==========================================
# 17. Confusion Matrix
# ==========================================

all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)

cm = np.zeros(
    (
        len(CLASS_NAMES),
        len(CLASS_NAMES)
    ),
    dtype=int
)

for true_label, predicted_label in zip(
    all_labels,
    all_predictions
):

    cm[
        true_label,
        predicted_label
    ] += 1


print()
print(
    "===== Confusion Matrix ====="
)

print(cm)


# ==========================================
# 18. Confusion Matrix 이미지
# ==========================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "sEMG ResNet18 Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

for i in range(
    len(CLASS_NAMES)
):

    for j in range(
        len(CLASS_NAMES)
    ):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "resnet18_confusion_matrix.png",
    dpi=150
)

plt.close()

print(
    "Confusion Matrix 저장 완료: "
    "resnet18_confusion_matrix.png"
)


# ==========================================
# 19. 클래스별 정확도
# ==========================================

print()
print(
    "===== 클래스별 정확도 ====="
)

class_accuracies = []

for i, class_name in enumerate(
    CLASS_NAMES
):

    class_total = cm[i].sum()

    class_correct = cm[i, i]

    if class_total > 0:

        class_accuracy = (
            class_correct /
            class_total *
            100
        )

    else:

        class_accuracy = 0

    class_accuracies.append(
        class_accuracy
    )

    print(
        f"{class_name}: "
        f"{class_accuracy:.2f}% "
        f"({class_correct}/{class_total})"
    )


# ==========================================
# 20. Precision / Recall / F1
# ==========================================

precision_list = []

recall_list = []

f1_list = []

for i in range(
    len(CLASS_NAMES)
):

    tp = cm[i, i]

    fp = (
        cm[:, i].sum()
        - tp
    )

    fn = (
        cm[i, :].sum()
        - tp
    )

    precision = (
        tp /
        (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp /
        (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    if (
        precision + recall
        > 0
    ):

        f1 = (
            2 *
            precision *
            recall /
            (
                precision +
                recall
            )
        )

    else:

        f1 = 0

    precision_list.append(
        precision
    )

    recall_list.append(
        recall
    )

    f1_list.append(
        f1
    )


macro_precision = (
    np.mean(
        precision_list
    ) * 100
)

macro_recall = (
    np.mean(
        recall_list
    ) * 100
)

macro_f1 = (
    np.mean(
        f1_list
    ) * 100
)


print()
print(
    "===== 전체 평가 지표 ====="
)

print(
    f"Accuracy : "
    f"{accuracy:.2f}%"
)

print(
    f"Precision: "
    f"{macro_precision:.2f}%"
)

print(
    f"Recall   : "
    f"{macro_recall:.2f}%"
)

print(
    f"F1 Score : "
    f"{macro_f1:.2f}%"
)


# ==========================================
# 21. 결과 저장
# ==========================================

with open(
    "resnet18_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ResNet18 sEMG Classification Results\n"
    )

    f.write(
        "=" * 50 + "\n\n"
    )

    f.write(
        f"Accuracy : "
        f"{accuracy:.2f}%\n"
    )

    f.write(
        f"Precision: "
        f"{macro_precision:.2f}%\n"
    )

    f.write(
        f"Recall   : "
        f"{macro_recall:.2f}%\n"
    )

    f.write(
        f"F1 Score : "
        f"{macro_f1:.2f}%\n\n"
    )

    f.write(
        "Confusion Matrix\n"
    )

    f.write(
        "-" * 30 + "\n"
    )

    for row in cm:

        f.write(
            " ".join(
                map(str, row)
            ) + "\n"
        )

    f.write(
        "\nClass Accuracy\n"
    )

    f.write(
        "-" * 30 + "\n"
    )

    for class_name, class_accuracy in zip(
        CLASS_NAMES,
        class_accuracies
    ):

        f.write(
            f"{class_name}: "
            f"{class_accuracy:.2f}%\n"
        )


print()
print(
    "결과 저장 완료: "
    "resnet18_results.txt"
)

print()
print(
    "===== ResNet18 학습 및 평가 완료 ====="
)