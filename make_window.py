import os
import glob
import random
import numpy as np

# =========================
# 기본 설정
# =========================

data_dir = "data"

fs = 1000
window_ms = 300
window_size = int(fs * window_ms / 1000)

# 50% overlap
stride = window_size // 2

classes = ["A", "B", "C", "D", "E"]

# 같은 Train/Test 분할을 사용하기 위해 seed 고정
random.seed(42)


# =========================
# Train / Test 파일 분리
# =========================

train_files = []
test_files = []

for label in classes:

    label_dir = os.path.join(data_dir, label)

    files = glob.glob(os.path.join(label_dir, "*.csv"))
    files.sort()

    random.shuffle(files)

    train = files[:40]
    test = files[40:]

    train_files.extend([(file, label) for file in train])
    test_files.extend([(file, label) for file in test])


# =========================
# Window 생성 함수
# =========================

def make_windows(file_list):

    windows = []
    labels = []

    for file_path, label in file_list:

        # CSV 읽기
        data = np.loadtxt(
            file_path,
            delimiter=",",
            skiprows=1
        )

        # 데이터 길이 확인
        signal_length = data.shape[0]

        # Window 생성
        count = 0

        for start in range(
            0,
            signal_length - window_size + 1,
            stride
        ):

            end = start + window_size

            window = data[start:end, :]

            windows.append(window)
            labels.append(label)

            count += 1

        print(
            f"{label}: {os.path.basename(file_path)} "
            f"-> {count}개 window"
        )

    return windows, labels


# =========================
# Train Window 생성
# =========================

print("===== Train Window 생성 =====")

train_windows, train_labels = make_windows(train_files)

print()


# =========================
# Test Window 생성
# =========================

print("===== Test Window 생성 =====")

test_windows, test_labels = make_windows(test_files)

print()


# =========================
# 결과 확인
# =========================

print("===== 결과 =====")

print(f"Train 파일 수 : {len(train_files)}")
print(f"Test 파일 수  : {len(test_files)}")

print()

print(f"Train window 수 : {len(train_windows)}")
print(f"Test window 수  : {len(test_windows)}")

print()

print(f"Window 하나의 크기 : {train_windows[0].shape}")

print()

print("예상 결과")
print("Train window : 3800")
print("Test window  : 950")
print("Window shape : (300, 2)")