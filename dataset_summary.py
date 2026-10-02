import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pywt

# 데이터 폴더
data_dir = "data"

# 설정
fs = 1000
window_ms = 300
window_size = int(fs * window_ms / 1000)

scales = np.arange(1, 33)

# 전체 데이터 확인
all_files = []

for label in sorted(os.listdir(data_dir)):
    label_dir = os.path.join(data_dir, label)

    if os.path.isdir(label_dir):
        files = glob.glob(os.path.join(label_dir, "*.csv"))

        print(f"{label}: {len(files)}개")

        for file in files:
            all_files.append((file, label))

print()
print(f"전체 CSV 파일 수: {len(all_files)}개")

# 첫 번째 파일 사용
file_path, label = all_files[0]

data = np.loadtxt(
    file_path,
    delimiter=",",
    skiprows=1
)

print(f"사용한 파일: {file_path}")
print(f"데이터 배열 크기: {data.shape}")

# 첫 번째 채널 사용
signal = data[:, 0]

# 300 ms 윈도우
window = signal[:window_size]

print(f"윈도우 크기: {window.shape}")

# CWT
coefficients, frequencies = pywt.cwt(
    window,
    scales,
    "morl"
)

# CWT 이미지
plt.figure(figsize=(10, 6))

plt.imshow(
    np.abs(coefficients),
    extent=[0, window_ms, scales[-1], scales[0]],
    aspect="auto",
    cmap="jet"
)

plt.xlabel("Time (ms)")
plt.ylabel("Scale")
plt.title(f"CWT of 300 ms Window - Class {label}")

plt.colorbar(label="Magnitude")
plt.tight_layout()

plt.savefig(
    "dataset_cwt_example.png",
    dpi=300
)

plt.show()

print("CWT 이미지 저장 완료")