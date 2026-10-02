import numpy as np
import matplotlib.pyplot as plt
import pywt

# 데이터 파일
file_path = "data/A/a (1).csv"

# 데이터 불러오기
data = np.loadtxt(
    file_path,
    delimiter=",",
    skiprows=1
)

# 첫 번째 채널 사용
signal = data[:, 0]

# CWT 설정
scales = np.arange(1, 33)

coefficients, frequencies = pywt.cwt(
    signal,
    scales,
    "morl"
)

# CWT 결과
plt.figure(figsize=(10, 6))

plt.imshow(
    np.abs(coefficients),
    extent=[0, len(signal), scales[-1], scales[0]],
    aspect="auto",
    cmap="jet"
)

plt.xlabel("Time")
plt.ylabel("Scale")
plt.title("CWT of sEMG Signal")

plt.colorbar(label="Magnitude")
plt.tight_layout()

# 이미지 저장
plt.savefig("cwt_example.png", dpi=300)

plt.show()

print("CWT 이미지 저장 완료")