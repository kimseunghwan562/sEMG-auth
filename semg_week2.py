import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, iirnotch, filtfilt, welch

# ==========================================
# 1. 기본 설정
# ==========================================

FS = 1000


# ==========================================
# 2. 60Hz 노치 + 20~499Hz 대역통과 필터
# ==========================================

def preprocess(x, fs=FS):

    # 1) 60Hz 전원선 잡음 제거
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)

    # 2) 20~499Hz 대역만 통과 (4차)
    b, a = butter(
        4,
        [20 / (fs / 2), 499 / (fs / 2)],
        btype='band'
    )

    return filtfilt(b, a, x, axis=0)


# ==========================================
# 3. CSV 파일 불러오기
# ==========================================

files = list(Path("data").glob("*/*.csv"))

if len(files) == 0:
    print("CSV 파일을 찾을 수 없습니다.")
    exit()

file = files[0]

print("사용한 파일:", file)

# 첫 번째 줄은 헤더이므로 제외
x = np.loadtxt(
    file,
    delimiter=",",
    skiprows=1
)

print("원본 데이터 모양:", x.shape)


# ==========================================
# 4. 필터 적용
# ==========================================

xf = preprocess(x)

print("필터 후 데이터 모양:", xf.shape)
print("전:", x.std())
print("후:", xf.std())


# ==========================================
# 5. 주파수 분석
# ==========================================

# 원본 신호
f0, P0 = welch(
    x[:, 0],
    fs=FS,
    nperseg=512
)

# 필터 후 신호
f1, P1 = welch(
    xf[:, 0],
    fs=FS,
    nperseg=512
)


# ==========================================
# 6. 필터 검증 그래프
# ==========================================

plt.figure(figsize=(9, 4))

plt.semilogy(
    f0,
    P0,
    label="before",
    linewidth=0.8
)

plt.semilogy(
    f1,
    P1,
    label="after",
    linewidth=0.8
)

# 60Hz 위치 표시
plt.axvline(
    60,
    color="red",
    linestyle="--",
    label="60 Hz"
)

# 0~300Hz 표시
plt.xlim(0, 300)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")

plt.title("sEMG Filter Verification")

plt.legend()

plt.tight_layout()


# ==========================================
# 7. 이미지 저장
# ==========================================

plt.savefig(
    "filter_check.png",
    dpi=150
)

print("결과 이미지 저장 완료: filter_check.png")

plt.show()
# ==========================================
# 8. CWT 변환
# ==========================================

import pywt

# CWT scale 설정
SCALES = np.arange(1, 33)


def to_cwt(one_window, wavelet='morl'):
    """
    one_window: (300, 2)
    결과: (3, 32, 300)
    """

    maps = []

    # 채널 1, 2를 각각 CWT 변환
    for ch in range(one_window.shape[1]):
        coef, _ = pywt.cwt(
            one_window[:, ch],
            SCALES,
            wavelet
        )

        maps.append(np.abs(coef))

    # 두 채널의 평균을 3번째 채널로 추가
    maps.append(
        (maps[0] + maps[1]) / 2
    )

    return np.stack(maps).astype(np.float32)


# ==========================================
# 9. 300ms 윈도우 하나 만들기
# ==========================================

WIN = 300

one_window = xf[:WIN]

print("윈도우 모양:", one_window.shape)


# ==========================================
# 10. CWT 변환
# ==========================================

cwt_result = to_cwt(one_window)

print("CWT 입력 텐서 모양:", cwt_result.shape)


# ==========================================
# 11. CWT 이미지 저장
# ==========================================

plt.figure(figsize=(10, 4))

plt.imshow(
    cwt_result[0],
    aspect='auto',
    origin='lower'
)

plt.xlabel("Time (samples)")
plt.ylabel("Scale")

plt.title("CWT Example")

plt.colorbar(label="Magnitude")

plt.tight_layout()

plt.savefig(
    "cwt_example.png",
    dpi=150
)

print("CWT 이미지 저장 완료: cwt_example.png")

plt.show()
# ==========================================
# 12. 학습 / 테스트 데이터 분할
# ==========================================

from sklearn.model_selection import train_test_split

# 전체 CSV 파일
files = list(Path("data").glob("*/*.csv"))

# 각 파일의 클래스 확인
labels = [f.parent.name for f in files]

# 학습 80%, 테스트 20%
tr_files, te_files = train_test_split(
    files,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

print()
print("===== 학습 / 테스트 데이터 분할 =====")
print("전체 파일 수:", len(files))
print("학습 파일 수:", len(tr_files))
print("테스트 파일 수:", len(te_files))

print()
print("학습 데이터 클래스별 개수:")

for cls in ["A", "B", "C", "D", "E"]:
    count = sum(f.parent.name == cls for f in tr_files)
    print(cls, ":", count)

print()
print("테스트 데이터 클래스별 개수:")

for cls in ["A", "B", "C", "D", "E"]:
    count = sum(f.parent.name == cls for f in te_files)
    print(cls, ":", count)