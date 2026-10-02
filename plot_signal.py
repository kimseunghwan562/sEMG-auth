import numpy as np
import glob
import matplotlib.pyplot as plt

subjects = ['A', 'B', 'C']

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for i, subject in enumerate(subjects):

    subject_files = sorted(glob.glob(f'data/{subject}/*.csv'))
    x = np.loadtxt(subject_files[0], delimiter=',', skiprows=1)

    t = np.arange(len(x)) / 1000.0

    axes[i].plot(t, x[:, 0], lw=0.6, label='Channel 1 (APB)')
    axes[i].plot(t, x[:, 1], lw=0.6, label='Channel 2 (ADM)')

    # 구간 경계
    axes[i].axvline(1.0, color='r', linestyle='--')
    axes[i].axvline(2.0, color='r', linestyle='--')

    # 구간 이름
    axes[i].text(0.5, 0.95, '파지', ha='center',
                 transform=axes[i].transAxes)
    axes[i].text(1.5, 0.95, '회전', ha='center',
                 transform=axes[i].transAxes)
    axes[i].text(2.5, 0.95, '정지', ha='center',
                 transform=axes[i].transAxes)

    axes[i].set_ylabel(f'피험자 {subject}')
    axes[i].legend(loc='upper right')

axes[0].set_title('피험자별 sEMG 신호 비교')
axes[2].set_xlabel('시간 (초)')

plt.tight_layout()
plt.savefig('signal_3subjects.png', dpi=150)