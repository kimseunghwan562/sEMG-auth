import matplotlib.pyplot as plt


# Epoch별 Loss
epochs = [1, 2, 3, 4, 5]

losses = [
    1.0540,
    0.7495,
    0.5848,
    0.5242,
    0.4189
]


# 그래프 생성
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    losses,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "DenseNet161 Training Loss"
)

plt.xticks(epochs)

plt.grid(True)

plt.tight_layout()


# 저장
plt.savefig(
    "loss_curve.png",
    dpi=150
)

plt.show()

print("Loss 그래프 저장 완료: loss_curve.png")