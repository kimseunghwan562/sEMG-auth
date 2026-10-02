import os
import glob
import random

# 데이터 경로
data_dir = "data"

# 재현 가능하도록 고정
random.seed(42)

# 클래스
classes = ["A", "B", "C", "D", "E"]

# 결과 저장
train_files = []
test_files = []

for label in classes:
    label_dir = os.path.join(data_dir, label)

    files = glob.glob(os.path.join(label_dir, "*.csv"))
    files.sort()

    # 파일 순서 섞기
    random.shuffle(files)

    # 40개 Train / 10개 Test
    train = files[:40]
    test = files[40:]

    train_files.extend([(file, label) for file in train])
    test_files.extend([(file, label) for file in test])

    print(f"{label}: Train {len(train)}개 / Test {len(test)}개")

print()
print(f"전체 Train 파일: {len(train_files)}개")
print(f"전체 Test 파일: {len(test_files)}개")

print()
print("Train 클래스별 개수")

for label in classes:
    count = sum(1 for file, y in train_files if y == label)
    print(f"{label}: {count}")

print()
print("Test 클래스별 개수")

for label in classes:
    count = sum(1 for file, y in test_files if y == label)
    print(f"{label}: {count}")