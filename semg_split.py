from pathlib import Path

# 데이터 폴더에서 CSV 파일 찾기
files = list(Path("data").glob("*/*.csv"))

# 클래스별 파일 분류
class_files = {}

for f in files:
    cls = f.parent.name

    if cls not in class_files:
        class_files[cls] = []

    class_files[cls].append(f)


# 학습 / 테스트 파일
train_files = []
test_files = []

for cls in ["A", "B", "C", "D", "E"]:

    class_files[cls].sort()

    # 50개 중 40개 학습, 10개 테스트
    train_files.extend(class_files[cls][:40])
    test_files.extend(class_files[cls][40:])


# 결과 출력
print("===== 학습 / 테스트 데이터 분할 =====")

print("전체 파일 수:", len(files))
print("학습 파일 수:", len(train_files))
print("테스트 파일 수:", len(test_files))

print()
print("학습 데이터 클래스별 개수")

for cls in ["A", "B", "C", "D", "E"]:
    count = sum(f.parent.name == cls for f in train_files)
    print(cls, ":", count)

print()
print("테스트 데이터 클래스별 개수")

for cls in ["A", "B", "C", "D", "E"]:
    count = sum(f.parent.name == cls for f in test_files)
    print(cls, ":", count)