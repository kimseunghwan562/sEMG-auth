import torch
from torchvision.models import densenet161

print("PyTorch 버전:", torch.__version__)
print("CUDA 사용 가능:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA를 사용할 수 없습니다.")

model = densenet161(weights=None)

print("DenseNet161 생성 완료")