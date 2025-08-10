import torch

print("PyTorch version:", torch.__version__)

# 测试是否支持CUDA（GPU）
print("CUDA available:", torch.cuda.is_available())

# 创建一个张量并打印
x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", x)

# 如果CUDA可用，尝试将张量移动到GPU
if torch.cuda.is_available():
    x = x.to("cuda")
    print("Tensor on GPU:", x)
else:
    print("CUDA not available, running on CPU")