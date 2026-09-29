# check_gpu.py
import torch
print("torch version:      ", torch.__version__)
print("torch CUDA build:   ", torch.version.cuda)   # None = CPU-only build
print("cuda available:     ", torch.cuda.is_available())
if not torch.cuda.is_available():
    try:
        torch.zeros(1).cuda()
    except Exception as e:
        print("CUDA error:", e)   # gives the actual reason