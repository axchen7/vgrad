import os
import math
import torch
from export_vgtensor import export_vgtensor

DATA_DIR = "../vgrad/measurements/data"
os.makedirs(DATA_DIR, exist_ok=True)

torch.manual_seed(3)

N = 100
x = torch.randn(N, N)

export_vgtensor(x, os.path.join(DATA_DIR, "rand_matrix.vgtensor"))


for i in range(10):
    x = x @ x / math.sqrt(N)
    print(f"Variance: {x.var().item()}")
