import torch
import time


def timeit(fn, label):
    start = time.time()
    fn()
    end = time.time()
    print(f"{label} took {(end - start) * 1000:.1f}ms")


for N in range(1_000_000, 10_000_000 + 1, 1_000_000):
    x = torch.zeros(N)
    timeit(lambda: x + 1, f"unary, N={N}")


for N in range(100, 1000 + 1, 100):
    x = torch.zeros(N, N)
    timeit(lambda: x @ x, f"matmul, N={N}")
