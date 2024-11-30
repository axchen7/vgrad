from export_vgtensor import export_vgtensor
import math
import matplotlib.pyplot as plt
import torch

N_VALS = 1000

# 1s, 500 readings / sec -> 2ms between readings
x = torch.linspace(0, 2, N_VALS)

baseline = x

noise_freq = 20
noise1 = torch.sin(noise_freq * x)
noise2 = torch.sin(noise_freq * x + math.pi * 3 / 4)

# actual readings is baseline + 1/2 probability of noise1 or noise2
bern = torch.bernoulli(torch.full_like(x, 0.5))
readings = baseline + torch.where(bern == 1, noise1, noise2)

export_vgtensor(x, "readings_x.vgtensor")
export_vgtensor(readings, "readings_y.vgtensor")

plt.scatter(x, readings, s=1)
plt.xlabel("Time (s)")
plt.ylabel("Force (pounds)")
plt.show()
