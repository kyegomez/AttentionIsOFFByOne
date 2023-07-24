import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from softmax_one import softmax1
import numpy as np

def benchmark(func, x, dim):
    start = time.time()
    for _ in range(1000):
        func(x, dim)
    end = time.time()
    return end - start

# Define the sizes to test
sizes = [(10, 10), (100, 100), (1000, 1000), (10000, 10000)]

# Arrays to store results
times_softmax = []
times_softmax1 = []

# Run the benchmark
for size in sizes:
    x = torch.rand(size)
    time_softmax = benchmark(F.softmax, x, dim=-1)
    time_softmax1 = benchmark(softmax1, x, dim=-1)
    
    times_softmax.append(time_softmax)
    times_softmax1.append(time_softmax1)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot([np.prod(s) for s in sizes], times_softmax, label='F.softmax')
plt.plot([np.prod(s) for s in sizes], times_softmax1, label='softmax1')
plt.legend()
plt.xlabel('Tensor Size')
plt.ylabel('Time (s)')
plt.title('Traditional Softmax vs SoftmaxOne')
plt.show()
