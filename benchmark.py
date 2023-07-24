import timeit
import torch
import numpy as np

import torch.nn.functional as F
import matplotlib.pyplot as plt
from softmax_one import softmax1

def benchmark(func, x):
    start = timeit.default_timer()
    func(x)
    return timeit.default_timer() - start

# Define the sizes to test
sizes = [(10, 10), (100, 100), (1000, 1000), (10000, 10000)]

# Arrays to store results
times_softmax = []
times_softmax1 = []

# Run the benchmark
for size in sizes:
    x = torch.rand(size)
    time_softmax = benchmark(F.softmax, x)
    time_softmax1 = benchmark(softmax1, x)
    
    times_softmax.append(time_softmax)
    times_softmax1.append(time_softmax1)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sizes, times_softmax, label='F.softmax')
plt.plot(sizes, times_softmax1, label='softmax1')
plt.legend()
plt.xlabel('Tensor Size')
plt.ylabel('Time (s)')
plt.title('Benchmarking Results')
plt.show()