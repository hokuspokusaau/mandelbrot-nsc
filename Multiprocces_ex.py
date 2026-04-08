from multiprocessing import Pool
import time
import numpy as np
"""
GIT TESTING
"""


def square(x):
    time.sleep(0.1)
    return x * x

if __name__ == '__main__':
    numbers = list(range(100))

    start = time.time()
    results_serial = [square(x) for x in numbers]
    time_serial = time.time() - start
    print(f"Serial: {time_serial: .2f}s")

    with Pool(processes=3) as pool:
        start = time.time()
        results_parallel = pool.map(square, numbers)
        time_parallel = time.time() - start
    print(f"Parallel: {time_parallel: .2f}s")
    speedup = time_serial / time_parallel
    print(f"Speedup: {speedup: .2f}x")
