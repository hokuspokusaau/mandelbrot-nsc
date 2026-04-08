import math
import random
import time
import statistics
import os
from multiprocessing import Pool

def estimate_pi_serial(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return 4 * inside_circle / num_samples


def estimate_pi_chunk(num_samples):
    # Estimate pi contributions for num_samples random points.
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return inside_circle
    
def estimate_pi_parallel(num_samples, num_processes=4):
    samples_per_process, remainder = divmod(num_samples, num_processes)
    tasks = [samples_per_process] * num_processes
    tasks[-1] += remainder
    with Pool(processes=num_processes) as pool:
        results = pool.map(estimate_pi_chunk, tasks)
    return 4 * sum(results) / num_samples

def test_granularity(total_work, chunk_size, n_proc):
    n_chunks = total_work // chunk_size
    tasks = [chunk_size] * n_chunks
    t0 = time.perf_counter()
    if n_proc == 1:
        results = [estimate_pi_chunk(s) for s in tasks]
    else:
        with Pool(processes=n_proc) as pool:
            results = pool.map(estimate_pi_chunk, tasks)
    return time.perf_counter() - t0, 4 * sum(results) / total_work

if __name__ == '__main__':
    total_work =  1_000_000
    n_proc = os.cpu_count() // 2
    chunk_sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
    print(f"{'L' : >12}  | {'serial (s)' : >12} | {'parallel (s)' : >12}")
    for L in chunk_sizes:
        t_ser, _ = test_granularity(total_work, L, n_proc=1)
        t_par, pi = test_granularity(total_work, L, n_proc=n_proc)
        print(f"{L:12d} | {t_ser:12.4f} | {t_par:12.4f}  pi={pi:.4f}")
