import numpy as np
import time
import statistics
import os
import argparse
import matplotlib

# Use a non-interactive backend when running on headless machines (typical VM setup).
if os.environ.get("DISPLAY") is None and os.environ.get("MPLBACKEND") is None:
     matplotlib.use("Agg")

import matplotlib.pyplot as plt
from numba import njit
from multiprocessing import Pool
from dask import delayed
from dask.distributed import Client, LocalCluster
import dask

"""
Mandelbrot set generator
By [Marcus d Almeida]
Course: Numerical Scientific Computing 2026
AI has been used extensively for fixing errors throughout the code. Everything has been made and written by the author.

run in nsc2026

My Spec:
Apple M1 pro:
8-Core CPU (6p and 2e)
14-Core GPU
16-Core Neural Engine
200 GB/s memory bandwidth
"""
##### PARAMETERS #####
xmin    = -2
xmax    = 1
ymin    = -1.5
ymax    = 1.5
width   = 4096 #passed as N in later functions
height  = 4096
max_iter = 100
######################

#####  #####
N = 10000

                                              
#region Main
#region MP1
def mandelbrot_point(c, max_iter):
    z = 0+0j
    for i in range(max_iter):
        z = z**2 + c
        if abs(z) > 2:
            return i
    return max_iter

def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
        result = np.zeros((height, width), dtype=int)

        xs = np.linspace(xmin, xmax, width)
        ys = np.linspace(ymin, ymax, height)

        for i, y in enumerate(ys) :
             for j, x in enumerate(xs):
                  c = complex(x,y)
                  result[i, j] = mandelbrot_point(c, max_iter)
        
        return result

@njit
def mandelbrot_point_njit(c, max_iter=100):
     z = 0j
     n = 0
     for n in range(max_iter):
          if z.real * z.real + z.imag * z.imag > 4.0:
               return n
          z = z * z + c
     return max_iter

@njit
def compute_mandelbrot_njit(xmin, xmax, ymin, ymax, width, height, max_iter, dtype=np.float64):
    result = np.zeros((height, width), dtype=np.int32)
    xs = np.linspace(xmin, xmax, width).astype(dtype)
    ys = np.linspace(ymin, ymax, height).astype(dtype)

    for i in range(height):
        for j in range(width):
            c = xs[j] + 1j * ys[i]
            result[i, j] = mandelbrot_point_njit(c, max_iter)

    return result

def compute_mandelbrot_hybrid(xmin, xmax, ymin, ymax, width, height, max_iter):
        result = np.zeros((height, width), dtype=int)

        xs = np.linspace(xmin, xmax, width)
        ys = np.linspace(ymin, ymax, height)

        for i, y in enumerate(ys) :
             for j, x in enumerate(xs):
                  c = complex(x,y)
                  result[i, j] = mandelbrot_point_njit(c, max_iter)
        
        return result

def compute_mandelbrot_vectorize(xmin, xmax, ymin, ymax, width, height, max_iter):
     x = np.linspace(xmin, xmax, width)
     y = np.linspace(ymin, ymax, height)
     X, Y = np.meshgrid(x,y)
     C = X + 1j*Y
     
     # Complex array
     Z = np.zeros_like(C)

     # Iter count
     M = np.zeros(C.shape, dtype=int)

     for _ in range(max_iter):
          mask = np.abs(Z) <= 2
          Z[mask] = Z[mask]**2 + C[mask]
          M[mask] += 1
     return M

def column_sum(ar):
     for j in range(ar.shape[1]):
          s = np.sum(ar[:, j])

def row_sum(ar):
     for i in range(ar.shape[0]):
          s = np.sum(ar[i, :])


def maybe_show_plot():
     # Avoid GUI-related errors when no display is available.
     if os.environ.get("DISPLAY") is not None:
          plt.show()

#region MP2
@njit(cache=True)
# Returns the escape iteration count for single point complex point
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = 0.0
    z_imag = 0.0
    for i in range(max_iter):
          z_sq = z_real * z_real + z_imag * z_imag
          if z_sq > 4.0:
               return i
          old_real = z_real
          old_imag = z_imag
          z_imag = 2.0 * old_real * old_imag + c_imag
          z_real = old_real * old_real - old_imag * old_imag + c_real
    return max_iter

@njit(cache=True) 
# Loops over rows and all columns. Computes pixel coordinates from index + bounds -- No arrays received as input.
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = 0.0 if N <= 1 else (x_max - x_min) / (N - 1)
    dy = 0.0 if N <= 1 else (y_max - y_min) / (N - 1)
    for r in range(row_end - row_start):
       c_imag = y_min + (r + row_start) * dy
       for col in range(N):
          c_real = x_min + col * dx
          out[r, col] = mandelbrot_pixel(c_real, c_imag, max_iter)
    return out

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    # Thin wrapper calls mandebrot_chunk -- Whole grid as one chunk
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

def benchmark_mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100, n_runs=3):
    mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)

    times = []
    result = None
    for _ in range(n_runs):
       t0 = time.perf_counter()
       result = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
       times.append(time.perf_counter() - t0)

    return statistics.median(times), result

def _worker(args):
     return mandelbrot_chunk(*args)


def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, num_processes=4):
     chunk_size = max(1, N // num_processes)
     chunks = []
     row = 0
     while row < N:
          row_end = min(row + chunk_size, N)
          chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
          row = row_end

     with Pool(processes=num_processes) as pool:
          parts = pool.map(_worker, chunks)

     return np.vstack(parts)

def mandelbrot_parallel_l4(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4, n_chunks=None, pool=None):
     if n_chunks is None:
          n_chunks = n_workers
     chunk_size = max(1, N // n_chunks)
     chunks, row = [], 0
     while row < N:
          row_end = min(row + chunk_size, N)
          chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
          row = row_end
     if pool is not None:          # Caller manages pool; skip startup + warmup
          return np.vstack(pool.map(_worker, chunks))
     tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
     with Pool(processes=n_workers) as p:
          p.map(_worker, tiny)     # warm-up: load JIT cache in workers
          parts = p.map(_worker, chunks)
     return np.vstack(parts)


# region  L6 
def mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter=100, n_chunks=32):
     chunk_size = max(1, N // n_chunks)
     tasks, row = [], 0
     while row < N:
          row_end = min(row + chunk_size, N)
          tasks.append(delayed(mandelbrot_chunk)(row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
          row = row_end
     parts = dask.compute(*tasks)
     return np.vstack(parts)

#region General testing
def results_sanity(one_result, another_result):
     if np.allclose(one_result, another_result):
          print("Results match!")
     else:
          print("Results differ!")

def compare_results(one_result, another_result):
     diff = np.abs(one_result - another_result)
     print(f"Max difference : {diff.max()}")
     print(f"Different pixels: {(diff > 0).sum()}")

def runtime_gridsize (xmin, xmax, ymin, ymax, max_iter, n_runs=3):
     gridsize = [256, 512, 1024, 2048, 4096]
     median_times = []


     for i in range(5):
          times = []
          gs = gridsize[i]
          gs_width, gs_height = gs, gs

          for _ in range(n_runs):
               t0 = time.perf_counter()
               result = compute_mandelbrot_vectorize(xmin, xmax, ymin, ymax, gs_width, gs_height, max_iter)
               times.append(time.perf_counter() - t0)

          median_t = statistics.median(times)
          median_times.append(median_t)
          print(f"Median for {gs}x{gs}: {median_t :.4f}s "
               f"(min={min(times) :.4f}, max={max(times) :.4f})")
     return gridsize, median_times, result

def benchmark(func, *args, n_runs=3, **kwargs):
     times = []
     result = None
     for _ in range(n_runs):
          t0 = time.perf_counter()
          result = func(*args, **kwargs)
          times.append(time.perf_counter() - t0)
     return statistics.median(times), result



def benchmark_mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, num_processes=4, n_runs=3):
     chunk_size = max(1, N // num_processes)
     chunks = []
     row = 0
     while row < N:
          row_end = min(row + chunk_size, N)
          chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
          row = row_end

     times = []
     result = None
     with Pool(processes=num_processes) as pool:
          # Warm up JIT compilation every worker before timed measurements.
          pool.map(_worker, chunks)

          for _ in range(n_runs):
               t0 = time.perf_counter()
               parts = pool.map(_worker, chunks)
               times.append(time.perf_counter() - t0)
               result = np.vstack(parts)

     return statistics.median(times), result


def sweep_mandelbrot_parallel( N, x_min, x_max, y_min, y_max, max_iter=100, n_runs=3, serial_time=None):
     cpu_count = os.cpu_count() or 1
     if serial_time is None:
          serial_time, _ = benchmark_mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=max_iter, n_runs=n_runs)

     print("Parallel worker sweep:")
     print("workers    time      speedup    efficiency")
     best_workers = 1
     best_time = float('inf')
     sweep_data = []
     for num_workers in range(1, cpu_count + 1):
          t_parallel, _ = benchmark_mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=max_iter, num_processes=num_workers, n_runs=n_runs)
          speedup = serial_time / t_parallel
          efficiency = speedup / num_workers
          print(f"{num_workers:2d}           {t_parallel:.3f}s               {speedup:.2f}x                  {efficiency:.2f}")
          sweep_data.append((num_workers, t_parallel, speedup, efficiency))
          if t_parallel < best_time:
               best_time = t_parallel
               best_workers = num_workers
     return best_workers, sweep_data


def sweep_mandelbrot_l4_chunks(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4, n_runs=3, serial_time=None):
     if serial_time is None:
          serial_time, _ = benchmark_mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=max_iter, n_runs=n_runs)

     print(f"L4 chunk sweep (fixed n_workers={n_workers}):")
     print("n_chunks          time      speedup    LIF")
     chunk_configs = [1 * n_workers, 2 * n_workers, 4 * n_workers, 8 * n_workers, 16 * n_workers]
     
     best_n_chunks = chunk_configs[0]
     best_time = float('inf')
     with Pool(processes=n_workers) as pool:
          pool.map(_worker, [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]) # tiny warm
          for n_chunks in chunk_configs:
               times = []
               for _ in range(n_runs):
                    t0 = time.perf_counter()
                    result = mandelbrot_parallel_l4(N, x_min, x_max, y_min, y_max, max_iter=max_iter, n_workers=n_workers, n_chunks=n_chunks, pool=pool)
                    times.append(time.perf_counter() - t0)
               t_parallel = statistics.median(times)
               speedup = serial_time / t_parallel
               lif = n_workers / speedup - 1 # Using speedup thus deviding 
               print(f"{n_chunks:2d} ({n_chunks//n_workers:2d}x)           {t_parallel:.3f}s      {speedup:.2f}x       {lif:.3f}")
               if t_parallel < best_time:
                    best_time = t_parallel
                    best_n_chunks = n_chunks

     return best_n_chunks

def mandelbrot_dask_IB(N, x_min, x_max, y_min, y_max, max_iter=100, n_chunks=None, n_workers=8, n_runs=3):
     chunk_configs = n_chunks if n_chunks is not None else [1, 2, 4, 8, 16, 32, 64]
     cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
     client = Client(cluster)
     client.run(lambda: mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, 10)) #JIT Warm

     try:
          t_serial, ref = benchmark_mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=max_iter, n_runs=n_runs)

          print('\nL05 Dask chunk sweep')
          print("n_chunks | time (s) | vs 1x | speedup | LIF")
          sweep_data = []
          baseline = None
          best_n_chunks = chunk_configs[0]
          best_time = float('inf')
          best_lif = float('inf')

          for chunk_count in chunk_configs:
               times = []
               for _ in range(n_runs):
                    t0 = time.perf_counter()
                    result = mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter, n_chunks=chunk_count)
                    times.append(time.perf_counter() - t0)

               t_dask = statistics.median(times)
               if baseline is None:
                    baseline = t_dask
               speedup = t_serial / t_dask
               lif = n_workers * t_dask / t_serial - 1
               sweep_data.append((chunk_count, t_dask, baseline / t_dask, speedup, lif))
               print(f"{chunk_count:8d} | {t_dask:.3f} | {baseline / t_dask:.3f}x | {speedup:.3f}x | {lif:.3f}")
               print(f"Sanity: {np.array_equal(ref, result)}")

               if t_dask < best_time:
                    best_time = t_dask
                    best_n_chunks = chunk_count
                    best_lif = lif

          x = [row[0] for row in sweep_data]
          y = [row[1] for row in sweep_data]
          plt.figure()
          plt.plot(x, y, marker='o')
          plt.xscale('log', base=2)
          plt.xlabel('n_chunks')
          plt.ylabel('wall time (s)')
          plt.title('Dask Chunk Sweep')
          plt.grid(True)
          plt.savefig('dask_chunk_sweep.png', dpi=150)
          maybe_show_plot()
          plt.close()

          print(f"n_chunks_optimal={best_n_chunks}, t_min={best_time:.3f}s, LIF at t_min={best_lif:.3f}")
          return best_n_chunks, best_time, best_lif, sweep_data
     finally:
          client.close()
          cluster.close()


def mandelbrot_dask_IB_U(N, x_min, x_max, y_min, y_max, max_iter=100, n_chunks=None, n_workers=8, n_runs=3):
     chunk_configs = n_chunks if n_chunks is not None else [1, 2, 4, 8, 16, 32, 64]
     client = Client("tcp://10.92.0.194:8786")
     client.run(lambda: mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, 10)) #JIT Warm

     try:
          t_serial, ref = benchmark_mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=max_iter, n_runs=n_runs)

          print('\nL05 Dask chunk sweep')
          print("n_chunks | time (s) | vs 1x | speedup | LIF")
          sweep_data = []
          baseline = None
          best_n_chunks = chunk_configs[0]
          best_time = float('inf')
          best_lif = float('inf')

          for chunk_count in chunk_configs:
               times = []
               for _ in range(n_runs):
                    t0 = time.perf_counter()
                    result = mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter, n_chunks=chunk_count)
                    times.append(time.perf_counter() - t0)

               t_dask = statistics.median(times)
               if baseline is None:
                    baseline = t_dask
               speedup = t_serial / t_dask
               lif = n_workers * t_dask / t_serial - 1
               sweep_data.append((chunk_count, t_dask, baseline / t_dask, speedup, lif))
               print(f"{chunk_count:8d} | {t_dask:.3f} | {baseline / t_dask:.3f}x | {speedup:.3f}x | {lif:.3f}")
               print(f"Sanity: {np.array_equal(ref, result)}")

               if t_dask < best_time:
                    best_time = t_dask
                    best_n_chunks = chunk_count
                    best_lif = lif

          x = [row[0] for row in sweep_data]
          y = [row[1] for row in sweep_data]
          plt.figure()
          plt.plot(x, y, marker='o')
          plt.xscale('log', base=2)
          plt.xlabel('n_chunks')
          plt.ylabel('wall time (s)')
          plt.title('Dask Chunk Sweep')
          plt.grid(True)
          plt.savefig('dask_chunk_sweep.png', dpi=150)
          maybe_show_plot()
          plt.close()

          print(f"n_chunks_optimal={best_n_chunks}, t_min={best_time:.3f}s, LIF at t_min={best_lif:.3f}")
          return best_n_chunks, best_time, best_lif, sweep_data
     finally:
          client.close()


def benchmark_mandelbrot_dask_ibu(N, x_min, x_max, y_min, y_max, max_iter=100, n_chunks=32, n_runs=3):
     client = Client("tcp://10.92.0.194:8786")
     client.run(lambda: mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, 10))

     try:
          times = []
          result = None
          for _ in range(n_runs):
               t0 = time.perf_counter()
               result = mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter=max_iter, n_chunks=n_chunks)
               times.append(time.perf_counter() - t0)
          return statistics.median(times), result
     finally:
          client.close()


#t, M = benchmark(compute_mandelbrot,xmin, xmax, ymin, ymax, width, height, max_iter)
#region Run

def main():
     parser = argparse.ArgumentParser(description="Run Numba single-core vs IB_U Dask benchmark")
     parser.add_argument("--n-runs", type=int, default=3, help="Number of timing runs")
     parser.add_argument("--n-chunks", type=int, default=32, help="Number of Dask chunks for IB_U run")
     args = parser.parse_args()

     t_numba, numba_result = benchmark_mandelbrot_serial(width, xmin, xmax, ymin, ymax, max_iter=max_iter, n_runs=args.n_runs)
     t_ibu, ibu_result = benchmark_mandelbrot_dask_ibu(width, xmin, xmax, ymin, ymax, max_iter=max_iter, n_chunks=args.n_chunks, n_runs=args.n_runs)

     speedup = t_numba / t_ibu
     print(f"{'Implementation':<20}{'Time (s)':>12}{'Speedup':>12}")
     print(f"{'Numba single-core':<20}{t_numba:>12.3f}{'1.00x':>12}")
     print(f"{'Dask IB_U':<20}{t_ibu:>12.3f}{(f'{speedup:.2f}x'):>12}")
     print(f"Results match: {np.array_equal(numba_result, ibu_result)}")


if __name__ == "__main__":
     main()
###### Sanity ########
'''
#M_naive = compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
#M_vectorize = compute_mandelbrot_vectorize(xmin, xmax, ymin, ymax, width, height, max_iter)
'''


##### Compare #####
'''
results_sanity(M_naive, M_vectorize)
compare_results(M_naive, M_vectorize)
'''


#region Plotter
##### Mandelbrot #####
'''
plt.imshow(compute_mandelbrot_vectorize(xmin, xmax, ymin, ymax, width, height, max_iter), cmap='hot')
plt.colorbar()
plt.title('Mandelbrot')
plt.savefig('Mandelbrot.png')
plt.show()
'''

##### Visual Comparison #####
'''
r32 = compute_mandelbrot_njit(xmin, xmax, ymin, ymax, width, height, max_iter, dtype=np.float32)
r64 = compute_mandelbrot_njit(xmin, xmax, ymin, ymax, width, height, max_iter, dtype=np.float64)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, result, title in zip(axes, [r32, r64], ['float32', 'float64 (ref)']):
     ax.imshow(result, cmap='viridis')
     ax.set_title(title); ax.axis('off')
plt.savefig('float comparison.png', dpi=150)
plt.show()
     
print(f"Max diff float 32 vs float64: {np.abs(r32-r64).max()}")
'''
##### Problem Size Scaling ######
'''
x, y, _ = runtime_gridsize (xmin, xmax, ymin, ymax, max_iter, n_runs=3)

plt.figure()
plt.plot(x, y, marker='o')
plt.xlabel('Grid Size (N x N)')
plt.ylabel('Median Runtime (s)')
plt.title('Runtime vs Grid Size')
plt.grid(True)
plt.show()
'''