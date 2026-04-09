import numpy as np
import time
import statistics
import matplotlib.pyplot as plt
from numba import njit
from multiprocessing import Pool
import os
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
width   = 1024
height  = 1024
max_iter = 100
######################

#####  #####
N = 10000
A = np.random.rand(N, N)

                                              
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
     for j in range(N):
          s = np.sum(ar[:, j])

def row_sum(ar):
     for i in range(N):
          s = np.sum(ar[i, :])

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



def benchmark_mandelbrot_parallel(
     N, x_min, x_max, y_min, y_max, max_iter=100, num_processes=4, n_runs=3):
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
          # Warm up JIT compilation in every worker before timed measurements.
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
     for num_workers in range(1, cpu_count + 1):
          t_parallel, _ = benchmark_mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=max_iter, num_processes=num_workers, n_runs=n_runs)
          speedup = serial_time / t_parallel
          efficiency = speedup / num_workers
          print(f"{num_workers:2d}           {t_parallel:.3f}s               {speedup:.2f}x                  {efficiency:.2f}")
     return cpu_count
#t, M = benchmark(compute_mandelbrot,xmin, xmax, ymin, ymax, width, height, max_iter)
#region Run

def main():
     ##### Run #####
     num_processes = min(4, os.cpu_count() or 1)

     ##### Monte Carlo pi #####
     # t_serial, _ = benchmark(esti_pi_serial, 10_000_000)
     # print(f"time for t_serial          {t_serial: .3f}s")
     # print("True Pi = 3.14159265")

     ##### Mandelbrot serial benchmark #####
     t_mb_serial, M_serial = benchmark_mandelbrot_serial(width, xmin, xmax, ymin, ymax, max_iter=max_iter, n_runs=3)
     print(f"Mandelbrot serial (Numba chunk): {t_mb_serial:.3f}s, shape={M_serial.shape}")

     ##### Mandelbrot parallel benchmark #####
     t_mb_parallel, M_parallel = benchmark_mandelbrot_parallel(width, xmin, xmax, ymin, ymax, max_iter=max_iter, num_processes=num_processes, n_runs=3)
     print( f"Mandelbrot parallel ({num_processes} workers): " f"{t_mb_parallel:.3f}s, speedup={t_mb_serial / t_mb_parallel:.2f}x")
     print("Comparing serial output with parallel output")
     results_sanity(M_serial, M_parallel)
     compare_results(M_serial, M_parallel)

     sweep_mandelbrot_parallel(width, xmin, xmax, ymin, ymax, max_iter=max_iter, n_runs=3, serial_time=t_mb_serial)

     ##### Mandelbrot L4 parallel benchmark #####
     t_mb_l4, M_l4 = benchmark(mandelbrot_parallel_l4, width, xmin, xmax, ymin, ymax, max_iter=max_iter, n_workers=num_processes)
     print(f"Mandelbrot L4 parallel ({num_processes} workers): {t_mb_l4:.3f}s")
     print("Comparing L4 parallel output with serial output")
     results_sanity(M_serial, M_l4)
     compare_results(M_serial, M_l4)

     ##### Time comparisons #####
     t_old_numba, M_old_numba = benchmark(
          compute_mandelbrot_njit, xmin, xmax, ymin, ymax, width, height, max_iter
     )
     t_vectorize, M_vectorize = benchmark(
          compute_mandelbrot_vectorize, xmin, xmax, ymin, ymax, width, height, max_iter
     )

     print(f"Old Numba:           {t_old_numba: .3f}s")
     print(f"Vectorize:           {t_vectorize: .3f}s")
     print(f"Numba chunk serial:  {t_mb_serial:.3f}s")
     print(f"Numba chunk parallel:{t_mb_parallel:.3f}s")
     print(f"Ratio (Vectorize):   {t_old_numba/t_vectorize :.5f}x")
     print(f"Ratio (Numba chunk): {t_old_numba/t_mb_serial :.5f}x")

     print("Comparing old MP1 Numba output with serial output")
     results_sanity(M_old_numba, M_serial)
     compare_results(M_old_numba, M_serial)

     # Example parallel run and correctness check.
     # t_mb_parallel, M_parallel = benchmark_mandelbrot_parallel(
     #     width, xmin, xmax, ymin, ymax, max_iter=max_iter, num_processes=4, n_runs=3
     # )
     # print(f"Mandelbrot parallel (4 workers): {t_mb_parallel:.3f}s")
     # results_sanity(M_serial, M_parallel)

     ##### Problem Size Scaling #####
     # runtime_gridsize(xmin, xmax, ymin, ymax, max_iter, n_runs=3)

     ##### Performance #####
     t, M = benchmark(row_sum, A)
     t, M = benchmark(column_sum, A)
     t, M = benchmark(compute_mandelbrot_vectorize, xmin, xmax, ymin, ymax, width, height, max_iter)
     t, M = benchmark(compute_mandelbrot, xmin, xmax, ymin, ymax, width, height, max_iter)


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