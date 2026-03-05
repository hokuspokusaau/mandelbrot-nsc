import numpy as np
import time
import statistics
import matplotlib.pyplot as plt
from numba import njit
"""
Mandelbrot set generator
By [Marcus d Almeida]
Course: Numerical Scientific Computing 2026
AI has been used extensively for fixing errors throughout the code. Everything has been made and written by the author.
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

##### Memory Acces Patterns Array #####
N = 10000
A = np.random.rand(N, N)

                                              
######## Main ##########

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
     for n in range(max_iter):
          if z.real * z.real + z.imag * z.imag > 4.0:
               return n
          z = z * z + c
     return max_iter

@njit
def compute_mandelbrot_njit(xmin, xmax, ymin, ymax, width, height, max_iter):
        result = np.zeros((height, width), dtype=np.int32)

        xs = np.linspace(xmin, xmax, width)
        ys = np.linspace(ymin, ymax, height)

        for i in range(height):
                  for j in range(width):
                       c = xs[j] + 1j * ys[i]
                       z = 0j
                       n = 0
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

def results_sanity(one_result, another_result):
     if np.allclose(one_result, another_result):
          print("Results match!")
     else:
          print("Results differ!")

def compare_results(one_result, another_result):
     diff = np.abs(one_result - another_result)
     print(f"Max difference : {diff.max()}")
     print(f"Different pixels: {(diff > 0).sum()}")


def benchmark(func, *args, n_runs=3):
     # Time func, return median of n_runs.
     times = []
     for _ in range(n_runs):
          t0 = time.perf_counter()
          result = func(*args)
          times.append(time.perf_counter() - t0)
     median_t = statistics.median(times)
     print(f"Median for {func}: {median_t :.4f}s "
           f"(min={min(times) :.4f}, max={max(times) :.4f})")
     return median_t, result

def row_sum(ar):
     for i in range(N):
          s = np.sum(ar[i, :])
def column_sum(ar):
     for j in range(N):
          s = np.sum(ar[:, j])


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

#t, M = benchmark(compute_mandelbrot,xmin, xmax, ymin, ymax, width, height, max_iter)
######## Run ##########
##### line_profiler ######
#compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)

##### JIT / JIT_Hybrid #####
_ = compute_mandelbrot_hybrid(-2, 1, -1.5, 1.5, 64, 64, 100)
_ = compute_mandelbrot_njit(-2, 1, -1.5, 1.5, 64, 64, 100)

t_hybrid, _ = benchmark(compute_mandelbrot_hybrid,-2, 1, -1.5, 1.5, 64, 64, 100) 
t_full, _ = benchmark(compute_mandelbrot_njit,-2, 1, -1.5, 1.5, 64, 64, 100)

print(f"Hybrid:               {t_hybrid: .3f}s")
print(f"Full njit Numba:      {t_full:.3f}s")
print(f"Ratio:                {t_hybrid/t_full :.1f}x")


##### Problem Size Scaling #####
#runtime_gridsize (xmin, xmax, ymin, ymax, max_iter, n_runs=3)

##### Performance #####
#t, M = benchmark(row_sum,A)
#t, M = benchmark(column_sum,A)
#t, M = benchmark(compute_mandelbrot_vectorize,xmin, xmax, ymin, ymax, width, height, max_iter)
#t, M = benchmark(compute_mandelbrot,xmin, xmax, ymin, ymax, width, height, max_iter)

###### Sanity ########
#M_naive = compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
#M_vectorize = compute_mandelbrot_vectorize(xmin, xmax, ymin, ymax, width, height, max_iter)



##### Compare #####
#results_sanity(M_naive, M_vectorize)
#compare_results(M_naive, M_vectorize)



####### Plotter ########
##### Mandelbrot #####
#plt.imshow(compute_mandelbrot_vectorize(xmin, xmax, ymin, ymax, width, height, max_iter), cmap='hot')
#plt.colorbar()
#plt.title('Mandelbrot')
#plt.savefig('Mandelbrot.png')
#plt.show()


##### Problem Size Scaling ######
#x, y, _ = runtime_gridsize (xmin, xmax, ymin, ymax, max_iter, n_runs=3)

#plt.figure()
#plt.plot(x, y, marker='o')
#plt.xlabel('Grid Size (N x N)')
#plt.ylabel('Median Runtime (s)')
#plt.title('Runtime vs Grid Size')
#plt.grid(True)
#plt.show()
