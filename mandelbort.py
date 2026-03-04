import numpy as np
import time
import statistics
import matplotlib.pyplot as plt
"""
Mandelbrot set generator
By [Marcus d Almeida]
Course: Numerical Scientific Computing 2026
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


def benchmark(func, *args, n_runs=3):
     # Time func, return median of n_runs.
     times = []
     for _ in range(n_runs):
          t0 = time.perf_counter()
          result = func(*args)
          times.append(time.perf_counter() - t0)
     median_t = statistics.median(times)
     print(f"Median: {median_t :.4f}s "
           f"(min={min(times) :.4f}, max={max(times) :.4f})")
     return median_t, result


             
######## Run ##########
t, M = benchmark(compute_mandelbrot_vectorize,xmin, xmax, ymin, ymax, width, height, max_iter)

# Verify shape and dtype




####### Plotter ########
plt.imshow(compute_mandelbrot_vectorize(xmin, xmax, ymin, ymax, width, height, max_iter), cmap='hot')
plt.colorbar()
plt.title('Mandelbrot')
#plt.savefig('Mandelbrot.png')
plt.show()



