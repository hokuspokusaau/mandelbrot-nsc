import numpy as np

"""
Mandelbrot set generator
By [Marcus d Almeida]
Course: Numerical Scientific Computing 2026
"""

def mandelbrot_point(c, max_iter):
    z = 0
    for i in range(max_iter):
        z = z**2 + c
        if abs(z) > 2:
            print(f"{i} iterations before divergence")
            return
    print("not enough iterations")
    

mandelbrot_point(2+0j,20) 