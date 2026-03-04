import cProfile, pstats
from mandelbort import compute_mandelbrot, compute_mandelbrot_vectorize

cProfile.run('compute_mandelbrot(-2, 1, -1.5, 1.5, 512, 512, 100)',
             'naive_profile.prof')

cProfile.run('compute_mandelbrot_vectorize(-2, 1, -1.5, 1.5, 512, 512, 100)',
             'numpy_profile.prof')

for name in ('naive_profile.prof', 'numpy_profile.prof'):
    stats = pstats.Stats(name)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
