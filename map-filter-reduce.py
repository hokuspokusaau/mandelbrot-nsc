import random, time
from functools import reduce
from multiprocessing import Pool

N = 1_000_000
data = [random.randint(10, 100) for _ in range(N)]

def subtract_seven(x):
    return x - 7

if __name__ == '__main__':
    # Part 1 -- serial pipeline (map / filter / reduce chained)
    t0 = time.perf_counter()
    result_ser = reduce(lambda a, b: a + b,
                        filter(lambda x: x % 2 == 1,
                               map(subtract_seven, data)))
    t_serial = time.perf_counter() - t0

    # Part 2 -- replace map() with Pool.map()
    t0 = time.perf_counter()
    with Pool() as pool:
        mapped = pool.map(subtract_seven, data)
    result_par = reduce(lambda a, b: a + b,
                        filter(lambda x: x % 2 == 1, mapped))
    t_parallel = time.perf_counter() - t0

    print(f"Serial:     {t_serial:.4f}s     result={result_ser}")
    print(f"Parallel:   {t_parallel:.4f}s   result={result_par}")
    print(f"Speedup:    {t_serial / t_parallel:.2f}x")

