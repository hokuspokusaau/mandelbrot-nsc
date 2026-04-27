import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from mandelbort import mandelbrot_point, mandelbrot_point_njit, compute_mandelbrot, compute_mandelbrot_njit

xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5


# 1: Known points test
implementations = [mandelbrot_point, mandelbrot_point_njit]

@pytest.mark.parametrize("impl", implementations)
@pytest.mark.parametrize("c, max_iter, expected", [
    (0+0j, 100, 100),    # inside set 
    (5+0j, 100, 1),      # far outside
    (-2.5+0j, 100, 1),   # left tip of set
])
def test_known_points(impl, c, max_iter, expected):
    result = impl(c, max_iter)
    assert result == expected



# 2: Result is always within [0, max_iter]
@pytest.mark.parametrize("impl", implementations)
def test_result_in_valid_range(impl):
    c = -0.75 + 0.1j   # slow escape point in seahorse valley
    max_iter = 100

    result = impl(c, max_iter)

    assert 0 <= result <= max_iter


# 3: Implementation agreement
def test_implementations_agree():
    width, height, max_iter = 32, 32, 100

    naive = compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
    numba = compute_mandelbrot_njit(xmin, xmax, ymin, ymax, width, height, max_iter)

    np.testing.assert_array_equal(numba, naive)