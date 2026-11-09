import numpy as np
import pytest
from hideseek.env.visibility import has_line_of_sight, bresenham_line, compute_visibility_mask

def test_bresenham_line():
    points = bresenham_line(0, 0, 3, 0)
    assert points == [(0, 0), (1, 0), (2, 0), (3, 0)]
    points = bresenham_line(0, 0, 2, 2)
    assert points == [(0, 0), (1, 1), (2, 2)]

def test_los_clear():
    grid = np.zeros((5, 5), dtype=int)
    assert has_line_of_sight((1, 1), (3, 3), grid, vision_radius=5) == True

def test_los_blocked_by_wall():
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 2] = 1  # Wall in the middle
    assert has_line_of_sight((1, 1), (3, 3), grid, vision_radius=5) == False

def test_los_blocked_by_box():
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 1] = 2  # Box at (row=2, col=1)
    assert has_line_of_sight((1, 1), (1, 3), grid, vision_radius=5) == False

def test_los_radius_cutoff():
    grid = np.zeros((10, 10), dtype=int)
    assert has_line_of_sight((1, 1), (6, 1), grid, vision_radius=4) == False
    assert has_line_of_sight((1, 1), (6, 1), grid, vision_radius=5) == True

def test_visibility_mask():
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 2] = 1 # Wall
    mask = compute_visibility_mask((1, 1), grid, vision_radius=2)
    assert mask[1, 1] == True # Self
    assert mask[1, 2] == True # Next to self
    assert mask[2, 2] == True # Wall itself
    assert mask[3, 3] == False # Behind the wall