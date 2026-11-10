import pytest
import numpy as np
from hideseek.env.core import HideSeekCore

def test_initialization():
    env = HideSeekCore(seed=42)
    assert env.map_size == 15
    assert len(env.hider_positions) == 2
    assert len(env.seeker_positions) == 2
    # Check that there are exactly n_boxes in the grid
    assert np.sum(env.grid == 2) == 2

def test_wall_collision():
    env = HideSeekCore(seed=42)
    # Put hider at (1,1) and try to move up (into wall)
    env.hider_positions[0] = [1, 1]
    actions = {"hider_0": 1}  # Move up
    env.step(actions)
    assert env.hider_positions[0] == [1, 1]  # Should not have moved

def test_push_box():
    env = HideSeekCore(seed=42)
    # Manually set up a scenario
    env.grid.fill(0)
    env.hider_positions[0] = [5, 5]
    env.grid[5, 6] = 2  # Box in front of hider
    
    actions = {"hider_0": 4}  # Move right (push box)
    env.step(actions)
    
    # Hider should be at [6, 5], box should be at [7, 5]
    assert env.hider_positions[0] == [6, 5]
    assert env.grid[5, 7] == 2
    assert env.grid[5, 6] == 0

def test_locked_box():
    env = HideSeekCore(seed=42)
    env.grid.fill(0)
    env.hider_positions[0] = [5, 5]
    env.grid[5, 6] = 3  # Locked box in front
    
    actions = {"hider_0": 4}  # Try to push
    env.step(actions)
    
    # Hider should not move
    assert env.hider_positions[0] == [5, 5]
    assert env.grid[5, 6] == 3

def test_toggle_lock():
    env = HideSeekCore(seed=42)
    env.grid.fill(0)
    env.hider_positions[0] = [5, 5]
    env.grid[5, 6] = 2  # Unlocked box adjacent
    
    actions = {"hider_0": 5}  # Toggle lock
    env.step(actions)
    
    # Box should now be locked (3)
    assert env.grid[5, 6] == 3