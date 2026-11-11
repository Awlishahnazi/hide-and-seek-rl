import numpy as np
from hideseek.env.core import HideSeekCore
from hideseek.metrics import StrategyDetector

def test_door_blocked():
    core = HideSeekCore(seed=42)
    detector = StrategyDetector(core)
    
    # Initially, door is not blocked
    assert detector.check_door_blocked() == False
    
    # Manually block the door with a locked box
    mid = core.map_size // 2
    core.grid[detector.door_y, mid - 1] = 3 # Locked box next to door
    
    assert detector.check_door_blocked() == True

def test_locked_boxes_count():
    core = HideSeekCore(seed=42)
    detector = StrategyDetector(core)
    
    core.grid[2, 2] = 3
    core.grid[3, 3] = 3
    assert detector.count_locked_boxes() == 2