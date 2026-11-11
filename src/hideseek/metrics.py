import numpy as np

class StrategyDetector:
    def __init__(self, core):
        self.core = core
        # Find the door position (where the central wall has a gap)
        mid = self.core.map_size // 2
        self.door_y = None
        for y in range(1, self.core.map_size - 1):
            if self.core.grid[y, mid] == 0:
                self.door_y = y
                break

    def check_door_blocked(self):
        """Checks if the door is blocked by a locked box at the end of prep phase."""
        if self.door_y is None:
            return False
        mid = self.core.map_size // 2
        # Check if the cell in front of the door is a locked box
        # Assuming door is at (mid, door_y), we check left and right sides
        left_cell = self.core.grid[self.door_y, mid - 1]
        right_cell = self.core.grid[self.door_y, mid + 1]
        
        return (left_cell == 3) or (right_cell == 3)

    def count_locked_boxes(self):
        return int(np.sum(self.core.grid == 3))

    def hider_wall_hugging(self):
        """Calculates how many hiders are adjacent to a wall."""
        hugging_count = 0
        for hx, hy in self.core.hider_positions:
            # Check 4 adjacent cells for a wall
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = hx + dx, hy + dy
                if 0 <= nx < self.core.map_size and 0 <= ny < self.core.map_size:
                    if self.core.grid[ny, nx] == 1:
                        hugging_count += 1
                        break
        return hugging_count