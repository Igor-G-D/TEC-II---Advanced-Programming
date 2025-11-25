import numpy as np
import math
from typing import Tuple, List
from interfaces import Grid
from abc import ABCMeta

class Simulation:
    def __init__(self, grid_factory, object_factory, algorithm_factory, grid_shape=0, cell_shape=0, allow_diagonals: bool = False):
        self.grid = grid_factory.create_grid(grid_shape, cell_shape, allow_diagonals)
        self.object_factory = object_factory
        self.algorithm_factory = algorithm_factory
        self.robots = []
        self.goals = []
        self.paths = []

    def add_robot(self, position: Tuple[int, int]) -> None:
        robot = self.object_factory.create_robot(position)
        self.robots.append(robot)

    def add_goal(self, position: Tuple[int, int], robot) -> None:
        goal = self.object_factory.create_goal(position)
        self.goals.append(goal)
        robot.goal = goal

    def run(self, allow_diagonals: bool) -> None:
        for robot, goal in zip(self.robots, self.goals):
            algorithm = self.algorithm_factory.create_algorithm("astar")
            path = algorithm.find_path(self.grid, robot.position, goal.position, allow_diagonals)
            self.paths.append(path)

class Object:
    def __init__(self, position: Tuple[int, int], color: Tuple[int, int, int] = (0, 0, 0)):
        self.position = position
        self.color = color

class Robot(Object):
    def __init__(self, position: Tuple[int, int], color: Tuple[int, int, int], goal = None):
        super().__init__(position, color)
        self.goal = goal

class Goal(Object):
    def __init__(self, position: Tuple[int, int]):
        super().__init__(position)

class SingletonGridMeta(ABCMeta):  # Inherit from ABCMeta
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class RectangleGrid(Grid, metaclass=SingletonGridMeta):
    def __init__(self, shape: Tuple[int, int], allow_diagonals: bool = False):
        self.rows, self.cols = shape
        self.matrix = np.zeros(shape)
        self.allow_diagonals = allow_diagonals # stored state so the class is conforming to the interfaces
    
    def toggle_obstacle(self, position: Tuple[int, int]) -> None:
        x, y = position
        self.matrix[x][y] = 1 if self.matrix[x][y] == 0 else 0
    
    def is_obstacle(self, position: Tuple[int, int]) -> bool:
        x, y = position
        return self.matrix[x][y] == 1
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]: # removed allow_diagonals from parameters to conform to interface
        x, y = position
        neighbors = []
        # Cardinal moves
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                neighbors.append((nx, ny))
        if self.allow_diagonals:
            for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.rows and 0 <= ny < self.cols:
                    neighbors.append((nx, ny))
        return neighbors
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        (x1, y1), (x2, y2) = a, b
        return math.hypot(x2 - x1, y2 - y1)

class HexGrid(Grid, metaclass=SingletonGridMeta):
    def __init__(self, shape: Tuple[int, int]):
        self.rows, self.cols = shape
        self.matrix = np.zeros(shape)
    
    def toggle_obstacle(self, position: Tuple[int, int]) -> None:
        r, q = position
        self.matrix[r][q] = 1 if self.matrix[r][q] == 0 else 0
    
    def is_obstacle(self, position: Tuple[int, int]) -> bool:
        r, q = position
        return self.matrix[r][q] == 1
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        row, col = position  # position is (row, col)
        
        if row % 2 == 1:  # odd row - shifted right
            neighbor_deltas = [
                (0, -1),               
                (-1, 0),     
                (-1, +1),  
                (0, +1),               
                (+1, +1),  
                (+1, 0)    
            ]
        else:  # even row
            neighbor_deltas = [
                (0, -1), 
                (-1, -1),
                (-1, 0),  
                (0, +1),  
                (+1, 0),  
                (+1, -1) 
            ]

        result = []
        for dr, dc in neighbor_deltas:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                result.append((new_row, new_col))
        return result

    def offset_to_cube(self, position: Tuple[int, int]) -> Tuple[int, int, int]:
        q, r = position
        x = q - (r - (r & 1)) // 2
        z = r
        y = -x - z
        return (x, y, z)

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        x1, y1, z1 = self.offset_to_cube(a)
        x2, y2, z2 = self.offset_to_cube(b)
        return max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2))
