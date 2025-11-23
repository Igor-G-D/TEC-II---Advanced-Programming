from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

class Grid(ABC):
    def __init__(self):
        self.rows, self.cols = (0, 0)
        self.matrix = None
    
    @abstractmethod
    def toggle_obstacle(self, position: Tuple[int, int]) -> None: 
        pass
    
    @abstractmethod
    def is_obstacle(self, position: Tuple[int, int]) -> bool: 
        pass
    
    @abstractmethod
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]: 
        pass
    
    @abstractmethod
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float: 
        pass

class PathfindingAlgorithm(ABC):
    @abstractmethod
    def find_path(self, grid: Grid, start: Tuple[int, int], goal: Tuple[int, int], allow_diagonals: bool) -> List[Tuple[int, int]]:
        pass