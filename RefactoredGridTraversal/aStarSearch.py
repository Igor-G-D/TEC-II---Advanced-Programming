import heapq
from typing import Tuple, List, Optional
from interfaces import Grid, PathfindingAlgorithm

class Cell:
    def __init__(self):
        self.parent: Optional[Tuple[int, int]] = None
        self.f: float = float('inf')
        self.g: float = float('inf')
        self.h: float = 0.0

def is_valid(grid: Grid, position: Tuple[int, int]) -> bool:
    x, y = position
    return (0 <= x < grid.rows) and (0 <= y < grid.cols)

def is_destination(position: Tuple[int, int], dest: Tuple[int, int]) -> bool:
    return position[0] == dest[0] and position[1] == dest[1]

def trace_path(cell_details: List[List[Cell]], dest: Tuple[int, int], start: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    x, y = dest
    while cell_details[x][y].parent is not None and (x, y) != start:
        path.append((x, y))
        px, py = cell_details[x][y].parent
        x, y = px, py
    path.append(start)
    path.reverse()
    print("Path found:", " -> ".join(str(p) for p in path))
    return path

def a_star_search(grid: Grid, start: Tuple[int, int], goal: Tuple[int, int], allow_diagonals: bool) -> List[Tuple[int, int]]:
    rows = grid.rows
    cols = grid.cols

    # Validate start and goal
    if not is_valid(grid, start) or not is_valid(grid, goal):
        print("Start or goal is invalid.")
        return []
    if grid.is_obstacle(start) or grid.is_obstacle(goal):
        print("Start or goal is blocked.")
        return []
    if is_destination(start, goal):
        print("Start is the destination.")
        return [start]

    closed_list = [[False for _ in range(cols)] for _ in range(rows)]
    cell_details = [[Cell() for _ in range(cols)] for _ in range(rows)]

    sx, sy = start
    cell_details[sx][sy].parent = (sx, sy)
    cell_details[sx][sy].g = 0.0
    cell_details[sx][sy].h = 0.0
    cell_details[sx][sy].f = 0.0

    open_list = []
    heapq.heappush(open_list, (0.0, sx, sy))

    found_dest = False

    while open_list:
        f, x, y = heapq.heappop(open_list)
        if closed_list[x][y]:
            continue

        closed_list[x][y] = True

        # Get neighbors based on grid type
        from classes import RectangleGrid, HexGrid
        if isinstance(grid, RectangleGrid):
            neighbors = grid.get_neighbors((x, y), allow_diagonals)
        elif isinstance(grid, HexGrid):
            neighbors = grid.get_neighbors((x, y))
        else:
            neighbors = grid.get_neighbors((x, y))

        for (nx, ny) in neighbors:
            if is_destination((nx, ny), goal):
                cell_details[nx][ny].parent = (x, y)
                print("Found destination!")
                return trace_path(cell_details, goal, start)

            if not closed_list[nx][ny] and not grid.is_obstacle((nx, ny)):
                new_g = cell_details[x][y].g + 1.0
                new_h = grid.heuristic((nx, ny), goal)
                new_f = new_g + new_h

                if cell_details[nx][ny].f > new_f:
                    cell_details[nx][ny].g = new_g
                    cell_details[nx][ny].h = new_h
                    cell_details[nx][ny].f = new_f
                    cell_details[nx][ny].parent = (x, y)
                    heapq.heappush(open_list, (new_f, nx, ny))

    print("Failed to reach destination.")
    return []


class AStarAlgorithm(PathfindingAlgorithm):
    def find_path(self, grid: Grid, start: Tuple[int, int], goal: Tuple[int, int], allow_diagonals: bool) -> List[Tuple[int, int]]:
        return a_star_search(grid, start, goal, allow_diagonals)