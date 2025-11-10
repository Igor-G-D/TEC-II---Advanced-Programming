import heapq
import numpy as np
from typing import Tuple, List, Optional

class Cell:
    def __init__(self):
        # Parent pointer: the cell from which we came
        self.parent: Optional[Tuple[int, int]] = None
        # Total cost of the cell (f = g + h)
        self.f: float = float('inf')
        # Cost from start to this cell
        self.g: float = float('inf')
        # Heuristic cost from this cell to destination
        self.h: float = 0.0

def is_valid(grid: List[List[int]], position: Tuple[int, int]) -> bool:
    rows = len(grid)
    cols = len(grid[0])
    x, y = position
    return (0 <= x < rows) and (0 <= y < cols)

def is_unblocked(grid: List[List[int]], position: Tuple[int, int]) -> bool:
    return grid[position[0]][position[1]] == 0

def is_destination(position: Tuple[int, int], dest: Tuple[int, int]) -> bool:
    return position[0] == dest[0] and position[1] == dest[1]

def calculate_heuristic(position: Tuple[int, int], dest: Tuple[int, int]) -> float:
    # Euclidean distance
    return ((position[0] - dest[0]) ** 2 + (position[1] - dest[1]) ** 2) ** 0.5

def get_valid_neighbors(grid: List[List[int]], position: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = position
    moves = [
        (x + 1, y), (x - 1, y),
        (x, y + 1), (x, y - 1),
        (x + 1, y + 1), (x - 1, y - 1),
        (x + 1, y - 1), (x - 1, y + 1)
    ]
    valid_neighbors = []
    for nx, ny in moves:
        if is_valid(grid, (nx, ny)) and is_unblocked(grid, (nx, ny)):
            valid_neighbors.append((nx, ny))
    return valid_neighbors

def trace_path(cell_details: List[List[Cell]], dest: Tuple[int, int], start: Tuple[int, int]) -> None:
    path = []
    x, y = dest
    # Trace back from destination to start via parent pointers
    while cell_details[x][y].parent is not None and (x, y) != start:
        path.append((x, y))
        px, py = cell_details[x][y].parent
        x, y = px, py
    path.append(start)
    path.reverse()
    print("Path found:", " -> ".join(str(p) for p in path))
    return path

def a_star_search(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> None:
    rows = len(grid)
    cols = len(grid[0])

    # Validate start and goal
    if not is_valid(grid, start) or not is_valid(grid, goal):
        print("Start or goal is invalid.")
        return
    if not is_unblocked(grid, start) or not is_unblocked(grid, goal):
        print("Start or goal is blocked.")
        return
    if is_destination(start, goal):
        print("Start is the destination.")
        return

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
        # If this cell is already closed, skip it
        if closed_list[x][y]:
            continue

        closed_list[x][y] = True

        # Explore neighbors
        for (nx, ny) in get_valid_neighbors(grid, (x, y)):
            if is_destination((nx, ny), goal):
                cell_details[nx][ny].parent = (x, y)
                print("Found destination!")
                res = trace_path(cell_details, goal, start)
                found_dest = True
                return res

            if not closed_list[nx][ny]:
                new_g = cell_details[x][y].g + 1.0
                new_h = calculate_heuristic((nx, ny), goal)
                new_f = new_g + new_h

                # If this path to neighbor is better than previous best
                if cell_details[nx][ny].f > new_f:
                    cell_details[nx][ny].g = new_g
                    cell_details[nx][ny].h = new_h
                    cell_details[nx][ny].f = new_f
                    cell_details[nx][ny].parent = (x, y)
                    heapq.heappush(open_list, (new_f, nx, ny))

    if not found_dest:
        print("Failed to reach destination.")

def main():
    grid = [
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 1, 1, 0]
        ]

    src = (8, 0)
    dest = (0, 0)

    a_star_search(grid, src, dest)

if __name__ == "__main__":
    main()
