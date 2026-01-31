import heapq
from typing import Tuple, List, Optional
from interfaces import Grid, PathfindingAlgorithm

class DijkstraCell:
    def __init__(self):
        self.parent: Optional[Tuple[int, int]] = None
        self.dist: float = float('inf')

def trace_path_dijkstra(cell_details: List[List[DijkstraCell]], dest: Tuple[int, int], start: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    x, y = dest
    while cell_details[x][y].parent is not None and (x, y) != start:
        path.append((x, y))
        px, py = cell_details[x][y].parent
        x, y = px, py
    path.append(start)
    path.reverse()
    return path

def dijkstra_search(grid: Grid, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    rows = grid.rows
    cols = grid.cols

    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < rows and 0 <= sy < cols) or not (0 <= gx < rows and 0 <= gy < cols):
        print("Start or goal out of bounds")
        return []
    if grid.is_obstacle(start) or grid.is_obstacle(goal):
        print("Start or goal is blocked")
        return []
    if start == goal:
        return [start]

    visited = [[False] * cols for _ in range(rows)]
    cell_details = [[DijkstraCell() for _ in range(cols)] for _ in range(rows)]

    cell_details[sx][sy].dist = 0.0
    cell_details[sx][sy].parent = (sx, sy)

    pq = []
    heapq.heappush(pq, (0.0, sx, sy))

    while pq:
        current_dist, x, y = heapq.heappop(pq)

        if visited[x][y]:
            continue

        visited[x][y] = True

        if (x, y) == goal:
            print("Reached goal via Dijkstra")
            return trace_path_dijkstra(cell_details, goal, start)

        neighbors = grid.get_neighbors((x, y)) # now no type checking is required

        for (nx, ny) in neighbors:
            if not visited[nx][ny] and not grid.is_obstacle((nx, ny)):
                move_cost = 1.0
                new_dist = current_dist + move_cost

                if new_dist < cell_details[nx][ny].dist:
                    cell_details[nx][ny].dist = new_dist
                    cell_details[nx][ny].parent = (x, y)
                    heapq.heappush(pq, (new_dist, nx, ny))

    print("Failed to find a path with Dijkstra")
    return []

class DijkstraAlgorithm(PathfindingAlgorithm):
    def find_path(grid: Grid, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        return dijkstra_search(grid, start, goal)
