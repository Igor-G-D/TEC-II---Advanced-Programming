import math
from typing import Tuple, List
import numpy as np

class Point:
    def __init__(self, x: float, y: float, color: Tuple[int, int, int] = (0,0,0), vx: float = 0.0, vy: float = 0.0, ax: float = 0.0, ay: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.color = color
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay
        
    def update(self, dt: float = 1.0, widthImage: int = 0, heightImage: int = 0) -> None:
        # Update velocity from acceleration
        self.vx += self.ax * dt
        self.vy += self.ay * dt

        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Bounce on X axis
        if self.x < 0:
            self.x = 0
            self.vx *= -1
        elif self.x > widthImage:
            self.x = widthImage
            self.vx *= -1

        # Bounce on Y axis
        if self.y < 0:
            self.y = 0
            self.vy *= -1
        elif self.y > heightImage:
            self.y = heightImage
            self.vy *= -1
            
    def __lt__(self, other):
            if (self.x == other.x):
                return self.y < other.y
            else:
                return self.x < other.x
            
    def __repr__(self):
            return f"Point({self.x},{self.y})"
        
    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar: float) -> 'Point':
        return Point(self.x * scalar, self.y * scalar)
    
    def dot(self, other: 'Point') -> float:
        return self.x * other.x + self.y * other.y

    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)


    
class Line:
    def __init__(self, p1: Point, p2: Point, color: Tuple[int, int, int] = (0,0,0)) -> None:
        self.point_1 = p1
        self.point_2 = p2
        self.color = color
    
    def update(self, dt: float = 1.0, widthImage: int = 0, heightImage: int = 0) -> None:
        self.point_1.update(dt, widthImage, heightImage)
        self.point_2.update(dt, widthImage, heightImage)

class Edge: # edge formed between two sites (THIS IS REALLY ONLY NEEDED FOR THE VORONOI DIAGRAM, USE LINE OTHERWISE)
    def __init__(self, p1: Point, p2: Point, startx: float = None):
        
        dy = p1.y - p2.y
        if abs(dy) < 1e-9: # rounding to a vertical line to prevent math weirdness
            self.m = math.inf
        else:
            self.m = -(p1.x - p2.x) / dy # slope of edge
        
        if abs(dy) < 1e-9:
            self.q = None # vertical lines don't intersect the y axis
        else:
            self.q = (0.5 * (p1.x**2 - p2.x**2 + p1.y**2 - p2.y**2)) / dy # y intersection of the slopes

        # sites to the left and right of the edge
        self.arc = {'left': p1, 'right': p2}

        self.start = None
        self.end = None

        if startx is not None: # if there already is a starting x, calculate the corresponding y
            if self.m != math.inf:
                y = self.get_y(startx)
            else:
                y = None
            self.start = Point(startx, y)

    def get_y(self, x: float):
        if self.m == math.inf or self.q is None:
            return None # if line is vertical
        return self.m * x + self.q # use equation to return y based on x

    def get_x(self, y: float):
        # If the line is vertical, return the x-coordinate of the start point
        if self.m == math.inf:
            if self.start is None:
                return None
            return self.start.x
        if self.m == 0:
            return None # if the line is horizontal, return none
        return (y - self.q) / self.m # use equation to return x based on y

class Arc:
    def __init__(self, left: 'Arc', right: 'Arc', focus: Point, edge_left: Edge, edge_right: Edge):
        self.left = left
        self.right = right
        self.focus = focus
        self.edge = {'left': edge_left, 'right': edge_right}
        self.event = None  # for circle event

class Event:
    def __init__(self, etype: str, position: Point, caller: Arc = None, vertex: Point = None):
        self.type = etype  # "point" or "circle"
        self.position = position
        self.caller = caller
        self.vertex = vertex
        self.active = True

    def __lt__(self, other: 'Event') -> bool:
        if self.position.y != other.position.y:
            return self.position.y < other.position.y
        return self.position.x < other.position.x

class Triangle:
    def __init__(self, a: Point, b: Point, c: Point):
        self.vertices = [a, b, c]
    
    def circumcircle_contains(self, point: Point):
        ax, ay = self.vertices[0].x, self.vertices[0].y
        bx, by = self.vertices[1].x, self.vertices[1].y
        cx, cy = self.vertices[2].x, self.vertices[2].y
        dx, dy = point.x, point.y

        mat = np.array([
            [ax - dx, ay - dy, (ax - dx) ** 2 + (ay - dy) ** 2],
            [bx - dx, by - dy, (bx - dx) ** 2 + (by - dy) ** 2],
            [cx - dx, cy - dy, (cx - dx) ** 2 + (cy - dy) ** 2]
        ])

        return np.linalg.det(mat) > 0
    
class Polygon:
    def __init__(self, points: List[Point], color: Tuple[int, int, int] = (0,0,0)):
        self.points = points
        self.color = color
        
        
    def is_point_inside(self, point: Point) -> bool: # ray casking algorithm
        x, y = point.x, point.y
        inside = False
        
        n = len(self.points)
        if n < 3:
            return False
        
        j = n - 1
        for i in range(n):
            xi, yi = self.points[i].x, self.points[i].y
            xj, yj = self.points[j].x, self.points[j].y
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside # if the ray is inside the polygon, it will intersect and odd number of times, and even if it is outside
        
    def shortest_distance_to_point(self, point: Point) -> float:
        
        if self.is_point_inside(point):
            return 0.0 # dist = 0 since it is inside the polygon
        
        min_dist = float('inf')
        
        # iterate through all edges of the polygon
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)]
            
            # vectors
            edge_vector = p2 - p1 
            point_vector = point - p1 
            
            r = edge_vector.dot(point_vector)
            
            edge_magnitude_sq = edge_vector.x**2 + edge_vector.y**2
            
            # Avoid division by zero
            if edge_magnitude_sq < 1e-9:
                dist = point_vector.magnitude()
            else:
                r /= edge_magnitude_sq
                
                if r < 0:
                    # Closest to p1
                    dist = point_vector.magnitude()
                elif r > 1:
                    # Closest to p2
                    dist = (point - p2).magnitude()
                else:
                    # Closest to the edge itself
                    projection_magnitude = r * math.sqrt(edge_magnitude_sq)
                    dist = math.sqrt(point_vector.magnitude()**2 - projection_magnitude**2)
            
            min_dist = min(min_dist, dist)
        
        return min_dist

