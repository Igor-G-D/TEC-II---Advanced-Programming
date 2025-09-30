import heapq
import math
from polygonSimulator import Point

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

class Edge: # edge formed between two sites
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


class Voronoi:
    def __init__(self, points: list[Point], width: float, height: float):
        self.point_list = points
        self.box_x = width
        self.box_y = height
        self.reset()
        
    def reset(self):
        self.event_list = []
        self.beachline_root = None
        self.voronoi_vertex: list[Point] = []
        self.edges: list[Edge] = []
        
    def update(self):
        self.reset()
        for p in self.point_list:
            ev = Event("point", p)
            heapq.heappush(self.event_list, ev)

        last_event_pos = None
        
        while self.event_list: # main loop
            e = heapq.heappop(self.event_list)
            last_event_pos = e.position
            if e.type == "point":
                self.point_event(e.position)
            else:
                if e.active:
                    self.circle_event(e)

        # After all events, complete the infinite edges
        if last_event_pos:
            self.complete_segments(last_event_pos)
