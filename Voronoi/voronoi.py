import heapq
import math
import matplotlib.pyplot as plt
from dataStructures import Point, Arc, Event, Edge

'''
IMPORTANT NOTE

This was adapted from https://github.com/ridoluc/Voronoi-Diagram/tree/master
'''

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

    def complete_segments(self, last): # this was pretty much copied over from the javascript code in the references
        r = self.beachline_root
        while r.right:
            e = r.edge['right']
            x = self.parabola_intersection(last.y * 1.1, e.arc['left'], e.arc['right'])
            y = e.get_y(x)

            # Determine if the edge is valid
            if (e.start.y < 0 and y < e.start.y) or (e.start.x < 0 and x < e.start.x) or (e.start.x > self.box_x and x > e.start.x):
                e.end = e.start
            else:
                if e.m == 0:
                    x = 0 if x - e.start.x <= 0 else self.box_x
                    e.end = Point(x, e.start.y)
                    self.voronoi_vertex.append(e.end)
                else:
                    if e.m == float('inf'):
                        y = self.box_y
                    else:
                        y = 0 if e.m * (x - e.start.x) <= 0 else self.box_y
                    e.end = self.edge_end(e, y)

            r = r.right

        # iterates through all edges, determines if the start or end points are outside the bounding box, adjusts accordingly
        for e in self.edges:
            option = 1 * self.point_outside(e.start) + 2 * self.point_outside(e.end)
            if option == 3:
                self.edges.remove(e)
            elif option == 1:
                y = 0 if e.start.y < e.end.y else self.box_y
                e.start = self.edge_end(e, y)
            elif option == 2:
                y = 0 if e.end.y <= e.start.y else self.box_y
                e.end = self.edge_end(e, y)
                
    def edge_end(self, e, y_lim):
        x = max(0, min(self.box_x, e.get_x(y_lim)))
        y = e.get_y(x)
        if y is None:
            y = y_lim
        p = Point(x, y)
        self.voronoi_vertex.append(p)
        return p

    def point_outside(self, p):
        # Check if the point is outside the canvas boundaries
        return p.x < 0 or p.x > self.box_x or p.y < 0 or p.y > self.box_y
    

    def parabola_intersection(self, y, f1, f2):
        fy_diff = f1.y - f2.y
        if fy_diff == 0:
            return (f1.x + f2.x) / 2
        fx_diff = f1.x - f2.x
        b1md = f1.y - y  
        b2md = f2.y - y  
        h1 = (-f1.x * b2md + f2.x * b1md) / fy_diff
        h2 = math.sqrt(b1md * b2md * (fx_diff ** 2 + fy_diff ** 2)) / fy_diff
        return h1 + h2 

    def edge_intersection(self, e1, e2):
        if e1.m == math.inf:
            return Point(e1.start.x, e2.get_y(e1.start.x))
        elif e2.m == math.inf:
            return Point(e2.start.x, e1.get_y(e2.start.x))
        else:
            mdif = e1.m - e2.m
            if mdif == 0:
                return None
            x = (e2.q - e1.q) / mdif
            y = e1.get_y(x)
            return Point(x, y)
        
    def point_event(self, p):
        q = self.beachline_root
        if q is None:
            self.beachline_root = Arc(None, None, p, None, None)
        else:
            while q.right is not None and self.parabola_intersection(p.y, q.focus, q.right.focus) <= p.x:
                q = q.right

            e_qp = Edge(q.focus, p, p.x)
            e_pq = Edge(p, q.focus, p.x)

            arc_p = Arc(q, None, p, e_qp, e_pq)
            arc_qr = Arc(arc_p, q.right, q.focus, e_pq, q.edge['right'])
            if q.right:
                q.right.left = arc_qr
            arc_p.right = arc_qr
            q.right = arc_p
            q.edge['right'] = e_qp

            if q.event:
                q.event.active = False

            self.add_circle_event(p, q)
            self.add_circle_event(p, arc_qr)

            self.edges.append(e_qp)
            self.edges.append(e_pq)

    def circle_event(self, e):
        arc = e.caller
        p = e.position
        edge_new = Edge(arc.left.focus, arc.right.focus)

        if arc.left.event:
            arc.left.event.active = False
        if arc.right.event:
            arc.right.event.active = False

        arc.left.edge['right'] = edge_new
        arc.right.edge['left'] = edge_new
        arc.left.right = arc.right
        arc.right.left = arc.left

        self.edges.append(edge_new)

        if not self.point_outside(e.vertex):
            self.voronoi_vertex.append(e.vertex)

        arc.edge['left'].end = arc.edge['right'].end = edge_new.start = e.vertex

        self.add_circle_event(p, arc.left)
        self.add_circle_event(p, arc.right)

    def add_circle_event(self, p, arc): 
        if arc.left and arc.right:
            a = arc.left.focus
            b = arc.focus
            c = arc.right.focus

            if (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y) > 0:
                new_inters = self.edge_intersection(arc.edge['left'], arc.edge['right'])
                circle_radius = math.sqrt(
                    (new_inters.x - arc.focus.x) ** 2 +
                    (new_inters.y - arc.focus.y) ** 2)
                event_pos = circle_radius + new_inters.y
                if event_pos > p.y and new_inters.y < self.box_y:
                    e = Event("circle", Point(new_inters.x, event_pos), arc, new_inters)
                    arc.event = e
                    heapq.heappush(self.event_list, e)


def plot_voronoi(voronoi, points, width, height):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x_points = [p.x for p in points]
    y_points = [p.y for p in points]
    ax.scatter(x_points, y_points, c='red', s=50, zorder=5, label='Sites')

    for edge in voronoi.edges:
        if edge.start and edge.end:
            start = (edge.start.x, edge.start.y)
            end = (edge.end.x, edge.end.y)
            ax.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=1)
    
    if voronoi.voronoi_vertex:
        x_vertices = [v.x for v in voronoi.voronoi_vertex]
        y_vertices = [v.y for v in voronoi.voronoi_vertex]
        ax.scatter(x_vertices, y_vertices, c='green', s=30, zorder=4, label='Vertices')
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Voronoi Diagram')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Test case
if __name__ == "__main__":
    points = [
        Point(2, 3),
        Point(5, 8),
        Point(8, 4),
        Point(4, 6),
        Point(7, 2),
        Point(3, 7)
    ]
    
    width, height = 10, 10
    
    voronoi = Voronoi(points, width, height)
    voronoi.update()

    print(f"Generated {len(voronoi.edges)} edges")
    print(f"Generated {len(voronoi.voronoi_vertex)} vertices")
    
    plot_voronoi(voronoi, points, width, height)