import matplotlib.pyplot as plt
from dataStructures import Point
def convexHull(points):
    if(len(points) < 3):
        return [] # needs at least 3 points
    sorted_points = sorted(points)
    lower = []
    for p in sorted_points:
        while len(lower) >= 2 and orientation(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
        
    upper = []
    for p in reversed(sorted_points):
        while len(upper) >= 2 and orientation(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
        
    return lower[:-1] + upper[:-1]

def orientation(p, q, r):
    # negative -> clockwise, positive -> counterclockwise, zero -> collinear
    return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)