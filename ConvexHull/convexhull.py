import random
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


points = [Point(3,5), Point(1, 2), Point(6,5), Point(8,4), Point(3, 6)]


def orientation(p, q, r):
    # negative -> clockwise, positive -> counterclockwise, zero -> collinear
    return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)

def random_points(n, x_range=(0, 10), y_range=(0, 10)):
    pts = []
    for _ in range(n):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        pts.append(Point(x, y))
    return pts

def plot_hull(points, hull_points):
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    plt.scatter(xs, ys, color='blue', label='All points')
    # highlight hull points
    hx = [p.x for p in hull_points]
    hy = [p.y for p in hull_points]
    plt.scatter(hx, hy, color='red', label='Convex hull points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Convex Hull of Random Points')
    plt.legend()
    plt.axis('equal')  # preserve aspect ratio
    plt.show()

if __name__ == '__main__':
    points = random_points(30, x_range=(0, 10), y_range=(0, 10))
    hull = convexHull(points)
    print("Hull has {} points: {}".format(len(hull), hull))
    plot_hull(points, hull)