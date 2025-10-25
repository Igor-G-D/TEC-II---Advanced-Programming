from dataStructures import Polygon, Point
from typing import List

def find_min_point_index(points: List[Point]) -> int:

    min_idx = 0
    for i in range(1, len(points)):
        if points[i].y < points[min_idx].y or \
            (points[i].y == points[min_idx].y and points[i].x < points[min_idx].x):
            min_idx = i
    return min_idx

def center_polygon_at_origin(polygon: Polygon) -> Polygon:
    # this is needed because of the canvas being 200x200 when creating the robot, so it needs to be centered first so the expanded polygon isn't offset

    # Calculate center of polygon
    center_x = sum(p.x for p in polygon.points) / len(polygon.points)
    center_y = sum(p.y for p in polygon.points) / len(polygon.points)
    
    # Translate all points to center at origin
    centered_points = []
    for point in polygon.points:
        centered_points.append(Point(
            point.x - center_x,
            point.y - center_y
        ))
    
    return Polygon(centered_points, color=polygon.color)

def reflect_polygon_through_origin(polygon: Polygon) -> Polygon:
    reflected_points = []
    for point in polygon.points:
        reflected_points.append(Point(-point.x, -point.y))
    return Polygon(reflected_points, color=polygon.color)

def minkowski(poly1: Polygon, poly2: Polygon) -> Polygon:
    # find the point with minimum y (and leftmost if tie) in each polygon
    min_idx1 = find_min_point_index(poly1.points)
    min_idx2 = find_min_point_index(poly2.points)
    
    n = len(poly1.points)
    m = len(poly2.points)
    
    i = 0
    j = 0
    result_points = []
    
    while i < n or j < m:
        #add current sum point
        p1 = poly1.points[(min_idx1 + i) % n]
        p2 = poly2.points[(min_idx2 + j) % m]
        result_points.append(Point(p1.x + p2.x, p1.y + p2.y))
        
        #Compare angles of next edges to decide which polygon to advance
        if i < n and j < m:
            next_p1 = poly1.points[(min_idx1 + i + 1) % n]
            next_p2 = poly2.points[(min_idx2 + j + 1) % m]
            
            #vector from current to next point
            vec1 = Point(next_p1.x - p1.x, next_p1.y - p1.y)
            vec2 = Point(next_p2.x - p2.x, next_p2.y - p2.y)
            
            # cross product to compare angles
            cross = vec1.x * vec2.y - vec1.y * vec2.x
            
            if cross >= 0:
                i += 1
            if cross <= 0:
                j += 1
        elif i < n:
            i += 1
        else:
            j += 1
    
    return Polygon(result_points)

def minkowski_robot(obstacle: Polygon, robot: Polygon) -> Polygon:

    #reflect robot points
    reflected_robot = reflect_polygon_through_origin(robot)
    #center robot points
    centered_robot = center_polygon_at_origin(reflected_robot)
    
    result = minkowski(obstacle, centered_robot)
    
    return result

def calc_distance_between_polygons(poly1: Polygon, poly2:Polygon) -> float:
    reversed_poly2 = reflect_polygon_through_origin(poly2)
    
    mink_sum = minkowski(poly1, reversed_poly2)
    
    distance = mink_sum.shortest_distance_to_point(Point(0,0))
    
    return distance