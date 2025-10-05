import numpy as np
import math
from typing import Tuple
import random
import cv2
import time
import pandas as pd
import os
import delaunay_bowyer_watson as delaunay
from dataStructures import Point, Line, Triangle
from voronoi import Voronoi

# Logging Info
MAX_MOUSE_MOVE_POINTS = 100000 # this many points should be good for ~30 mins of runtime at 60 points per second
MAX_MOUSE_CLICK_POINTS = 10000 # this is probably overkill

mouse_move_xs = np.zeros((MAX_MOUSE_MOVE_POINTS,), dtype = np.int32)
mouse_move_ys = np.zeros((MAX_MOUSE_MOVE_POINTS,), dtype = np.int32)
mouse_move_timestamp = np.zeros((MAX_MOUSE_MOVE_POINTS,), dtype=np.float64)
mouse_move_size = 0 # how many points stored so far

mouse_click_xs = np.zeros((MAX_MOUSE_CLICK_POINTS,), dtype = np.int32)
mouse_click_ys = np.zeros((MAX_MOUSE_CLICK_POINTS,), dtype = np.int32)
mouse_click_timestamp = np.zeros((MAX_MOUSE_MOVE_POINTS,), dtype=np.float64)
mouse_click_size = 0 # how many points stored so far

objects_clicked_index = np.zeros((MAX_MOUSE_CLICK_POINTS,), dtype = np.int32)
objects_clicked_type = np.zeros((MAX_MOUSE_CLICK_POINTS,), dtype=np.int8) 
objects_clicked_timestamp = np.zeros((MAX_MOUSE_MOVE_POINTS,), dtype=np.float64)
objects_clicked_size = 0 # how many oobjects have been clicked so far

# Simulation Info
heightImage = 800
widthImage = 1200

line_width = 6
voronoi_edge_width = 2
delaunay_edge_width = 1
point_radius = 5

# Mapping from class to type code
class_to_type = {
    Point: 0,
    Line: 1
}

# Reverse mapping type code -> class
type_to_class = {v: k for (k, v) in class_to_type.items()}

num_points = 0
num_lines = 0

pointList = []
lineList = []
voronoiEdges = []
delaunayEdges = []
voronoiTime = []
delaunayTime = []

# creating image
image = np.ones((heightImage, widthImage, 3), dtype=np.uint8) * 255

def point_line_distance(P, A, B): # this calculates the distance between a line defined by points A and B to a point P
    # Vector from A to B
    AB = B - A
    # Vector from A to P
    AP = P - A
    # Projection length (normalized)
    t = np.dot(AP, AB) / np.dot(AB, AB)
    # Clamp t to [0,1] to stay within the segment
    t = max(0, min(1, t))
    # Closest point on segment
    closest = A + t * AB
    # Distance from P to closest point
    return np.linalg.norm(P - closest)

def mouse_callback(event, x, y, flags, param):
    global mouse_move_size, mouse_click_size, objects_clicked_size, pointList, num_points, voronoiEdges, delaunayEdges, voronoiTime, delaunayTime

    if event == cv2.EVENT_MOUSEMOVE and mouse_move_size < MAX_MOUSE_MOVE_POINTS:
        mouse_move_xs[mouse_move_size] = x
        mouse_move_ys[mouse_move_size] = y
        mouse_move_timestamp[mouse_move_size] = time.time()
        mouse_move_size += 1
    
    if event == cv2.EVENT_LBUTTONDOWN:
        
        mouse_click_xs[mouse_click_size] = x
        mouse_click_ys[mouse_click_size] = y
        mouse_click_timestamp[mouse_click_size] = time.time()
        mouse_click_size += 1
        
        point_clicked = False
        
        # checking points
        for i,point in enumerate(pointList):
            distance = ((x - point.x)**2 + (y - point.y)**2)**0.5
            if distance <= point_radius:
                print(f"Mouse inside point {i} at ({x}, {y})")
                point_clicked = True
                objects_clicked_index[objects_clicked_size] = i
                objects_clicked_type[objects_clicked_size] = class_to_type.get(Point)
                objects_clicked_timestamp[objects_clicked_size] = time.time()
                objects_clicked_size += 1
                
        if not point_clicked:
            point = Point(x, y, (0,0,0))
            pointList.append(point)
            print(f"Created point {len(pointList)} at ({x},{y})")
            print(point.x)
            num_points += 1
            
            
            if num_points > 1: # start calculating voronoi and delunay triangulation
                voronoiEdges.clear()
                voronoi = Voronoi(pointList, widthImage, heightImage)
                voronoi_start = time.perf_counter()
                voronoi = Voronoi(pointList, widthImage, heightImage)
                voronoi.update()
                voronoi_end = time.perf_counter()
                
                voronoiTime.append((voronoi_end - voronoi_start) * 1000)
                
                for edge in voronoi.edges:
                    if edge.start and edge.end:
                        start = Point(edge.start.x, edge.start.y)
                        end = Point(edge.end.x, edge.end.y)
                        voronoiEdges.append(Line(start, end))
                        
            if num_points > 2:
                delaunayEdges.clear()
                delaunay_start = time.perf_counter()
                triangles = delaunay.bowyer_watson(pointList)
                delaunay_end = time.perf_counter()
                
                delaunayTime.append((delaunay_end - delaunay_start) * 1000)
                for tri in triangles:
                    vertices = tri.vertices
                    l1 = Line(vertices[0], vertices[1])
                    l2 = Line(vertices[0], vertices[2])
                    l3 = Line(vertices[1], vertices[2]) 
                    
                    delaunayEdges.extend([l1,l2,l3])
                
        '''
        # checking lines
        P = np.array([x, y])
        for i, line in enumerate(lineList):
            dist = point_line_distance(P, np.array([line.point_1.x, line.point_1.y]), np.array([line.point_2.x, line.point_2.y]))
            if dist <= line_width / 2:
                print(f"Inside line {i} at ({x},{y})")
                objects_clicked_index[objects_clicked_size] = i
                objects_clicked_type[objects_clicked_size] = class_to_type.get(Line)
                objects_clicked_timestamp[objects_clicked_size] = time.time()
                objects_clicked_size += 1
        '''
                
window_name = "Simulation"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)


start_time = time.time()
running = True # in case i want to add a pause function later
while True:
    if running:
        # Clear the image
        image.fill(255)

        # Update positions
        for point in pointList:
            point.update(1.0, widthImage, heightImage)
            
        for line in lineList:
            line.update(1.0,  widthImage, heightImage)

        # Draw points
        for point in pointList:
            cv2.circle(image, (int(point.x), int(point.y)), point_radius, point.color, -1)

        # Draw voronoi edges
        for line in voronoiEdges:
            cv2.line(image, (int(line.point_1.x), int(line.point_1.y)),
                    (int(line.point_2.x), int(line.point_2.y)), [0,0,0], voronoi_edge_width)
            
        # Draw delaunay edges
        for line in delaunayEdges:
            cv2.line(image, (int(line.point_1.x), int(line.point_1.y)),
                    (int(line.point_2.x), int(line.point_2.y)), [0,0,255], delaunay_edge_width)

        # Display the image
        cv2.imshow("Simulation", image)
        
        # Check for ESC key to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

end_time = time.time()
execution_duration = end_time - start_time
print(f"Simulation ran for {execution_duration:.2f} seconds.")

#storing logged info

timestamp = time.strftime("%Y%m%d_%H%M%S")
folder_name = f"simulation_data_{timestamp}"
os.makedirs(folder_name, exist_ok=True)

#mouse movement data
mouse_move_df = pd.DataFrame({
    'timestamp': mouse_move_timestamp[:mouse_move_size],
    'x': mouse_move_xs[:mouse_move_size],
    'y': mouse_move_ys[:mouse_move_size]
})
mouse_move_df.to_csv(os.path.join(folder_name, 'mouse_movements.csv'), index=False)
print(f"Exported {mouse_move_size} mouse movement points")

#mouse click data
mouse_click_df = pd.DataFrame({
    'timestamp': mouse_click_timestamp[:mouse_click_size],
    'x': mouse_click_xs[:mouse_click_size],
    'y': mouse_click_ys[:mouse_click_size]
})
mouse_click_df.to_csv(os.path.join(folder_name, 'mouse_clicks.csv'), index=False)
print(f"Exported {mouse_click_size} mouse click points")

#objects click data
objects_clicked_df = pd.DataFrame({
    'timestamp': objects_clicked_timestamp[:objects_clicked_size],
    'object_index': objects_clicked_index[:objects_clicked_size],
    'object_type': objects_clicked_type[:objects_clicked_size],
    'object_type_name': ['Point' if t == 0 else 'Line' for t in objects_clicked_type[:objects_clicked_size]]
})
objects_clicked_df.to_csv(os.path.join(folder_name, 'objects_clicked.csv'), index=False)
print(f"Exported {objects_clicked_size} object click events")

#simulation parameters
simulation_info_df = pd.DataFrame({
    'parameter': ['timestamp_start','timestamp_end','duration_seconds', 'width', 'height', 'num_points', 'num_lines'],
    'value': [start_time, end_time, execution_duration, widthImage, heightImage, num_points, num_lines]
})
simulation_info_df.to_csv(os.path.join(folder_name, 'simulation_info.csv'), index=False)
'''
#initial point info
points_df = pd.DataFrame(points_info)
points_df.to_csv(os.path.join(folder_name, 'points_info.csv'), index=False)

#initial line info
lines_df = pd.DataFrame(lines_info)
lines_df.to_csv(os.path.join(folder_name, 'lines_info.csv'), index=False)
'''
print(f"All data exported to folder: {folder_name}")

print(voronoiTime)
print(delaunayTime)

cv2.destroyAllWindows()