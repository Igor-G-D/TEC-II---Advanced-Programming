import numpy as np
import math
from typing import Tuple
import random
import cv2
import time
import pandas as pd
import os

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
point_radius = 8

class Point:
    def __init__(self, x: float, y: float, color: Tuple[int, int, int] = (0,0,0), vx: float = 0.0, vy: float = 0.0, ax: float = 0.0, ay: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.color = color
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay
    def update(self, dt: float = 1.0) -> None:
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


    
class Line:
    def __init__(self, p1: Point, p2: Point, color: Tuple[int, int, int] = (0,0,0)) -> None:
        self.point_1 = p1
        self.point_2 = p2
        self.color = color
    
    def update(self, dt: float = 1.0) -> None:
        self.point_1.update(dt)
        self.point_2.update(dt)
        

# Mapping from class to type code
class_to_type = {
    Point: 0,
    Line: 1
}

# Reverse mapping type code -> class
type_to_class = {v: k for (k, v) in class_to_type.items()}


# create random lines and points
num_points = 8
num_lines = 4

pointList = [Point] * num_points
lineList = [Line] * num_lines

for i in range(0, num_points):
    point_x = random.randint(0, widthImage)
    point_y = random.randint(0, heightImage)
    point_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    point_acc_x = random.uniform(-1, 1) * 0.2

    point_acc_y = random.uniform(-1, 1) * 0.2
    
    pointList[i] = Point(point_x, point_y, point_color, point_acc_x, point_acc_y)

for i in range(0, num_lines):
    point_x_1 = random.randint(0, widthImage)
    point_y_1 = random.randint(0, heightImage)
    
    point_acc_x_1 = random.uniform(-1, 1) * 0.2
    point_acc_y_1 = random.uniform(-1, 1) * 0.2
    
    point_x_2 = random.randint(0, widthImage)
    point_y_2 = random.randint(0, heightImage)
    point_acc_x_2 = random.uniform(-1, 1) * 0.2
    point_acc_y_2 = random.uniform(-1, 1) * 0.2
    
    line_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    lineList[i] = Line(Point(point_x_1, point_y_1, (0,0,0), point_acc_x_1, point_acc_y_1), Point(point_x_2, point_y_2, (0,0,0), point_acc_x_2, point_acc_y_2), line_color)

#storing initial positions to log starting parameters later

points_info = []
for i, point in enumerate(pointList):
    points_info.append({
        'index': i,
        'initial_x': point.x,
        'initial_y': point.y,
        'initial_vx': point.vx,
        'initial_vy': point.vy,
        'initial_ax': point.ax,
        'initial_ay': point.ay,
        'color_r': point.color[0],
        'color_g': point.color[1],
        'color_b': point.color[2]
    })
    
lines_info = []
for i, line in enumerate(lineList):
    lines_info.append({
        'index': i,
        'point1_x': line.point_1.x,
        'point1_y': line.point_1.y,
        'point1_vx': line.point_1.vx,
        'point1_vy': line.point_1.vy,
        'point1_ax': line.point_1.ax,
        'point1_ay': line.point_1.ay,
        'point2_x': line.point_2.x,
        'point2_y': line.point_2.y,
        'point2_vx': line.point_2.vx,
        'point2_vy': line.point_2.vy,
        'point2_ax': line.point_2.ax,
        'point2_ay': line.point_2.ay,
        'color_r': line.color[0],
        'color_g': line.color[1],
        'color_b': line.color[2]
    })

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
    global mouse_move_size
    global mouse_click_size
    global objects_clicked_size
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
        
        # checking points
        for i,point in enumerate(pointList):
            distance = ((x - point.x)**2 + (y - point.y)**2)**0.5
            if distance <= point_radius:
                print(f"Mouse inside point {i} at ({x}, {y})")
                objects_clicked_index[objects_clicked_size] = i
                objects_clicked_type[objects_clicked_size] = class_to_type.get(Point)
                objects_clicked_timestamp[objects_clicked_size] = time.time()
                objects_clicked_size += 1
                
            
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
            point.update()
            
        for line in lineList:
            line.update()

        # Draw points
        for point in pointList:
            cv2.circle(image, (int(point.x), int(point.y)), point_radius, point.color, -1)

        # Draw lines
        for line in lineList:
            cv2.line(image, (int(line.point_1.x), int(line.point_1.y)),
                    (int(line.point_2.x), int(line.point_2.y)), line.color, line_width)

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

#initial point info
points_df = pd.DataFrame(points_info)
points_df.to_csv(os.path.join(folder_name, 'points_info.csv'), index=False)

#initial line info
lines_df = pd.DataFrame(lines_info)
lines_df.to_csv(os.path.join(folder_name, 'lines_info.csv'), index=False)

print(f"All data exported to folder: {folder_name}")

cv2.destroyAllWindows()