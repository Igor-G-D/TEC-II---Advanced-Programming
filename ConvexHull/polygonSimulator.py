import numpy as np
import cv2
import time
import pandas as pd
import os
import math
from dataStructures import Point, Line
from convexhull import convexHull

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
convexHull_time = []
convexHull_points = []
pointList_n = []
convexHull_points_n = []


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

def clearSimulation(clearData = False):
    global pointList, lineList
    global convexHull_time, convexHull_points
    global pointList_n, convexHull_points_n
    global num_points
    global image
    
    print("Clearing all points and lines")
    pointList.clear()
    lineList.clear()
    convexHull_points.clear()
    num_points = 0
    image.fill(255)
    
    if clearData:
        convexHull_time.clear()
        pointList_n.clear()
        convexHull_points_n.clear()

def populate_triangle_grid(stage):
    new_points = []
    
    # Scale parameters with stage
    rows = 3 + stage * 2 
    size = min(widthImage, heightImage) * 0.7
    
    center_x = widthImage // 2
    center_y = heightImage // 2
    height = size
    side_length = (2 * height) / math.sqrt(3)
    
    top = Point(center_x, center_y - height/2, (0, 0, 0))
    left = Point(center_x - side_length/2, center_y + height/2, (0, 0, 0))
    right = Point(center_x + side_length/2, center_y + height/2, (0, 0, 0))
    
    # triangular grid
    for row in range(rows):
        t = row / (rows - 1) if rows > 1 else 0
        left_top = Point(
            left.x + t * (top.x - left.x),
            left.y + t * (top.y - left.y),
            (0, 0, 0)
        )
        left_right = Point(
            left.x + t * (right.x - left.x),
            left.y + t * (right.y - left.y),
            (0, 0, 0)
        )
        
        points_in_row = row + 1
        
        for col in range(points_in_row):
            u = col / (points_in_row - 1) if points_in_row > 1 else 0
            x = left_top.x + u * (left_right.x - left_top.x)
            y = left_top.y + u * (left_right.y - left_top.y)
            new_points.append(Point(x, y, (0, 0, 0)))
    
    return new_points

def populate_square_grid(stage):
    new_points = []
    
    grid_size = 3 + stage * 2 
    spacing = min(widthImage, heightImage) // (grid_size + 2)
    
    # Calculate centered grid
    center_x = widthImage // 2
    center_y = heightImage // 2
    total_width = (grid_size - 1) * spacing
    total_height = (grid_size - 1) * spacing
    offset_x = center_x - total_width / 2
    offset_y = center_y - total_height / 2
    
    for r in range(grid_size):
        for c in range(grid_size):
            x = offset_x + c * spacing
            y = offset_y + r * spacing
            if 0 <= x <= widthImage and 0 <= y <= heightImage:
                new_points.append(Point(x, y, (0, 0, 0)))
    return new_points

def populate_circle_grid(stage):
    new_points = []
    
    num_circles = 2 + stage  
    points_per_circle = 8 + stage * 4  
    max_radius = min(widthImage, heightImage) * 0.4
    
    center_x = widthImage // 2
    center_y = heightImage // 2
    
    # Create points for each circle
    for circle_idx in range(num_circles):
        radius = max_radius * (circle_idx + 1) / num_circles
        
        for point_idx in range(points_per_circle):
            angle = 2 * math.pi * point_idx / points_per_circle
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            if 0 <= x <= widthImage and 0 <= y <= heightImage:
                new_points.append(Point(x, y, (0, 0, 0)))
    
    return new_points


def calculate_convex_hull():
    global convexHull_points, convexHull_time, lineList, pointList_n, convexHull_points_n
    
    if len(pointList) > 2:
        lineList.clear()
        convexHull_points.clear()
        t0 = time.perf_counter()
        convexHull_points.extend(convexHull(pointList))
        t1 = time.perf_counter()
        convexHull_time.append((t1 - t0) * 1000)
        
        # Draw convex hull edges
        if len(convexHull_points) > 1:
            for i in range(len(convexHull_points)):
                start_pt = convexHull_points[i]
                end_pt = convexHull_points[(i + 1) % len(convexHull_points)]
                lineList.append(Line(start_pt, end_pt, (255, 0, 0)))
        
        # Update tracking lists
        pointList_n.append(len(pointList))
        print(pointList_n)
        convexHull_points_n.append(len(convexHull_points))
def mouse_callback(event, x, y, flags, param):
    global mouse_move_size, mouse_click_size, objects_clicked_size, pointList, pointList_n, lineList, num_points, convexHull_points, convexHull_points_n, convexHull_time

    
    if event == cv2.EVENT_LBUTTONDOWN:
        
        
        point_clicked = False
        
        # checking points
        for i,point in enumerate(pointList):
            distance = ((x - point.x)**2 + (y - point.y)**2)**0.5
            if distance <= point_radius:
                print(f"Mouse inside point {i} at ({x}, {y})")
                point_clicked = True
                
        if not point_clicked:
            point = Point(x, y, (0,0,0))
            pointList.append(point)
            print(f"Created point {len(pointList)} at ({x},{y})")
            print(point.x)
            num_points += 1
                    
            # Calculate convex hull if we have enough points
            if num_points > 2:
                calculate_convex_hull()
            else:
                pointList_n.append(num_points)
                    
window_name = "Simulation"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)


start_time = time.time()
keypress = False
running = True # in case i want to add a pause function later
triangle_stage = 0
square_stage = 0
circle_stage = 0

while True:
    if running:
        # Clear the image
        image.fill(255)

        # Update positions
        for point in pointList:
            point.update(1.0, widthImage, heightImage)

        # Draw points
        for point in pointList:
            cv2.circle(image, (int(point.x), int(point.y)), point_radius, (0,0,255) if point in convexHull_points else (255,0,0), -1)

        # Draw voronoi edges
        for line in lineList:
            cv2.line(image, (int(line.point_1.x), int(line.point_1.y)),
                    (int(line.point_2.x), int(line.point_2.y)), [0,0,0], voronoi_edge_width)
            

        # Display the image
        cv2.imshow("Simulation", image)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 255:
            keypress = False
        else:
            if not keypress:
                if key == 27:  # ESC
                    break
                elif key == 8:  # Backspace
                    clearSimulation(True)
                elif key == ord('1'):  # Triangle grid
                    if (triangle_stage == 0):
                        clearSimulation(True)

                    triangle_stage += 1
                    square_stage = 0
                    circle_stage = 0
                    
                    clearSimulation()

                    pts = populate_triangle_grid(triangle_stage)
                    pointList.extend(pts)
                    num_points = len(pointList)

                    # Compute convex hull using helper function
                    if num_points > 2:
                        calculate_convex_hull()

                    keypress = True

                elif key == ord('2'):  # Square grid
                    if (square_stage == 0):
                        clearSimulation(True)
                    square_stage += 1
                    triangle_stage = 0
                    circle_stage = 0
                    
                    clearSimulation()

                    pts = populate_square_grid(square_stage)
                    pointList.extend(pts)
                    num_points = len(pointList)

                    if num_points > 2:
                        calculate_convex_hull()

                    keypress = True

                elif key == ord('3'):  # Circle grid
                    if (circle_stage == 0):
                        clearSimulation(True)
                        
                    circle_stage += 1
                    triangle_stage = 0
                    square_stage = 0
                    
                    clearSimulation()

                    pts = populate_circle_grid(circle_stage)
                    pointList.extend(pts)
                    num_points = len(pointList)

                    if num_points > 2:
                        calculate_convex_hull()

                    keypress = True

end_time = time.time()
execution_duration = end_time - start_time
print(f"Simulation ran for {execution_duration:.2f} seconds.")

#storing logged info

timestamp = time.strftime("%Y%m%d_%H%M%S")
folder_name = f"simulation_data_{timestamp}"
os.makedirs(folder_name, exist_ok=True)


simulation_info_df = pd.DataFrame({
    'parameter': ['timestamp_start','timestamp_end','duration_seconds', 'width', 'height', 'num_points'],
    'value': [start_time, end_time, execution_duration, widthImage, heightImage, num_points]
})
simulation_info_df.to_csv(os.path.join(folder_name, 'simulation_info.csv'), index=False)

# Export timing and bad triangle data per point
performance_data = []
for i, pointNum in enumerate(pointList_n):
    performance_data.append({
        'convexhull_time_ms': convexHull_time[i-2] if i > 1 and i-2 < len(convexHull_time) else None,
        'total_points': pointNum,
        'convexHull_points': convexHull_points_n[i-2] if i > 1 and i-2 < len(convexHull_time) else None
    })

performance_df = pd.DataFrame(performance_data)
performance_df.to_csv(os.path.join(folder_name, 'performance_log.csv'), index=False)

cv2.imwrite(os.path.join(folder_name, f"simulation_result.png"), image)

print(f"Exported simulation parameters to {folder_name}/simulation_info.csv")
print(f"Exported performance log to {folder_name}/performance_log.csv")
print(f"Exported final resulting graph to {folder_name}/simulation_result.png")


cv2.destroyAllWindows()