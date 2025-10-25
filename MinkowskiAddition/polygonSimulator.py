import numpy as np
import cv2
import time
import pandas as pd
import os
import seaborn as sns
from dataStructures import Point, Polygon
from typing import List
from minkowski import minkowski_robot, calc_distance_between_polygons
from convexhull import convexHull

# Simulation Info
heightImage = 800
widthImage = 1200

point_radius = 5

num_points = 0

pointList = []
currentPointList= []
polygonList = []
color_palette = sns.color_palette("husl", 20)
bgr_palette = [(int(b * 255), int(g * 255), int(r * 255)) for r, g, b in color_palette]
rgb_palette = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in color_palette]


# creating image
main_image = np.ones((heightImage, widthImage, 3), dtype=np.uint8) * 255

#creating object image
obj_image = np.ones((200, 200, 3), dtype=np.uint8) * 255

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
    global pointList, currentPointList, polygonList, obj_points, expandedPolygons
    global num_points, obj_poly
    global main_image, obj_image
    global minkowski_time
    
    print("Clearing all points and polygons")
    pointList.clear()
    currentPointList.clear()
    polygonList.clear()
    num_points = 0
    main_image.fill(255)
    
    obj_points.clear()
    obj_poly = None
    obj_image.fill(255)
    
    #logging info
    
    minkowski_time.clear()
    
    expandedPolygons = []
    
def ensure_counter_clockwise(points: List[Point]) -> List[Point]:
    if len(points) < 3:
        return points
    
    # signed area
    area = 0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        area += points[i].x * points[j].y - points[j].x * points[i].y
    
    if area < 0:
        return points[::-1]
    return points
        
def main_mouse_callback(event, x, y, flags, param):
    global mouse_move_size, mouse_click_size, objects_clicked_size, pointList, num_points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        
        point_clicked = False
        
        # checking points
        for i,point in enumerate(pointList):
            distance = ((x - point.x)**2 + (y - point.y)**2)**0.5
            if distance <= point_radius:
                print(f"(Main) Mouse inside point {i} at ({x}, {y})")
                point_clicked = True
                
        if not point_clicked:
            point = Point(x, y, (0,0,0))
            currentPointList.append(point)
            pointList.append(point)
            print(f"(Main) Created point {len(pointList)} at ({x},{y})")
            print(point.x)
            num_points += 1
            
            
obj_points = []
obj_poly = None

def obj_mouse_callback(event, x, y, flags, param):
    global mouse_move_size, mouse_click_size, objects_clicked_size, obj_points, obj_poly
    
    if event == cv2.EVENT_LBUTTONDOWN and obj_poly == None:
        
        point_clicked = False
        
        # checking points
        for i,point in enumerate(obj_points):
            distance = ((x - point.x)**2 + (y - point.y)**2)**0.5
            if distance <= point_radius:
                print(f"(Obj) Mouse inside point {i} at ({x}, {y})")
                point_clicked = True
                
        if not point_clicked:
            point = Point(x, y, (0,0,0))
            obj_points.append(point)
            print(f"(Obj) Created point {len(obj_points)-1} at ({x},{y})")
            print(point.x)

cv2.namedWindow("Simulation")
cv2.namedWindow("Object")

cv2.setMouseCallback("Simulation", main_mouse_callback)
cv2.setMouseCallback("Object", obj_mouse_callback)



start_time = time.time()
keypress = False
running = True # in case i want to add a pause function later
expandedPolygons = []

# logging info
minkowski_time = []

def compute_distance_between_pairs(polygons: List[Polygon]) -> List[List[float]]:
    n = len(polygons)
    dist_mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = calc_distance_between_polygons(polygons[i], polygons[j])
            dist_mat[i][j] = d
            dist_mat[j][i] = d
    return dist_mat

# Main loop
while True:
    if running:
        # Clear the images
        main_image.fill(255)
        obj_image.fill(255)

        # main simulation window
        for point in currentPointList:
            cv2.circle(main_image, (int(point.x), int(point.y)), point_radius, (255, 0, 0), -1)

        # expanded polygons
        for exp_poly in expandedPolygons:
            pts = np.array([[p.x, p.y] for p in exp_poly.points], dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(main_image, [pts], (0,0,0))

        for polygon in polygonList: 
            pts = np.array([[point.x, point.y] for point in polygon.points], dtype=np.int32)
            pts = pts.reshape((-1, 1, 2)) 
            cv2.fillPoly(main_image, [pts], polygon.color)
            
            
        # object window
        for point in obj_points:
            cv2.circle(obj_image, (int(point.x), int(point.y)), point_radius, (0, 0, 0), -1)

        if obj_poly != None:
            pts = np.array([[point.x, point.y] for point in obj_poly.points], dtype=np.int32)
            pts = pts.reshape((-1, 1, 2)) 
            cv2.fillPoly(obj_image, [pts], (0,0,0))
        

        cv2.imshow("Simulation", main_image)
        cv2.imshow("Object", obj_image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == 8:  # Backspace to clear
            clearSimulation(True)
        elif key == 13:  # Enter to create polygon
            if len(currentPointList) > 2:
                polygonList.append(Polygon(ensure_counter_clockwise(convexHull(currentPointList)), bgr_palette[len(polygonList)])) # algorithm assumes convex polygon
                currentPointList = []
                print("(Main) Created obstacle")
                
            if len(obj_points) > 2 and obj_poly == None:
                obj_poly = Polygon(ensure_counter_clockwise(convexHull(obj_points))) # algorithm assumes convex polygon, also ensuring counter clockwise orientation
                obj_points = []
                print("(Obj) Created object")
        elif key == 32: # space to calculate minkowski addition
            if currentPointList == [] and polygonList != [] and obj_poly != None and expandedPolygons == []:
                for polygon in polygonList:
                    t0 = time.perf_counter()
                    expandedPoly = minkowski_robot(polygon, obj_poly)
                    t1 = time.perf_counter()
                    expandedPolygons.append(expandedPoly)
                    
                    # logging info
                    minkowski_time.append((t1 - t0)*1000)
                    
                    

end_time = time.time()
execution_duration = end_time - start_time
print(f"Simulation ran for {execution_duration:.2f} seconds.")

#storing logged info

if expandedPolygons != []:

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"simulation_data_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)


    simulation_info_df = pd.DataFrame({
        'parameter': ['timestamp_start','timestamp_end','duration_seconds', 'width', 'height', 'num_points', 'robot_points_n', 'pallete'],
        'value': [start_time, end_time, execution_duration, widthImage, heightImage, num_points, len(obj_poly.points), rgb_palette]
    })
    simulation_info_df.to_csv(os.path.join(folder_name, 'simulation_info.csv'), index=False)

    # exporting simulation data
    performance_data = []
    for i, poly in enumerate(polygonList):
        performance_data.append({
            'minkowski_time_ms': minkowski_time[i],
            'polygon_size': len(polygonList[i].points)
        })

    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(os.path.join(folder_name, 'performance_log.csv'), index=False)

    cv2.imwrite(os.path.join(folder_name, f"simulation_result.png"), main_image)
    cv2.imwrite(os.path.join(folder_name, f"robot_result.png"), obj_image)

    distance_matrix = compute_distance_between_pairs(expandedPolygons)
    distance_matrix_df = pd.DataFrame(distance_matrix)
    
    distance_matrix_df.columns = [f"Polygon_{i}" for i in range(len(expandedPolygons))]
    distance_matrix_df.index = [f"Polygon_{i}" for i in range(len(expandedPolygons))]
    
    distance_matrix_df.to_csv(os.path.join(folder_name, 'distance_matrix.csv'), index=False)

    print(f"Exported simulation parameters to {folder_name}/simulation_info.csv")
    print(f"Exported performance log to {folder_name}/performance_log.csv")
    print(f"Exported final resulting graph to {folder_name}/simulation_result.png")
    print(f"Exported final resulting robot to {folder_name}/robot_result.png")
    print(f"Exported distance matrix to {folder_name}/distance_matrix.csv")

    compute_distance_between_pairs(expandedPolygons)

cv2.destroyAllWindows()