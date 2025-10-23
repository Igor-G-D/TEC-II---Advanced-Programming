import numpy as np
import cv2
import time
import pandas as pd
import os
import copy
from dataStructures import Point, Polygon

# Simulation Info
heightImage = 800
widthImage = 1200

point_radius = 5

num_points = 0

pointList = []
currentPointList= []
polygonList = []


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
    global pointList, currentPointList, polygonList, obj_points
    global num_points, obj_poly
    global main_image, obj_image
    
    print("Clearing all points and polygons")
    pointList.clear()
    currentPointList.clear()
    polygonList.clear()
    num_points = 0
    main_image.fill(255)
    
    obj_points.clear()
    obj_poly = None
    obj_image.fill(255)
    

        
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

# Main loop
while True:
    if running:
        # Clear the images
        main_image.fill(255)
        obj_image.fill(255)

        # main simulation window
        for point in currentPointList:
            cv2.circle(main_image, (int(point.x), int(point.y)), point_radius, (255, 0, 0), -1)

        for polygon in polygonList: 
            pts = np.array([[point.x, point.y] for point in polygon.points], dtype=np.int32)
            pts = pts.reshape((-1, 1, 2)) 
            cv2.fillPoly(main_image, [pts], (255,0,0))
            
            
        # object window
        for point in obj_points:
            cv2.circle(obj_image, (int(point.x), int(point.y)), point_radius, (0, 0, 255), -1)

        if obj_poly != None:
            pts = np.array([[point.x, point.y] for point in obj_poly.points], dtype=np.int32)
            pts = pts.reshape((-1, 1, 2)) 
            cv2.fillPoly(obj_image, [pts], (0,0,255))

        cv2.imshow("Simulation", main_image)
        cv2.imshow("Object", obj_image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == 8:  # Backspace to clear
            clearSimulation(True)
        elif key == 13:  # Enter to create polygon
            if len(currentPointList) > 2:
                polygonList.append(Polygon(copy.deepcopy(currentPointList)))
                currentPointList = []
                print("(Main) Created obstacle")
                
            if len(obj_points) > 2 and obj_poly == None:
                obj_poly = Polygon(copy.deepcopy(obj_points))
                obj_points = []
                print("(Obj) Created object")


'''
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

'''
cv2.destroyAllWindows()