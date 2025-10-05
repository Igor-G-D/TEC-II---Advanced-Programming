import numpy as np
import cv2
import time
import pandas as pd
import os
import delaunay_bowyer_watson as delaunay
from dataStructures import Point, Line
from voronoi import Voronoi

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
delaunay_bad_triangle_count = []


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
    global mouse_move_size, mouse_click_size, objects_clicked_size, pointList, num_points, voronoiEdges, delaunayEdges, voronoiTime, delaunayTime, delaunay_bad_triangle_count

    
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
                triangles, delaunay_bad_triangle_count = delaunay.bowyer_watson(pointList)
                delaunay_end = time.perf_counter()
                
                delaunayTime.append((delaunay_end - delaunay_start) * 1000)
                for tri in triangles:
                    vertices = tri.vertices
                    l1 = Line(vertices[0], vertices[1])
                    l2 = Line(vertices[0], vertices[2])
                    l3 = Line(vertices[1], vertices[2]) 
                    
                    delaunayEdges.extend([l1,l2,l3])
                    
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

# Export timing and bad triangle data per point
performance_data = []
for i in range(num_points):
    performance_data.append({
        'point_index': i + 1,
        'point_x': pointList[i].x,
        'point_y': pointList[i].y,
        'voronoi_time_ms': voronoiTime[i-1] if i > 0 and i-1 < len(voronoiTime) else None,
        'delaunay_time_ms': delaunayTime[i-2] if i > 1 and i-2 < len(delaunayTime) else None,
        'bad_triangle_count': delaunay_bad_triangle_count[i-2] if i > 1 and i-2 < len(delaunay_bad_triangle_count) else None
    })

performance_df = pd.DataFrame(performance_data)
performance_df.to_csv(os.path.join(folder_name, 'performance_log.csv'), index=False)

print(f"Exported performance log to {folder_name}/performance_log.csv")


cv2.destroyAllWindows()