import numpy as np
import cv2
import time
import pandas as pd
import os
import seaborn as sns
import math
# Simulation Info
heightImage = 800
widthImage = 1200

point_radius = 5
grid_shape = (20, 30) # each square will be 40x40 pixels 

'''
for the grid:

0 => empty
1 => obstacle

'''

grid = np.zeros(grid_shape)
print(grid)

num_points = 0

color_palette = sns.color_palette("husl", 20)
bgr_palette = [(int(b * 255), int(g * 255), int(r * 255)) for r, g, b in color_palette]
rgb_palette = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in color_palette]

robots = []
goals = []

# creating image
main_image = np.ones((heightImage, widthImage, 3), dtype=np.uint8) * 255

def clearSimulation(clearData = False):
    print("cleared!")
        
def main_mouse_callback(event, x, y, flags, param):
    global main_image, robots, goals
    
    cell_clicked = which_cell_clicked(heightImage, widthImage, grid_shape, x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"left mouse button clicked on cell {cell_clicked}")
        if grid.item(cell_clicked) == 0:
            gx, gy = cell_clicked
            grid[gx][gy] = 1
            print(f"cell {cell_clicked} now is an obstacle")
            paint_cell(main_image, grid_shape, cell_clicked)
    if event == cv2.EVENT_RBUTTONDOWN:
        print(f"right mouse button clicked on cell {cell_clicked}")
        
        if grid.item(cell_clicked) == 0:
            if len(robots) > len(goals):
                print(f"set goal for robot {len(goals)} at {cell_clicked}")
                goals.append(cell_clicked)
                paint_cell(main_image, grid_shape, cell_clicked, bgr_palette[(len(goals)-1)], 1)
            else:
                print(f"robot {len(robots)} created at {cell_clicked}")
                robots.append(cell_clicked)
                paint_cell(main_image, grid_shape, cell_clicked, bgr_palette[(len(robots)-1)], 0)
        
def which_cell_clicked(height, width, grid_shape, x, y):
    rows, cols = grid_shape  
    cell_width = width / cols
    cell_height = height / rows
    
    cell_col = math.floor(x / cell_width)   # x determines column
    cell_row = math.floor(y / cell_height)  # y determines row
    
    return (cell_row, cell_col) 

def draw_grid(img, grid_shape, color=(0, 0, 0), thickness=1):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

def paint_cell(img, grid_shape, cell, color=(0,0,0), shape = 0):
    img_h = img.shape[0]
    img_w = img.shape[1]  
    rows, cols = grid_shape
    
    cell_h = img_h / rows
    cell_w = img_w / cols  
    
    if shape == 0:
        tl = (int(cell[1] * cell_w), int(cell[0] * cell_h)) 
        br = (int((cell[1] + 1) * cell_w), int((cell[0] + 1) * cell_h)) 
        
        cv2.rectangle(img, tl, br, color, -1)
    elif shape == 1:
        center = (int(cell[1] * cell_w + cell_w/2), int(cell[0] * cell_h + cell_h/2)) 
        radius = int(cell_h/2)
        
        cv2.circle(main_image, center, radius, color, -1)

cv2.namedWindow("Simulation", flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMALs)

cv2.setMouseCallback("Simulation", main_mouse_callback)



start_time = time.time()
running = True # in case i want to add a pause function later
initialized = False
# logging info

# Main loop
while True:
    if running:
        # Clear the images
        if(not initialized):
            main_image.fill(255)
            draw_grid(main_image, grid_shape)
            # main simulation window
            initialized = True

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == 8:  # Backspace to clear
            clearSimulation(True)
        elif key == 13:  # Enter to create polygon
            print("Displaying calculated paths")
            
        cv2.imshow("Simulation", main_image)
end_time = time.time()
execution_duration = end_time - start_time
print(f"Simulation ran for {execution_duration:.2f} seconds.")

#storing logged info
'''
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
'''
cv2.destroyAllWindows()