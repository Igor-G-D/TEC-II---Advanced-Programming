import numpy as np
import cv2
import time
import pandas as pd
import os
import seaborn as sns
import math
from aStarSearch import calculate_heuristic
from factories import DefaultSimulationFactory
from classes import Simulation

# Simulation Info
heightImage = 800
widthImage = 1200

point_radius = 5
grid_shape = (20, 30) # each square will be 40x40 pixels 

color_palette = sns.color_palette("husl", 20)
bgr_palette = [(int(b * 255), int(g * 255), int(r * 255)) for r, g, b in color_palette]
rgb_palette = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in color_palette]

# Create simulation instance
factory = DefaultSimulationFactory()
simulation = Simulation(
    factory.create_grid_factory(),
    factory.create_object_factory(bgr_palette),
    factory.create_algorithm_factory()
)

# creating image
main_image = np.ones((heightImage, widthImage, 3), dtype=np.uint8) * 255

# aux variables
img_h = main_image.shape[0]
img_w = main_image.shape[1]  
rows, cols = grid_shape

cell_h = img_h / rows
cell_w = img_w / cols 

# logs
astar_execution_time = []
euclidian_distance = []
path_size = []

def clearSimulation(clearData=False):
    global simulation, astar_execution_time, euclidian_distance, path_size
    print("cleared!")
    # Reset simulation
    factory = DefaultSimulationFactory()
    simulation = Simulation(
        factory.create_grid_factory(),
        factory.create_robot_factory(bgr_palette),
        factory.create_algorithm_factory()
    )
    astar_execution_time = []
    euclidian_distance = []
    path_size = []
    
def main_mouse_callback(event, x, y, flags, param):
    global main_image, simulation
    
    cell_clicked = which_cell_clicked(x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"left mouse button clicked on cell {cell_clicked}")
        # Check if position is occupied by robot or goal
        position_occupied = any(robot.position == cell_clicked for robot in simulation.robots) or \
                            any(goal == cell_clicked for goal in simulation.goals)
        
        if not position_occupied:
            if simulation.grid.is_obstacle(cell_clicked):
                simulation.grid.toggle_obstacle(cell_clicked)
                print(f"cell {cell_clicked} is no longer an obstacle")
            else:
                simulation.grid.toggle_obstacle(cell_clicked)
                print(f"cell {cell_clicked} now is an obstacle")
            
            reset_window(main_image, simulation.robots, simulation.goals, simulation.paths)
        else:
            print(f"position is invalid - occupied by robot or goal")
            
    if event == cv2.EVENT_RBUTTONDOWN:
        print(f"right mouse button clicked on cell {cell_clicked}")
        
        if not simulation.grid.is_obstacle(cell_clicked):
            if len(simulation.robots) > len(simulation.goals):
                print(f"set goal for robot {len(simulation.goals)} at {cell_clicked}")
                simulation.add_goal(cell_clicked, simulation.robots[-1]) 
            else:
                print(f"robot {len(simulation.robots)} created at {cell_clicked}")
                simulation.add_robot(cell_clicked)
        reset_window(main_image, simulation.robots, simulation.goals, simulation.paths)
    
def which_cell_clicked(x, y):
    cell_col = math.floor(x / cell_w)   # x determines column
    cell_row = math.floor(y / cell_h)  # y determines row
    
    return (cell_row, cell_col) 

def draw_grid(img, color=(0, 0, 0), thickness=1):
    # draw vertical lines
    for x in np.linspace(start=cell_w, stop=img_w-cell_w, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, img_h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=cell_h, stop=img_h-cell_h, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (img_w, y), color=color, thickness=thickness)

    return img

def paint_cell(img, cell, color=(0,0,0), shape=0):
    if shape == 0:
        tl = (int(cell[1] * cell_w), int(cell[0] * cell_h)) 
        br = (int((cell[1] + 1) * cell_w), int((cell[0] + 1) * cell_h)) 
        
        cv2.rectangle(img, tl, br, color, -1)
    elif shape == 1:
        center = (int(cell[1] * cell_w + cell_w/2), int(cell[0] * cell_h + cell_h/2)) 
        radius = int(cell_h/2)
        
        cv2.circle(main_image, center, radius, color, -1)
        
def draw_x(img, grid_pos, color=(0, 0, 255), thickness=2):
    row, col = grid_pos

    x1 = int(col * cell_w)
    y1 = int(row * cell_h)
    x2 = int((col + 1) * cell_w)
    y2 = int((row + 1) * cell_h)

    cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 4)
    cv2.line(img, (x1, y2), (x2, y1), (0,0,255), 4)

def reset_window(img, robots, destinations, paths):
    img.fill(255)
    draw_grid(img)
    
    # Draw obstacles from grid
    for i in range(simulation.grid.rows):
        for j in range(simulation.grid.cols):
            if simulation.grid.is_obstacle((i, j)):
                paint_cell(img, (i, j))
    
    # Draw robots and goals
    for i, robot in enumerate(robots):
        paint_cell(img, robot.position, bgr_palette[i], 0)
        
    for i, dest in enumerate(destinations):
        paint_cell(img, dest.position, bgr_palette[i], 1)
        
    # Draw paths
    for i, path in enumerate(paths): # black points and lines
        if path == []: # couldn't reach destination
            if i < len(robots):
                draw_x(img, robots[i].position)
            if i < len(destinations):
                draw_x(img, destinations[i])
        
        prev_center = None
        for j, node in enumerate(path):
            row, column = node
            center = (int((column * cell_w) + (cell_w/2)), int((row * cell_h) + (cell_h/2)))
            cv2.circle(img, center, point_radius+2, (0,0,0), -1)
            if j > 0 and prev_center is not None:
                cv2.line(img, prev_center, center, (0,0,0), 6)
            prev_center = center
    
    # Draw colored paths
    for i, path in enumerate(paths):
        prev_center = None
        for j, node in enumerate(path):
            row, column = node
            center = (int((column * cell_w) + (cell_w/2)), int((row * cell_h) + (cell_h/2)))
            cv2.circle(main_image, center, point_radius, bgr_palette[i], -1)
            if j > 0 and prev_center is not None:
                cv2.line(img, prev_center, center, bgr_palette[i], 2)
            prev_center = center

# Initialize window
cv2.namedWindow("Simulation", flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback("Simulation", main_mouse_callback)

start_time = time.time()
running = True
initialized = False
allow_diagonals = False
pressed = False
# Main loop
while True:
    if running:
        if not initialized:
            reset_window(main_image, simulation.robots, simulation.goals, simulation.paths)
            initialized = True

        key = cv2.waitKey(1) & 0xFF
        if not pressed:
            if key == 27 and not pressed:  # ESC to exit
                break
            elif key == 8:  # Backspace to clear
                clearSimulation(True)
                reset_window(main_image, simulation.robots, simulation.goals, simulation.paths)
            elif key == 13:  # Enter to calculate paths
                astar_execution_time = []
                print("Displaying calculated paths")
                simulation.paths = []  # Clear existing paths
                
                if len(simulation.robots) == len(simulation.goals):
                    for i, (robot, goal) in enumerate(zip(simulation.robots, simulation.goals)):
                        print(f"Calculating path for robot {i}")
                        t0 = time.perf_counter()
                        
                        algorithm = simulation.algorithm_factory.create_algorithm("astar")
                        path = algorithm.find_path(simulation.grid, robot.position, goal.position, allow_diagonals)
                        
                        t1 = time.perf_counter()
                        astar_execution_time.append((t1 - t0) * 1000)
                        simulation.paths.append(path)
                        
                        euclidian_distance.append(calculate_heuristic(robot.position, goal.position))
                        path_size.append(len(path))
                        
                    reset_window(main_image, simulation.robots, simulation.goals, simulation.paths)
                else:
                    print("Number of robots and goals must match!")
                    
            elif key == 9 and not pressed: #tab
                pressed = True
                allow_diagonals = not allow_diagonals
                print(f"Diagonal Movement = {allow_diagonals}")
        else:
            if key == 255:
                pressed = False
        cv2.imshow("Simulation", main_image)

end_time = time.time()
execution_duration = end_time - start_time
print(f"Simulation ran for {execution_duration:.2f} seconds.")

# Store logged info
timestamp = time.strftime("%Y%m%d_%H%M%S")
folder_name = f"simulation_data_{timestamp}"
os.makedirs(folder_name, exist_ok=True)

simulation_info_df = pd.DataFrame({
    'parameter': ['timestamp_start', 'timestamp_end', 'duration_seconds', 'width', 'height', 'num_robots', 'grid_shape', 'palette'],
    'value': [start_time, end_time, execution_duration, widthImage, heightImage, len(simulation.robots), grid_shape, rgb_palette]
})
simulation_info_df.to_csv(os.path.join(folder_name, 'simulation_info.csv'), index=False)

# Export performance data
performance_data = []
for i in range(len(astar_execution_time)):
    if i < len(simulation.robots) and i < len(simulation.goals):
        performance_data.append({
            'robot_id': i,
            'start_position': simulation.robots[i].position,
            'goal_position': simulation.goals[i].position,
            'astar_exec_time': astar_execution_time[i],
            'euclidian_distance': euclidian_distance[i] if i < len(euclidian_distance) else 0,
            'path_size': path_size[i] if i < len(path_size) else 0
        })

performance_df = pd.DataFrame(performance_data)
performance_df.to_csv(os.path.join(folder_name, 'performance_log.csv'), index=False)

cv2.imwrite(os.path.join(folder_name, f"simulation_result.png"), main_image)

print(f"Exported simulation parameters to {folder_name}/simulation_info.csv")
print(f"Exported performance log to {folder_name}/performance_log.csv")
print(f"Exported final resulting graph to {folder_name}/simulation_result.png")

cv2.destroyAllWindows()