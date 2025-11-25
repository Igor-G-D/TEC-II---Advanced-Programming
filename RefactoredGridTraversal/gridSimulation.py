import numpy as np
import cv2
import time
import pandas as pd
import os
import seaborn as sns
import math
from factories import DefaultSimulationFactory
from classes import Simulation

# Simulation Info
heightImage = 800
widthImage = 1200

point_radius = 5
grid_shape = (10, 15) # rows x cols 

color_palette = sns.color_palette("husl", 20)
bgr_palette = [(int(b * 255), int(g * 255), int(r * 255)) for r, g, b in color_palette]
rgb_palette = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in color_palette]

cell_shape = 0 # 0 for rectangle, 1 for hex

# Create simulation instance
factory = DefaultSimulationFactory()
simulation = Simulation(
    factory.create_grid_factory(),
    factory.create_object_factory(bgr_palette),
    factory.create_algorithm_factory(),
    grid_shape,
    cell_shape
)

# creating image
main_image = np.ones((heightImage, widthImage, 3), dtype=np.uint8) * 255

# aux variables
img_h = main_image.shape[0]
img_w = main_image.shape[1]  
rows, cols = grid_shape


r_from_width = widthImage / (cols * math.sqrt(3))
r_from_height = heightImage / (rows * 1.5 + 0.5)

hex_r = min(r_from_width, r_from_height) * 0.8


x_step = math.sqrt(3) * hex_r
y_step = 1.5 * hex_r
x_stagger = x_step / 2
hex_h_full = 2 * hex_r 

cell_h = img_h / rows
cell_w = img_w / cols 

# logs
astar_execution_time = []
euclidian_distance = []
path_size = []

def clearSimulation(clearData=False):
    global simulation, astar_execution_time, euclidian_distance, path_size, main_image, cell_shape
    print("cleared!")
    # Reset simulation
    factory = DefaultSimulationFactory()
    simulation = Simulation(
        factory.create_grid_factory(),
        factory.create_object_factory(bgr_palette),
        factory.create_algorithm_factory(),
        grid_shape,
        cell_shape # rectangle = 0, hex = 1
    )
    astar_execution_time = []
    euclidian_distance = []
    path_size = []
    
    reset_window(main_image, simulation.robots, simulation.goals, simulation.paths)
    
def main_mouse_callback(event, x, y, flags, param):
    global main_image, simulation
    
    
    if event == cv2.EVENT_LBUTTONDOWN:
        cell_clicked = which_cell_clicked(x, y) 
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
        cell_clicked = which_cell_clicked(x, y) 
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
    global cell_shape
    if cell_shape == 0:  # Square grid
        cell_col = math.floor(x / cell_w)
        cell_row = math.floor(y / cell_h)
        return (cell_row, cell_col)
    
    elif cell_shape == 1:  # Hex grid - distance-based approach
        min_distance = float('inf')
        closest_cell = (0, 0)
        
        for row in range(rows):
            for col in range(cols):
                # Get the center of this hex cell - ensure this returns proper screen coordinates
                cx, cy = get_hex_center(row, col)
                
                # Calculate distance to center
                distance = math.sqrt((x - cx)**2 + (y - cy)**2)
                
                # If we're very close to this center, return immediately
                if distance < hex_r * 0.5:  # Within the inner circle of the hex
                    return (row, col)
                
                # Otherwise track the closest one
                if distance < min_distance:
                    min_distance = distance
                    closest_cell = (row, col)
        return closest_cell

def hex_corners(cx, cy, r=hex_r):
    corners = []
    for i in range(6):
        angle_deg = 60 * i - 30 
        angle = math.radians(angle_deg)
        x_i = cx + r * math.cos(angle)
        y_i = cy + r * math.sin(angle)
        corners.append((int(round(x_i)), int(round(y_i))))
    return corners

def get_hex_center(row, col):
    
    x_step = math.sqrt(3) * hex_r
    y_step = 1.5 * hex_r
    x_stagger = x_step / 2
    hex_h_full = 2 * hex_r 
    
    # Calculate total dimensions for centering
    total_grid_w = cols * x_step + x_stagger # Max width 
    total_grid_h = (rows - 1) * y_step + hex_h_full # total height
    
    # Calculate offsets
    offset_x = (widthImage - total_grid_w) / 2
    offset_y = (heightImage - total_grid_h) / 2
    
    cx = offset_x + col * x_step
    
    if row % 2 == 1:
        cx += x_stagger
    cy = offset_y + row * y_step + hex_r
    
    return cx, cy

def draw_grid_hex(img, color=(0, 0, 0), thickness=1):
    
    for row in range(rows):
        for col in range(cols):
            # Calculate center position
            cx, cy = get_hex_center(row, col)
            
            # Draw hexagon outline
            pts = np.array(hex_corners(cx, cy, hex_r), np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

def draw_grid_rec(img, color=(0, 0, 0), thickness=1):
    # draw vertical lines
    for x in np.linspace(start=cell_w, stop=img_w-cell_w, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, img_h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=cell_h, stop=img_h-cell_h, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (img_w, y), color=color, thickness=thickness)

def draw_grid(img, color=(0, 0, 0), thickness=1, cell_shape=0): # 0 for square grid, 1 for hexagon
    if cell_shape == 0:
        draw_grid_rec(img, color, thickness)
    elif cell_shape == 1:
        draw_grid_hex(img, color, thickness)

    return img


def paint_cell(img, cell, color=(0,0,0), shape=0, cell_shape=0):
    if cell_shape == 0: # Rectangle logic (uses cell_w/cell_h)
        tl = (int(cell[1] * cell_w), int(cell[0] * cell_h)) 
        br = (int((cell[1] + 1) * cell_w), int((cell[0] + 1) * cell_h)) 
        
        if shape == 0:
            cv2.rectangle(img, tl, br, color, -1)
        elif shape == 1:
            center = (int(cell[1] * cell_w + cell_w/2), int(cell[0] * cell_h + cell_h/2)) 
            radius = int(cell_h/2)
            cv2.circle(main_image, center, radius, color, -1)

    elif cell_shape == 1: # Hexagon logic
        row, col = cell
        cx, cy = get_hex_center(row, col)
        
        if shape == 0: # Robot (filled hexagon)
            pts = np.array(hex_corners(cx, cy, hex_r), np.int32)
            cv2.fillPoly(img, [pts], color)
        elif shape == 1: # Goal (circle in the center)
            center = (int(cx), int(cy))
            cv2.circle(img, center, int(hex_r * 0.4), color, -1)

        
def draw_x(img, grid_pos, color=(0, 0, 255), thickness=2, cell_shape=0):
    if cell_shape == 0: # Rectangle logic
        row, col = grid_pos
        x1 = int(col * cell_w)
        y1 = int(row * cell_h)
        x2 = int((col + 1) * cell_w)
        y2 = int((row + 1) * cell_h)
        cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 4)
        cv2.line(img, (x1, y2), (x2, y1), (0,0,255), 4)
    elif cell_shape == 1: # Hexagon logic
        row, col = grid_pos
        cx, cy = get_hex_center(row, col)
        
        # Draw a diagonal X inside the cell (using a slightly smaller area)
        size = int(hex_r * 0.6)
        c = (int(cx), int(cy))
        cv2.line(img, (c[0] - size, c[1] - size), (c[0] + size, c[1] + size), color, thickness)
        cv2.line(img, (c[0] - size, c[1] + size), (c[0] + size, c[1] - size), color, thickness)


def reset_window(img, robots, destinations, paths):
    img.fill(255)
    draw_grid(img, cell_shape=cell_shape)
    
    # Draw obstacles from grid
    for i in range(simulation.grid.rows):
        for j in range(simulation.grid.cols):
            if simulation.grid.is_obstacle((i, j)):
                paint_cell(img, (i, j), (0, 0, 0), shape=0, cell_shape=cell_shape)
    
    # Draw robots and goals
    for i, robot in enumerate(robots):
        paint_cell(img, robot.position, bgr_palette[i], 0, cell_shape)
        
    for i, dest in enumerate(destinations):
        paint_cell(img, dest.position, bgr_palette[i], 1, cell_shape)
        
    # Draw paths
    for i, path in enumerate(paths): # black points and lines
        if path == []: # couldn't reach destination
            if i < len(robots):
                draw_x(img, robots[i].position, cell_shape=cell_shape)
            if i < len(destinations):
                draw_x(img, destinations[i].position, cell_shape=cell_shape)
        
        prev_center = None
        for j, node in enumerate(path):
            row, column = node
            
            # Get center coordinates based on cell_shape
            if cell_shape == 0:
                center = (int((column * cell_w) + (cell_w/2)), int((row * cell_h) + (cell_h/2)))
            else: # cell_shape == 1 (Hex)
                cx, cy = get_hex_center(row, column)
                center = (int(cx), int(cy))

            cv2.circle(img, center, point_radius+2, (0,0,0), -1)
            if j > 0 and prev_center is not None:
                cv2.line(img, prev_center, center, (0,0,0), 6)
            prev_center = center
    
    # Draw colored paths
    for i, path in enumerate(paths):
        prev_center = None
        for j, node in enumerate(path):
            row, column = node
            
            if cell_shape == 0:
                center = (int((column * cell_w) + (cell_w/2)), int((row * cell_h) + (cell_h/2)))
            else: # 1 => hex
                cx, cy = get_hex_center(row, column)
                center = (int(cx), int(cy))

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

algorithms = ["astar", "dijkstra"]
algorithm_index = 0
cell_shape_names = ["rectangular", "hexagonal"]
cell_shape_index = 0

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
            elif key == 13:  # Enter to calculate paths
                astar_execution_time = []
                print("Displaying calculated paths")
                simulation.paths = []  # Clear existing paths
                
                if len(simulation.robots) == len(simulation.goals):
                    for i, (robot, goal) in enumerate(zip(simulation.robots, simulation.goals)):
                        print(f"Calculating path for robot {i}")
                        t0 = time.perf_counter()
                        
                        algorithm = simulation.algorithm_factory.create_algorithm(algorithms[algorithm_index % len(algorithms)])
                        path = algorithm.find_path(simulation.grid, robot.position, goal.position, allow_diagonals)
                        
                        t1 = time.perf_counter()
                        astar_execution_time.append((t1 - t0) * 1000)
                        simulation.paths.append(path)
                        
                        euclidian_distance.append(simulation.grid.heuristic(robot.position, goal.position))
                        path_size.append(len(path))
                        
                    reset_window(main_image, simulation.robots, simulation.goals, simulation.paths)
                else:
                    print("Number of robots and goals must match!")
            elif key == 49 and not pressed: #tab
                pressed = True
                cell_shape_index += 1
                cell_shape = cell_shape_index % len(cell_shape_names)
                print(f"board cell shape switched to {cell_shape_names[cell_shape]}")
                clearSimulation(True)
            elif key == 50 and not pressed: #tab
                pressed = True
                allow_diagonals = not allow_diagonals
                print(f"Diagonal Movement = {allow_diagonals}")
            elif key == 51 and not pressed:
                algorithm_index += 1
                print(f"changed pathfinding algorithm to: {algorithms[algorithm_index % len(algorithms)]}")
            
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