import numpy as np
import math
from typing import Tuple, List
from interfaces import Grid
from abc import ABCMeta, ABC, abstractmethod
from enum import Enum

class EventType(Enum):
    MOVEMENT = "movement"
    ARRIVAL = "arrival"

class Event:
    def __init__(self, event_type: EventType, source, data=None):
        self.event_type = event_type
        self.source = source
        self.data = data or {}

class EventManager:
    def __init__(self):
        self.listeners = {}
        
    def subscribe(self, event_type: EventType, listener):
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(listener)
        
    def unsubscribe(self, event_type: EventType, listener):
        if event_type in self.listeners:
            if listener in self.listeners[event_type]:
                self.listeners[event_type].remove(listener)
                
    def notify(self, event: Event):
        event_type = event.event_type
        if event_type in self.listeners:
            for listener in self.listeners[event_type]:
                listener.on_event(event)

class ValidationHandler(ABC):
    def __init__(self):
        self._next_handler = None

    def set_next(self, handler):
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle(self, simulation) -> Tuple[bool, str]:
        if self._next_handler:
            return self._next_handler.handle(simulation)
        return True, ""  
    

class RobotExistsHandler(ValidationHandler):
    def handle(self, simulation) -> Tuple[bool, str]:
        if len(simulation.robots) == 0:
            return False, "No robots have been created yet!"
        return super().handle(simulation)

class RobotGoalHandler(ValidationHandler):
    def handle(self, simulation) -> Tuple[bool, str]:
        for i, robot in enumerate(simulation.robots):
            if robot.goal is None:
                return False, f"Robot {i} doesn't have a goal!"
        return super().handle(simulation)

class RobotPathHandler(ValidationHandler):
    def handle(self, simulation) -> Tuple[bool, str]:
        for i, robot in enumerate(simulation.robots):
            if robot.path is None:
                return False, f"Robot {i} doesn't have a path!"
        return super().handle(simulation)

class Simulation:
    def __init__(self, grid_factory, object_factory, algorithm_factory, grid_shape=0, cell_shape=0, allow_diagonals: bool = False):
        self.grid = grid_factory.create_grid(grid_shape, cell_shape, allow_diagonals)
        self.object_factory = object_factory
        self.algorithm_factory = algorithm_factory
        self.robots = []
        self.goals = []
        self.command_history = CommandHistory()
        self.event_manager = EventManager()
        self.validation_chain = self._build_validation_chain()
        
        # Track robot positions for collision detection
        self.robot_positions = {}


    def _build_validation_chain(self):
            chain = RobotExistsHandler()
            chain.set_next(RobotGoalHandler()).set_next(RobotPathHandler())
            return chain
        
    def scenario_ready(self):  # checks to see if everything is calculated and that the pathfinding algorithms were run
        return self.validation_chain.handle(self)[0]
    
    def get_validation_details(self):
        return self.validation_chain.handle(self)
    
    def add_robot(self, position: Tuple[int, int], algorithm_type: str) -> None:
        base_robot = self.object_factory.create_robot(position, self.grid)
        
        if algorithm_type == "astar":
            decorated_robot = AStarRobot(base_robot)
        elif algorithm_type == "dijkstra":
            decorated_robot = DijkstraRobot(base_robot)
        else:
            decorated_robot = base_robot 
        
        self.robots.append(decorated_robot)
        # subscribe robot to movement events
        self.event_manager.subscribe(EventType.MOVEMENT, decorated_robot)
        #update position tracking
        self.robot_positions[decorated_robot] = position

    def add_goal(self, position: Tuple[int, int], robot) -> None:
        goal = self.object_factory.create_goal(position)
        self.goals.append(goal)
        robot.goal = goal
        #subscribe goal to arrival events
        self.event_manager.subscribe(EventType.ARRIVAL, goal)

    def get_paths(self) -> List[List[Tuple[int, int]]]:
        paths = []
        for robot in self.robots:
            paths.append(robot.path)
        return paths
    
    def clear_paths(self) -> None:
        for robot in self.robots:
            robot.clear_path()
            
    def step_robots(self):
        for robot in self.robots:
            old_position = self.robot_positions[robot]
            robot.step()
            new_position = robot.get_curr_pos()
            
            #Update position tracking
            self.robot_positions[robot] = new_position
            
            # Notify movement event
            self.event_manager.notify(Event(
                EventType.MOVEMENT,
                robot,
                {"old_position": old_position, "new_position": new_position}
            ))
            
            # Check for arrival at goal
            if robot.goal and new_position == robot.goal.position:
                self.event_manager.notify(Event(
                    EventType.ARRIVAL,
                    robot,
                    {"goal": robot.goal,"old_position": old_position, "new_position": new_position}
                ))
            
    def step_back_robots(self):
        for robot in self.robots:
            old_position = self.robot_positions[robot]
            robot.step_back()
            new_position = robot.get_curr_pos()
            
            # Update position tracking
            self.robot_positions[robot] = new_position
            
            # Notify movement event (for undo)
            self.event_manager.notify(Event(
                EventType.MOVEMENT,
                robot,
                {"old_position": old_position, "new_position": new_position}
            ))
    
    def run_command(self, command):
        if self.scenario_ready(): 
            command.execute()
            self.command_history.register_command(command)
        else:
            success, message = self.get_validation_details()
            print(f"Cannot execute command: {message}")
        
    def undo(self):
        try:
            last_command = self.command_history.history.pop() 
            last_command.reverse()
        except IndexError as e:
            print("Command history is empty!")

    def run(self) -> None:
        self.paths = [] 
        
        for robot in self.robots:
            if robot.goal is None:
                continue

            algorithm_type = robot.get_pathfinding_algorithm_type()
            
            algorithm = self.algorithm_factory.create_algorithm(algorithm_type)
            
            path = algorithm.find_path(self.grid, robot.position, robot.goal.position)
            robot.set_path(path)
            
            # Initialize robot position tracking
            self.robot_positions[robot] = robot.position
            
        self.command_history.clear_history()

class Object(ABC):
    def __init__(self, position: Tuple[int, int], color: Tuple[int, int, int] = (0, 0, 0)):
        self.position = position
        self.color = color
        
    @abstractmethod
    def on_event(self, event: Event):
        pass

class Robot(Object):
    def __init__(self, position: Tuple[int, int], grid, color: Tuple[int, int, int] = (0, 0, 0), goal=None):
            super().__init__(position, color)
            self.grid = grid 
            self.goal = goal
            self.algorithm_type = "astar"
            self.path = None
            self.path_step = 0
        
    def get_pathfinding_algorithm_type(self) -> str:
        return self.algorithm_type
    
    def set_path(self, path):
        self.path = path
        
    def clear_path(self):
        self.path = None
        self.path_step = 0
    
    def step(self):
        self.path_step += 1
            
    def step_back(self):
        self.path_step -= 1
    
    def get_curr_pos(self, offset = 0) -> Tuple[int, int]:
        
        if self.path == None:
            return self.position
        
        idx = self.path_step + offset
        if idx < 0:
            return self.path[0]
        elif idx < len(self.path):
            return self.path[idx]
        return self.path[-1]
        
    def on_event(self, event: Event):
        if event.event_type == EventType.MOVEMENT:
            other_robot_decorated = event.source
            
            if hasattr(other_robot_decorated, 'undecorated'):
                other_robot = other_robot_decorated.undecorated
            else:
                other_robot = other_robot_decorated

            this_robot_moved = self.get_curr_pos(-1) != self.get_curr_pos()
            other_robot_moved = event.data["old_position"] != event.data["new_position"]

            if other_robot != self and (this_robot_moved or other_robot_moved):
                
                my_position = self.get_curr_pos()
                other_position = event.data["new_position"]

                distance = self.grid.distance(my_position, other_position)
                
                if distance == 0:
                    print(f"**COLLISION (Distance 0):** Robot {self} at {my_position} collided with robot {other_robot}!")
                elif distance <= 3.0:
                    print(f"**PROXIMITY WARNING (Distance <= 3):** Robot {self} at {my_position} is close to robot {other_robot} (Distance: {distance:.1f})!")

class RobotDecorator(Robot):
    def __init__(self, decorated_robot: Robot):
        self._robot = decorated_robot
        
    @property
    def undecorated(self):
        return self._robot    
    
    @property
    def position(self): return self._robot.position

    @property
    def color(self): return self._robot.color

    @property
    def goal(self): return self._robot.goal
    
    @property
    def path(self): return self._robot.path
    
    @property
    def path_step(self): return self._robot.path_step
    
    @goal.setter
    def goal(self, value): self._robot.goal = value

    def get_pathfinding_algorithm_type(self) -> str:
        return self._robot.get_pathfinding_algorithm_type()
    
    def set_path(self, path):
        return self._robot.set_path(path)
        
    def clear_path(self):
        return self._robot.clear_path()
    
    def step(self):
        return self._robot.step()
            
    def step_back(self):
        return self._robot.step_back()
    
    def get_curr_pos(self, offset=0):
        return self._robot.get_curr_pos(offset)
        
    def on_event(self, event: Event):
        return self._robot.on_event(event)

class AStarRobot(RobotDecorator):
    def get_pathfinding_algorithm_type(self) -> str:
        return "astar"

class DijkstraRobot(RobotDecorator):
    def get_pathfinding_algorithm_type(self) -> str:
        return "dijkstra"

class Goal(Object):
    def __init__(self, position: Tuple[int, int]):
        super().__init__(position)
        
    def on_event(self, event: Event):
        if event.event_type == EventType.ARRIVAL:
            if event.data["goal"] == self and event.data["old_position"] != event.data["new_position"]: # prevent multiple arrivals if the robot isn't moving
                print(f"ARRIVAL: Robot {self} arrived at goal at position {self.position}!")

class SingletonGridMeta(ABCMeta):  # Inherit from ABCMeta
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class RectangleGrid(Grid, metaclass=SingletonGridMeta):
    def __init__(self, shape: Tuple[int, int], allow_diagonals: bool = False):
        self.rows, self.cols = shape
        self.matrix = np.zeros(shape)
        self.allow_diagonals = allow_diagonals # stored state so the class is conforming to the interfaces
    
    def toggle_obstacle(self, position: Tuple[int, int]) -> None:
        x, y = position
        self.matrix[x][y] = 1 if self.matrix[x][y] == 0 else 0
    
    def is_obstacle(self, position: Tuple[int, int]) -> bool:
        x, y = position
        return self.matrix[x][y] == 1
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]: # removed allow_diagonals from parameters to conform to interface
        x, y = position
        neighbors = []
        # Cardinal moves
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                neighbors.append((nx, ny))
        if self.allow_diagonals:
            for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.rows and 0 <= ny < self.cols:
                    neighbors.append((nx, ny))
        return neighbors
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        (x1, y1), (x2, y2) = a, b
        return math.hypot(x2 - x1, y2 - y1)
    
    def distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:

        (x1, y1), (x2, y2) = a, b
        dx, dy = abs(x1 - x2), abs(y1 - y2)

        if self.allow_diagonals:
            return max(dx, dy)
        else:
            return dx + dy

class HexGrid(Grid, metaclass=SingletonGridMeta):
    def __init__(self, shape: Tuple[int, int]):
        self.rows, self.cols = shape
        self.matrix = np.zeros(shape)
    
    def toggle_obstacle(self, position: Tuple[int, int]) -> None:
        r, q = position
        self.matrix[r][q] = 1 if self.matrix[r][q] == 0 else 0
    
    def is_obstacle(self, position: Tuple[int, int]) -> bool:
        r, q = position
        return self.matrix[r][q] == 1
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        row, col = position  # position is (row, col)
        
        if row % 2 == 1:  # odd row - shifted right
            neighbor_deltas = [
                (0, -1),               
                (-1, 0),     
                (-1, +1),  
                (0, +1),               
                (+1, +1),  
                (+1, 0)    
            ]
        else:  # even row
            neighbor_deltas = [
                (0, -1), 
                (-1, -1),
                (-1, 0),  
                (0, +1),  
                (+1, 0),  
                (+1, -1) 
            ]

        result = []
        for dr, dc in neighbor_deltas:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                result.append((new_row, new_col))
        return result

    def offset_to_cube(self, position: Tuple[int, int]) -> Tuple[int, int, int]:
        q, r = position
        x = q - (r - (r & 1)) // 2
        z = r
        y = -x - z
        return (x, y, z)

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        x1, y1, z1 = self.offset_to_cube(a)
        x2, y2, z2 = self.offset_to_cube(b)
        return max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2))
    
    def distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:

        x1, y1, z1 = self.offset_to_cube(a)
        x2, y2, z2 = self.offset_to_cube(b)
        
        return (abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) / 2.0

class Command(ABC):
    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def reverse(self):
        pass
    
class StepCommand(Command):
    def execute(self):
        self.simulation.step_robots()
    def reverse(self):
        self.simulation.step_back_robots()

class StepBackCommand(Command):
    def execute(self):
        self.simulation.step_back_robots()
    def reverse(self):
        self.simulation.step_robots()
        
class CommandHistory:
    def __init__(self):
        self.history = []
    def register_command(self, command: Command):
        self.history.append(command)
    def clear_history(self):
        self.history = []