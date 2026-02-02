import numpy as np
import math
import time
import copy
from typing import Tuple, List
from interfaces import Grid
from abc import ABCMeta, ABC, abstractmethod
from enum import Enum


class EventType(Enum):
    MOVEMENT = "movement"
    ARRIVAL = "arrival"
    NEGOTIATION = "negotiation"  
    RESERVATION = "reservation"
    GOAL_REACHED_BLOCK = "goal_reached_block"

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
    
    def is_position_safe(self, current_robot, target_position) -> bool:

        for other_robot in self.robots:
            if other_robot == current_robot:
                continue
            other_pos = other_robot.get_curr_pos()
            
            # Calculate distance
            distance = self.grid.distance(target_position, other_pos)
            
            if distance < 1.0: 
                return False
        return True
    
    def add_robot(self, position: Tuple[int, int], algorithm_type: str, avoidance_method: str) -> None:
        base_robot = self.object_factory.create_robot(position, self.grid)
        
        # pathfinding decorators
        if algorithm_type == "astar":
            decorated_robot = AStarRobot(base_robot)
        elif algorithm_type == "dijkstra":
            decorated_robot = DijkstraRobot(base_robot)
        else:
            decorated_robot = base_robot 
        
        # avoidance decorator
        
        if avoidance_method == "no_communication":
            decorated_robot = NoCommunicationAvoidanceRobot(decorated_robot, self)
        elif avoidance_method == "direct_communication":
            decorated_robot = DirectCommunicationAvoidanceRobot(decorated_robot, self)
        elif avoidance_method == "indirect_communication":
            decorated_robot = IndirectCommunicationAvoidanceRobot(decorated_robot, self)
        
        self.robots.append(decorated_robot)
        self.event_manager.subscribe(EventType.MOVEMENT, decorated_robot)
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
            self.robot_positions[robot] = new_position
            
            # Notify movement and check arrival
            self.event_manager.notify(Event(EventType.MOVEMENT, robot, {"old_position": old_position, "new_position": new_position}))
            
            if robot.goal and new_position == robot.goal.position:
                self.event_manager.notify(Event(EventType.ARRIVAL, robot, {"goal": robot.goal, "old_position": old_position, "new_position": new_position}))
    
    def get_robot_log_data(self):
        log_data = {}
        for i, robot in enumerate(self.robots):
            # Retrieve the overhead list if it exists, otherwise return an empty list
            log_data[i] = {
                "overhead": getattr(robot, 'overhead', [])
            }
        return log_data
        
    
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
    def __init__(self, id: int, position: Tuple[int, int], color: Tuple[int, int, int] = (0, 0, 0)):
        self.id = id
        self.position = position
        self.color = color
        
    @abstractmethod
    def on_event(self, event: Event):
        pass

class Robot(Object):
    def __init__(self, id: int, position: Tuple[int, int], grid, color: Tuple[int, int, int] = (0, 0, 0), goal=None):
            super().__init__(id, position, color)
            self.grid = grid 
            self.goal = goal
            self.algorithm_type = "astar"
            self.path = None
            self.path_step = 0
            self.movement_history = []
        
    def get_pathfinding_algorithm_type(self) -> str:
        return self.algorithm_type
    
    def set_path(self, path):
        self.path = path
        self.path_step = 0
        
    def clear_path(self):
        self.path = None
        self.path_step = 0
        self.movement_history = []
    
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
    
    def is_at_goal(self) -> bool:
        if not self.goal or not self.path:
            return False
        return self.get_curr_pos() == self.goal.position
        
    def on_event(self, event: Event):
        if event.source.id == self.id:
            return # ignore events coming from itself
        if event.event_type == EventType.MOVEMENT:
            other_robot_decorated = event.source
            
            if hasattr(other_robot_decorated, 'undecorated'):
                other_robot = other_robot_decorated.undecorated
            else:
                other_robot = other_robot_decorated

            my_pos = self.get_curr_pos()
            other_pos = event.data["new_position"]
            
            if other_robot != self and event.data["old_position"] != other_pos:
                distance = self.grid.distance(my_pos, other_pos)
                
                if distance == 0:
                    print(f"**COLLISION :** Robot {self.id} and {other_robot.id} are at {my_pos}")
                elif distance < 3: 
                    print(f"**PROXIMITY:** Robot {self.id} near {other_robot.id} (Dist: {distance:.1f})")

class RobotDecorator(Robot):
    def __init__(self, decorated_robot: Robot):
        self._robot = decorated_robot
        
    @property
    def undecorated(self):
        return self._robot    
    
    @property
    def id(self): return self._robot.id
    
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
    
    @property
    def movement_history(self): return self._robot.movement_history
    
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
    def __init__(self, id: int, position: Tuple[int, int]):
        super().__init__(id, position)
        
    def on_event(self, event: Event):
        if event.event_type == EventType.ARRIVAL:
            if event.data["goal"] == self and event.data["old_position"] != event.data["new_position"]: # prevent multiple arrivals if the robot isn't moving
                print(f"ARRIVAL: Robot {self.id} arrived at goal at position {self.position}!")
        
class RobotDecoratorAvoidance(RobotDecorator):
    def __init__(self, decorated_robot: Robot):
        super().__init__(decorated_robot)
        self._overhead_history = [] 
        self._blocked_squares = []
        
    @property
    def overhead(self): 
        return self._overhead_history 
    @property
    def blocked_squares(self): 
        return self._blocked_squares
    
    def _attempt_escape_and_replan(self, blocked_square) -> bool:
        current_pos = self.get_curr_pos()
        grid = self.simulation.grid
        
        self.blocked_squares.append(blocked_square)
        
        # Get all squares currently occupied by other robots

        neighbors = grid.get_neighbors(current_pos)
        np.random.shuffle(neighbors)
        
        for neighbor in neighbors:
            # safety check
            if not grid.is_obstacle(neighbor) and self.simulation.is_position_safe(self, neighbor):
                
                avoidance_grid = DynamicObstacleGrid(grid, self.blocked_squares)
                
                algorithm = self.simulation.algorithm_factory.create_algorithm(
                    self.get_pathfinding_algorithm_type()
                )
                
                new_path = algorithm.find_path(avoidance_grid, neighbor, self.goal.position)
                
                if new_path:
                    self.set_path(new_path)
                    self.simulation.event_manager.notify(Event(
                        EventType.RESERVATION, self, {"position": neighbor}
                    ))
                    return True
        return False
        
class NoCommunicationAvoidanceRobot(RobotDecoratorAvoidance):
    def __init__(self, decorated_robot: Robot, simulation: Simulation):
        super().__init__(decorated_robot)
        self.simulation = simulation
        self.consecutive_waits = 0 

    def step(self):
        old_position = self.get_curr_pos()
        next_pos = self.get_curr_pos(offset=1)
        
        if next_pos == old_position:
            self._robot.step()
            self.consecutive_waits = 0
            return

        start_time = time.perf_counter()
        
        is_safe = self.simulation.is_position_safe(self, next_pos)
        
        if is_safe:
            self.overhead.append(time.perf_counter() - start_time)
            
            self._robot.step()
            self.consecutive_waits = 0 
            if old_position != self.goal.position:
                self.movement_history.append(True)
        else:
            self.consecutive_waits += 1
            print(f"NO COMMUNICATION: Robot {self.id} blocked at {next_pos}. Wait count: {self.consecutive_waits}")
            
            if self.consecutive_waits >= 3:
                print(f"DEADLOCK (NO COMMUNICATION): Robot {self.id} stuck. Attempting escape...")
                
                escaped = self._attempt_escape_and_replan(next_pos)
                
                # end timing after escape attempt (successful or failed)
                self.overhead.append(time.perf_counter() - start_time)
                
                if escaped:
                    self.consecutive_waits = 0 
                    return 
            else:
                self.overhead.append(time.perf_counter() - start_time)
            
            print(f"NO COMMUNICATION: Robot {self.id} waiting at {old_position}")
            self.movement_history.append(False)
    
class DirectCommunicationAvoidanceRobot(RobotDecoratorAvoidance):
    def __init__(self, decorated_robot: Robot, simulation: Simulation):
        super().__init__(decorated_robot)
        self.simulation = simulation
        self.waiting_for_peer = False
        self.simulation.event_manager.subscribe(EventType.NEGOTIATION, self)
        self.simulation.event_manager.subscribe(EventType.GOAL_REACHED_BLOCK, self)

    def on_event(self, event: Event):
        if event.source.id == self.id:
            return
        
        self._robot.on_event(event)
        
        if event.event_type == EventType.NEGOTIATION:
            target_pos = event.data.get("target_pos")
            curr_pos = self.get_curr_pos()
            next_pos = self.get_curr_pos(offset=1)
            
            # another robot wants to occupy the space I want to go in
            if target_pos == curr_pos or target_pos == next_pos:
                if self.is_at_goal():
                    print(f"DIRECT: Robot {self.id} at goal. Blocking robot {event.source.id}.")
                    self.simulation.event_manager.notify(Event(
                        EventType.GOAL_REACHED_BLOCK, 
                        self, 
                        {"blocked_robot_id": event.source.id}
                    ))
                # if I don't have prority, wait
                elif self.id < event.source.id:
                    self.waiting_for_peer = True
                else:
                    # if I have priority, don't wait
                    self.waiting_for_peer = False

        if event.event_type == EventType.GOAL_REACHED_BLOCK:
            if event.data.get("blocked_robot_id") == self.id:
                print(f"DIRECT: Robot {self.id} blocked by finished robot, replanning route")
                self._attempt_escape_and_replan(next_pos)

    def step(self):
        self.waiting_for_peer = False
        
        next_pos = self.get_curr_pos(offset=1)
        if not next_pos or next_pos == self.get_curr_pos():
            self._robot.step()
            return

        # notify the intention to move before verifying safety of next square
        self.simulation.event_manager.notify(Event(
            EventType.NEGOTIATION, 
            self, 
            {"target_pos": next_pos}
        ))

        # decide movement based on negotiation and safety of next square
        if not self.waiting_for_peer:
            if self.simulation.is_position_safe(self, next_pos):
                self._robot.step()
            else:
                print(f"DIRECT: Robot {self.id} has priority but {next_pos} is occupied, waiting for the way to be clear")
                self.movement_history.append(False)
        else:
            print(f"DIRECT: Robot {self.id} waiting for other robot to clear the way")
            self.movement_history.append(False)
            
class IndirectCommunicationAvoidanceRobot(RobotDecoratorAvoidance):
    def __init__(self, decorated_robot: Robot, simulation: Simulation):
        super().__init__(decorated_robot)
        self.simulation = simulation
        self.reservation_map = {} 
        self.consecutive_waits = 0
        self.simulation.event_manager.subscribe(EventType.RESERVATION, self)

    def on_event(self, event: Event):
        if event.event_type == EventType.RESERVATION:
            reserver_id = event.source.id
            if reserver_id != self.id:
                reserved_pos = event.data.get("position")
                self.reservation_map[reserved_pos] = reserver_id

    def step(self):
        next_pos = self.get_curr_pos(offset=1)
        
        if not next_pos or next_pos == self.get_curr_pos():
            self._robot.step()
            self.consecutive_waits = 0 # reset if arrived at goal
            return

        # check map
        is_reserved = next_pos in self.reservation_map and self.reservation_map[next_pos] != self.id
        
        if not is_reserved and self.simulation.is_position_safe(self, next_pos):
            # leave reserved marker for others
            self.simulation.event_manager.notify(Event(
                EventType.RESERVATION, 
                self, 
                {"position": next_pos}
            ))
            self._robot.step()
            self.consecutive_waits = 0 
        else:
            self.consecutive_waits += 1
            print(f"INDIRECT: Robot {self.id} blocked at {next_pos}. Wait count: {self.consecutive_waits}")
            
            # check for deadlock, and if so, replan route
            if self.consecutive_waits >= 3:
                print(f"DEADLOCK (INDIRECT): Robot {self.id} stuck for 3 turns. Replanning route")
                if self._attempt_escape_and_replan(next_pos):
                    # if there is a replan, reserve the next spot immediately
                    new_escape_pos = self.get_curr_pos(offset=1)
                    if new_escape_pos:
                        self.simulation.event_manager.notify(Event(
                            EventType.RESERVATION, 
                            self, 
                            {"position": new_escape_pos}
                        ))
                    
                    self.consecutive_waits = 0
            
            self.movement_history.append(False)
            
            # ckear up old reservations for next step
            self.reservation_map.clear()
            
class SingletonGridMeta(ABCMeta): 
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

class DynamicObstacleGrid: # wrapper that lets obstacles be added dynamically
    def __init__(self, original_grid, blocked_positions):
        self._grid = original_grid
        self._blocked = set(blocked_positions)

    def is_obstacle(self, position: Tuple[int, int]) -> bool:
        return self._grid.is_obstacle(position) or position in self._blocked

    def __getattr__(self, name): # everything else is from the base class
        return getattr(self._grid, name)

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