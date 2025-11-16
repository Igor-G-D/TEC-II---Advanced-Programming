from aStarSearch import a_star_search
import numpy as np

class Simulation:
    def __init__(self, grid_factory, object_factory, algorithm_factory):
        self.grid = grid_factory.create_grid()
        self.object_factory = object_factory
        self.algorithm_factory = algorithm_factory
        self.robots = []
        self.goals = []
        self.paths = []

    def add_robot(self, position):
        robot = self.object_factory.create_robot(position)
        self.robots.append(robot)

    def add_goal(self, position, robot):
        goal = self.object_factory.create_goal(position)
        self.goals.append(goal)
        robot.goal = goal

    def run(self, allow_diagonals):
        for robot, goal in zip(self.robots, self.goals):
            algorithm = self.algorithm_factory.create_algorithm("astar")
            path = algorithm.find_path(self.grid, robot.position, goal, allow_diagonals)
            self.paths.append(path)

class Grid:
    def __init__(self, shape):
        self.rows, self.cols = shape
        self.matrix = np.zeros(shape)

    def toggle_obstacle(self, position):
        x, y = position
        self.matrix[x][y] = 1 if self.matrix[x][y] == 0 else 0

    def is_obstacle(self, position):
        x, y = position
        return self.matrix[x][y] == 1

class Object:
    def __init__(self, position, color = (0,0,0)):
        self.position = position
        self.color = color

class Robot(Object):
    def __init__(self, position, color, goal = None):
        self.position = position
        self.color = color
        self.goal = goal

class Goal(Object):
    def __init__ (self, position):
        self.position = position

class PathfindingAlgorithm:
    def find_path(self, grid, start, goal, allow_diagonals):
        raise NotImplementedError

class AStarAlgorithm(PathfindingAlgorithm):
    def find_path(self, grid, start, goal, allow_diagonals):
        return a_star_search(grid.matrix, start, goal, allow_diagonals)
