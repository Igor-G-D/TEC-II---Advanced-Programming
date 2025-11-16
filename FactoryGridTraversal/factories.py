from classes import Robot, Goal, AStarAlgorithm, Grid

class AbstractFactory:
    def create_grid_factory(self): pass
    def create_object_factory(self): pass
    def create_algorithm_factory(self): pass

class ObjectFactory:
    def __init__(self, colors):
        self.colors = colors
        self.index = 0

    def create_robot(self, position, goal = None):
        color = self.colors[self.index % len(self.colors)]
        self.index += 1
        return Robot(position, color)
    
    def create_goal(self, position):
        new_goal = Goal(position)
        return new_goal
    
class AlgorithmFactory:
    def create_algorithm(self, type_name):
        if type_name == "astar":
            return AStarAlgorithm()
        raise ValueError(f"Unknown algorithm type: {type_name}")
    
class DefaultSimulationFactory(AbstractFactory):
    def create_grid_factory(self):
        return GridFactory()

    def create_object_factory(self, colors):
        return ObjectFactory(colors)

    def create_algorithm_factory(self):
        return AlgorithmFactory()

class GridFactory:
    def create_grid(self, shape = (20,30)):
        return Grid(shape)

