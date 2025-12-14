Added distance calculation to the grid, as well as a reference to the grid inside of the robot class. Now robots detect when other robots are near

Possible improvements:
- during a step, each robot moves in sequence, so when a "step" is issued, one robot will move, detect proximity, and then the other robot will move and it'll be a collision when they both should only be colliding instead
- Look into the other robot's path to see if there is an interception between the planned paths