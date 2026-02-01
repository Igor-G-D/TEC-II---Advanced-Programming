Added collision avoidance through a no-communication method where the robot uses a helper function to check ahead of itself to make sure the space in front of it is not occupied by another robot. If it is, it skips it's turn to move until it is allowed to move into a free space.

The graphs display the overhead needed in every step taken by the robots to use the helper function to check ahead in the path it is about to take, as well as when each robot had to wait until it's path was free.

If it happens that a robot is stuck for more than 2 movement attempts, it then attempts to move to a random adjacent square, and then it recalculates it's path towards the goal from this new position. This has a much higher overhead than the "waiting" approach, but it is needed give that a robot can block a path after reaching it's goal.

Video link: https://drive.google.com/file/d/1Nr0sKr4mlcEpE0pBTYm0VlN--emQ1XsN/view?usp=drive_link