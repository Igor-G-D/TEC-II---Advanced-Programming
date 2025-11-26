Implemented more functionalities and included a few more design patterns in the project

Right arrow key makes robots move one step ahead in their path, left arrow key makes them go back a step, with tab being an undo button for those actions, added verification for the simulation's state to ensure that commands are only issued when all robots have goals and all paths were properly calculated, as well as printing in the console whenever two robots occupy the same space, and when a robot reaches it's goal

As far as design patterns, the command design pattern was used to implement the movement as the undo functions, observers were used for collision detection and arrival for robots, and a chain of responsibility was used to implement a verification chain for the simulation's state before commands are permitted to be issued.

https://refactoring.guru/design-patterns/command
https://refactoring.guru/design-patterns/observer
https://refactoring.guru/design-patterns/chain-of-responsibility