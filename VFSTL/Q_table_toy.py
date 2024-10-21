import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Initialize the grid and value function
grid_size = 11
value_function = np.zeros((grid_size, grid_size))
value_function[5, 5] = 1

# Define actions
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # top, bottom, left, right

# BFS to update the value function
def bfs(start, value_function, gamma):
    queue = deque([start])
    visited = set()
    visited.add(start)
    
    while queue:
        current = queue.popleft()
        current_value = value_function[current]
        
        for action in actions:
            next_state = (current[0] + action[0], current[1] + action[1])
            if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size:
                if next_state not in visited:
                    value_function[next_state] = max(value_function[next_state], gamma * current_value)
                    queue.append(next_state)
                    visited.add(next_state)

# Perform BFS from (5, 5)
bfs((5, 5), value_function, gamma=0.9)

# Plot the value function
plt.imshow(value_function, cmap='viridis', origin='lower')
plt.colorbar(label='Value Function')
plt.title('Value Function in Grid World')
plt.show()