import numpy as np

def generate_task(num_reach, num_avoid, num_zones):
    '''
        generate random target and avoid zones
        num_reach: number of target zones, >=1
        num_avoid: number of avoid zones, can be 0

        return target_zones, avoid_zones are list contains index of zones
        (i.e. target_zones = [0, 1], avoid_zones = [2, 3] reach black white avoid red yellow)
    '''
    # Ensure num_reach + num_avoid does not exceed num_zones
    assert num_reach + num_avoid <= num_zones, "Total number of reach and avoid zones cannot exceed the number of zones."

    # Generate random target zones
    zone_indices = np.arange(num_zones)
    target_zones = np.random.choice(zone_indices, size=num_reach, replace=False)

    # Remove target zones from the list of available zones
    remaining_zones = np.setdiff1d(zone_indices, target_zones)

    # Generate random avoid zones from the remaining zones
    avoid_zones = np.random.choice(remaining_zones, size=num_avoid, replace=False)

    return target_zones.tolist(), avoid_zones.tolist()

# heler functions
def divide_time_slots(nt, n):
    """
    Divide the time slot into n slices.
    
    Args:
    nt (int): Total number of time steps.
    n (int): Number of slices.
    
    Returns:
    List[Tuple[int, int]]: List of tuples representing the start and end indices for each slice.
    """
    slices = []
    step = nt // n
    for i in range(n):
        start = i * step
        end = (i + 1) * step if i < n - 1 else nt
        slices.append((start, end))
    return slices


def is_success(env, states, reach_zones, avoid_zones):
    '''
        this function is used to check if the task is completed
        states (np.ndarray): The states of the game, shape (time_step, dim) where dim is 2 representing (x, y).
        reach_zones (list): List of target zones to reach.
        avoid_zones (list): List of zones to avoid.
        
        returns
        sucess: True if visit reach zones sequentially
        collide: int, number of time hit the avoid zones
    '''
    reach_zones = reach_zones.copy()  # Create a copy to avoid modifying the original list
    avoid_collisions = set()  # To track avoid zone collisions
    collision_count = 0  # To count the number of collisions with avoid zones
    complete_tasks = []

    zone_poses = env.env.zones_pos
    zone_sizes = env.env.zones_size

    # Convert zone positions to numpy arrays for vectorized operations
    reach_zone_positions = np.array([zone_poses[zone][:2] for zone in reach_zones])
    

    # Calculate squared distances to reach zones
    reach_distances_squared = np.sum((states[:, np.newaxis, :] - reach_zone_positions[np.newaxis, :, :])**2, axis=2)
    reach_mask = reach_distances_squared <= zone_sizes

    # Check if each reach zone is reached sequentially
    for i, zone in enumerate(reach_zones):
        if np.any(reach_mask[:, i]):
            complete_tasks.append(reach_zones.pop(0))  # Remove the first zone if it is reached
        else:
            break  # Stop if the current zone is not reached

    if len(avoid_zones) > 0:
        # Convert zone positions to numpy arrays for vectorized operations
        avoid_zone_positions = np.array([zone_poses[zone][:2] for zone in avoid_zones])

        # Calculate squared distances to avoid zones
        avoid_distances_squared = np.sum((states[:, np.newaxis, :] - avoid_zone_positions[np.newaxis, :, :])**2, axis=2)
        avoid_mask = avoid_distances_squared <= zone_sizes

        # Check for collisions with avoid zones
        for i, zone in enumerate(avoid_zones):
            if np.any(avoid_mask[:, i]):
                avoid_collisions.add(zone)
                collision_count += np.sum(avoid_mask[:, i])

    # Check if all target zones are reached sequentially and no avoid zone collisions
    success = len(reach_zones) == 0
    return success, collision_count, complete_tasks