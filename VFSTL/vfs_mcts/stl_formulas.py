from lib_stl_core import *
from copy import deepcopy

'''
    J = states[:,:, 0]
    W = states[:,:, 1]
    R = states[:,:, 2]
    Y = states[:,:, 3]
'''


# we considering V >= 0.8 as complete the task
threshold = 0.9
avoid_treshold = 0.2

index_to_vfs = {
    0: "J",
    1: "W",
    2: "R",
    3: "Y"
}

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


'''
    below are STL measure for vfs states
'''
def reach_avoid_vfs(nt, reach_vfs, avoid_vfs, reach_threshold, avoid_threshold):
    '''
        nt: number of time steps
        reach_vfs: a list of vfs index that we want to reach, cannot be empty
        avoid_vfs: a list of vfs index that we want to avoid can be empty
    '''
    reach_stls = []
    num_to_reach = len(reach_vfs)
    time_slices = divide_time_slots(nt, num_to_reach)

    # for i, (start, end) in enumerate(time_slices):
    #     print("start: ", start)
    #     print("end: ", end)
    #     print("nt: ", nt)
    #     print("time_slices: ", time_slices)
    #     print("i: ", i)
    #     reach_stls.append(Eventually(start, end, AP(lambda x: x[..., reach_vfs[i]] - reach_threshold, comment=f"REACH {index_to_vfs[reach_vfs[i]]}")))


    for i in range(len(time_slices)):
        integer = deepcopy(i)
        start, end = time_slices[i]

        # print("start: ", start)
        # print("end: ", end)
        # print("nt: ", nt)
        # print("time_slices: ", time_slices)
        # print("i: ", i)
        # print("integer: ", integer)

        reach_stls.append(Always(start, end, AP(lambda x: x[..., reach_vfs[integer]] - reach_threshold, comment=f"REACH {index_to_vfs[reach_vfs[integer]]}")))

    avoid_stls = []
    for i in avoid_vfs:
        avoid_stls.append(Always(0, nt, AP(lambda x: avoid_threshold - x[..., i], comment=f"AVOID {index_to_vfs[i]}")))

    stl = ListAnd(reach_stls + avoid_stls)
    stl.update_format("word")
    return stl

def reach_avoid_states(env, nt, reach_zone=None, avoid_zone=None):
    '''
        env: gym environment
        states: trajectory of robot in shape (batch_size, time_steps, state_dim)            state_dim: (x, y, z)
        reach_zone: index of zone that we want to reach, cannnot be empty 
        avoid_zone: index of zone that we want to avoid, can be empty


        env.env.robot_pos........   # need go 2 steps
        robot_pos: ndarray ([x, y, z])
        zones_pos: a list [zone1, zone2, zone3, zone4]
        zones_pos[i]: ndarray ([x, y, z])
        zones_size: scalar of the size
    '''

    zone_r = env.env.zones_size
    zone_poses = env.env.zones_pos


    # for i in range(len(time_slices)):
    #     integer = deepcopy(i)
    #     start, end = time_slices[i]

    #     print("start: ", start)
    #     print("end: ", end)
    #     print("nt: ", nt)
    #     print("time_slices: ", time_slices)
    #     print("i: ", i)
    #     print("integer: ", integer)

    #     reach_stls.append(Eventually(start, end, AP(lambda x: x[..., reach_vfs[integer]] - reach_threshold, comment=f"REACH {index_to_vfs[reach_vfs[integer]]}")))

    reach_stls = []
    time_slices = divide_time_slots(nt, len(reach_zone))
    for i in range(len(time_slices)):
        integer = deepcopy(i)
        start, end = time_slices[i]

        # print("start: ", start)
        # print("end: ", end)
        # print("nt: ", nt)
        # print("time_slices: ", time_slices)
        # print("i: ", i)
        # print("integer: ", integer)

        reach_stls.append(Always(start, end, AP(lambda x: zone_r**2 - ((x[..., 0] - zone_poses[reach_zone[integer]][0])**2 + (x[...,1]- zone_poses[reach_zone[integer]][1])**2), comment=f"REACH {reach_zone[i]}")))

    
    # reach_stls = []
    # for i, (start, end) in enumerate(divide_time_slots(nt, len(reach_zone))):
    #     reach_stls.append(Eventually(start, end, AP(lambda x: zone_r**2 - ((x[..., 0] - zone_poses[reach_zone[i]][0])**2 + (x[...,1]- zone_poses[reach_zone[i]][1])**2), comment=f"REACH {reach_zone[i]}")))

    avoid_stls = []
    for i in avoid_zone:
        avoid_stls.append(Always(0, nt, AP(lambda x: ((x[..., 0] - zone_poses[i][0])**2 + (x[...,1]- zone_poses[i][1])**2 - zone_r**2), comment=f"AVOID {i}")))
    

    stl = ListAnd(reach_stls + avoid_stls)
    stl.update_format("word")
    return stl

    '''
    code for cpo, trpo, ppo might need implement on server
    
    elif env.task == 'button':          # env for sequential reach
        
    elif env.task == '':                # env for reach avoid
    '''


def reach_avoid_stl_formula(nt):
    '''
       reach red while avoiding black 
    '''
    Reach = Eventually(0, nt, AP(lambda x: x[..., 2] - threshold, comment="REACH RED"))
    Avoid = Always(0, nt, AP(lambda x: threshold - x[..., 0], comment="AVOID BLACK"))               # TODO: adjust threshold for avoid
    stl = ListAnd([Reach, Avoid])

    # check the STL content
    stl.update_format("word")
    print(stl)

    return stl

def sequential_reach_stl_formula(nt):
    '''
        reach red, reach yellow, reach black
    '''
    ReachRed = Eventually(0, nt//3, AP(lambda x: x[..., 2] - threshold, comment="REACH RED"))
    ReachYellow = Eventually(nt//3+1, 2*nt//3, AP(lambda x: x[..., 3] - threshold, comment="REACH YELLOW"))
    ReachBlack = Eventually(2*nt//3+1, nt, AP(lambda x: x[..., 0] - threshold, comment="REACH BLACK"))
    stl = ListAnd([ReachRed, ReachYellow, ReachBlack])

    # check the STL content
    stl.update_format("word")
    print(stl)

    return stl

def reach_black_avoid_others_stl_formula(nt):
    ReachBlack = Eventually(0, nt, AP(lambda x: x[..., 0] - threshold, comment="REACH BLACK"))
    AvoidRed = Always(0, nt, AP(lambda x: avoid_treshold - x[..., 2], comment="AVOID Red"))
    AvoidYellow = Always(0, nt, AP(lambda x: avoid_treshold - x[..., 3], comment="AVOID Yellow"))
    AvoidWhite = Always(0, nt, AP(lambda x: avoid_treshold - x[..., 1], comment="AVOID White"))
    stl = ListAnd([ReachBlack, AvoidRed, AvoidYellow, AvoidWhite])

    # check the STL content
    stl.update_format("word")
    print(stl)

    return stl


def sequential_avoid_stl_formula(nt):
    '''
       reach red, reach yellow, while avoiding black
    '''
    ReachRed = Eventually(0, nt//2, AP(lambda x: x[..., 2] - threshold, comment="REACH RED"))
    ReachYellow = Eventually(nt//2+1, nt, AP(lambda x: x[..., 3] - threshold, comment="REACH YELLOW"))
    Avoid = Always(0, nt, AP(lambda x: threshold - x[..., 0], comment="AVOID BLACK"))               # TODO: adjust threshold for avoid
    stl = ListAnd([ReachRed, ReachYellow, Avoid])

    # check the STL content
    stl.update_format("word")
    print(stl)

    return stl
