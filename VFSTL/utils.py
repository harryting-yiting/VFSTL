from vfs_mcts.lib_stl_core import *
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


class GameStatistic:
    '''
        This class record if game is success or not, and how many time hit the avoid zones
    '''

    def __init__(self, reach_zones, avoid_zones) -> None:
        self.reach_zones = deepcopy(reach_zones)
        self.avoid_zones = avoid_zones
        
        # need to update during the game
        self.complete_tasks = []
        self.num_collisions = 0
        self.is_success = False
        
    def check(self, env):
        '''
            at each step, obtain distance to zones.
            check if any reach_zone is reached, if so, remove the zone from reach_zones
            check if any avoid_zone is collide, if so, increment num_collisions
        '''

        zone_size = env.env.zones_size
        
        dist = env.env.get_distance_to_zones()

        # reach task is complete, the game is complete
        if len(self.reach_zones) == 0:
            self.is_success = True
            return

        # check if reach first reach_zone
        if dist[self.reach_zones[0]*2] <= zone_size or dist[self.reach_zones[0]*2+1] <= zone_size:
            self.complete_tasks.append(self.reach_zones.pop(0))
            if len(self.reach_zones) == 0:
                self.is_success = True
                return

        # should avoid some states, record collision
        if len(self.avoid_zones) > 0:
            for avoid_zone in self.avoid_zones:
                if dist[avoid_zone*2] <= zone_size or dist[avoid_zone*2+1] <= zone_size:
                    self.num_collisions += 1

    def get_result(self):
        return self.is_success, self.num_collisions, self.complete_tasks

def generate_reference_vfs(nt, reach_vfs, reach_threshold, avoid_vfs, avoid_threshold):
    '''
        This function generate reference vfs trajectory for reach task, 
        we do not know how to consider avoid task in this setting
        nt: number of time steps
        reach_vfs: a list of vfs index that we want to reach, cannot be empty
        reach_threshold: the threshold to reach the target

        return reference vfs trajectory in shape (nt, 4), e.g. only reach J in form [[0.8, 0, 0, 0], [0.8, 0, 0, 0], ...]
        only measure mse for corresponding vfs
    '''

    ref_vfs = torch.zeros((nt+1, 4))

    num_to_reach = len(reach_vfs)
    time_slices = divide_time_slots(nt+1, num_to_reach)

    for index, (start, end) in enumerate(time_slices):
        ref_vfs[start:end, reach_vfs[index]] = reach_threshold

    if len(avoid_vfs) > 0:
        for i in avoid_vfs:
            ref_vfs[:, i] = avoid_threshold

    return ref_vfs

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

    for i in range(len(time_slices)):
        integer = deepcopy(i)
        start, end = time_slices[i]

        reach_stls.append(Eventually(start, end, AP(lambda x: x[..., reach_vfs[integer]] - reach_threshold, comment=f"REACH {index_to_vfs[reach_vfs[integer]]}")))

    avoid_stls = []
    for i in avoid_vfs:
        avoid_stls.append(Always(0, nt, AP(lambda x: avoid_threshold - x[..., i], comment=f"AVOID {index_to_vfs[i]}")))

    stl = ListAnd(reach_stls + avoid_stls)
    stl.update_format("word")
    return stl

def reach_avoid_states(dist_vec, nt, zones_size=0, reach_zone=None, avoid_zone=None):
    '''
        dist: in shape (8, time_steps), each color has 2 zones
        nt: number of time steps
        reach_zone: index of zone that we want to reach, cannnot be empty [0, 1, 2, 3]
        avoid_zone: index of zone that we want to avoid, can be empty [0, 1, 2, 3]
    '''

    reach_stls = []
    time_slices = divide_time_slots(nt, len(reach_zone))
        
    # compress (8, time_steps) to (4, time_steps), by using the min distance of 2 zones
    min_dist = torch.zeros((4, dist_vec.shape[1]))
    for i in range(4):
        min_dist[i] = torch.min(dist_vec[0, :, 2*i], dist_vec[0, :, 2*i+1])

    for i in range(len(time_slices)):
        integer = deepcopy(i)
        start, end = time_slices[i]

        reach_stls.append(Eventually(start, end, AP(lambda x: zones_size - x[..., reach_zone[integer]], comment=f"REACH {index_to_vfs[reach_zone[i]]}")))

    avoid_stls = []
    for i in avoid_zone:
        avoid_stls.append(Always(0, nt, AP(lambda x: x[..., i] - zones_size, comment=f"AVOID {index_to_vfs[i]}")))
    

    stl = ListAnd(reach_stls + avoid_stls)
    stl.update_format("word")

    rob = stl(min_dist.unsqueeze(0).view(1, -1, 4), 100)[:, 0].item()
    
    return rob