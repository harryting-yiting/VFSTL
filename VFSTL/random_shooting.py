import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import gym
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset
import sys
from collect_skill_trajectories import get_all_goal_value, from_real_dict_to_vector
from stable_baselines3 import PPO
from train_dynamics import VFDynamics, VFDynamicsMLP
import rtamt
sys.path.append("/app/vfstl/src/GCRL-LTL/zones")
from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector
from rl.traj_buffer import TrajectoryBufferDataset

class RandomShootingOptimization():

    def __init__(self, dynamics, cost_fn, constraints, timesteps) -> None:
        self.dynamics = dynamics
        self.cost_fn = cost_fn
        self.constraints = constraints
        self.timesteps = timesteps

    def optimize(self, num_sample_batches, batch_size, multiprocessing, init_state, device):
        # return the best sample and there costs
        mini_control = torch.randint(0, self.dynamics.size_discrete_actions, (self.timesteps,), device=device)
        mini_state = []
        mini_cost = 1000000
        for i in range(0, num_sample_batches):
            # generate random action sequence with batch_size * timesteps
            controls = torch.randint(0, self.dynamics.size_discrete_actions, (batch_size, self.timesteps), device=device) 
            # run simulation
            states = self.dynamics.forward_simulation(controls, init_state)
            costs = self.cost_fn(states)
            mini_index = torch.argmin(costs)
             # get best control and cost
            if costs[mini_index] < mini_cost:
                mini_cost = costs[mini_index]
                mini_control = controls[mini_index]
                mini_state = states[mini_index]
        
        return mini_control, mini_state, mini_cost


def stl_cost_fn(states):
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.declare_var('a0', 'float')
    spec.spec = 'eventually[0,10](a0 >= 0.8)'
    try:
        spec.parse()
        spec.pastify()
    except rtamt.RTAMTException as err:
        print('RTAMT Exception: {}'.format(err))
        sys.exit()
    a0 = states[:,:, 0]
    rob = spec.evaluate(['a0', a0])
    return rob
    

def test_random_shooting():
        # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        print("CUDA is available. Training on GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training on CPU.")

    def cost_fn(state):
        return torch.randn(state.size()[0])
    
    model_path = '/app/vfstl/src/GCRL-LTL/zones/models/goal-conditioned/best_model_ppo_8'
    model = PPO.load(model_path, device=device)
    timeout = 10000
    env = ZoneRandomGoalEnv(
        env=gym.make('Zones-8-v0', timeout=timeout), 
        primitives_path='/app/vfstl/src/GCRL-LTL/zones/models/primitives', 
        goals_representation=get_zone_vector(),
        use_primitves=True,
        rewards=[0, 1],
        device=device,
    )

    obs = env.reset()
    init_values = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, model.policy, get_zone_vector(), device))).to(device)

    vf_num = 4
    model = VFDynamicsMLP(vf_num)
    model.load_state_dict(torch.load("/app/vfstl/src/VFSTL/dynamic_models/test_model_20240307_085639_11"))
    dynamics = VFDynamics(model.to(device), vf_num)
    op = RandomShootingOptimization(dynamics, stl_cost_fn, cost_fn, 10)
    print(op.optimize(1024, 1024, False, init_values, device))


if __name__ == "__main__":
    test_random_shooting()
    