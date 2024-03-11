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
import time
#from gym.wrappers import RecordVideo
from gym.wrappers.monitor import video_recorder as VR
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
    # state N* T * M
    spec = rtamt.StlDiscreteTimeOfflineSpecification()
    spec.declare_var('J0', 'float')
    spec.declare_var('W0', 'float')
    spec.declare_var('R0', 'float')
    spec.declare_var('Y0', 'float')
    #spec.spec = 'eventually[0,2]((J0 >= 0.8) and eventually[0,2](W0 >= 0.8))'
    spec.spec = 'not ((J0 > 0.8) or (R0 > 0.8) or (Y0 > 0.8)) until[0, 3] ((W0 > 0.8) and ((not ((J0 > 0.8) or (R0 > 0.8) or (W0 > 0.8))) until[0, 3] (Y0 > 0.8)))'
    #spec.spec = 'eventually[0,10](W0 >= 0.8)'
    try:
        spec.parse()
        #spec.pastify()
    except rtamt.RTAMTException as err:
        print('RTAMT Exception: {}'.format(err))
        sys.exit()

    J = states[:,:, 0]
    W = states[:,:, 1]
    R = states[:,:, 2]
    Y = states[:,:, 3]

    dataset = {
    'time': torch.tensor([i for i in range(0, J.size()[1])]).repeat((states.size()[0], 1)).T,
    'J0': J.T,
    'W0': W.T,
    'R0': R.T,
    'Y0': Y.T,
    }
    # spec.evaluate(dataset) return [[0, ro], [1, ro], ...], we need the robustness of the last timestep only.
    # minimize the means -
    m = spec.evaluate(dataset)
    m = torch.vstack(m)
    # m: T*N
    end_tiem_step = 3
    #m = m[:3+1, :]

    #print(m)
    robs = m[0,:]
    #robs = torch.max(m, 0).values
    #robs = m[-1][1] * -1
    return robs * -1
    

def test_random_shooting():
        # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA is available. Training on GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training on CPU.")

    def cost_fn(state):
        return torch.randn(state.size()[0])
    
    model_path = '/app/vfstl/src/GCRL-LTL/zones/models/goal-conditioned/best_model_ppo_8'
    policy_model = PPO.load(model_path, device=device)
    timeout = 10000
    env = ZoneRandomGoalEnv(
        env=gym.make('Zones-8-v0', timeout=timeout), 
        primitives_path='/app/vfstl/src/GCRL-LTL/zones/models/primitives', 
        goals_representation=get_zone_vector(),
        use_primitves=True,
        rewards=[0, 1],
        device=device,
    )
    env.metadata['render.modes'] = ['rgb_array']
    video_rec = VR.VideoRecorder(env, path = "./test_not((J0 > 0.8) or (R0 > 0.8) or (Y0 > 0.8)) until[0, 3] ((W0 > 0.8) and ((not ((J0 > 0.8) or (R0 > 0.8) or (W0 > 0.8))) until[0, 3] (Y0 > 0.8))).mp4")
    obs = env.reset()
    init_values = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, policy_model.policy, get_zone_vector(), device))).to(device)

    vf_num = 4
    T_horizon = 10
    skill_timesteps = 100
    model = VFDynamicsMLP(vf_num)
    model.load_state_dict(torch.load("/app/vfstl/src/VFSTL/dynamic_models/test_model_20240307_085639_11"))
    dynamics = VFDynamics(model.to(device), vf_num)
    op = RandomShootingOptimization(dynamics, stl_cost_fn, cost_fn, T_horizon)
   
    controls, states, cost = op.optimize(1, 20480, False, init_values, device)
    print(controls)
    print(states)
    print(cost)

    for i in range(0, T_horizon):
        frame = env.fix_goal(env.goals[controls[i]])
        print(np.shape(frame))
        for j in range(0, skill_timesteps):
            action, _ = policy_model.predict(env.current_observation(), deterministic=True)
            obs, reward, eval_done, info = env.step(action)
            video_rec.capture_frame()
        real_value = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, policy_model.policy, get_zone_vector(), device))).to(device)
        print('------------real value--------------')
        print(real_value)
        print('------------predict value--------------')
        print(states[i])
    video_rec.close()
    env.close()
    
if __name__ == "__main__":
    test_random_shooting()
    