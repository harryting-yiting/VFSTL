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
from random_shooting import RandomShootingController
sys.path.append("/app/vfstl/src/GCRL-LTL/zones")
from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector
from rl.traj_buffer import TrajectoryBufferDataset
from envs.safety.zones_env import zone

# return the distances from the robot to regions +
# parse the value function stl into ground truth stl (right now, we can manul to do this) +
# construct StlOfflineMonitor from stl_ground_truth
# compute the ground truth robutenss

def get_env_ground_truth_robustness_value(stl_env_spec, states: torch.Tensor, zones:list, zone_types):
    # input, stl_env: states and timesteps come from environmet directely
    # states: N* T * M, M: the number of zones, for each row and each type we only need the one with samllest value
    # output, robutness value
    # state N* T * M
    spec = rtamt.StlDiscreteTimeOfflineSpecification()
    spec.declare_var('J0', 'float')
    spec.declare_var('W0', 'float')
    spec.declare_var('R0', 'float')
    spec.declare_var('Y0', 'float')
    #spec.spec = 'eventually[0,2]((J0 >= 0.8) and eventually[0,2](W0 >= 0.8))'
    #spec.spec = 'not ((J0 > 0.8) or (R0 > 0.8) or (Y0 > 0.8)) until[0, 3] ((W0 > 0.8) and ((not ((J0 > 0.8) or (R0 > 0.8) or (W0 > 0.8))) until[0, 3] (Y0 > 0.8)))'
    #spec.spec = 'eventually[0,10](W0 >= 0.8)'
    spec.spec = stl_env_spec
    try:
        spec.parse()
        #spec.pastify()
    except rtamt.RTAMTException as err:
        print('RTAMT Exception: {}'.format(err))
        sys.exit()
    
    # get the corresponding types
    # [J; W; Y: R]
    state_types = {}
    for z_type in zone_types:
        row_index = [ind for ind, ele in enumerate(zones) if ele == z_type]
        tmp = states[:,:, row_index] # N*T*R
        state_types[z_type] = torch.min(tmp, 2).values # N*T*1

    J = 1 - state_types[zone.JetBlack] # N * T
    W = 1 - state_types[zone.White]
    R = 1 - state_types[zone.Red]
    Y = 1 - state_types[zone.Yellow]
    
    
    dataset = {
        'time': torch.tensor([i for i in range(0, J.size()[1])]).repeat((states.size()[0], 1)).T,
        'J0': J.Tmm,
        'W0': W.T,
        'R0': R.T,
        'Y0': Y.T,
    }
    m = spec.evaluate(dataset)
    # m = torch.vstack(m)
    # robs = m[0,:]
    return m


class ControllerEvaluator:
    # evaluate different controller 
    # evalutate matrices
    # success_rate timestep real_robustness value
    def __init__(self, controller, env: gym.Env) -> None:
        self.controller = controller
        self.env = env
        pass
    
    def evaluate(self, stl) -> None:
        # evaluate the controller a given stl
        # state should be the labelling function(boolen)or predicate(real) result R4 [j, w, r, y]
        state_trajectory = []
        action_trajectory = []
        # state = reset env
        obs = self.env.reset()
        state_trajectory.append(self.env.get_distance_to_zones())
        # set stl for controller
        self.controller.reset()
        self.controller.setTarget(stl)
        # run a control loop
        done = False
        total_timestep = 0
        total_reward = 0
        while not done:
            action, contoller_done = self.controller.predict(obs)
            obs, reward, env_done, info = self.env.step(action)
            state_trajectory.append(self.env.get_distance_to_zones())
            action_trajectory.append(action)
            done = contoller_done #or env_done
            total_timestep+=1
            total_reward += reward
            # prev_state = next_state
            # result.real_states, controls,
        #rob = get_env_ground_truth_robustness_value()
        #get_timestep, get_robustness_value, get_success_rate
        return state_trajectory, action_trajectory, total_reward, total_timestep
    
    def random_evaluate() -> None:
        pass
    

def main(stl, stl_env):
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
    
    vf_num = 4
    T_horizon = 10
    skill_timesteps = 100
    
    model = VFDynamicsMLP(env.goal_types)
    model.load_state_dict(torch.load("/app/vfstl/src/VFSTL/dynamic_models/test_model_20240307_085639_11"))
    dynamics = VFDynamics(model.to(device), vf_num)
    


    controller = RandomShootingController(skill_timesteps, policy_model, dynamics, env.goals, T_horizon, 65536, 100, device)
    
    evaluator = ControllerEvaluator(controller, env)
    s,c,r,t = evaluator.evaluate(stl)
    tensor_s = torch.tensor(s, device=device)
    tensor_s = tensor_s[None, :, :]
    ro = get_env_ground_truth_robustness_value(stl_env, tensor_s, env.zones, env.zone_types)
    
    print(len(s))
    print(ro)
    return

if __name__ == "__main__":
    stl_spec =  'eventually[0,4](R0 >= 0.8 and eventually[0,5] (Y0 >= 0.8))'
    skill_timesteps = 100
    vf_to_ditance = 0.2
    stl_spec_env = 'eventually[0,400](R0 >= 0.8 and eventually[0,500] (Y0 >= 0.8))'
    
    main(stl_spec, stl_spec_env)

