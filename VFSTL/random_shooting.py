import numpy as np
import json
import torch
import torch.nn as nn
from datetime import datetime
import gym
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset
import sys
from collect_skill_trajectories import get_all_goal_value, from_real_dict_to_vector, ZONE_OBS_DIM
from stable_baselines3 import PPO
from train_dynamics import VFDynamics, VFDynamicsMLPLegacy
import rtamt
import time
#from gym.wrappers import RecordVideo
from gym.wrappers.monitor import video_recorder as VR
import math
from stl_core_lib import *
sys.path.append("/app/vfstl/src/GCRL-LTL/zones")
from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector
from rl.traj_buffer import TrajectoryBufferDataset

def get_stl_cost_function(stl_spec: str):

    def stl_cost_fn(states):

        J = states[:,:, 0]
        W = states[:,:, 1]
        R = states[:,:, 2]
        Y = states[:,:, 3]

        nt = J.size()[1]
        batch_size = states.size()[0]

        # reach Y -> reach R 
        Reach1 = Eventually(0, nt//2, AP(lambda x: x[..., 3] - 0.8, comment="REACH YELLOW"))
        Reach2 = Eventually(nt//2, nt, AP(lambda x: x[..., 2] - 0.8, comment="REACH RED"))
        stl = ListAnd([Reach1, Reach2])
        
        # print(stl)
        stl.update_format("word")
        # print(stl)
        robs = stl(states, 100, d={"hard":True})[..., 0]

        return robs * -1
    return stl_cost_fn


class RandomShootingOptimization():

    def __init__(self, dynamics, cost_fn, constraints, timesteps, device) -> None:
        self.dynamics = dynamics
        self.cost_fn = cost_fn
        self.constraints = constraints
        self.timesteps = timesteps
        self.mini_control = torch.randint(0, self.dynamics.size_discrete_actions, (self.timesteps,), device=device)

    def optimize(self, num_sample_batches, batch_size, multiprocessing, init_state, device, pre_states = None):
        # return the best sample and there costs
        self.mini_control = torch.randint(0, self.dynamics.size_discrete_actions, (self.timesteps,), device=device)
        mini_state = []
        mini_cost = 1000000
        for i in range(0, num_sample_batches):
            # generate random action sequence with batch_size * timesteps
            controls = torch.randint(0, self.dynamics.size_discrete_actions, (batch_size, self.timesteps), device=device) 
            # run simulation
            # states dimsion: N * (T + 1) * M
            # pre_states: t*M
            states = self.dynamics.forward_simulation(controls, init_state)
            states_add_init = torch.cat((init_state.repeat(batch_size, 1, 1 ), states), dim=1)
            if pre_states != None:
                 states_add_init = torch.cat((pre_states.repeat(batch_size, 1, 1 ), states_add_init), dim=1)
            costs = self.cost_fn(states_add_init)
            mini_index = torch.argmin(costs)
             # get best control and cost
            if costs[mini_index] < mini_cost:
                mini_cost = costs[mini_index]
                mini_control = controls[mini_index]
                mini_state = states[mini_index]
        
        return mini_control, mini_state, mini_cost


def get_stl_cost_function(stl_spec: str):


    def stl_cost_fn(states):

        J = states[:,:, 0]
        W = states[:,:, 1]
        R = states[:,:, 2]
        Y = states[:,:, 3]

        nt = J.size()[1]
        batch_size = states.size()[0]

        # reach Y -> reach R 
        Reach1 = Eventually(0, nt//2, AP(lambda x: x[..., 3] - 0.8, comment="REACH YELLOW"))
        Reach2 = Eventually(nt//2, nt, AP(lambda x: x[..., 2] - 0.8, comment="REACH RED"))
        stl = ListAnd([Reach1, Reach2])
        
        # print(stl)
        stl.update_format("word")
        # print(stl)
        robs = stl(states, 100, d={"hard":True})[..., 0]

        return robs * -1



    # def stl_cost_fn(states):
    #     # state N* T+1(the one is the initail state) * M
    #     spec = rtamt.StlDiscreteTimeOfflineSpecification()
    #     spec.declare_var('J0', 'float')
    #     spec.declare_var('W0', 'float')
    #     spec.declare_var('R0', 'float')
    #     spec.declare_var('Y0', 'float')
    #     #spec.spec = 'eventually[0,2]((J0 >= 0.8) and eventually[0,2](W0 >= 0.8))'
    #     #spec.spec = 'not ((J0 > 0.8) or (R0 > 0.8) or (Y0 > 0.8)) until[0, 3] ((W0 > 0.8) and ((not ((J0 > 0.8) or (R0 > 0.8) or (W0 > 0.8))) until[0, 3] (Y0 > 0.8)))'
    #     #spec.spec = 'eventually[0,10](W0 >= 0.8)'
    #     spec.spec = stl_spec
    #     try:
    #         spec.parse()
    #         #spec.pastify()
    #     except rtamt.RTAMTException as err:
    #         print('RTAMT Exception: {}'.format(err))
    #         sys.exit()

    #     J = states[:,:, 0]
    #     W = states[:,:, 1]
    #     R = states[:,:, 2]
    #     Y = states[:,:, 3]

    #     horizon = J.size()[1]
    #     batch_size = states.size()[0]
    #     # tiemar = torch.tensor([i for i in range(0, J.size()[1])]).repeat((states.size()[0], 1)).T.to(device=states.device)
    #     s = time.time()
    #     timer = torch.arange(0, horizon, device=states.device).repeat((batch_size, 1))#.to(device=states.device)
    #     e = time.time()
    #     print('generating time: {}'.format(e-s))
    #     dataset = {
    #     'time':  timer,
    #     'J0': J.T,
    #     'W0': W.T,
    #     'R0': R.T,
    #     'Y0': Y.T,
    #     }
    #     m = spec.evaluate(dataset)
    #     # m = torch.vstack(m)
    #     robs = m[0,:]
    #     return robs * -1
    
    return stl_cost_fn
    

#def MPC(env, policy, time)


def test_random_shooting():
        # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    env.metadata['render.modes'] = ['rgb_array']
    
    stl_spec = 'not ((J0 > 0.8) or (R0 > 0.8) or (Y0 > 0.8)) until[0, 3] ((W0 > 0.8) and ((not ((J0 > 0.8) or (R0 > 0.8) or (W0 > 0.8))) until[0, 3] (Y0 > 0.8)))'
    video_rec = VR.VideoRecorder(env, path = "./test_{}_{}.mp4".format(stl_spec, timestamp))
    
    obs = env.reset()
    init_values = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, policy_model.policy, get_zone_vector(), device))).to(device)

    vf_num = 4
    T_horizon = 10
    skill_timesteps = 100
    
    model = VFDynamicsMLP(vf_num)
    model.load_state_dict(torch.load("/app/vfstl/src/VFSTL/dynamic_models/test_model_20240307_085639_11"))
    dynamics = VFDynamics(model.to(device), vf_num)
    op = RandomShootingOptimization(dynamics, get_stl_cost_function(stl_spec), cost_fn, T_horizon, device)
   
    controls, states, cost = op.optimize(5, 65536, False, init_values, device)
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


def trivial_fn(state):
    return torch.randn(state.size()[0])
    
    
class RandomShootingController():
    def __init__(self, timesteps_pre_policy: int,  nnPolicy: torch.nn.Module, dynamics, goals ,horizon: int, batch_size: int, epoch: int, device ):
        # timesteps_pre_action: the numer of env timesteps needed per action in the controller 
        # NNPolicy: goal_one_hot + obs -> action (one env step) or values
        self.timesteps_pre_policy = timesteps_pre_policy
        self.NNPolicy = nnPolicy
        self.batch_size = batch_size
        self.epoch = epoch
        self.horizon = horizon
        self.zone_vector = get_zone_vector()
        self.device = device
        self.dynamics = dynamics
        self.goals = goals
        
        
        # updated
        self.op = None # updated by setTarget
        self.current_timestep = 0
        self.current_controls_plans = []
        self.prev_n_in_horizon = 0
        
        return
        
    def setTarget(self, stl:str):
        self.op = RandomShootingOptimization(self.dynamics, get_stl_cost_function(stl), trivial_fn, self.horizon, self.device)
        return
    
    def predict(self, obs):
        
        done = False
        
        if self.current_timestep == 0:
            init_values = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, self.NNPolicy.policy, get_zone_vector(), self.device))).to(self.device)
            controls, states, cost = self.op.optimize(self.epoch, self.batch_size, False, init_values, self.device)
            self.current_controls_plans = controls
            print(controls)
            # print(init_values)
            # print(states)
            print("cost")
            print(cost)

            with open('./test_random', 'w') as f:
                json.dump(controls.tolist(), f)
        
        new_n_horizon = math.floor(self.current_timestep / self.timesteps_pre_policy)
        current_goal_index = self.current_controls_plans[new_n_horizon]
        obs = np.concatenate((obs[:-ZONE_OBS_DIM], self.zone_vector[self.goals[current_goal_index]]))
        action, _ = self.NNPolicy.predict(obs, deterministic=True)
        
        # update class data
        self.current_timestep += 1
        self.prev_n_in_horizon = new_n_horizon
        
        if self.current_timestep > self.horizon * self.timesteps_pre_policy - 1:
            done = True
            self.reset()
            
        return action, done, current_goal_index
    
    def reset(self):
        self.op = None # updated by setTarget
        self.current_timestep = 0
        self.current_controls_plans = []
        self.prev_n_in_horizon = 0


class MPController():
    def __init__(self, timesteps_pre_policy: int,  nnPolicy: torch.nn.Module, dynamics, goals ,horizon: int, batch_size: int, epoch: int, device ):
        # timesteps_pre_action: the numer of env timesteps needed per action in the controller 
        # NNPolicy: goal_one_hot + obs -> action (one env step) or values
        self.timesteps_pre_policy = timesteps_pre_policy
        self.NNPolicy = nnPolicy
        self.batch_size = batch_size
        self.epoch = epoch
        self.horizon = horizon
        self.zone_vector = get_zone_vector()
        self.device = device
        self.dynamics = dynamics
        self.goals = goals
        
        # updated
        self.op = None # updated by setTarget
        self.current_timestep = 0
        self.current_controls_plans = []
        self.prev_n_in_horizon = 0
        self.pre_values = []
        return
    
    def setTarget(self, stl:str):
        self.op = RandomShootingOptimization(self.dynamics, get_stl_cost_function(stl), trivial_fn, self.horizon, self.device)
        return
    
    def predict(self, obs):
        
        done = False
        # MPC
        # for each big time step, re-optimize control sequnece
        
        if self.current_timestep == 0:
            init_values = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, self.NNPolicy.policy, get_zone_vector(), self.device))).to(self.device)
            controls, states, cost = self.op.optimize(self.epoch, self.batch_size, False, init_values, self.device)
            self.pre_values.append(init_values)
            self.current_controls_plans = controls
            # print(controls)
            # # print(init_values)
            # # print(states)
            # print("cost")
            # print(cost)
        
        new_n_horizon = math.floor(self.current_timestep / self.timesteps_pre_policy)
        if(self.prev_n_in_horizon != new_n_horizon):
            self.op.timesteps -= 1
            init_values = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, self.NNPolicy.policy, get_zone_vector(), self.device))).to(self.device)
            controls, states, cost = self.op.optimize(self.epoch, self.batch_size, False, init_values, self.device, torch.stack(self.pre_values).to(self.device))
            # print(controls)
            # print(new_n_horizon)
            self.current_controls_plans = controls
            self.pre_values.append(init_values)
            # print(torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, self.NNPolicy.policy, get_zone_vector(), self.device))).to(self.device))
            # print(current_goal_index)
        
        current_goal_index = self.current_controls_plans[0]
        obs = np.concatenate((obs[:-ZONE_OBS_DIM], self.zone_vector[self.goals[current_goal_index]]))
        action, _ = self.NNPolicy.predict(obs, deterministic=True)
        

        self.current_timestep += 1
        self.prev_n_in_horizon = new_n_horizon
        
        if self.current_timestep > self.horizon * self.timesteps_pre_policy - 1:
            done = True
            self.reset()
            
        return action, done, current_goal_index
    
    def reset(self):
        self.op = None # updated by setTarget
        self.current_timestep = 0
        self.current_controls_plans = []
        self.prev_n_in_horizon = 0
        self.pre_values = []

def test_random_shooting_controller(stl_spec:str):
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
        env=gym.make('Zones-8-v1', timeout=timeout, map_seed=123), 
        primitives_path='/app/vfstl/src/GCRL-LTL/zones/models/primitives', 
        goals_representation=get_zone_vector(),
        use_primitves=True,
        rewards=[0, 1],
        device=device,
    )
    
    vf_num = 4
    T_horizon = 10
    skill_timesteps = 100
    
    model = VFDynamicsMLPLegacy(vf_num)
    model.load_state_dict(torch.load("/app/vfstl/src/VFSTL/dynamic_models/test_model_20240307_085639_11"))
    dynamics = VFDynamics(model.to(device), vf_num)
    
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    env.metadata['render.modes'] = ['rgb_array']
    # video_rec = VR.VideoRecorder(env, path = "./test_{}_{}.mp4".format(stl_spec, timestamp))
    video_rec = VR.VideoRecorder(env, path = "./test_random.mp4")
    controller = RandomShootingController(skill_timesteps, policy_model, dynamics, env.goals, T_horizon, 60000, 100, device)
    controller.setTarget(stl_spec)
    obs = env.reset()
    done = False
    while not done:
        action, controller_done, _ = controller.predict(obs)
        obs, reward, eval_done, info = env.step(action)
        done = controller_done
        video_rec.capture_frame()
    video_rec.close()
    env.close()
    

if __name__ == "__main__":
    #stl_spec = 'not ((J0 > 0.8) or (R0 > 0.8) or (Y0 > 0.8)) until[0, 3] ((W0 > 0.8) and ((not ((J0 > 0.8) or (R0 > 0.8) or (W0 > 0.8))) until[0, 3] (Y0 > 0.8)))'
    stl_spec =  'eventually[0,4](R0 >= 0.8 and eventually[0,5] (Y0 >= 0.8))'
    test_random_shooting_controller(stl_spec=stl_spec)
    #test_random_shooting()
    