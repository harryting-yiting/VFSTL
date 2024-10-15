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
# import rtamt
import time
#from gym.wrappers import RecordVideo
from gym.wrappers.monitor import video_recorder as VR
import math

from vfs_mcts.lib_stl_core import *
from utils import *

from tqdm import tqdm

sys.path.append("/app/vfstl/src/GCRL-LTL/zones")
from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector
from rl.traj_buffer import TrajectoryBufferDataset


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
            with torch.no_grad():
                states = self.dynamics.forward_simulation(controls, init_state)
            states_add_init = torch.cat((init_state.repeat(batch_size, 1, 1 ), states), dim=1)
            if pre_states != None:
                 states_add_init = torch.cat((pre_states.repeat(batch_size, 1, 1 ), states_add_init), dim=1)
            costs = -1 * self.cost_fn(states_add_init, 100, d={"hard":True})[..., 0]
            mini_index = torch.argmin(costs)
             # get best control and cost
            if costs[mini_index] < mini_cost:
                mini_cost = costs[mini_index]
                mini_control = controls[mini_index]
                mini_state = states[mini_index]
        
        return mini_control, mini_state, mini_cost

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
        
    def setTarget(self, stl):
        self.op = RandomShootingOptimization(self.dynamics, stl, trivial_fn, self.horizon, self.device)
        return
    
    def predict(self, obs):
        
        done = False
        
        if self.current_timestep == 0:
            init_values = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, self.NNPolicy.policy, get_zone_vector(), self.device))).to(self.device)
            
            start_time = time.time()
            controls, states, cost = self.op.optimize(self.epoch, self.batch_size, False, init_values, self.device)
            end_time = time.time()
            execution_time = end_time - start_time
            
            self.current_controls_plans = controls.tolist()
            self.predicted_vfs = states
            self.vfs_robs = cost.item() * -1
            self.optimizing_time = execution_time
        
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

def test_random_shooting_controller(num_reach, num_avoid, num_experiments, render):
    # initialize the constants
    vf_num = 4
    T_horizon = 24
    skill_timesteps = 100
    smoothing_factor = 100

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA is available. Training on GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training on CPU.")
    
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
    
    model = VFDynamicsMLPLegacy(vf_num)
    model.load_state_dict(torch.load("/app/vfstl/src/VFSTL/dynamic_models/test_model_20240307_085639_11", map_location=device))
    dynamics = VFDynamics(model.to(device), vf_num)
    
    # Create a folder for the current run of experiments
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_folder = f"./data/always_reach_T=24/rs_experiment_results_{timestamp}"
    os.makedirs(save_folder, exist_ok=True)

    # Initialize accumulators for statistics
    total_pred_vfs_stl = 0
    total_state_stl = 0
    total_success = 0
    total_collide = 0

    if render:
        env.metadata['render.modes'] = ['rgb_array']

    # Open a file to record each experiment's data
    data_file_path = os.path.join(save_folder, "experiment_data.txt")

    for i in tqdm(range(num_experiments)):
        # generate random task
        target_zones, avoid_zones = generate_task(num_reach, num_avoid, 4)

        # reset the environment
        game_checker = GameStatistic(target_zones, avoid_zones)
        obs = env.reset()
        game_checker.check(env)
        # init_vfs = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, policy_model.policy, get_zone_vector(), device))).to(device)

        # initialize the stl formulas
        vfs_stl = reach_avoid_vfs(T_horizon, target_zones, avoid_zones, 0.9, 0.2)

        if render:
            video_rec = VR.VideoRecorder(env, path=f"{save_folder}/{str(vfs_stl) + str(i)}.mp4")

        controller = RandomShootingController(skill_timesteps, policy_model, dynamics, env.goals, T_horizon, 10000, 100, device)
        controller.setTarget(vfs_stl)

        gt_states = [torch.from_numpy(np.array(env.env.get_distance_to_zones()))]

        done = False
        while not done:
            action, controller_done, _ = controller.predict(obs)
            
            if controller.current_timestep == 1:
                # recording statistics
                controls = controller.current_controls_plans
                pred_vfs_stl = controller.vfs_robs
                pred_vfs = controller.predicted_vfs
                optimizing_time = controller.optimizing_time

            obs, reward, eval_done, info = env.step(action)
            game_checker.check(env)
            # record states
            gt_states.append(torch.from_numpy(np.array(env.env.get_distance_to_zones())))
            done = controller_done
            if render:
                video_rec.capture_frame()
        if render:
            video_rec.close()
        env.close()
    
        gt_states = torch.stack(gt_states).unsqueeze(0)
        # Calculate metrics
        state_stl_val = reach_avoid_states(gt_states, T_horizon*skill_timesteps, env.env.zones_size, target_zones, avoid_zones)

        # convert gt_states to numpy
        success, collide, complete_tasks = game_checker.get_result()
        
        # Record metrics
        with open(data_file_path, "a") as data_file:
            data_file.write(f"target_zones: {target_zones}, \
                            avoid_zones: {avoid_zones}, \
                            actions: {controls}, \
                            pred_vfs_stl: {pred_vfs_stl}, \
                            state_vfs_stl: {state_stl_val}, \
                            is_success: {1 if success else 0}, \
                            complete_tasks: {complete_tasks}, \
                            number_collide: {collide}, \
                            stl_formula: {str(vfs_stl)}, \
                            optimizing_time: {optimizing_time}\n")

        # Update accumulators
        total_pred_vfs_stl += pred_vfs_stl
        total_state_stl += state_stl_val
        total_success += 1 if success else 0
        total_collide += collide

    # Calculate averages and success rate
    avg_pred_vfs_stl = total_pred_vfs_stl / num_experiments
    avg_state_stl = total_state_stl / num_experiments
    success_rate = total_success / num_experiments
    avg_collide = total_collide / num_experiments

    # Record summary statistics
    summary_file_path = os.path.join(save_folder, "summary_statistics.txt")
    with open(summary_file_path, "w") as summary_file:
        summary_file.write(f"Average pred_vfs_stl: {avg_pred_vfs_stl}\n")
        summary_file.write(f"Average state_stl: {avg_state_stl}\n")
        summary_file.write(f"Success rate: {success_rate}\n")
        summary_file.write(f"Average number of collide: {avg_collide}\n")

if __name__ == "__main__":
    num_reach = 3
    render = False
    num_experiments = 100
    
    for num_avoid in [0, 1]:
        test_random_shooting_controller(num_reach=num_reach, num_avoid=num_avoid, num_experiments=num_experiments, render=render)