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

def reach_avoid_mse(ref_vfs, pred_vfs):
    '''
        ref_vfs: (nt, 4)
        pred_vfs: (batch_size, nt, 4)

        calculate the mse for each batch only if index of ref_vfs is not 0
    '''

    batch_size, nt, _ = pred_vfs.shape

    # Create a mask for non-zero elements in the last dimension of ref_vfs
    mask = ref_vfs != 0

    mse_list = []
    for i in range(batch_size):
        # Apply the mask to the batch
        masked_pred = pred_vfs[i][mask]
        masked_ref = ref_vfs[mask]
        
        # Calculate MSE
        mse = ((masked_pred - masked_ref) ** 2).mean()
        
        # Append MSE to the list
        mse_list.append(mse)

    # Convert mse_list to a tensor
    mse_tensor = torch.tensor(mse_list)

    return mse_tensor


class RandomShootingOptimization():

    def __init__(self, dynamics, cost_fn, timesteps, device) -> None:
        self.dynamics = dynamics
        self.cost_fn = cost_fn
        self.timesteps = timesteps                                 
        self.mini_control = torch.randint(0, self.dynamics.size_discrete_actions, (self.timesteps,), device=device)

    def optimize(self, num_sample_batches, batch_size, ref_vfs, init_state, device, pre_states = None):
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
            states_add_init = states_add_init.cpu()
            costs = self.cost_fn(ref_vfs, states_add_init)
            mini_index = torch.argmin(costs)
             # get best control and cost
            if costs[mini_index] < mini_cost:
                mini_cost = costs[mini_index]
                mini_control = controls[mini_index]
                mini_state = states_add_init[mini_index]
        
        return mini_control, mini_state, mini_cost

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
        
    def setTarget(self, cost_fn, ref_vfs, vfs_stl):
        self.op = RandomShootingOptimization(self.dynamics, cost_fn, self.horizon, self.device)
        self.ref_vfs = ref_vfs
        self.vfs_stl = vfs_stl
        return
    
    def predict(self, obs):
        
        done = False
        
        if self.current_timestep == 0:
            init_values = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, self.NNPolicy.policy, get_zone_vector(), self.device))).to(self.device)
            
            start_time = time.time()
            controls, states, cost = self.op.optimize(self.epoch, self.batch_size, self.ref_vfs, init_values, self.device)
            end_time = time.time()
            execution_time = end_time - start_time
            
            self.current_controls_plans = controls.tolist()
            self.predicted_vfs = states
            self.mse = cost.item()
            self.vfs_rob = self.vfs_stl(states.unsqueeze(0), 100, d={"hard":True})[..., 0]
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
    save_folder = f"./data/always_reach_T=24/rs_mse_experiment_results_{timestamp}"
    os.makedirs(save_folder, exist_ok=True)

    # Initialize accumulators for statistics
    total_pred_vfs_stl = 0
    total_state_stl = 0
    total_success = 0
    total_collide = 0
    total_vfs_mse = 0

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
        ref_vfs = generate_reference_vfs(T_horizon, target_zones, 0.9, avoid_zones, 0.2)

        if render:
            video_rec = VR.VideoRecorder(env, path=f"{save_folder}/{str(vfs_stl) + str(i)}.mp4")

        controller = RandomShootingController(skill_timesteps, policy_model, dynamics, env.goals, T_horizon, 10000, 100, device)
        controller.setTarget(reach_avoid_mse, ref_vfs, vfs_stl)

        gt_states = [torch.from_numpy(np.array(env.env.get_distance_to_zones()))]

        done = False
        while not done:
            action, controller_done, _ = controller.predict(obs)
            
            if controller.current_timestep == 1:
                # recording statistics
                controls = controller.current_controls_plans
                vfs_mse = controller.mse
                vfs_rob = controller.vfs_rob.item()
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
                            pred_vfs_stl: {vfs_rob}, \
                            vfs_mse: {vfs_mse}, \
                            state_vfs_stl: {state_stl_val}, \
                            is_success: {1 if success else 0}, \
                            complete_tasks: {complete_tasks}, \
                            number_collide: {collide}, \
                            stl_formula: {str(vfs_stl)}, \
                            optimizing_time: {optimizing_time}\n")

        # Update accumulators
        total_pred_vfs_stl += vfs_rob
        total_vfs_mse += vfs_mse
        total_state_stl += state_stl_val
        total_success += 1 if success else 0
        total_collide += collide

    # Calculate averages and success rate
    avg_pred_vfs_stl = total_pred_vfs_stl / num_experiments
    avg_vfs_mse = total_vfs_mse / num_experiments
    avg_state_stl = total_state_stl / num_experiments
    success_rate = total_success / num_experiments
    avg_collide = total_collide / num_experiments

    # Record summary statistics
    summary_file_path = os.path.join(save_folder, "summary_statistics.txt")
    with open(summary_file_path, "w") as summary_file:
        summary_file.write(f"Average pred_vfs_stl: {avg_pred_vfs_stl}\n")
        summary_file.write(f"Average state_stl: {avg_state_stl}\n")
        summary_file.write(f"Average vfs_mse: {avg_vfs_mse}\n")
        summary_file.write(f"Success rate: {success_rate}\n")
        summary_file.write(f"Average number of collide: {avg_collide}\n")


if __name__ == "__main__":
    num_reach = 3
    render = False
    num_experiments = 100
    
    for num_avoid in [0, 1]:
        test_random_shooting_controller(num_reach=num_reach, num_avoid=num_avoid, num_experiments=num_experiments, render=render)