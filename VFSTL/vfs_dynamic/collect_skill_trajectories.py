import argparse
import random

import torch
import numpy as np
import gym
from stable_baselines3 import PPO
import sys

from tqdm import tqdm, trange
 
import concurrent.futures

sys.path.append("/app/vfstl/src/GCRL-LTL/zones")
from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector
from rl.traj_buffer import TrajectoryBufferDataset

ZONE_OBS_DIM = 24


def get_goal_value(state, policy, zone_vector, device):
    # get values of other goals except the current goal
    goal_value = {'J': None, 'W': None, 'R': None, 'Y': None}
    for zone in zone_vector:
        if not np.allclose(state[-ZONE_OBS_DIM:], zone_vector[zone]):
            with torch.no_grad():
                obs = torch.from_numpy(np.concatenate((state[:-ZONE_OBS_DIM], zone_vector[zone]))).unsqueeze(dim=0).to(device)
                goal_value[zone] = policy.predict_values(obs)[0].item()
    
    return goal_value

def get_all_goal_value(state, policy, zone_vector, device):
    # get values of all goals
    goal_value = {'J': None, 'W': None, 'R': None, 'Y': None}
    for zone in zone_vector:
            with torch.no_grad():
                obs = torch.from_numpy(np.concatenate((state[:-ZONE_OBS_DIM], zone_vector[zone]))).unsqueeze(dim=0).to(device)
                goal_value[zone] = policy.predict_values(obs)[0].item()
    
    return goal_value

def from_real_dict_to_vector(dict_real):
    v = np.array([])
    for key in dict_real:
        v = np.append(v, dict_real[key])
    return v

def main(args, surfix):

    device = torch.device(args.device)
    timeout = args.timeout
    buffer_size = args.buffer_size
    model_path = args.model_path
    exp_name = args.exp_name
        
    # build dataset
    skipped_steps = args.skipped_steps
    random_goal = args.random_goal

    save_path = "/app/vfstl/src/VFSTL/dynamic_models/datasets/new_zone_dynamic_{}_{}_{}_{}_{}".format(
        skipped_steps, timeout, buffer_size, random_goal, surfix)
    
    model = PPO.load(model_path, device=device)
    env = ZoneRandomGoalEnv(
        env=gym.make('Zones-8-v0', timeout=timeout), 
        primitives_path='/app/vfstl/src/GCRL-LTL/zones/models/primitives', 
        goals_representation=get_zone_vector(),
        use_primitves=True,
        rewards=[0, 1],
        device=device,
    )

    current_size = 0
    
    vf_dimension = 4
    vf_dynamic_dataset = np.zeros((buffer_size, vf_dimension+1))
    with torch.no_grad():
        with tqdm(total=buffer_size) as pbar:
            while current_size < buffer_size:
                prev_obs = env.reset()
                obs = prev_obs
                prev_values = from_real_dict_to_vector(get_all_goal_value(prev_obs, model.policy, get_zone_vector(), device))
                local_steps = 0
                eval_done = False
                # we want the model to keep running even the one signal has already been true
                while local_steps < timeout and current_size < buffer_size:
                    action, _ = model.predict(env.current_observation(), deterministic=True)
                    obs, reward, eval_done, info = env.step(action)
                    local_steps = local_steps + 1
                    
                    values = from_real_dict_to_vector(get_all_goal_value(obs, model.policy, get_zone_vector(), device))
                    goal_index = env.goal_index
                    tmp_row = np.concatenate(([goal_index], values))
                    vf_dynamic_dataset[current_size] = tmp_row
                    current_size += 1
                    prev_values = values
                    # change to another goal and execute policy
                    if local_steps % skipped_steps == 0 :
                        if random_goal:
                            env.fix_goal(np.random.choice(env.goals))
                    pbar.update(1)

    np.save(save_path, vf_dynamic_dataset)
    
    return save_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--skipped_steps', type=int, default=100) #100
    parser.add_argument('--timeout', type=int, default=1000)
    parser.add_argument('--buffer_size', type=int, default=40000) #100,000/50 ,500,000 / 80 = 6250
    parser.add_argument('--random_goal', type=int, default=1)
    

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--model_path', type=str, default='/app/vfstl/src/GCRL-LTL/zones/models/goal-conditioned/best_model_ppo_8')
    parser.add_argument('--exp_name', type=str, default='traj_dataset')
    parser.add_argument('--execution_mode', type=str, default='primitives', choices=('primitives'))
    
    args = parser.parse_args()

    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    max_thread = 60
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_thread) as executor:
        futures = []
        for i in range(0, max_thread):
            futures.append(executor.submit(main, args, i))
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
        print(results)
