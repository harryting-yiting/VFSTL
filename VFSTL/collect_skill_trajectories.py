import argparse
import random

import torch
import numpy as np
import gym
from stable_baselines3 import PPO
import sys
sys.path.append("/app/vfstl/src/GCRL-LTL/zones")

from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector
from rl.traj_buffer import TrajectoryBufferDataset

ZONE_OBS_DIM = 24


def get_goal_value(state, policy, zone_vector, device):
    goal_value = {'J': None, 'W': None, 'R': None, 'Y': None}
    for zone in zone_vector:
        if not np.allclose(state[-ZONE_OBS_DIM:], zone_vector[zone]):
            with torch.no_grad():
                obs = torch.from_numpy(np.concatenate((state[:-ZONE_OBS_DIM], zone_vector[zone]))).unsqueeze(dim=0).to(device)
                goal_value[zone] = policy.predict_values(obs)[0].item()
    
    return goal_value

def from_real_dict_to_vector(dict_real):
    v = np.array([])
    for key in dict_real:
        v = np.append(v, dict_real[key])
    return v

def main(args):

    device = torch.device(args.device)
    timeout = args.timeout
    buffer_size = args.buffer_size
    model_path = args.model_path
    exp_name = args.exp_name
    
    model = PPO.load(model_path, device=device)
    env = ZoneRandomGoalEnv(
        env=gym.make('Zones-8-v0', timeout=timeout), 
        primitives_path='/app/vfstl/src/GCRL-LTL/zones/models/primitives', 
        goals_representation=get_zone_vector(),
        use_primitves=True,
        rewards=[0, 1],
        device=device,
    )
    
    # build dataset
    skipped_steps = 40
    current_size = 0
    
    vf_dimension = 4
    vf_dynamic_dataset = np.zeros((buffer_size, 2*vf_dimension+1))
    with torch.no_grad():
        while current_size < buffer_size:
            prev_obs = env.reset()
            obs = prev_obs
            prev_values = from_real_dict_to_vector(get_goal_value(prev_obs, model.policy, get_zone_vector(), device))
            local_steps = 0
            eval_done = False
            while not eval_done and local_steps < timeout:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, eval_done, info = env.step(action)
                local_steps = local_steps + 1
                if local_steps % skipped_steps == 0 :
                    values = from_real_dict_to_vector(get_goal_value(obs, model.policy, get_zone_vector(), device))
                    goal_index = env.goals.index(env.current_goal())
                    tmp_row = np.concatenate(([goal_index], prev_values, values))
                    vf_dynamic_dataset[current_size] = tmp_row
                    current_size += 1
                    prev_values = values
                    prev_obs = obs

    np.save("/app/vfstl/src/VFSTL/dynamic_models/datasets/test", vf_dynamic_dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--timeout', type=int, default=1000)
    parser.add_argument('--buffer_size', type=int, default=20)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--model_path', type=str, default='/app/vfstl/src/GCRL-LTL/zones/models/goal-conditioned/best_model_ppo_8')
    parser.add_argument('--exp_name', type=str, default='traj_dataset')
    parser.add_argument('--execution_mode', type=str, default='primitives', choices=('primitives'))
    
    args = parser.parse_args()

    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    main(args)
