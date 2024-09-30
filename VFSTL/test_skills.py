# test skills

# for each skill
    # for n tests
    #   reset env
    #   execute skill
    #   timeout or success
# success rate for each skill

import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import gym
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset
import sys
from VFSTL.vfs_dynamic.collect_skill_trajectories import get_all_goal_value, from_real_dict_to_vector, ZONE_OBS_DIM
from stable_baselines3 import PPO
from VFSTL.vfs_dynamic.train_dynamics import VFDynamics, VFDynamicsMLP
import rtamt
import time
#from gym.wrappers import RecordVideo
from gym.wrappers.monitor import video_recorder as VR
import math
from tqdm import tqdm
sys.path.append("/app/vfstl/src/GCRL-LTL/zones")
from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector
from rl.traj_buffer import TrajectoryBufferDataset

def main():
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

    # [success_num, total_num, success_steps_num]
    run_stastics = {'J':[0, 0, []], "W": [0, 0, []], "R":[0, 0, []], "Y": [0, 0, []]}
    
    epochs = 1000
    for i in tqdm(range(epochs)):
        obs = env.reset()
        done = False
        success = False
        num_steps = 0
        while not (done or success):
            action, _ = policy_model.predict(env.current_observation(), deterministic=True)
            obs, reward, done, info = env.step(action) 
            success = (reward == 1)
            num_steps+=1
            if success:
                run_stastics[env.current_goal()][0]+=1
                run_stastics[env.current_goal()][2].append(num_steps)
        run_stastics[env.current_goal()][1]+=1
        print(run_stastics)
    print(run_stastics)
if __name__ == '__main__':
    
    main()