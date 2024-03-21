import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import gym
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset
import sys
from collect_skill_trajectories import get_all_goal_value, from_real_dict_to_vector
from stable_baselines3 import PPO
from train_dynamics import VFDynamicsMLPWithDropout, VFDynamicsMLPLegacy
# sys.path.insert(0, '/app/vfstl/lib/rtamt/build/lib/')
import rtamt
import time
from datetime import datetime
from random_shooting import RandomShootingController, MPController

from mcts import MCTSController, VFDynamics, MPCMCTSController, stl_cost
sys.path.append("/app/vfstl/src/GCRL-LTL/zones")
from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector
from rl.traj_buffer import TrajectoryBufferDataset
from envs.safety.zones_env import zone
import matplotlib.pyplot as plt
from datetime import datetime
import random
import copy


class TaskSampler:
    def __init__(self, task, aps, aps_env, sequence):
        self.task = task
        self.aps = aps
        self.aps_env = aps_env
        self.sequence = sequence

    def sample(self):
        
        #stl_spec =  'eventually[0,4](R0 >= 0.8)'
        #stl_spec_env = 'eventually[0,401](R0 >= 0.8)'
        #stl_spec = 'not ((J0 > 0.8) or (R0 > 0.8) or (Y0 > 0.8)) until[0, 3] ((W0 > 0.8) and ((not ((J0 > 0.8) or (R0 > 0.8) or (W0 > 0.8))) until[0, 3] (Y0 > 0.8)))'
        
        aps = copy.copy(self.aps)
        aps_env = copy.copy(self.aps_env)
        indices = np.arange(len(aps))
        # sequence = ['J', 'W', 'R', 'Y']

        # sequence = ['J', 'R', 'Y']
        sequence = self.sequence
        
        random.shuffle(indices)
        new_sequence = []

        vfs_step = 15
        low_level_step = 300
        if self.task == 'avoid':
            # task_info = random.choice([('not (+) until[0, {}] ((+) and (not (+) until[0, {}] (+)))', 4), ('not (+) until[0, {}] (+)', 2)])
            # low_level_task_info = random.choice([('not (+) until[0, {}] ((+) and (not (+) until[0, {}] (+)))', 4), ('not (+) until[0, ] (+)', 2)])
            task_info = ('not (+) until[0, {}] ((+) and (not (+) until[0, {}] (+)))'.format(vfs_step, vfs_step), 4)
            low_level_task_info = ('not (+) until[0, {}] ((+) and (not (+) until[0, {}] (+)))'.format(low_level_step, low_level_step), 4)
            # task_info = ('not (+) until[0, {}] (+)'.format(vfs_step), 2)
            # low_level_task_info = ('not (+) until[0, {}] (+)'.format(low_level_step), 2)
            sketch, num_ap = task_info
            low_level, num_ap = low_level_task_info
            print(indices)
            print(num_ap)
            aps_index = random.sample(indices.tolist(), k=num_ap)
            print(aps_index)
            for ap_index in aps_index:
                sketch = sketch.replace('+', aps[ap_index], 1)
                low_level = low_level.replace('+', aps_env[ap_index], 1)
                new_sequence.append(sequence[ap_index])

        elif self.task == 'chain':
            # sketch, num_ap = 'eventually[0, {}]( (+) and eventually[0, {}]((+) and eventually[0, {}]((+) and eventually[0, {}](+))))'.format(vfs_step), 4
            # low_level, num_ap = 'eventually[0, 300]( (+) and eventually[0, 300]((+) and eventually[0, 300]((+) and eventually[0, 300](+))))'.format(low_level_step), 4

            sketch, num_ap = 'eventually[0, {}]( (+) and eventually[0, {}]((+) and eventually[0, {}]((+))))'.format(vfs_step, vfs_step, vfs_step), 3
            low_level, num_ap = 'eventually[0, {}]( (+) and eventually[0, {}]((+) and eventually[0, {}]((+))))'.format(low_level_step, low_level_step, low_level_step), 3
            # sketch, num_ap = 'eventually[0, 15](+) and eventually[16, 31](+) and eventually[32, 47](+)', 3
            # low_level, num_ap = 'eventually[0, 300](+) and eventually[401, 701](+) and eventually[702, 1002](+)', 3

            for i in indices:
                print(aps[i])
                print(aps_env[i])
                print(sequence[i])
                sketch = sketch.replace('+', aps[i], 1)
                low_level = low_level.replace('+', aps_env[i], 1)
                new_sequence.append(sequence[i])

        elif self.task == 'stable':
            sketch  = 'eventually[0, {}](always[0, {}](+)) '.format(15, 10)
            low_level = 'eventually[0, 500](always[0, 300](+))'
            ap_i = random.choice(indices)
            sketch = sketch.replace('+', aps[ap_i])
            low_level = low_level.replace('+', aps_env[ap_i])
            new_sequence.append(sequence[ap_i])

        elif self.task == 'traverse':
            sketch, num_ap = 'GF(+ && XF +) && G(!+)', 3
            aps = random.sample(aps, k=num_ap)
            for ap in aps:
                sketch = sketch.replace('+', ap, 1)

        return sketch, low_level, new_sequence
        

def find_s1_in_s2(s1 , s2):
    # r: list
    # l; list
    j = 0
    for i in range(len(s2)):
        if j >= len(s1):
            break
        if s1[j] == s2[i]:
            j += 1
    return j == len(s1)
        

def test_task_simpler():
    print('testing: -----------chian------------')
    ts = TaskSampler("chain", ['J0 >= 0.9', 'W0 >= 0.9', 'R0 >= 0.9', 'Y0 >= 0.9'], ['J0 >= 0.75', 'W0 >= 0.75', 'R0 >= 0.75', 'Y0 >= 0.75'])
    stl, low, seq = ts.sample()
    print(stl)
    print(low)
    print(seq)
    print('testing: ------------avoid------------')
    ts = TaskSampler("avoid", ['J0 >= 0.9', 'W0 >= 0.9', 'R0 >= 0.9', 'Y0 >= 0.9'], ['J0 >= 0.75', 'W0 >= 0.75', 'R0 >= 0.75', 'Y0 >= 0.75'])
    stl, low, seq = ts.sample()
    print(stl)
    print(low)
    print(seq)
    print('testing: ------------stable------------')
    ts = TaskSampler("stable", ['J0 >= 0.9', 'W0 >= 0.9', 'R0 >= 0.9', 'Y0 >= 0.9'], ['J0 >= 0.75', 'W0 >= 0.75', 'R0 >= 0.75', 'Y0 >= 0.75'])
    stl, low, seq = ts.sample()
    print(stl)
    print(low)
    print(seq)
# return the distances from the robot to regions +
# parse the value function stl into ground truth stl (right now, we can manul to do this) +
# construct StlOfflineMonitor from stl_ground_truth
# compute the ground truth robutenss

def get_env_ground_truth_robustness_value(stl_env_spec, states: torch.Tensor, zones:list, zone_types):
    # input, stl_env: states and timesteps come from environmet directely
    # states: N* T * M, M: the number of zones, for each row and each type we only need the one with samllest value
    # output, robutness value
    # state N* T * M
    states = states.to(torch.device('cuda'))
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
        state_types[z_type] = torch.min(tmp, 2).values.to(states.device) # N*T*1

    J = 1 - state_types[zone.JetBlack] # N * T
    W = 1 - state_types[zone.White]
    R = 1 - state_types[zone.Red]
    Y = 1 - state_types[zone.Yellow]
    
    # print((state_types[zone.Red][0] < 0.2).nonzero(as_tuple=True)[0])
    # print(state_types[zone.Red][0][(state_types[zone.Red][0] < 0.2).nonzero(as_tuple=True)[0]])
    dataset = {
        'time': torch.tensor([i for i in range(0, J.size()[1])]).repeat((states.size()[0], 1)).T.to(states.device),
        'J0': J.T,
        'W0': W.T,
        'R0': R.T,
        'Y0': Y.T,
    }
    m = spec.evaluate(dataset)
    # m = torch.vstack(m)
    # robs = m[0,:]
    return m


def plot_traj_2d(env, traj, goals, filename):
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.set_xlim(xmin=-2.5, xmax=2.5)
    ax.set_ylim(ymin=-2.5, ymax=2.5)
    # plot circle ranges
    circle_color = env.env.zones
    circle_pos = env.env.zones_pos
    circles = []
    for i in range(len(circle_color)):
        if circle_color[i].name == 'JetBlack':
            circles.append(plt.Circle((circle_pos[i][0], circle_pos[i][1]), 0.25, color='black'))
        elif circle_color[i].name == 'Red':
            circles.append(plt.Circle((circle_pos[i][0], circle_pos[i][1]), 0.25, color='red'))
        elif circle_color[i].name == 'White':
            circles.append(plt.Circle((circle_pos[i][0], circle_pos[i][1]), 0.25, color='pink'))
        elif circle_color[i].name == 'Yellow':
            circles.append(plt.Circle((circle_pos[i][0], circle_pos[i][1]), 0.25, color='gold'))
    for circle in circles:
        ax.add_patch(circle)

    # plot agent position
    # traj = np.delete(traj, range(1, traj.shape[0], 10), axis=0)
    # x, y = traj.T
    x = traj[:, 0]
    y = traj[:, 1]
    for i in range(len(x)):
        if i == 0:
            color = 'pink'
        else:
            color = 'red'
            if goals[i-1] == 0:
                color = 'black'
            if goals[i-1] == 1:
                color = 'pink'
            if goals[i-1] == 2:
                color = 'red'
            if goals[i-1] == 3:
                color = 'gold'
        ax.plot(x[i], y[i], linestyle='dashed', marker='x', color=color, linewidth=3)

    ax.grid(True)
    # ax.scatter(x, y)

    fig.savefig('/app/vfstl/src/VFSTL/trajectory_plot/{}.png'.format(filename))
    
class ControllerEvaluator:
    # evaluate different controller 
    # evalutate matrices
    # success_rate timestep real_robustness value
    def __init__(self, controller, env: gym.Env) -> None:
        self.controller = controller
        self.env = env
        pass
    
    def one_epoch_prediction(self, stl) -> None:
        # evaluate the controller a given stl
        # state should be the labelling function(boolen)or predicate(real) result R4 [j, w, r, y]
        distance_trajectory = []
        action_trajectory = []
        position_trajectory = []
        goals = []
        # state = reset env
        obs = self.env.reset()
        distance_trajectory.append(self.env.get_distance_to_zones())
        position_trajectory.append(self.env.robot_pos)

        zone_positions = self.env.zones_pos
        # set stl for controller
        self.controller.reset()
        self.controller.setTarget(stl)
        # run a control loop
        done = False
        total_timestep = 0
        total_reward = 0
        robot_in_zones = []
        stl_c = 0
        high_level_controls = []
        while not done:
            action, contoller_done, goal, high_level_controls, vfs_states = self.controller.predict(obs)
            obs, reward, env_done, info = self.env.step(action)
            if info['zone']:
                robot_in_zones.append(info['zone'])
            distance_trajectory.append(self.env.get_distance_to_zones())
            position_trajectory.append(self.env.robot_pos)
            if isinstance(goal, int):
                goals.append(np.asarray(goal))
            else:
                goals.append(goal.detach().cpu().numpy())
            action_trajectory.append(action)
            done = contoller_done #or env_done
            total_timestep+=1
            total_reward += reward
            # prev_state = next_state
            # result.real_states, controls,
        #rob = get_env_ground_truth_robustness_value()
        #get_timestep, get_robustness_value, get_success_rate
        return distance_trajectory, action_trajectory, total_reward, total_timestep, position_trajectory, zone_positions, goals, robot_in_zones, stl_c, high_level_controls, vfs_states
    
    def random_evaluate(self, task, experiment_num, device, plot=False, exp_name='') -> None:
        # task: any of [avoid, chain]
        # ts = TaskSampler(task, ['J0 >= 0.9', 'W0 >= 0.9', 'R0 >= 0.9', 'Y0 >= 0.9'], ['J0 >= 0.75', 'W0 >= 0.75', 'R0 >= 0.75', 'Y0 >= 0.75'])
        # ts = TaskSampler(task, ['R0 >= 0.9', 'Y0 >= 0.9'], ['R0 >= 0.75', 'Y0 >= 0.75'])
        sequence = ['J', 'R', 'Y']
        ts = TaskSampler(task, ['J0 >= 0.9', 'R0 >= 0.9', 'Y0 >= 0.9'], ['J0 >= 0.75', 'R0 >= 0.75', 'Y0 >= 0.75'], sequence)
        robs = []
        truth = []
        stl_cs = []
        vfs_robs = []
        #torch.tensor(s, device=device)
        succ = 0

        for i in tqdm(range(experiment_num), desc='running experiment'):
            stl, low, zone_sequence= ts.sample()
            print(stl)
            print(low)
            print(zone_sequence)
            # stl =  'eventually[0,4](R0 >= 0.8 and eventually[0,5] (Y0 >= 0.8))'
            # low = 'eventually[0,401](R0 >= 0.8 and eventually[0,501] (Y0 >= 0.8))'
            # stl =  'eventually[0,4](R0 >= 0.8)'
            # low = 'eventually[0,401](R0 >= 0.8)'
            distances ,contorls, reward, timesteps, traj, zones_pos, goals, robot_in_zones, stl_c, high_level_controls, vfs_states = self.one_epoch_prediction(stl)
            
            tensor_s = torch.tensor(distances)
            tensor_s = tensor_s[None, :, :]
            ro = get_env_ground_truth_robustness_value(low, tensor_s, self.env.zones, self.env.zone_types)
            
            stl_cs.append(stl_c)
            robs.append(ro[0])
            print(high_level_controls)

            truth.append(find_s1_in_s2(zone_sequence, robot_in_zones))
            print(robot_in_zones)
          
            print(ro[0])

            vfs_rob = stl_cost(vfs_states, stl)
            vfs_robs.append(vfs_rob)

            print(succ)
            print(len(robs))
            print(find_s1_in_s2(zone_sequence, robot_in_zones))
            if plot:
                plot_traj_2d(self.env, np.array(traj), np.array(goals), '{}_traj_{}'.format(exp_name, low))

        
        return robs, stl_cs, truth, vfs_robs
        
        

def main(stl, stl_env):
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
    # env.metadata['render.modes'] = ['rgb_array']
    
    vf_num = 4
    T_horizon = 11
    skill_timesteps = 100
    with torch.no_grad():
        model = VFDynamicsMLPLegacy(len(env.goals)).to(device=device)
        model.load_state_dict(torch.load("/app/vfstl/src/VFSTL/dynamic_models/test_model_20240307_085639_11", map_location=device))
        # model.load_state_dict(torch.load("/app/vfstl/src/VFSTL/dynamic_models/new_20_timsteps_direct_model_20240320_075521_49", map_location=device)) 
        dynamics = VFDynamics(model.to(device), len(env.goals))
        # controller = RandomShootingController(skill_timesteps, policy_model, dynamics, env.goals, T_horizon, 32768, 100, device)
        # controller = MPController(skill_timesteps, policy_model, dynamics, env.goals, T_horizon, 16384, 50, device)
        # controller = MCTSController(skill_timesteps, policy_model, dynamics, env.goals, T_horizon, 16384, 50, device)
        controller = MPCMCTSController(skill_timesteps, policy_model, dynamics, env.goals, T_horizon, 10000, 2, device)
        # controller = RandomShootingController(skill_timesteps, policy_model, dynamics, env.goals, T_horizon, 32768, 20, device)
        #controller = MPController(skill_timesteps, policy_model, dynamics, env.goals, T_horizon, 16384, 50, device)
        evaluator = ControllerEvaluator(controller, env)
        # s,c,r,t = evaluator.one_step_prediction(stl)
        # tensor_s = torch.tensor(s, device=device)
        # tensor_s = tensor_s[None, :, :]
        # ro = get_env_ground_truth_robustness_value(stl_env, tensor_s, env.zones, env.zone_types)
        #np.save("/app/vfstl/src/VFSTL/robs_mpc_10000_chain",robs)
        robs, stl_cs, truth, vfs_robs = evaluator.random_evaluate('chain', 100 , device)

        print('statis ----------------------------------------------------------------------------------------------------')
        print(robs)
        print(stl_cs)
        robs= torch.stack(robs)
        vfs_robs = torch.stack(vfs_robs)
        print(f'Total Success in Zone Seq: {sum(truth)}')
        # stl_cs = torch.stack(stl_cs)
        path = '/app/vfstl/src/VFSTL/controller_evaluation_result/{}.pt'
        # torch.save(stl_cs, path.format('vf_estimated'))
        torch.save(truth, path.format('zone_truth'))
        torch.save(robs, path.format('ground_truth'))
        torch.save(vfs_robs, path.format('vfs_robs'))
        # robs = torch.stack(robs)
        # print(robs[robs > 0].size())
        # print(len(robs))
        # np.save("/app/vfstl/src/VFSTL/robs_mpc_10000_chain",robs)
        # print(len(s))
        # print(ro[0])
    return

if __name__ == "__main__":
    
    # skill_timesteps = 100
    # vf_to_ditance = 0.2
    # # stl_spec =  'eventually[0,4](R0 >= 0.8 and eventually[0,5] (Y0 >= 0.8))'
    # # stl_spec_env = 'eventually[0,401](R0 >= 0.8 and eventually[0,501] (Y0 >= 0.8))'

    stl_spec =  'eventually[0,4](R0 >= 0.8)'
    stl_spec_env = 'eventually[0,401](R0 >= 0.8)'
    # #stl_spec = 'not ((J0 > 0.8) or (R0 > 0.8) or (Y0 > 0.8)) until[0, 3] ((W0 > 0.8) and ((not ((J0 > 0.8) or (R0 > 0.8) or (W0 > 0.8))) until[0, 3] (Y0 > 0.8)))'
    main(stl_spec, stl_spec_env)
    #est_task_simpler()
    # print(find_s1_in_s2(['a', 'b'], ['a', 'c', 'a', 'a', 'b', 'd','b']))

