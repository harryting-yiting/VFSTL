# reference https://ai-boson.github.io/mcts/

import sys
from tqdm import tqdm

# sys.path.insert(0, '/app/vfstl/lib/rtamt/build/lib/')
import rtamt
import time
from datetime import datetime
import math
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from collections import defaultdict
import gym
import random
import matplotlib.pyplot as plt

# sys.path.append("/home/leo/Desktop/VFSTL/GCRL-LTL/zones/")
sys.path.append("/app/vfstl/src/GCRL-LTL/zones")
from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector
from gym.wrappers.monitor import video_recorder as VR
from collect_skill_trajectories import get_all_goal_value, from_real_dict_to_vector
from train_dynamics import VFDynamicsMLPLegacy, VFDynamics, VFDynamicsMLPWithDropout

ZONE_OBS_DIM = 24

def stl_cost(states, stl_spec):
        device = torch.device('cuda')
        if isinstance(states, list):
            states = torch.stack(states).view(-1, 4)
        if len(states.size()) < 3:
            states = states[None, :, :].to(device)


        # state N* T+1(the one is the initail state) * M
        spec = rtamt.StlDiscreteTimeOfflineSpecification()
        spec.declare_var('J0', 'float')
        spec.declare_var('W0', 'float')
        spec.declare_var('R0', 'float')
        spec.declare_var('Y0', 'float')
        #spec.spec = 'eventually[0,2]((J0 >= 0.8) and eventually[0,2](W0 >= 0.8))'
        #spec.spec = 'not ((J0 > 0.8) or (R0 > 0.8) or (Y0 > 0.8)) until[0, 3] ((W0 > 0.8) and ((not ((J0 > 0.8) or (R0 > 0.8) or (W0 > 0.8))) until[0, 3] (Y0 > 0.8)))'
        #spec.spec = 'eventually[0,10](W0 >= 0.8)'
        spec.spec = stl_spec
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
        'time': torch.tensor([i for i in range(0, J.size()[1])]).repeat((states.size()[0], 1)).T.to(device=states.device),
        'J0': J.T,
        'W0': W.T,
        'R0': R.T,
        'Y0': Y.T,
        }
        m = spec.evaluate(dataset)
        m = torch.vstack(m)
        robs = m[0,:]
        return robs

class MonteCarloTreeSearchNode:
    def __init__(self, state, dynamics, max_step, stl, batch_size, cur_step=0, parent=None, parent_action=None):
        self.dynamics = dynamics
        self.batch_size = batch_size
        self.device = torch.device('cuda')
        self.max_step = max_step
        self.cur_step = cur_step
        self.stl = stl
        self.state = state.to(self.device)
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        # self._results = defaultdict(int)
        self.score = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()

    def untried_actions(self):
        self._untried_actions = self.get_legal_actions()
        random.shuffle(self._untried_actions)
        return self._untried_actions

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.move(action, self.state)
        child_node = MonteCarloTreeSearchNode(
            next_state, self.dynamics, self.max_step, self.stl, self.batch_size, self.cur_step+1, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        if self.cur_step >= self.max_step:
            return True
        return False
        

    def rollout(self, states_outof_tree):
        # batch N of current_state, T = max_step - cur_step, init_vfs = cur_state
        device = torch.device('cuda')
        # start_time = datetime.now()

        T = self.max_step - self.cur_step
        N = self.batch_size
        cur_vfs = self.state.to(device).view(self.state.shape[0])

        rand_controls = torch.randint(0, self.state.shape[0], (N , T)).to(device)
        roll_out_states = self.dynamics.forward_simulation(rand_controls, cur_vfs)

        # put previous states to torch
        prev_states = self.traverse_state_seq()
        prev_states = torch.stack(prev_states).view(-1, 4)
        prev_states = prev_states.repeat(N, 1, 1).to(device)

        # stack cur states to batch
        cur_vfs = cur_vfs.repeat(N, 1).to(device)
        cur_vfs = cur_vfs[:, None, :]

        # stack all states
        if states_outof_tree is not None:
            states_outof_tree = states_outof_tree.repeat(N, 1, 1).to(device)
            all_states = torch.cat((states_outof_tree, prev_states, cur_vfs, roll_out_states), 1)
        else:
            all_states = torch.cat((prev_states, cur_vfs, roll_out_states), 1)

        # evaluate as a batch
        stl_robs = stl_cost(all_states, self.stl)

        # end_time = datetime.now()
        # print('Time Of RollOut Once {}'.format(end_time - start_time))

        return torch.max(stl_robs)

    def backpropagate(self, reward):
        self._number_of_visits += 1.
        self.score += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    
    def best_child_max_score(self):
        choices_weights = torch.tensor([(c.score / c.n()) for c in self.children], device=self.device)
        return self.children[torch.argmax(choices_weights)]

    def best_child(self, c_param=0.1):

        choices_weights = torch.tensor( [(c.score / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children], device=self.device)
        return self.children[torch.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):

        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():

            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def build_tree(self, iteration, states_outof_tree=None):
        # start_time = datetime.now()

        # for i in tqdm(range(iteration), desc='building tree'):
        for i in range(iteration):
            v = self._tree_policy()
            reward = v.rollout(states_outof_tree)
            v.backpropagate(reward)

        # end_time = datetime.now()
        # print('Time of Building One Tree: {}'.format(end_time - start_time))

    def move(self, action, state):
        device = torch.device('cuda')
        # return self.dynamics.forward_simulation(action, state)
        return self.dynamics.one_step_simulation(torch.tensor(action).to(device).view(1), state[None, :].to(device)).view(4)

    def get_legal_actions(self):
        '''
        Modify according to your game or
        needs. Constructs a list of all
        possible actions from current state.
        Returns a list.
        '''

        return list(range(self.dynamics.size_discrete_actions))

    def game_result(self, states_sequence):
        '''
        Modify according to your game or
        needs. Returns 1 or 0 or -1 depending
        on your state corresponding to win,
        tie or a loss.
        '''
        return stl_cost(states_sequence, self.stl)

    def traverse_state_seq(self):
        states_sequence = []
        tmp = self
        # record state values and time
        while tmp.parent:
            tmp = tmp.parent
            # record state values and time
            states_sequence.append(tmp.state)

        return states_sequence[::-1]

def best_action_sequence(root):
    action_sequence = []
    states_sequence = []
    score_sequence = []
    cur = root
    states_sequence.append(cur.state)
    score_sequence.append(cur.score)
    while len(cur.children) > 0:
        cur = cur.best_child_max_score()
        states_sequence.append(cur.state)
        action_sequence.append(cur.parent_action)
        score_sequence.append(cur.score)

    return action_sequence, states_sequence, score_sequence

def plot_traj_2d(env, traj):
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
            circles.append(plt.Circle((circle_pos[i][0], circle_pos[i][1]), 0.25, color='darkgrey'))
        elif circle_color[i].name == 'Yellow':
            circles.append(plt.Circle((circle_pos[i][0], circle_pos[i][1]), 0.25, color='gold'))
    for circle in circles:
        ax.add_patch(circle)

    # plot agent position
    traj = np.delete(traj, range(1, traj.shape[0], 10), axis=0)
    x, y = traj.T
    ax.plot(x, y, linestyle='dashed', marker='x', color='limegreen', linewidth=3)

    ax.grid(True)
    # ax.scatter(x, y)

    fig.savefig('plot_traj.png')

class MPCMCTSController:
    def __init__(self, timesteps_pre_policy:int, nnPolicy: torch.nn.Module, dynamics, goals, horizon:int, batch_size:int, tree_nodes, device):
        self.timesteps_pre_policy = timesteps_pre_policy
        self.NNPolicy = nnPolicy
        self.batch_size = batch_size
        self.horizon = horizon
        self.zone_vector = get_zone_vector()
        self.device = device
        self.dynamics = dynamics
        self.goals = goals
        self.tree_nodes = tree_nodes

        # updated
        self.stl = None # updated by setTarget
        self.current_timestep = 0
        self.current_controls_plans = []
        self.prev_n_in_horizon = 0
        self.pre_values = []
        self.high_level_controls = []

    def setTarget(self, stl:str):
        self.stl = stl
        self.still_process_T = self.horizon

    def predict(self, obs):
        done = False

        if self.current_timestep == 0:
            init_values = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, self.NNPolicy.policy, get_zone_vector(), self.device))).to(self.device)
            # controls, states, cost = self.op.optimize(self.epoch, self.batch_size, False, init_values, self.device)
            root = MonteCarloTreeSearchNode(init_values, self.dynamics, self.still_process_T, self.stl, self.batch_size)
            root.build_tree(self.tree_nodes)
            controls, states, scores = best_action_sequence(root)
            
            self.pre_values.append(init_values)
            self.current_controls_plans = controls
            self.high_level_controls.append(self.current_controls_plans[0])

        new_n_horizon = math.floor(self.current_timestep / self.timesteps_pre_policy)
        if(self.prev_n_in_horizon != new_n_horizon):
            self.still_process_T -= 1
            init_values = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, self.NNPolicy.policy, get_zone_vector(), self.device))).to(self.device)
            root = MonteCarloTreeSearchNode(init_values, self.dynamics, self.still_process_T, self.stl, self.batch_size)

            root.build_tree(self.tree_nodes, torch.stack(self.pre_values).to(self.device))
            controls, states, scores = best_action_sequence(root)
            # print(controls)
            # print(new_n_horizon)
            self.current_controls_plans = controls
            self.high_level_controls.append(self.current_controls_plans[0])
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
            
            
        # may need to calculate vfs stl robust
        return action, done, current_goal_index, self.high_level_controls
  
    def reset(self):
        self.stl = None # updated by setTarget
        self.still_process_T = 0
        self.current_timestep = 0
        self.current_controls_plans = []
        self.prev_n_in_horizon = 0
        self.pre_values = []
        self.high_level_controls = []
    
class MCTSController:
    def __init__(self, timesteps_pre_policy: int,  nnPolicy: torch.nn.Module, dynamics, goals ,horizon: int, batch_size: int, epoch: int, tree_nodes, device) -> None:
        self.timesteps_pre_policy = timesteps_pre_policy
        self.NNPolicy = nnPolicy
        self.batch_size = batch_size
        self.epoch = epoch
        self.horizon = horizon
        self.zone_vector = get_zone_vector()
        self.device = device
        self.dynamics = dynamics
        self.goals = goals
        self.tree_noes =tree_nodes

        # updated
        self.op = None # updated by setTarget
        self.current_timestep = 0
        self.current_controls_plans = []
        self.prev_n_in_horizon = 0
        self.stl_c = 0

    def setTarget(self, stl:str):
        self.op = MonteCarloTreeSearchNode(torch.zeros(1).to(self.device), self.dynamics, self.horizon, stl, self.batch_size)

    def predict(self, obs):
        done = False
        if self.current_timestep == 0:
            init_values = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, self.NNPolicy.policy, get_zone_vector(), self.device))).to(self.device)
            self.op.state = init_values
            self.op.build_tree(self.tree_nodes)
            controls, states, scores = best_action_sequence(self.op)
            
            self.stl_c = stl_cost(states, self.op.stl)
            print(f'{self.stl_c}-------------------------------------------------')
            print(controls)
            print(states)
            
            # print(scores)
            self.current_controls_plans = controls

        new_n_horizon = math.floor(self.current_timestep / self.timesteps_pre_policy)
        current_goal_index = self.current_controls_plans[new_n_horizon]
        obs = np.concatenate((obs[:-ZONE_OBS_DIM], self.zone_vector[self.goals[current_goal_index]]))
        action, _ = self.NNPolicy.predict(obs, deterministic=True)
        
        # if(self.prev_n_in_horizon != new_n_horizon or self.current_timestep == 0):
        #     print(torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, self.NNPolicy.policy, get_zone_vector(), self.device))).to(self.device))
        #     print(current_goal_index)
        # update class data
        self.current_timestep += 1
        self.prev_n_in_horizon = new_n_horizon
        
        if (self.current_timestep > self.horizon * self.timesteps_pre_policy - 1) or (self.current_timestep > len(self.current_controls_plans) * self.timesteps_pre_policy - 1):
            done = True
            
        return action, done, current_goal_index, self.stl_c

    def reset(self):
        self.op = None # updated by setTarget
        self.current_timestep = 0
        self.current_controls_plans = []
        self.prev_n_in_horizon = 0
        self.stl_c = 0

# def test_mcts_controller(stl_spec:str):
#     # Check if CUDA is available
#     # if torch.cuda.is_available():
#     #     device = torch.device("cuda:0")
#     #     print("CUDA is available. Training on GPU.")
#     # else:
#     device = torch.device("cpu")
#     # print("CUDA is not available. Training on CPU.")

    
#     model_path = '/app/vfstl/src/GCRL-LTL/zones/models/goal-conditioned/best_model_ppo_8'
#     policy_model = PPO.load(model_path, device=device)
#     timeout = 10000
#     env = ZoneRandomGoalEnv(
#         env=gym.make('Zones-8-v0', timeout=timeout), 
#         primitives_path='/app/vfstl/src/GCRL-LTL/zones/models/primitives', 
#         goals_representation=get_zone_vector(),
#         use_primitves=True,
#         rewards=[0, 1],
#         device=device,
#     )
    
#     vf_num = 4
#     T_horizon = 24
#     skill_timesteps = 100
    

#     # model = VFDynamicsMLP(vf_num).to(device=device)
#     # model.load_state_dict(torch.load('/app/vfstl/src/VFSTL/dynamic_models/test_model_20240307_085639_11', map_location=device))
#     # dynamic = VFDynamics(model, vf_num)

#     model = VFDynamicsMLPLegacy(vf_num).to(device=device)
#     model.load_state_dict(torch.load("/app/vfstl/src/VFSTL/dynamic_models/test_model_20240307_085639_11", map_location=device))
#     dynamics = VFDynamics(model.to(device), vf_num)
    
    
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     env.metadata['render.modes'] = ['rgb_array']
#     video_rec = VR.VideoRecorder(env, path = "./test_{}_{}.mp4".format(stl_spec, timestamp))
#     controller = MCTSController(skill_timesteps, policy_model, dynamics, env.goals, T_horizon, 65536, 100, device)
#     controller.setTarget(stl_spec)
#     obs = env.reset()
#     done = False
#     while not done:
#         action, controller_done, _ = controller.predict(obs)
#         obs, reward, eval_done, info = env.step(action)
#         done = controller_done
#         video_rec.capture_frame()
#     video_rec.close()
#     env.close()

# if __name__ == '__main__':
    # stl_spec = 'eventually[0,4](W0 >= 0.8 and eventually[0,5] (J0 >= 0.8))'
    # stl_spec = 'eventually[0,5] (J0 >= 0.8)'
    # test_mcts_controller(stl_spec=stl_spec)

    # device = torch.device("cpu")

    # # lower level policy
    # # model_path = '../GCRL-LTL/zones/models/goal-conditioned/best_model_ppo_8'
    # model_path = '/app/vfstl/src/GCRL-LTL/zones/models/goal-conditioned/best_model_ppo_8'
    # sys.path.append("/app/vfstl/src/GCRL-LTL/zones")
    # policy_model = PPO.load(model_path, device=device)
    # timeout = 10000
    # env = ZoneRandomGoalEnv(
    #     env=gym.make('Zones-8-v0', timeout=timeout),
    #     primitives_path='/app/vfstl/src/GCRL-LTL/zones/models/primitives', 
    #     goals_representation=get_zone_vector(),
    #     use_primitves=True,
    #     rewards=[0, 1],
    # )

    # # set up video
    # env.metadata['render.modes'] = ['rgb_array']
    # video_rec = VR.VideoRecorder(env, path = "./result.mp4")
    

    # # upper level policy
    # vf_num = 4
    # max_step = 10
    # model = VFDynamicsMLP(vf_num).to(device=device)
    # model.load_state_dict(torch.load('/app/vfstl/src/VFSTL/dynamic_models/test_model_20240307_085639_11', map_location=device))
    # dynamic = VFDynamics(model, vf_num)

    # obs = env.reset()
    # init_state = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, policy_model.policy, get_zone_vector(), device))).to(device)

    # root = MonteCarloTreeSearchNode(init_state, dynamic, max_step, stl_spec, cur_step=0)
    # root.build_tree(50000)
    # controls, states = best_action_sequence(root)

    # # print(controls)
    # # print(states)

    # # plot
    # T_horizon = 10
    # skill_timesteps = 100
    # traj = env.env.robot_pos[0:2]

    # for i in range(0, T_horizon):
    #     frame = env.fix_goal(env.goals[controls[i]])
    #     print(np.shape(frame))
    #     for j in range(0, skill_timesteps):
    #         action, _ = policy_model.predict(env.current_observation(), deterministic=True)
    #         obs, reward, eval_done, info = env.step(action)
    #         traj = np.vstack((traj, env.env.robot_pos[0:2]))
    #         video_rec.capture_frame()
    #         time.sleep(0.0001)
    #     real_value = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, policy_model.policy, get_zone_vector(), device))).to(device)
    #     print('------------real value--------------')
    #     print(real_value)

    # video_rec.close()
    # plot_traj_2d(env, traj)
    # env.close()




