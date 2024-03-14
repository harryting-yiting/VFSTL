# reference https://ai-boson.github.io/mcts/

from tqdm import tqdm
import rtamt
import time
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from collections import defaultdict
import gym
import sys
import matplotlib.pyplot as plt

# sys.path.append("/home/leo/Desktop/VFSTL/GCRL-LTL/zones/")
sys.path.append("/app/vfstl/src/GCRL-LTL/zones")
from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector
from gym.wrappers.monitor import video_recorder as VR
from collect_skill_trajectories import get_all_goal_value, from_real_dict_to_vector
from train_dynamics import VFDynamicsMLP

def stl_cost(states):
    # from list of tensor, to one tensor
    states_tensor = torch.zeros([len(states), states[0].size()[0]])         # T x vfs_dim
    for i in range(len(states)):
        states_tensor[i, :] = states[i]
    
    spec = rtamt.StlDiscreteTimeOfflineSpecification()
    spec.declare_var('J0', 'float')
    spec.declare_var('W0', 'float')
    spec.declare_var('R0', 'float')
    spec.declare_var('Y0', 'float')
    # spec.spec = 'eventually[0,5]((W0 >= 0.8) and eventually[5,10](R0 >= 0.8))'
    # spec.spec = 'eventually[0,10](W0 >= 0.8)'
    spec.spec = 'eventually[0,2]((J0 >= 0.8) and eventually[0,2](W0 >= 0.8))'
 

    time = torch.arange(0, states_tensor.size()[0])
    J = states_tensor[:, 0]
    W = states_tensor[:, 1]
    R = states_tensor[:, 2]
    Y = states_tensor[:, 3]

    dataset = {
    'time': time,
    'J0': J,
    'W0': W,
    'R0': R,
    'Y0': Y,
    }

    try:
        spec.parse()
        #spec.pastify()
    except rtamt.RTAMTException as err:
        print('RTAMT Exception: {}'.format(err))
        sys.exit()

    rob = spec.evaluate(dataset)

    return rob[0][1].item()

class MonteCarloTreeSearchNode:
    def __init__(self, state, dynamics, max_step, cur_step, parent=None, parent_action=None):
        self.dynamics = dynamics
        self.max_step = max_step
        self.cur_step = cur_step
        self.state = state
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
        return self._untried_actions

    # need to change to robustness value
    # def q(self):
    #     wins = self._results[1]
    #     loses = self._results[-1]
    #     return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.move(action, self.state)
        child_node = MonteCarloTreeSearchNode(
            next_state, self.dynamics, self.max_step, self.cur_step+1, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        if self.cur_step >= self.max_step:
            return True
        return False
        

    def rollout(self):
        state_sequences = []

        current_rollout_state = self.state
        state_sequences.append(current_rollout_state)
        rollout_count = self.cur_step
        while rollout_count < self.max_step: 
            possible_moves = self.get_legal_actions()

            action = self.rollout_policy(possible_moves)
            current_rollout_state = self.move(action, current_rollout_state)
            state_sequences.append(current_rollout_state)
            rollout_count += 1

        return self.game_result(self.traverse_state_seq() + state_sequences)

    def backpropagate(self, reward):
        self._number_of_visits += 1.
        self.score += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    
    def best_child_max_score(self):
        choices_weights =[c.score for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def best_child(self, c_param=0.1):

        choices_weights = [(c.score / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

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

    def build_tree(self, iteration):

        for i in tqdm(range(iteration)):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        # return self.best_child(c_param=0.)

    def move(self, action, state):
        return self.dynamics.forward_simulation(action, state)

    # def is_game_over(self):
    #     '''
    #     Modify according to your game or
    #     needs. It is the game over condition
    #     and depends on your game. Returns
    #     true or false
    #     '''
    #     if self.cur_step >= self.max_step:
    #         return True
    #     return False

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
        return stl_cost(states_sequence)

    def traverse_state_seq(self):
        states_sequence = []
        tmp = self
        # record state values and time
        while tmp.parent:
            tmp = tmp.parent
            # record state values and time
            states_sequence.append(tmp.state)

        return states_sequence[::-1]


class VFDynamics:

    def __init__(self, NNModel, size_discrete_actions) -> None:
        self.NNModel = NNModel
        self.size_discrete_actions = size_discrete_actions

    def forward_simulation(self, control, vfs):
        # one hot control, concate with vfs, call model forward
        control = nn.functional.one_hot(torch.tensor(control, dtype=torch.int64), num_classes=vfs.size()[0])
        nn_input = torch.concatenate((control, vfs), 0)
        return self.NNModel.predict(nn_input.to(torch.float32))



def best_action_sequence(root):
    action_sequence = []
    states_sequence = []
    cur = root
    states_sequence.append(cur.state)
    while len(cur.children) > 0:
        cur = cur.best_child_max_score()
        states_sequence.append(cur.state)
        action_sequence.append(cur.parent_action)

    return action_sequence, states_sequence

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
    

if __name__ == '__main__':
    device = torch.device("cpu")

    # lower level policy
    # model_path = '../GCRL-LTL/zones/models/goal-conditioned/best_model_ppo_8'
    model_path = '/app/vfstl/src/GCRL-LTL/zones/models/goal-conditioned/best_model_ppo_8'
    sys.path.append("/app/vfstl/src/GCRL-LTL/zones")
    policy_model = PPO.load(model_path, device=device)
    timeout = 10000
    env = ZoneRandomGoalEnv(
        env=gym.make('Zones-8-v0', timeout=timeout),
        primitives_path='/app/vfstl/src/GCRL-LTL/zones/models/primitives', 
        goals_representation=get_zone_vector(),
        use_primitves=True,
        rewards=[0, 1],
    )

    # set up video
    env.metadata['render.modes'] = ['rgb_array']
    video_rec = VR.VideoRecorder(env, path = "./result.mp4")
    

    # upper level policy
    vf_num = 4
    max_step = 10
    model = VFDynamicsMLP(vf_num).to(device=device)
    model.load_state_dict(torch.load('/app/vfstl/src/VFSTL/dynamic_models/test_model_20240307_085639_11', map_location=device))
    dynamic = VFDynamics(model, vf_num)

    obs = env.reset()
    init_state = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, policy_model.policy, get_zone_vector(), device))).to(device)

    root = MonteCarloTreeSearchNode(init_state, dynamic, max_step, cur_step=0)
    root.build_tree(50000)
    controls, states = best_action_sequence(root)

    # print(controls)
    # print(states)

    # plot
    T_horizon = 10
    skill_timesteps = 100
    traj = env.env.robot_pos[0:2]

    # controls = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    for i in range(0, T_horizon):
        frame = env.fix_goal(env.goals[controls[i]])
        print(np.shape(frame))
        for j in range(0, skill_timesteps):
            action, _ = policy_model.predict(env.current_observation(), deterministic=True)
            obs, reward, eval_done, info = env.step(action)
            traj = np.vstack((traj, env.env.robot_pos[0:2]))
            video_rec.capture_frame()
            time.sleep(0.0001)
        real_value = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, policy_model.policy, get_zone_vector(), device))).to(device)
        print('------------real value--------------')
        print(real_value)

    video_rec.close()
    plot_traj_2d(env, traj)
    env.close()




