# reference source: https://github.com/ciamic/MCTS

# torch 
import torch

# math tools
import random
from math import sqrt, log

# plot tools

# system tools
from tqdm import tqdm
import sys
from copy import deepcopy

sys.path.append('/app/vfstl/src/GCRL-LTL/zones')
sys.path.append('/app/vfstl/src/VFSTL/vfs_dynamic')
# our packages
from train_dynamics import VFDynamicsMLPLegacy, VFDynamics
from vfs_env import *

# constants
c = 1.0                                 # exploration parameter of mcts
smoothing_factor = 100                  # smoothing factor for the stl robustness
batch_size = 32                         # batch size for the rollout in mcts

nodes_count = 0

class MCTSNode:
    def __init__(self, state, dynamic, action, device, parent=None) -> None:
        self.state = state                  # vfs
        self.action = action                # option lead to this node

        # dynamics f(vfs, option) -> next_vfs (i.e. a neural network)
        self.dynamic = dynamic
        self.device = device                # device for the dynamics

        self.parent = parent
        self.children = None

        # visit count
        self.visits = 0
        
        # rewards from MCTS exploration
        self.rewards = 0

    def getUCBscore(self):
        '''
        This is the formula that gives a value to the node.
        The MCTS will pick the nodes with the highest value.        
        '''
        # Unexplored nodes have maximum values so we favour exploration
        if self.visits == 0:
            return float('inf')
        
        # We need the parent node of the current node 
        top_node = self
        if top_node.parent:
            top_node = top_node.parent
            
        # We use one of the possible MCTS formula for calculating the node value
        return (self.rewards / self.visits) + c * sqrt(log(top_node.visits) / self.visits)
    
    def move(self, action):
        with torch.no_grad():
            child_state = self.dynamic.one_step_simulation(torch.tensor(action).to(self.device).view(1), self.state[None, :].to(self.device).view(-1, 4))
        return child_state[0].cpu()

    def get_possible_actions(self):
        '''
        extract possible actions from the dynamics
        '''
        return list(range(self.dynamic.size_discrete_actions))
    
    def next(self):
        ''' 
        Once we have done enough search in the tree, the values contained in it should be statistically accurate.
        We will at some point then ask for the next action to play from the current node, and this is what this function does.
        There may be different ways on how to choose such action, in this implementation the strategy is as follows:
        - pick at random one of the node which has the maximum visit count, as this means that it will have a good value anyway.
        '''
        children = self.children

        # Find the key-value pair with the max node.visits, and if node.visits is the same, then max node.rewards
        best_action, best_node = max(children.items(), key=lambda item: (item[1].visits, item[1].rewards))

        return best_action, best_node

    
    def create_children(self):
        raise NotImplementedError           # child node type is differnt for different games

    def explore(self):
        raise NotImplementedError           # child class might have different rollout strategy
    
    def rollout(self):
        raise NotImplementedError           # child class might have different rollout strategy


class MCTSNodeReward(MCTSNode):
    def __init__(self, state, dynamic, action, device, reward_env, parent=None) -> None:
        super().__init__(state, dynamic, action, device, parent)
        
        # reward environment object
        self.reward_env = reward_env

    def move(self, action):
        with torch.no_grad():
            child_state = self.dynamic.one_step_simulation(torch.tensor(action).to(self.device).view(1), self.state[None, :].to(self.device).view(-1, 4))
        return child_state[0].cpu()

    def create_children(self):
        
        actions = self.get_possible_actions()

        children = {}
        for action in actions:
            reward_env = deepcopy(self.reward_env)
            child_state = self.move(action)
            reward_env.current_step += 1
            children[action] = MCTSNodeReward(child_state, self.dynamic, action, self.device, reward_env, parent=self)

        self.children = children
    
    def explore(self, nodes_count):
        current = self

        while current.children:
            children = current.children
            action, current = max(children.items(), key=lambda x: x[1].getUCBscore())

        if current.visits < 1:
            reward = current.rollout()
        elif current.reward_env.is_termnial():
            reward = current.rollout()
        else:
            nodes_count += 1
            current.create_children()
            # debug print
            # print('building new nodes, this is the', nodes_count, 'th node')
            if current.children:
                current = random.choice(current.children)
            reward = current.rollout()

        # debug print
        # print('reward:', reward)
        # update statistics and backpropagate
        while current:
            current.visits += 1
            current.rewards += reward
            current = current.parent

        return nodes_count

    def rollout(self):
        reward = 0
        state = self.state
        reward_env = deepcopy(self.reward_env)
        actions = self.get_possible_actions()
        while not reward_env.is_termnial():
            reward += reward_env.reward(state)
            state = self.move(random.choice(actions))

        return reward


class MCTSNodeSTL(MCTSNode):
    def __init__(self, state, dynamic, max_step, cur_step, action, stl_formula, device, parent=None):
        super().__init__(state, dynamic, action, device, parent)
        '''
            This class is for STL guided MCTS, the reward is measured by calculating the average robustness of 
            the STL formula over the trajectory. This tree has a fixed length due to STL measure a fixed length.
        '''

        # steps in environment (e.g. tree depth)
        self.max_step = max_step        
        self.cur_step = cur_step

        # parent node
        self.parent = parent

        # stl formula
        self.stl_formula = stl_formula 

    def create_children(self):
        
        '''
        We create one children for each possible action of the game, 
        then we apply such action to a copy of the current node enviroment 
        and create such child node with proper information returned from the action executed
        '''
        if self.is_terminal():
            return 
        
        actions = self.get_possible_actions()

        children = {}
        for action in actions:
            child_state = self.move(action)
            children[action] = MCTSNodeSTL(child_state, self.dynamic, self.max_step, self.cur_step+1, action, self.stl_formula, self.device, parent=self)
            # self.dynamics.one_step_simulation(torch.tensor(action).to(device).view(1), state[None, :].to(device)).view(4)

        self.children = children


    def explore(self, nodes_count):
        
        '''
        The search along the tree is as follows:
        - from the current node, recursively pick the children which maximizes the value according to the MCTS formula
        - when a leaf is reached:
            - if it has never been explored before, do a rollout and update its current value
            - otherwise, expand the node creating its children, pick one child at random, do a rollout and update its value
        - backpropagate the updated statistics up the tree until the root: update both value and visit counts
        '''
        current = self

        while current.children:
            children = current.children
            action, current = max(children.items(), key=lambda x: x[1].getUCBscore())

        '''
            visits < 1 + terminal: rollout and update
            visits < 1 + not terminal: rollout and update
            visits >= 1 + terminal: rollout and update
            visits >= 1 + not terminal: expand
        '''
        if current.is_terminal():
            # current.rewards += current.rollout()
            reward = current.rollout()
        elif current.visits < 1:
            # current.rewards += current.rollout()
            reward = current.rollout()
        else:
            assert current.visits == 1, "Error: the node is not terminal neither a leaf"
            nodes_count += 1
            current.create_children()
            if current.children:
                current = random.choice(current.children)          # TODO: dictonary can not be sampled
            # current.T = current.T + current.rollout()
            reward = current.rollout()

        # current.visits += 1

        # update statistics and backpropagate
        while current:
            current.visits += 1
            current.rewards += reward
            current = current.parent
            
        return nodes_count
    
    
    def is_terminal(self):
        '''
        check if the current node is terminal (reach max_depth)
        '''
        return self.cur_step == self.max_step
    
    def rollout(self):
        
        '''
        This function concatenates previous states, current states, and random future states. Then 
        evaluate the average cost of the STL formula over the states.

        The evaluated trajecoty is in shape (vfs_dim, N, T), where N is the batch size and T is the time steps.
        '''
        # batch N of current_state, T = max_step - cur_step, init_vfs = cur_state
        # device = torch.device('cuda')
        device = self.device
        # start_time = datetime.now()

        T = self.max_step - self.cur_step
        N = batch_size
        cur_vfs = self.state.to(device).view(self.state.shape[0])

        if T > 0:
            rand_controls = torch.randint(0, self.state.shape[0], (N, T)).to(device)
            with torch.no_grad():
                roll_out_states = self.dynamic.forward_simulation(rand_controls, cur_vfs)
        else:
            # Handle the case where T is 0
            roll_out_states = torch.empty(N, 0, self.state.shape[0]).to(device)

        # put previous states to torch
        prev_states = self.traverse_state_seq()
        if prev_states:
            prev_states = torch.stack(prev_states).view(-1, 4)
            prev_states = prev_states.repeat(N, 1, 1).to(device)
        else:
            # Handle the case where prev_states is empty
            prev_states = torch.empty(0, 4).to(device)

        # stack cur states to batch
        cur_vfs = cur_vfs.repeat(N, 1).to(device)
        cur_vfs = cur_vfs[:, None, :]

        # stack all states
        if prev_states.numel() > 0:
            all_states = torch.cat((prev_states, cur_vfs, roll_out_states), 1).to(device)
        else:
            all_states = torch.cat((cur_vfs, roll_out_states), 1).to(device)

        # evaluate as a batch
        with torch.no_grad():
            stl_robs = self.stl_formula(all_states, smoothing_factor)[:, 0]

        # Move stl_robs to CPU and free GPU memory
        # stl_robs_cpu = stl_robs.detach().cpu()
        # del stl_robs, all_states, cur_vfs, prev_states, roll_out_states, rand_controls
        # torch.cuda.empty_cache()

        return torch.mean(stl_robs).item()
    
    def traverse_state_seq(self):
        '''
        This function returns the sequence of states from the root to the current node.
        '''
        states_sequence = []
        tmp = self
        # record state values and time
        while tmp.parent:
            tmp = tmp.parent
            # record state values and time
            states_sequence.append(tmp.state)

        return states_sequence[::-1]
