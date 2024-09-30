import torch
'''
    This class is a game env for the high level MCTS
'''

reach_threshold = 0.9
avoid_threshold = 0.2

class SparseReachAvoidReward:
    def __init__(self, reach_vf, avoid_vf, max_step=100) -> None:
        '''
            reach_vf: a list of option, the order of list is vising order
            avoid_vf: a list of option to avoid
            +1 for each reach option
            -1 for touch each avoid option
        '''
        self.reach_vf = reach_vf
        self.avoid_vf = avoid_vf
        self.max_step = max_step

        self.current_step = 0

    def reward(self, state):
        '''
            state: a numpy array, the state of the env in shape (batch, dim)
            J = states[:,:, 0]
            W = states[:,:, 1]
            R = states[:,:, 2]
            Y = states[:,:, 3]
        '''
        assert len(self.reach_vf) > 0, "No reach option"

        reward = 0
        current_reach = self.reach_vf[0]

        if self.is_termnial():
            # returen a reward indicating the end of the game
            return 1

        if state[current_reach] >= reach_threshold:
            reward += 1
            self.reach_vf.pop(0)
        if any(state[avoid] > avoid_threshold for avoid in self.avoid_vf):
            reward -= .5
        
        self.current_step += 1
        
        return reward
    
    def is_termnial(self):
        return len(self.reach_vf) == 0 or self.current_step >= self.max_step

class DenseReachAvoidReward(SparseReachAvoidReward):
    def __init__(self, reach_vf, avoid_vf, max_step) -> None:
        super().__init__(reach_vf, avoid_vf, max_step)
    
    def reward(self, state):
        '''
            state: a numpy array, the state of the env
            J = states[:,:, 0]
            W = states[:,:, 1]
            R = states[:,:, 2]
            Y = states[:,:, 3]
        '''

        assert len(self.reach_vf) > 0, "No reach option"

        reward = 0
        current_reach = self.reach_vf[0]

        if self.is_termnial():
            # returen a reward indicating the end of the game
            return 1
        
        # closer to the target, higher the reward
        reward += -(state[current_reach] - reach_threshold) **2
        
        if state[current_reach] >= reach_threshold:
            self.reach_vf.pop(0)

        # any avoid option is touched, the reward is negative
        if any(state[avoid] > avoid_threshold for avoid in self.avoid_vf):
            reward += -.5
        
        self.current_step += 1

        return reward
    

# class SparseSequentialReachReward:
#     def __init__(self, reach_vf, max_step=100) -> None:
#         '''
#             reach_vf: a list of option, the order of list is vising order
#             -1 for touch each avoid option
#         '''
#         self.reach_vf = reach_vf
#         self.max_step = max_step

#         self.current_step = 0

#     def is_termnial(self):
#         return len(self.reach_vf) == 0 or self.current_step >= self.max_step

#     def reward(self, state):
#         '''
#             state: a numpy array, the state of the env in shape (batch, dim)
#             J = states[:,:, 0]
#             W = states[:,:, 1]
#             R = states[:,:, 2]
#             Y = states[:,:, 3]
#         '''
#         reward = 0
#         current_reach = self.reach_vf[0]

#         if state[current_reach] >= reach_threshold:
#             reward += 1
#             self.reach_vf.pop(0)
        
#         self.current_step += 1

#         return reward
    

# class DenseReachAvoidReward(SparseReachAvoidReward):
#     def __init__(self, reach_vf, max_step) -> None:
#         super().__init__(reach_vf, max_step)
    
#     def reward(self, state):
#         '''
#             state: a numpy array, the state of the env
#             J = states[:,:, 0]
#             W = states[:,:, 1]
#             R = states[:,:, 2]
#             Y = states[:,:, 3]
#         '''
#         reward = 0
#         current_reach = self.reach_vf[0]
        
#         # closer to the target, higher the reward
#         reward += -(state[current_reach] - reach_threshold) **2
        
#         if state[current_reach] >= reach_threshold:
#             self.reach_vf.pop(0)
        
#         self.current_step += 1

#         return reward

