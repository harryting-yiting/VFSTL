from datetime import datetime
import math
from tqdm import tqdm

import torch
import torch.nn.functional as F
import gym
from gym.wrappers.monitor import video_recorder as VR
from stable_baselines3 import PPO

from collect_skill_trajectories import get_all_goal_value, from_real_dict_to_vector, ZONE_OBS_DIM
from train_dynamics import VFDynamics, VFDynamicsMLPLegacy

from stl_core_lib import *

sys.path.append("/app/vfstl/src/GCRL-LTL/zones")
from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector

torch.manual_seed(123)

def get_stl_cost_function(stl_spec: str):

    def stl_cost_fn(states):

        J = states[:,:, 0]
        W = states[:,:, 1]
        R = states[:,:, 2]
        Y = states[:,:, 3]

        nt = J.size()[1]
        batch_size = states.size()[0]

        # reach Y -> reach R 
        Reach1 = Eventually(0, nt//2, AP(lambda x: x[..., 3] - 0.8, comment="REACH YELLOW"))
        Reach2 = Eventually(nt//2, nt, AP(lambda x: x[..., 2] - 0.8, comment="REACH RED"))
        stl = ListAnd([Reach1, Reach2])
        
        # print(stl)
        stl.update_format("word")
        # print(stl)
        robs = stl(states, 100, d={"hard":True})[..., 0]

        return robs * -1
    return stl_cost_fn

def trivial_fn(state):
    return torch.randn(state.size()[0])

class TrajectoryOptimizer:
    def __init__(self, dynamics, cost_fn, timesteps, size_discrete_actions, device):
        self.dynamics = dynamics
        self.cost_fn = cost_fn
        self.timesteps = timesteps
        self.size_discrete_actions = size_discrete_actions
        self.possible_actions = torch.tensor(list(range(self.size_discrete_actions))).to(device)
        self.device = device

    # Function to discretize actions
    def discretize_actions(self, continuous_actions, possible_actions):
        possible_actions =  possible_actions.view(1, -1)
        distances = (continuous_actions.unsqueeze(-1) - possible_actions).abs()
        indices = distances.squeeze(dim=1).argmin(dim=-1)
        discrete_actions = possible_actions.squeeze(0)[indices]
        return discrete_actions

    def optimize(self, num_iterations, init_state, device):

        # actions = 3 * torch.rand(self.timesteps, 1, device=device)
        actions = 3 * torch.rand(self.timesteps, 1, device=device)
        # optimizer = torch.optim.Adam([actions], lr=0.01)

        # Optimization loop
        num_iterations = 1000
        for i in tqdm(range(num_iterations)):
            # optimizer.zero_grad()

            discrete_actions = self.discretize_actions(actions, self.possible_actions)            
            # Discretize actions for forward simulation
            states = self.dynamics.forward_simulation(discrete_actions, init_state)

            cost = self.cost_fn(states)
            cost.backward()

            actions = discrete_actions
            
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost.item()}, Actions: {discrete_actions}, C Actions: {actions}")

        # Discretize final optimized actions
        optimized_discrete_actions = self.discretize_actions(actions, self.possible_actions)
        print("Optimized discrete actions:", optimized_discrete_actions)
        return optimized_discrete_actions, states, cost
        
class DualDescentController():
    def __init__(self, timesteps_pre_policy: int,  nnPolicy: torch.nn.Module, dynamics, goals ,horizon: int, epoch: int, device ):
        # timesteps_pre_action: the numer of env timesteps needed per action in the controller 
        # NNPolicy: goal_one_hot + obs -> action (one env step) or values
        self.timesteps_pre_policy = timesteps_pre_policy
        self.NNPolicy = nnPolicy
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
        
    def setTarget(self, stl:str):
        self.op = TrajectoryOptimizer(self.dynamics, get_stl_cost_function(stl), self.horizon, 4, self.device)
        return
    
    def predict(self, obs):
        
        done = False
        
        if self.current_timestep == 0:
            init_values = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, self.NNPolicy.policy, get_zone_vector(), self.device))).to(self.device)
            controls, states, cost = self.op.optimize(self.epoch, init_values, self.device)
            self.current_controls_plans = controls
            print(controls)
            # print(init_values)
            # print(states)
            print("cost")
            print(cost)
        
        new_n_horizon = math.floor(self.current_timestep / self.timesteps_pre_policy)
        current_goal_index = self.current_controls_plans[new_n_horizon]
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
    
def test_dual_descent_controller(stl_spec:str):
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
        env=gym.make('Zones-8-v1', timeout=timeout, map_seed=123), 
        primitives_path='/app/vfstl/src/GCRL-LTL/zones/models/primitives', 
        goals_representation=get_zone_vector(),
        use_primitves=True,
        rewards=[0, 1],
        device=device,
    )
    
    vf_num = 4
    T_horizon = 10
    skill_timesteps = 100
    
    model = VFDynamicsMLPLegacy(vf_num)
    model.load_state_dict(torch.load("/app/vfstl/src/VFSTL/dynamic_models/test_model_20240307_085639_11"))
    dynamics = VFDynamics(model.to(device), vf_num)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    env.metadata['render.modes'] = ['rgb_array']
    # video_rec = VR.VideoRecorder(env, path = "./test_{}_{}.mp4".format(stl_spec, timestamp))
    video_rec = VR.VideoRecorder(env, path = "./test_descent.mp4")
    controller = DualDescentController(skill_timesteps, policy_model, dynamics, env.goals, T_horizon, 10000, device)
    controller.setTarget(stl_spec)
    obs = env.reset()
    done = False
    while not done:
        action, controller_done, _ = controller.predict(obs)
        obs, reward, eval_done, info = env.step(action)
        done = controller_done
        video_rec.capture_frame()
    video_rec.close()
    env.close()

if __name__ == "__main__":
    #stl_spec = 'not ((J0 > 0.8) or (R0 > 0.8) or (Y0 > 0.8)) until[0, 3] ((W0 > 0.8) and ((not ((J0 > 0.8) or (R0 > 0.8) or (W0 > 0.8))) until[0, 3] (Y0 > 0.8)))'
    stl_spec =  'eventually[0,4](R0 >= 0.8 and eventually[0,5] (Y0 >= 0.8))'
    test_dual_descent_controller(stl_spec=stl_spec)
    #test_random_shooting()
    

# Usage
# dynamics = YourDynamicsModel()
# cost_fn = YourCostFunction()
# timesteps = 10
# size_discrete_actions = 5
# optimizer = TrajectoryOptimizer(dynamics, cost_fn, timesteps, size_discrete_actions)
# best_control, best_state, best_cost = optimizer.optimize(num_iterations=100, num_samples=50, init_state=init_state, device=device, lr=0.01)
