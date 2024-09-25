# torch 
import torch

# stable baselines
from stable_baselines3 import PPO

# gym
import gym
from gym.wrappers.monitor import video_recorder as VR

# system tools
from tqdm import tqdm
import sys
from datetime import datetime

sys.path.append('/app/vfstl/src/GCRL-LTL/zones')
# our packages
from train_dynamics import VFDynamicsMLPLegacy, VFDynamics          # vfs dynamics
from mcts1 import MCTSNode                                          # mcts method
from stl_fomulas import *                                           # stl formulas
# environments
from collect_skill_trajectories import get_all_goal_value, from_real_dict_to_vector     
from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector


class OpenLoopOptions:
    def __init__(self, dynamics, max_step, stl_formula, device) -> None:
        self.dynamics = dynamics
        self.max_step = max_step
        self.stl_formula = stl_formula
        self.device = device

    def get_options(self, init_state):
        """
        Get the option sequences from the initial state
        """
        raise NotImplementedError
    
class MCTSOpenLoopOptions(OpenLoopOptions):
    def __init__(self, dynamics, max_step, stl_formula, device) -> None:
        super().__init__(dynamics, max_step, stl_formula, device)

    def get_options(self, init_state, iterations=80000):
        """
        starting from the initial state, build the tree for iterations
        """
        # initialize the root node
        self.root = MCTSNode(init_state, self.dynamics, self.max_step, 0, None, self.stl_formula, self.device)

        # build the tree
        for _ in tqdm(range(iterations)):
            self.root.explore()

        # get the best option sequence
        return self.get_actions_from_tree()


    def get_actions_from_tree(self):
        """
        Get the best option sequence from the tree (e.g. after the tree is built)
        """
        options = []
        vfs_hat = [self.root.state]

        cur_node = self.root
        while cur_node.children:
            cur_action, cur_node = cur_node.next()
            options.append(cur_action)
            vfs_hat.append(cur_node.state)

        return options, vfs_hat



def experiment():
    # initialize cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Training on GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training on CPU.")

    # initialize the environment
    model_path = '/app/vfstl/src/GCRL-LTL/zones/models/goal-conditioned/best_model_ppo_8'
    policy_model = PPO.load(model_path, device=device)
    timeout = 10000
    env = ZoneRandomGoalEnv(
        # env=gym.make('Zones-8-v0', timeout=timeout),
        env=gym.make('Zones-6-avoid-v0', timeout=timeout), 
        primitives_path='/app/vfstl/src/GCRL-LTL/zones/models/primitives', 
        goals_representation=get_zone_vector(),
        use_primitves=True,
        rewards=[0, 1],
        device=device,
    )
    obs = env.reset()
    # get init vfs
    init_vfs = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, policy_model.policy, get_zone_vector(), device))).to(device)

    # initialize the constants
    vf_num = 4                      # vfs dim
    T_horizon = 10                  # option steps
    skill_timesteps = 100           # env steps per option

    # initialize the stl formulas
    # stl_formula = reach_avoid_stl_formula(T_horizon)
    # stl_formula = sequential_reach_stl_formula(T_horizon)
    # stl_formula = sequential_avoid_stl_formula(T_horizon)
    stl_formula = reach_black_avoid_others_stl_formula(T_horizon)

    # initialize the env recording (TODO: need to write stl_spec and evaluation)
    env.metadata['render.modes'] = ['rgb_array']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_rec = VR.VideoRecorder(env, path = "./test_{}_{}.mp4".format(str(stl_formula), timestamp))

    # initialize the dynamics (TODO: VF Dyanmic need to be fixed)
    model = VFDynamicsMLPLegacy(vf_num)
    model.load_state_dict(torch.load("/app/vfstl/src/VFSTL/dynamic_models/test_model_20240307_085639_11", map_location=device))
    dynamics = VFDynamics(model.to(device), vf_num)

    # initialize the options methods (e.g. MCTS)
    open_loop_options = MCTSOpenLoopOptions(dynamics, T_horizon, stl_formula, device)

    # get the option sequences
    options, vfs_seq = open_loop_options.get_options(init_vfs.cpu())
    
    print(f'Option Sequences: {options}')

    # execute the option sequences in a loop
    for i in range(0, T_horizon):
        frame = env.fix_goal(env.goals[options[i]])
        for j in range(0, skill_timesteps):
            action, _ = policy_model.predict(env.current_observation(), deterministic=True)
            obs, reward, done, info = env.step(action)
            video_rec.capture_frame()
        real_vfs = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, policy_model.policy, get_zone_vector(), device))).to(device)

        print(f'Option Step {i}, GT vfs: {real_vfs.tolist()}, Predicted vfs: {vfs_seq[i].tolist()}')

    video_rec.close()
    env.close()
    

if __name__ == '__main__':
    experiment()