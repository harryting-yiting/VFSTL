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
import argparse

sys.path.append('/app/vfstl/src/GCRL-LTL/zones')
sys.path.append('/app/vfstl/src/VFSTL/vfs_dynamic')
# our packages
from utils import *
from train_dynamics import VFDynamicsMLPLegacy, VFDynamics          # vfs dynamics
from mcts import *                                   # mcts method
from stl_formulas import *                                           # stl formulas
# environments
from collect_skill_trajectories import get_all_goal_value, from_real_dict_to_vector     
from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector


class CloseLoopOptions:
    def __init__(self, dynamics, device) -> None:
        self.dynamics = dynamics
        self.device = device

    def get_options(self, init_state):
        """
        Get the option sequences from the initial state
        """
        raise NotImplementedError
    
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
    
class STLMCTSCloseLoopOptions(CloseLoopOptions):
    def __init__(self, dynamics, max_step, stl_formula, device) -> None:
        super().__init__(dynamics, device)
        self.max_step = max_step
        self.stl_formula = stl_formula

        self.node_count = 0

    def get_options(self, init_state, iterations=80000):
        """
        starting from the initial state, build the tree for iterations
        """
        # initialize the root node
        self.root = MCTSNodeSTL(init_state, self.dynamics, self.max_step, 0, None, self.stl_formula, self.device)

        # build the tree
        for _ in tqdm(range(iterations)):
            self.node_count = self.root.explore(self.node_count)

        action_seq, vfs_seq = self.get_actions_from_tree()
        if len(action_seq) < self.max_step:
            action_seq += [action_seq[-1]] * (self.max_step - len(action_seq))
        elif len(action_seq) > self.max_step:
            action_seq = action_seq[:self.max_step]

        assert len(action_seq) == self.max_step, f"Action sequence length {len(action_seq)} does not match the horizon {self.max_step}"

        # get the best option sequence
        return action_seq, vfs_seq, self.node_count


class RewardMCTSOpenLoopOptions(OpenLoopOptions):
    def __init__(self, dynamics, reward_env, device, max_step) -> None:
        super().__init__(dynamics, device)
        self.reward_env = reward_env
        self.node_count = 0

        self.max_step = max_step

    def get_options(self, init_state, iterations=80000):
        self.root = MCTSNodeReward(init_state, self.dynamics, None, self.device, self.reward_env)

        # build the tree
        for _ in tqdm(range(iterations)):
            self.node_count = self.root.explore(self.node_count)

        action_seq, vfs_seq = self.get_actions_from_tree()
        if len(action_seq) < self.max_step:
            action_seq += [action_seq[-1]] * (self.max_step - len(action_seq))
        elif len(action_seq) > self.max_step:
            action_seq = action_seq[:self.max_step]

        assert len(action_seq) == self.max_step, f"Action sequence length {len(action_seq)} does not match the horizon {self.max_step}"

        # get the best option sequence
        return action_seq, vfs_seq, self.node_count
    
def experiments(num_reach, num_avoid, num_experiment, method):
    # use to run multiple experiments and record statistics
    # args = args.parse_args()
    # num_reach = args.num_reach
    # num_avoid = args.num_avoid
    # num_experiment = args.num_experiment
    # method = args.method

    # initialize the constants
    vf_num = 4                      # vfs dim
    T_horizon = 10                  # option steps
    skill_timesteps = 100           # env steps per option

    smoothing_factor = 100          # slt smoothing factor

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
        env=gym.make('Zones-4-v0', timeout=timeout),
        primitives_path='/app/vfstl/src/GCRL-LTL/zones/models/primitives', 
        goals_representation=get_zone_vector(),
        use_primitves=True,
        rewards=[0, 1],
        device=device,
    )

    # initialize the dynamics
    model = VFDynamicsMLPLegacy(vf_num)
    model.load_state_dict(torch.load("/app/vfstl/src/VFSTL/dynamic_models/test_model_20240307_085639_11", map_location=device))
    dynamics = VFDynamics(model.to(device), vf_num)

    # Create a folder for the current run of experiments
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_folder = f".data/{method}_experiment_results_{timestamp}"
    os.makedirs(save_folder, exist_ok=True)

    # Initialize accumulators for statistics
    total_pred_vfs_stl = 0
    total_gt_vfs_stl = 0
    total_state_stl = 0
    total_success = 0
    total_collide = 0

    # Open a file to record each experiment's data
    data_file_path = os.path.join(save_folder, "experiment_data.txt")

    for _ in range(num_experiment):
        # generate random task
        target_zones, avoid_zones = generate_task(num_reach, num_avoid, 4)

        # reset the environment 
        obs = env.reset()
        init_vfs = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, policy_model.policy, get_zone_vector(), device))).to(device)

        # initialize the stl formulas
        vfs_stl = reach_avoid_vfs(T_horizon, target_zones, avoid_zones, 0.8, 0.2)
        state_stl = reach_avoid_states(env, T_horizon*skill_timesteps, target_zones, avoid_zones)
        
        if method == 'stl':
            open_loop_options = STLMCTSOpenLoopOptions(dynamics, T_horizon, vfs_stl, device)
        elif method == 'SparseR':
            reward_env = SparseReachAvoidReward(target_zones, avoid_zones, max_step=100)            # TODO: need to combine reach and reach_avoid
            open_loop_options = RewardMCTSOpenLoopOptions(dynamics, reward_env, device, T_horizon)
        elif method == 'DenseR':
            reward_env = DenseReachAvoidReward(target_zones, avoid_zones, max_step=100)              # TODO: need to combine reach and reach_avoid
            open_loop_options = RewardMCTSOpenLoopOptions(dynamics, reward_env, device, T_horizon)

        # get the option sequences
        options, vfs_seq, nodes_count = open_loop_options.get_options(init_vfs.cpu())
        vfs_seq  = torch.stack(vfs_seq).unsqueeze(0)                                                # convert to tensor

        # recoidng statistics
        gt_vfs = [init_vfs.cpu()]
        gt_states = [torch.from_numpy(env.env.robot_pos[:2])]

        print(f'Option Sequences: {options}')

        # run one simulation
        for i in range(0, T_horizon):
            frame = env.fix_goal(env.goals[options[i]])
            for j in range(0, skill_timesteps):
                action, _ = policy_model.predict(env.current_observation(), deterministic=True)
                obs, reward, done, info = env.step(action)
                # record state
                gt_states.append(torch.from_numpy(env.env.robot_pos[:2]))
            # record vfs
            real_vfs = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, policy_model.policy, get_zone_vector(), device))).cpu()
            gt_vfs.append(real_vfs)
        

        gt_vfs = torch.stack(gt_vfs).unsqueeze(0)                                                    # convert to tensor
        gt_states = torch.stack(gt_states).unsqueeze(0)                                              # convert to tensor

        # Calculate metrics
        pred_vfs_stl = vfs_stl(vfs_seq, smoothing_factor)[:, 0].item()
        gt_vfs_stl = vfs_stl(gt_vfs, smoothing_factor)[:, 0].item()
        state_stl_val = state_stl(gt_states, smoothing_factor)[:, 0].item()


        # convert gt_states to numpy
        gt_states = gt_states.squeeze(0).numpy()
        success, collide, complete_tasks = is_success(env, gt_states, target_zones, avoid_zones)

        # Record metrics
        with open(data_file_path, "a") as data_file:
            data_file.write(f"target_zones: {target_zones}, \
                            avoid_zones: {avoid_zones}, \
                            pred_vfs_stl: {pred_vfs_stl}, gt_vfs_stl: {gt_vfs_stl}, \
                            state_vfs_stl: {state_stl_val}, \
                            num_nodes: {nodes_count}, \
                            is_success: {1 if success else 0}, \
                            complete_tasks: {complete_tasks}, \
                            number_collide: {collide}\n")

        # Update accumulators
        total_pred_vfs_stl += pred_vfs_stl
        total_gt_vfs_stl += gt_vfs_stl
        total_state_stl += state_stl_val
        total_success += 1 if success else 0
        total_collide += collide

    # Calculate averages and success rate
    avg_pred_vfs_stl = total_pred_vfs_stl / num_experiment
    avg_gt_vfs_stl = total_gt_vfs_stl / num_experiment
    avg_state_stl = total_state_stl / num_experiment
    success_rate = total_success / num_experiment
    avg_collide = total_collide / num_experiment

    # Record summary statistics
    summary_file_path = os.path.join(save_folder, "summary_statistics.txt")
    with open(summary_file_path, "w") as summary_file:
        summary_file.write(f"Average pred_vfs_stl: {avg_pred_vfs_stl}\n")
        summary_file.write(f"Average gt_vfs_stl: {avg_gt_vfs_stl}\n")
        summary_file.write(f"Average state_stl: {avg_state_stl}\n")
        summary_file.write(f"Success rate: {success_rate}\n")
        summary_file.write(f"Average number of collide: {avg_collide}\n")