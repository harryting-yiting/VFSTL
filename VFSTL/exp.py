import argparse
import torch
from stable_baselines3 import PPO
from VFSTL.vfs_dynamic.train_dynamics import VFDynamicsMLPLegacy
import gym
from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector
from datetime import datetime

from mcts import MCTSController, VFDynamics, MPCMCTSController, stl_cost, test_mcts_controller
from controller_evaluator import ControllerEvaluator, TaskSampler, find_s1_in_s2, get_env_ground_truth_robustness_value, plot_traj_2d



if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='exp')   # exp or plot
    parser.add_argument('--num_tree_nodes', type=int, default=500)   # number of tree nodes to build to find optimal skill sequence
    parser.add_argument('--num_rollout', type=int, default=10000)     # number of rollouts when build leaf nodes
    parser.add_argument('--policy_model_path', type=str, default= '/app/vfstl/src/GCRL-LTL/zones/models/goal-conditioned/best_model_ppo_8')
    parser.add_argument('--primitives_path', type=str, default='/app/vfstl/src/GCRL-LTL/zones/models/primitives')
    parser.add_argument('--env', type=str, default='Zones-8-v0')

    # vfs dynamics
    parser.add_argument('--skill_timesteps', type=int, default=100)
    parser.add_argument('--vfs_dynamic_model', type=str, default="/app/vfstl/src/VFSTL/dynamic_models/test_model_20240307_085639_11")
    
    # evaluator constant
    parser.add_argument('--task', type=str, default='chain')
    parser.add_argument('--T_horizon', type=int, default=11)
    parser.add_argument('--num_exp', type=int, default=100)

    # data save path
    parser.add_argument('--res_path', type=str, default='/app/vfstl/src/VFSTL/controller_evaluation_result/{}.pt')

    # stl for plot
    parser.add_argument('--stl', type=str, default='eventually[0,5] (J0 >= 0.8)')

    args = parser.parse_args()

    if args.type == 'exp':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA is available. Training on GPU.")
        else:
            device = torch.device("cpu")
            print("CUDA is not available. Training on CPU.")

        policy_model = PPO.load(args.policy_model_path, device=device)
        timeout = 10000
        env = ZoneRandomGoalEnv(
            env=gym.make(args.env, timeout=timeout), 
            primitives_path=args.primitives_path, 
            goals_representation=get_zone_vector(),
            use_primitves=True,
            rewards=[0, 1],
            device=device,
        )

        with torch.no_grad():
            model = VFDynamicsMLPLegacy(len(env.goals)).to(device=device)
            model.load_state_dict(torch.load(args.vfs_dynamic_model, map_location=device))
            dynamics = VFDynamics(model.to(device), len(env.goals))
            controller = MPCMCTSController(args.skill_timesteps, policy_model, dynamics, env.goals, args.T_horizon, args.num_rollout, args.num_tree_nodes, device)
            evaluator = ControllerEvaluator(controller, env)

            robs, stl_cs, truth, vfs_robs = evaluator.random_evaluate(args.task, args.num_exp, device)

            robs = torch.stack(robs)
            vfs_robs = torch.stack(vfs_robs)

            date = datetime.now().strftime('%Y%m%d_%H%M%S')
            torch.save(truth, args.res_path.format('zone_truth_mcts_mpc_{}_{}'.format(args.task, date)))
            torch.save(robs, args.res_path.format('ground_truth_mcts_mpc_{}_{}'.format(args.task,date)))
            torch.save(vfs_robs, args.res_path.format('vfs_robs_mcts_mpc_{}_{}'.format(args.task, date)))

    elif args.type == 'plot':
        test_mcts_controller(args.stl)