import argparse

if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='mcts') # mcts and random shoot
    parser.add_argument('--policy_model_path', type=str, default='')