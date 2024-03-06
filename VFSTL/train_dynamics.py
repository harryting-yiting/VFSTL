import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import gym
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset
import sys
from collect_skill_trajectories import get_all_goal_value, from_real_dict_to_vector
from stable_baselines3 import PPO

sys.path.append("/app/vfstl/src/GCRL-LTL/zones")
from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector
from rl.traj_buffer import TrajectoryBufferDataset

def one_hot_encoding(arr:np.ndarray, num_class:int):
    encoded_arr = np.zeros((arr.size, num_class), dtype=int)
    encoded_arr[np.arange(arr.size),arr] = 1
    return encoded_arr


class VFDynamicsMLP(nn.Module):
    def __init__(self, state_dim) -> None:
        super().__init__()
        
        # input : one_hot + all vfs
        # output: all vfs
        self.model = nn.Sequential(
            nn.Linear(2*state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        with torch.no_grad():
            return self.model(x)


class VFDynamicDataset(Dataset):
    def __init__(self, data: np.ndarray, num_vf:int ) -> None:
        self.num_vf = num_vf
        self.vf_class = nn.functional.one_hot(torch.from_numpy(data[:, 0]).to(torch.int64), num_classes= self.num_vf)
        self.v_t = data[:, 1:num_vf+1]
        self.v_next_t = data[:, num_vf+1:2*num_vf+1]
        
    def __len__(self):
        return np.shape(self.v_t)[0]
    
    def __getitem__(self, index):
        return torch.concatenate((self.vf_class[index], 
                                  torch.from_numpy(self.v_t[index]))).to(torch.float32), torch.from_numpy(self.v_next_t[index]).to(torch.float32)
    
    
class VFDynamicTrainer():
    def __init__(self, model:nn.Module, train_loader, val_loader, epochs, model_name, device) -> None:
        self.device = device
        self.model = model.to(device)
        self.model_name = model_name
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    
    def train(self):

        def train_one_epoch(epoch_index, tb_writer):
            running_loss = 0.
            last_loss = 0.
            self.model.to(self.device)
            # Here, we use enumerate(training_loader) instead of
            # iter(training_loader) so that we can track the batch
            # index and do some intra-epoch reporting
            for i, data in enumerate(self.train_loader):
                # Every data instance is an input + label pair
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Zero your gradients for every batch!
                self.optimizer.zero_grad()

                # Make predictions for this batch
                outputs = self.model(inputs)

                # Compute the loss and its gradients
                loss = self.loss_fn(outputs, labels)
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                if i % 1000 == 999:
                    last_loss = running_loss / 1000 # loss per batch
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                    tb_x = epoch_index * len(self.train_loader) + i + 1
                    tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                    running_loss = 0.

            return last_loss
        
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('/app/vfstl/src/VFSTL/dynamic_models/runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0

        best_vloss = 1_000_000.

        for epoch in range(self.epochs):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = train_one_epoch(epoch_number, writer)


            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.val_loader):
                    vinputs, vlabels = vdata
                    vinputs, vlabels = vinputs.to(self.device), vlabels.to(self.device)
                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = '/app/vfstl/src/VFSTL/dynamic_models/{}_model_{}_{}'.format(self.model_name,timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1


class VFDynamics():

    def __init__(self, NNModel, size_discrete_actions) -> None:
        self.NNModel = NNModel
        self.size_discrete_actions = size_discrete_actions

    def one_step_simulation(self, controls, vfs) -> None:
        # controls: R N should a vector of RN for parallel prediction of multiple trajectories
        # vfs: a array of R N*M, with N is the number of samples and M is equal to the total amount of skills
        
        # one_hot to controls
        controls = nn.functional.one_hot(controls.to(torch.int64), num_classes= vfs.size()[1])

        # concat control with vfs
        nn_input = torch.concatenate((controls, vfs), 1)
        # feed them into NN and get prediction
        
        return self.NNModel(nn_input.to(torch.float32))

    def forward_simulation(self, control_seqs, init_vfs):
        # control_seqs: R: N * T, N: batch size, T: timesteps
        # init_vfs: R M, M: number of value functions
        # return N* T * M, no initial_vfs
        batch_num = control_seqs.size()[0]
        timesteps = control_seqs.size()[1]
        # s_t+1 = NN(st)
        
        prev_vfs = init_vfs.repeat((batch_num, 1))
        vf_prediction = torch.zeros((batch_num, timesteps, init_vfs.size()[0]))
        
        for i in range(0, timesteps):
            vfs = self.one_step_simulation(control_seqs[:, i], prev_vfs)
            prev_vfs = vfs
            vf_prediction[:,i,:] = vfs
        
        return vf_prediction


class RandomShootingOptimization():

    def __init__(self, dynamics, cost_fn, constraints, timesteps) -> None:
        self.dynamics = dynamics
        self.cost_fn = cost_fn
        self.constraints = constraints
        self.timesteps = timesteps

    def optimize(self, num_sample_batches, batch_size, multiprocessing, init_state, device):
        # return the best sample and there costs
        mini_control = torch.randint(0, self.dynamics.size_discrete_actions, (self.timesteps,), device=device)
        mini_state = []
        mini_cost = 1000000
        for i in range(0, num_sample_batches):
            # generate random action sequence with batch_size * timesteps
            controls = torch.randint(0, self.dynamics.size_discrete_actions, (batch_size, self.timesteps), device=device) 
            # run simulation
            states = self.dynamics.forward_simulation(controls, init_state)
            costs = self.cost_fn(states)
            mini_index = torch.argmin(costs)
             # get best control and cost
            if costs[mini_index] < mini_cost:
                mini_cost = costs[mini_index]
                mini_control = controls[mini_index]
                mini_state = states[mini_index]
        
        return mini_control, mini_state, mini_cost


def test_random_shooting():
        # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        print("CUDA is available. Training on GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training on CPU.")

    def cost_fn(state):
        return torch.randn(state.size()[0])
    model_path = '/app/vfstl/src/GCRL-LTL/zones/models/goal-conditioned/best_model_ppo_8'
    model = PPO.load(model_path, device=device)
    timeout = 10000
    env = ZoneRandomGoalEnv(
        env=gym.make('Zones-8-v0', timeout=timeout), 
        primitives_path='/app/vfstl/src/GCRL-LTL/zones/models/primitives', 
        goals_representation=get_zone_vector(),
        use_primitves=True,
        rewards=[0, 1],
        device=device,
    )

    obs = env.reset()
    init_values = torch.from_numpy(from_real_dict_to_vector(get_all_goal_value(obs, model.policy, get_zone_vector(), device))).to(device)

    vf_num = 4
    model = VFDynamicsMLP(vf_num)
    model.load_state_dict(torch.load("/app/vfstl/src/VFSTL/dynamic_models/test_model_20240306_152632_3"))
    dynamics = VFDynamics(model.to(device), 4)
    op = RandomShootingOptimization(dynamics, cost_fn, cost_fn, 10)
    print(op.optimize(1024, 1024, False, init_values, device))
    


def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        print("CUDA is available. Training on GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training on CPU.")
    
    raw_data = np.load("/app/vfstl/src/VFSTL/dynamic_models/datasets/zone_dynamic_100_10000_20_1_0.npy")
    num_vf = 4
    train_dataset_len = 8
    vali_dataset_len = 5
    train_raw_data = raw_data[:train_dataset_len]
    vali_raw_data = raw_data[train_dataset_len: train_dataset_len+vali_dataset_len]
    train_loader = torch.utils.data.DataLoader(VFDynamicDataset(train_raw_data, num_vf), batch_size=2, shuffle=True)
    val_loader = torch.utils.data.DataLoader(VFDynamicDataset(vali_raw_data, num_vf), batch_size=2, shuffle=True)
    
    
    dy_model = VFDynamicsMLP(num_vf)
    trainer = VFDynamicTrainer(dy_model, train_loader, val_loader, 5, "test", device)
    trainer.train()
    
if __name__ == "__main__":
    #main()
    test_random_shooting()
    
