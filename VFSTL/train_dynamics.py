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
import math
sys.path.append("/app/vfstl/src/GCRL-LTL/zones")
import pandas as pd
from envs import ZoneRandomGoalEnv
from envs.utils import get_zone_vector
from rl.traj_buffer import TrajectoryBufferDataset

def one_hot_encoding(arr:np.ndarray, num_class:int):
    encoded_arr = np.zeros((arr.size, num_class), dtype=int)
    encoded_arr[np.arange(arr.size),arr] = 1
    return encoded_arr


class VFDynamicsMLPLegacy(nn.Module):
    def __init__(self, state_dim) -> None:
        super().__init__()
        
        # input : one_hot + all vfs
        # output: all vfs
        self.model = nn.Sequential(
            nn.Linear(2*state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, state_dim)
        )

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        with torch.no_grad():
            return self.model(x)

# pleas ust this class for model trained after 3.20
class VFDynamicsMLPWithDropout(nn.Module):
    def __init__(self, state_dim) -> None:
        super().__init__()
        # input : one_hot + all vfs
        # output: all vfs
        self.model = nn.Sequential(
            nn.Linear(2*state_dim, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, state_dim)
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
                if i % 10 == 9:
                    last_loss = running_loss / 10 # loss per batch
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
                model_path = '/app/vfstl/src/VFSTL/dynamic_models/new_{}_model_{}_{}'.format(self.model_name,timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1


class VFDynamics():

    def __init__(self, NNModel, size_discrete_actions) -> None:
        self.NNModel = NNModel
        self.size_discrete_actions = size_discrete_actions

    def one_step_simulation(self, controls, vfs) -> None:
        # controls: R*N should a vector of RN for parallel prediction of multiple trajectories
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
        vf_prediction = torch.zeros((batch_num, timesteps, init_vfs.size()[0]), device=prev_vfs.device)
        
        for i in range(0, timesteps):
            vfs = self.one_step_simulation(control_seqs[:, i], prev_vfs)
            prev_vfs = vfs
            vf_prediction[:,i,:] = vfs
        
        return vf_prediction



def combine_load_data(paths):
    data = []
    for path in paths:
        data.append(np.load(path))
    return np.concatenate(data)

def pre_process_new_dataset(raw_data,dataset_skipped_steps, target_skippped_steps, num_vf):
    # data size n*(1+num_vf)
    # return data size  (dataset_dynamic_steps-dynamic_skipped_steps+1) * (n / dataset_dynamic_steps) + 2* num_vf: skill, pre_vf, after_f 
    n = raw_data.shape[0]
    epoch = math.floor(n / dataset_skipped_steps)
    num_samples_each_epoch = dataset_skipped_steps - target_skippped_steps + 1 
    new_dataset = np.zeros((epoch*num_samples_each_epoch, 1+2*num_vf))
    new_dataset_index = 0
    for i in range(epoch):
        for j in range(num_samples_each_epoch):
            pre_vf_index = i * target_skippped_steps + j
            after_vf_index = i * target_skippped_steps + j + target_skippped_steps - 1
            new_dataset[new_dataset_index, 0] = raw_data[pre_vf_index, 0]
            new_dataset[new_dataset_index, 1:num_vf+1] = raw_data[pre_vf_index, 1:num_vf+1]   
            new_dataset[new_dataset_index, num_vf+1 : 2*num_vf+1] = raw_data[after_vf_index, 1:num_vf+1]          
            new_dataset_index+=1
    # training_dataset = pd.DataFrame(raw_data)
    # training_dataset.to_csv("{}_to_{}_training_data.csv".format(dataset_skipped_steps, target_skippped_steps))
    return new_dataset


def create_loading_paths(skipped_steps, timeout, buffer_size, random_goal, max_surfix):
    paths = []
    for i in range(0, max_surfix):
        save_path = "/app/vfstl/src/VFSTL/dynamic_models/datasets/new_zone_dynamic_{}_{}_{}_{}_{}.npy".format(
            skipped_steps, timeout, buffer_size, random_goal, i)
        paths.append(save_path)
    return paths

def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        print("CUDA is available. Training on GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training on CPU.")
    
    dataset_skipped_steps = 100
    target_skippped_steps = 100
    timeout=1000
    buffer_size=40000
    random_goal=1
    max_surfix=60
    epochs = 1000
    # n_timesteps_direct
    # n_timesteps_difference
    model_name='{}_timsteps_direct'.format(target_skippped_steps)
    data_paths = create_loading_paths(dataset_skipped_steps, timeout, buffer_size, random_goal, max_surfix)
    raw_data = combine_load_data(data_paths)
    num_vf = 4
    raw_data = pre_process_new_dataset(raw_data,dataset_skipped_steps, target_skippped_steps, num_vf)
    np.random.shuffle(raw_data)
    vali_dataset_len = int(np.shape(raw_data)[0] * 0.2)
    vali_raw_data = raw_data[:vali_dataset_len]
    train_raw_data = raw_data[vali_dataset_len:]
    print(np.shape(vali_raw_data))
    print(np.shape(train_raw_data))
    print(np.shape(raw_data))
    train_loader = torch.utils.data.DataLoader(VFDynamicDataset(train_raw_data, num_vf), batch_size=1024, shuffle=True)
    val_loader = torch.utils.data.DataLoader(VFDynamicDataset(vali_raw_data, num_vf), batch_size=1024, shuffle=True)
    
    
    dy_model = VFDynamicsMLPWithDropout(num_vf)
    trainer = VFDynamicTrainer(dy_model, train_loader, val_loader, epochs, model_name, device)
    trainer.train()
    
if __name__ == "__main__":
    main()
    #test_random_shooting()
    
