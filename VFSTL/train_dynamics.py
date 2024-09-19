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
        
def get_index_from_multidiensional_one_hot(multidimensional_one_hot):
    ## one_hot_state: M*N, M is the number of dimension,  N is the number of discrete states
    ## the total number of states is N^M
    ## return the index of the one_hot_state in the range of N^M
    ## multidimensional_one_hot is a one_hot encoding of a multi-dimensional discrete state
    # Get the dimensions of the multidimensional one-hot tensor
    M, N = multidimensional_one_hot.shape
    
    # Convert the one-hot encoding to indices for each dimension
    indices = torch.argmax(multidimensional_one_hot, dim=1)
    
    # Calculate the index in the flattened space
    index = 0
    for i in range(M):
        index += indices[i] * (N ** (M - i - 1))
    
    return index

def get_multidimensional_one_hot_from_index(index, num_discrete_states, num_dimensions):
    ## index is the index in the range of N^M
    ## num_discrete_states is the number of discrete states for each dimension
    ## return a one_hot encoding of the index in the range of N^M
    
    # Calculate the indices for each dimension
    indices = []
    for i in range(num_dimensions):
        indices.append(index % num_discrete_states)
        index //= num_discrete_states
    
    # Create the one-hot encoding
    one_hot = torch.zeros(num_dimensions, num_discrete_states)
    for dim, idx in enumerate(reversed(indices)):
        one_hot[dim, idx] = 1
    
    return one_hot

def get_one_hot_from_multidimensional_index(index, num_discrete_states, num_dimensions):
    # Calculate the total number of possible states
    total_states = num_discrete_states ** num_dimensions
    
    # Initialize the one-hot tensor as a one-dimensional array
    one_hot = torch.zeros(total_states)
    
    # Set the corresponding index to 1
    one_hot[index] = 1
    
    return one_hot

class DiscreteVFDynamicsMLP(nn.Module):
    def __init__(self, num_discrete_states, num_discrete_actions, ) -> None:
        super().__init__()
        self.num_discrete_states = num_discrete_states
        self.num_discrete_actions = num_discrete_actions
        self.output_size = num_discrete_states ** num_discrete_actions
        # input : [action, state], state is an index
        # output: probabilities for each possible multi-dimensional state
        self.model = nn.Sequential(
            nn.Linear(num_discrete_actions + 1, 1024),  # +1 for the state index
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        with torch.no_grad():
            probabilities = self.forward(x)
            return torch.argmax(probabilities, dim=-1)
        
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


    
def discrete_state_to_one_hot(states, num_discrete_states):
    # states: N*M, N is the number of samples, M is the number of value functions
    # clip vfs to range [-1, 1]
    states_clipped = torch.clamp(states, -1, 1)
    # scale vfs to range [0, num_discrete_states-1] and then one_hot encode
    states_scaled = ((states_clipped + 1) * (num_discrete_states - 1) / 2).to(torch.int64)
    vfs_one_hot = nn.functional.one_hot(states_scaled, num_classes=num_discrete_states)
    
    return vfs_one_hot

def one_hot_to_one_dimension(one_hot_states):
    # one_hot_states: N*M*K, N is the number of samples, M is the number of value functions, K is the number of discrete states
    return one_hot_states.view(one_hot_states.size()[0], -1)

class VFDynamicDatasetDiscrete(Dataset):
    def __init__(self, data: np.ndarray, num_vf:int, num_discrete_states) -> None:
        self.num_vf = num_vf
        self.num_discrete_states = num_discrete_states
        self.vf_class = nn.functional.one_hot(torch.from_numpy(data[:, 0]).to(torch.int64), num_classes=self.num_vf)
        self.v_t = data[:, 1:num_vf+1]
        self.v_next_t = data[:, num_vf+1:2*num_vf+1]
        # the v_t and v_next_t should be  one_hot encoded
        self.v_t_one_hot = discrete_state_to_one_hot(torch.from_numpy(self.v_t).float(), self.num_discrete_states)
        self.v_next_t_one_hot = discrete_state_to_one_hot(torch.from_numpy(self.v_next_t).float(), self.num_discrete_states)
        
    def __len__(self):
        return np.shape(self.v_t)[0]
    
    def __getitem__(self, index):
        vf_class = self.vf_class[index]
        v_t_index = get_index_from_multidiensional_one_hot(self.v_t_one_hot[index]).float()
        input_tensor = torch.cat((vf_class, v_t_index.unsqueeze(0)), dim=0).to(torch.float32)
        
        v_next_t_index = get_index_from_multidiensional_one_hot(self.v_next_t_one_hot[index])
        output_tensor = get_one_hot_from_multidimensional_index(v_next_t_index, self.num_discrete_states, self.num_vf).to(torch.float32)
        
        return input_tensor, output_tensor
class VFDynamicTrainer():
    def __init__(self, model:nn.Module, train_loader, val_loader, epochs, model_name, device) -> None:
        self.device = device
        self.model = model.to(device)
        self.model_name = model_name
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train(self):

        def train_one_epoch(epoch_index, tb_writer):
            running_loss = 0.
            last_loss = 0.
            self.model.to(self.device)
            # Here, we use enumerate(training_loader) instead of
            # iter(training_loader) so that we can track the batch
            # index and do some intra-epoch reporting
            def transform_data_to_discrete_trainning(data):
                inputs, labels = data
                # the input is a list of [action, states], we need to transform it to [action, one diemnsional one_hot_states]
                # the label is a list of states, we need to transform it to one_hot_states
                
                
            for i, data in enumerate(self.train_loader):
                # Every data instance is an input + label pair
                # inputs: [[action, states_one_hot_one_dimension], [action, states_one_hot_one_dimension], ...]
                # action: 1*4, states_one_hot_one_dimension : 1*num_discrete_states*num_vf
                # labels: [states_one_hot_one_dimension, states_one_hot_one_dimension, ...]
                inputs, labels = data
                # print('shape of inputs')
                # print(inputs.shape)
                # print('shape of labels')
                # print(labels.shape)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Zero your gradients for every batch!
                self.optimizer.zero_grad()
                # Make predictions for this batch
                outputs = self.model(inputs)
                # print(outputs[0])
                # print('predicted')
                # print(self.model.predict(inputs))
                # print(inputs[0])
                # print(labels[0].argmax())

                
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

class DiscreteVFDynamics():

    
    def __init__(self, NNModel, size_discrete_actions, num_discrete_states) -> None:
        self.NNModel = NNModel
        self.size_discrete_actions = size_discrete_actions
        self.num_discrete_states = num_discrete_states

    def one_step_simulation(self, controls, vfs) -> None:
        # controls: R*N should a vector of RN for parallel prediction of multiple trajectories
        # vfs: a array of R N*M, with N is the number of samples and M is equal to the total amount of skills
        
        # one_hot to controls
        controls = nn.functional.one_hot(controls.to(torch.int64), num_classes=self.size_discrete_actions)

        # clip vfs to range [-1, 1]
        vfs_clipped = torch.clamp(vfs, -1, 1)
        
        # scale vfs to range [0, num_discrete_states-1] and then one_hot encode
        vfs_scaled = ((vfs_clipped + 1) * (self.num_discrete_states - 1) / 2).to(torch.int64)
        vfs_one_hot = nn.functional.one_hot(vfs_scaled, num_classes=self.num_discrete_states)

        # concat control with vfs
        nn_input = torch.cat((controls, vfs_one_hot), dim=1)
        print('shape of nn_input')
        print(nn_input.shape)
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

# def main():
    # Check if CUDA is available
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:1")
    #     print("CUDA is available. Training on GPU.")
    # else:
    #     device = torch.device("cpu")
    #     print("CUDA is not available. Training on CPU.")
    # train normal dynamics
    # dataset_skipped_steps = 100
    # target_skippped_steps = 100
    # timeout=1000
    # buffer_size=40000
    # random_goal=1
    # max_surfix=60
    # epochs = 1000
    # # n_timesteps_direct
    # # n_timesteps_difference
    # model_name='{}_timsteps_direct'.format(target_skippped_steps)
    # data_paths = create_loading_paths(dataset_skipped_steps, timeout, buffer_size, random_goal, max_surfix)
    # raw_data = combine_load_data(data_paths)
    # num_vf = 4
    # raw_data = pre_process_new_dataset(raw_data,dataset_skipped_steps, target_skippped_steps, num_vf)
    # np.random.shuffle(raw_data)
    # vali_dataset_len = int(np.shape(raw_data)[0] * 0.2)
    # vali_raw_data = raw_data[:vali_dataset_len]
    # train_raw_data = raw_data[vali_dataset_len:]
    # print(np.shape(vali_raw_data))
    # print(np.shape(train_raw_data))
    # print(np.shape(raw_data))
    # train_loader = torch.utils.data.DataLoader(VFDynamicDataset(train_raw_data, num_vf), batch_size=1024, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(VFDynamicDataset(vali_raw_data, num_vf), batch_size=1024, shuffle=True)
    
    
    # dy_model = VFDynamicsMLPWithDropout(num_vf)
    # trainer = VFDynamicTrainer(dy_model, train_loader, val_loader, epochs, model_name, device)
    # trainer.train()
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
    # n_timesteps means the number of skipped steps
    # discrete: if the states is discrete, how many discrete states for each value function
    num_discrete_states = 5  # Example value, adjust as needed
    model_name='{}_timsteps_direct_discrete_{}'.format(target_skippped_steps, num_discrete_states)
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
    train_loader = torch.utils.data.DataLoader(VFDynamicDatasetDiscrete(train_raw_data, num_vf, num_discrete_states), batch_size=1024, shuffle=True)
    val_loader = torch.utils.data.DataLoader(VFDynamicDatasetDiscrete(vali_raw_data, num_vf, num_discrete_states), batch_size=1024, shuffle=True)
    
    dy_model = DiscreteVFDynamicsMLP(num_discrete_states, num_vf)
    trainer = VFDynamicTrainer(dy_model, train_loader, val_loader, epochs, model_name, device)
    trainer.train()
    
def test_discrete_states():
    num_discrete_states = 100  # Example value, adjust as needed
    num_discrete_actions = 4  # Example value, adjust as needed
    num_vf = 4
    num_samples = 2
    states = torch.rand(num_samples, num_vf) * 2 - 1  # Random states in the range [-1, 1]
    actions = torch.randint(0, num_discrete_actions, (num_samples,))  # Random actions in the range [0, num_discrete_actions-1]

    one_hot_states = discrete_state_to_one_hot(states, num_discrete_states)
    # transform one_hot_states to one dimension
    one_hot_states = one_hot_states.view(num_samples, -1)
    print(one_hot_states.shape)
    
if __name__ == "__main__":
    # test_discrete_states()
    main()
    #test_random_shooting()
    
