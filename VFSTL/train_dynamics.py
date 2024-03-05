import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset

def one_hot_encoding(arr:np.ndarray, num_class:int):
    encoded_arr = np.zeros((arr.size, num_class), dtype=int)
    encoded_arr[np.arange(arr.size),arr] = 1
    return encoded_arr


class VFDynamics(nn.Module):
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
        self.vf_class:int = data[:, 0]
        self.v_t = data[:, 1:num_vf]
        self.v_next_t = data[:, num_vf:2*num_vf]
        
    def __len__(self):
        return np.shape(self.v_t)[0]
    
    def __getitem__(self, index):
        return torch.concatenate((
            nn.functional.one_hot(torch.from_numpy(self.vf_class), torch.from_numpy(self.num_vf)),
            torch.from_numpy(self.v_t[index])), 1), torch.from_numpy(self.v_next_t[index])
    
    
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

def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA is available. Training on GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training on CPU.")
    
    raw_data = np.load("/app/vfstl/src/VFSTL/dynamic_models/datasets/test.npy")

    print(raw_data)
    # num_vf = 10
    # train_dataset_len = 10000
    # vali_dataset_len = 300
    # train_raw_data = torch.rand(train_dataset_len, 4)
    # vali_raw_data = torch.rand(vali_dataset_len, 4)
    # train_loader = torch.utils.data.DataLoader(VFDynamicDataset(train_raw_data), batch_size=4, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(VFDynamicDataset(vali_raw_data), batch_size=4, shuffle=True)
    
    
    # dy_model = VFDynamics(2)
    # trainer = VFDynamicTrainer(dy_model, train_loader, val_loader, 5, "test", device)
    # trainer.train()
    
    # test one_hot
    # print(one_hot_encoding(np.array(([[1,2,3],[1,2,3]])), 4))
    
    
if __name__ == "__main__":
    main()
    
