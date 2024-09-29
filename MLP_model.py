"""

"""
from math import isclose

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """

    """
    def __init__(self, inp_dim:int, n_hidden_layers:int, n_neurons:int, device:str='cuda:0'):
        super().__init__()
        self.Inp_layers = nn.Linear(inp_dim, n_neurons)
        self.Hidden_layers = nn.ModuleList([nn.Linear(n_neurons, n_neurons) for _ in range(n_hidden_layers)])
        self.Out_layers = nn.Linear(n_neurons, 1)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = F.relu(self.Inp_layers(x))
        for _layer in self.Hidden_layers:
            x = F.relu(_layer(x))
        x:th.Tensor = self.Out_layers(x)
        return x.squeeze()

    def predict(self, x):
        if not isinstance(x, np.ndarray):
            x = th.from_numpy(x)
        elif not isinstance(x, th.Tensor):
            x = th.tensor(x)
        x = x.to(self.device)
        #self.to(self.device)
        self.eval()
        with th.no_grad():
            y:th.Tensor = self.forward(x)
        self.train()
        return y.numpy(force=True)

class TrainMLP:
    """

    """
    def __init__(self, model: nn.Module, model_name='untitled', batch_size:int=64, lr:float=0.001, val_ratio:float=0.1, n_iter=500, device='cuda:0'):
        self.Optim = th.optim.Adam(model.parameters(), lr=lr)
        self.model = model
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.model_name = model_name
        assert 0. < val_ratio < 1., ValueError('val_ratio must be between 0 and 1.')
        self.val_ratio = val_ratio
        self.device = device

    def run(self, n_epoch:int, data:th.Tensor, label:th.Tensor, shuffle:bool=True, save_thres=5.):
        self.model.train()
        loss_fn = nn.MSELoss()
        val_fn = nn.L1Loss()
        data = data.to(self.device)
        label = label.to(self.device)
        # shuffle
        if shuffle:
            shuff_indx = th.randperm(len(data), )
            data = data[shuff_indx]
            label = label[shuff_indx]
        # split validation set
        if not isclose(self.val_ratio, 0.):
            n_val = round(self.val_ratio * len(data))
            data = data[:-n_val]
            label = label[:-n_val]
            val_data = data[-n_val:]
            val_lable = label[-n_val:]
            val_data = th.split(val_data, self.batch_size)
            val_lable = th.split(val_lable, self.batch_size)
            old_val_error = save_thres
        else:
            val_data = None
        data = th.split(data, self.batch_size)
        label = th.split(label, self.batch_size)
        for i in range(n_epoch):
            print(f'epoch {i+1:<6d}: ')
            # shuffle
            if shuffle:
                data = th.cat(data, dim=0)
                label = th.cat(label)
                shuff_indx = th.randperm(len(data), )
                data = data[shuff_indx]
                label = label[shuff_indx]
                data = th.split(data, self.batch_size)
                label = th.split(label, self.batch_size)
            # train
            self.model.train()
            for j, _dat in enumerate(data):
                output = self.model(_dat)
                loss = loss_fn(output, label[j])
                # optimize
                self.Optim.zero_grad()
                loss.backward()
                self.Optim.step()
                # print
                print(f'\tbatch {j+1:<6d}, loss: {loss.item():>.4e}, MAE: {val_fn(output, label[j]).item():>.4e}')
            # val
            if val_data is not None:
                print('\tValidation ...')
                self.model.eval()
                with th.no_grad():
                    val_loss, val_error = 0., 0.
                    for j, _dat in enumerate(val_data):
                        output_ = self.model(_dat)
                        val_loss += loss_fn(output_, val_lable[j]).item()
                        val_error += val_fn(output_, val_lable[j]).item()
                    # print
                    print(f'\t\tval_loss: {val_loss/len(val_data):>.4e}, val_MAE: {val_error/len(val_data):>.4e}')
                if val_error/len(val_data) < old_val_error:
                    old_val_error = val_error/len(val_data)
                    th.save(self.model, f'MLP_models/MLP_valError_{val_error/len(val_data)}_{self.model_name}.pt', )
        th.save(self.model, f'MLP_models/MLP_last_{self.model_name}.pt')
