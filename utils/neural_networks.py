from matplotlib.pyplot import jet
import torch
import torch.nn as nn
from torch.nn.modules.linear import Identity
from torch.nn.utils import weight_norm
import math
import torch.functional as F


class mlp(nn.Module):
    def __init__(self, n_neurons, activation=nn.ReLU(), output_activation=nn.Identity(),
                all_actions=False):
        super(mlp, self).__init__()
        layers = []
        for i,j in zip(n_neurons[:-1],n_neurons[1:]):
            layers.append(nn.Linear(i, j))
            layers.append(activation)
        layers[-1] = output_activation
        self.linear =  nn.Sequential(*layers)

    def forward(self, state, actions):
        return self.linear(torch.cat((state,actions.reshape(state.size(0),-1)),dim=1))


# class mlp(nn.Module):
#     def __init__(self, n_neurons, activation=nn.ReLU(), output_activation=nn.Identity(),
#                 all_actions=False):
#         super(mlp, self).__init__()
#         self.first_linear = nn.Linear(n_neurons[0], n_neurons[1])
#         if len(n_neurons)>2:
#             layers = []
#             for i,j in zip(n_neurons[1:-1],n_neurons[2:]):
#                 layers.append(nn.Linear(i, j))
#                 layers.append(activation)
#             layers[-1] = output_activation
#             self.linear =  nn.Sequential(*layers)
#         else:
#             self.linear = nn.Identity()
#         # if all_actions:
#         #     self.mask = torch.triu(torch.ones())
#         # else:
#         #     self.mask = nn.Identity()

#     def forward(self, state, actions):
#         # F.linear(input, self.weight, self.bias)
#         return self.linear(torch.cat((state,actions.reshape(state.size(0),-1)),dim=1))


class cnn(nn.Module):
    @staticmethod
    def output_size(l_in, n_channels, kernel_size):
        return l_in - (kernel_size-1)*len(n_channels)


    def __init__(self, n_channels, n_neurons, kernel_size, activation=nn.ReLU(), output_activation=nn.Identity()):
        super(cnn, self).__init__()
        layers = []
        for i,j in zip(n_channels[:-1],n_channels[1:]):
            layers.append(nn.Conv1d(i,j,kernel_size=kernel_size))
            layers.append(activation)
        self.conv = nn.Sequential(*layers)

        layers = []
        for i,j in zip(n_neurons[:-1],n_neurons[1:]):
            layers.append(nn.Linear(i, j))
            layers.append(activation)
        layers[-1] = output_activation
        self.linear = nn.Sequential(*layers)

    def forward(self, state, actions):
        encoded_actions = self.conv(actions.transpose(1,2))
        return self.linear(torch.cat((state,encoded_actions.reshape(state.size(0),-1)),dim=1))

         