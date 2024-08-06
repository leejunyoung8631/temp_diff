import torch
import torch.nn as nn
import torch.nn.functional as F


class PowerNN(nn.Module):
    def __init__(self, in_features = 15, hidden = 64):
        super(PowerNN, self).__init__()
        '''it consist of 3 layers NN with relu function'''
        self.hidden = hidden
        self.in_features = in_features
        self.out = 1 # power value
        self.fc1 = nn.Linear(in_features, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        # self.fc2_1 = nn.Linear(self.hidden, self.hidden)
        # self.fc2_2 = nn.Linear(self.hidden, self.hidden)
        # self.fc2_3 = nn.Linear(self.hidden, self.hidden)
        # self.fc2_4 = nn.Linear(self.hidden, self.hidden)
        # self.fc2_5 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.out)
        
    def forward(self, x):
        '''here, do not use relu'''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc2_1(x))
        # x = F.relu(self.fc2_2(x))
        x = self.fc3(x)
        return x