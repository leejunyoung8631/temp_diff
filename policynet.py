import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN_FC(nn.Module):
    def __init__(self, in_dim, n_actions):
        super(DQN_FC, self).__init__()
        self.in_features = in_dim
        self.n_actions = n_actions
        
        
        self.fc1 = nn.Linear(self.in_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256,256)
        self.bn3 = nn.BatchNorm1d(256)
        # self.fc4 = nn.Linear(256,256)
        # self.bn4 = nn.BatchNorm1d(256)
        # self.fc5 = nn.Linear(256,256)
        # self.bn5 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256,64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc7 = nn.Linear(64, self.n_actions)
        
        
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        # x = F.relu(self.bn4(self.fc4(x)))
        # x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = self.fc7(x)
        
        return x
    
    

class DQN_LSTM(nn.Module):
    def __init__(self, in_dim, n_actions):
        super(DQN_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=128, num_layers=3, batch_first=True)
        self.fc1 = nn.Linear(128, 128)
        self.bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, n_actions)
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2) # for input shape of lstm
        
        x, (hidden_state, cell_state) = self.lstm(x)
        x = x[:, -1, :] # last hidden state
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x