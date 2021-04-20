import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=9*4*4, out_features=72)
        self.fc2 = nn.Linear(in_features=72, out_features=48)
        self.fc3 = nn.Linear(in_features=48, out_features=12)
        self.out = nn.Linear(in_features=12, out_features=3)

    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.fc1(t.reshape(-1, 9*4*4)))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = self.out(t)

        return t
    

