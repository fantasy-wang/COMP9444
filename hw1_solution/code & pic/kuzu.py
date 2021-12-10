# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.lin = nn.Linear(28*28, 10)
        self.log_s = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x_view = x.view(-1, 28*28)
        result = self.log_s(self.lin(x_view))
        return result  # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.first_lin = nn.Linear(28*28, 500)
        self.second_lin = nn.Linear(500, 10)
        self.tanh = nn.Tanh()
        self.log_s = nn.LogSoftmax(dim=1)
        # INSERT CODE HERE

    def forward(self, x):
        x_view = x.view(-1, 28*28)
        hidden = self.tanh(self.first_lin(x_view))
        result = self.log_s(self.second_lin(hidden))
        return result  # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=12,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=12,
            out_channels=20,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.pool = nn.MaxPool2d(2,2)
        self.lin1 = nn.Linear(3920, 2613)
        self.lin2 = nn.Linear(2613, 10)
        self.relu = nn.ReLU(True)
        self.log_s = nn.LogSoftmax(dim=1)


        # INSERT CODE HERE

    def forward(self, x):
        cv1 = self.conv1(x)
        r1 = self.relu(cv1)
        pool1 = self.pool(r1)
        cv2 = self.conv2(pool1)
        r2 = F.relu(cv2)
        pool2 = self.pool(r2)
        pool2_view = pool2.view(pool2.size(0), -1)
        lin1 = self.lin1(pool2_view)
        r3 = F.relu(lin1)
        lin2 = self.lin2(r3)
        result = self.log_s(lin2)
        return result # CHANGE CODE HERE
