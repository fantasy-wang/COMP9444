# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.lin1 = nn.Linear(2, num_hid)
        self.lin2 = nn.Linear(num_hid, 1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        # INSERT CODE HERE

    def forward(self, input):
        input_x = input[:, 0]
        input_y = input[:, 1]
        r = torch.sqrt(input_x*input_x+input_y*input_y).reshape(-1, 1)
        a = torch.atan2(input_y, input_x).reshape(-1, 1)
        concat = torch.cat((r, a), 1)
        lin1 = self.lin1(concat)
        self.hid1 = self.tanh(lin1)
        lin2 = self.lin2(self.hid1)
        result = self.sig(lin2)
        #output = 0*input[:,0] # CHANGE CODE HERE
        return result

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.lin1 = nn.Linear(2, num_hid)
        self.lin2 = nn.Linear(num_hid, num_hid)
        self.lin3 = nn.Linear(num_hid, 1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        # INSERT CODE HERE

    def forward(self, input):
        lin1 = self.lin1(input)
        self.hid1 = self.tanh(lin1)
        lin2 = self.lin2(self.hid1)
        self.hid2 = self.tanh(lin2)
        lin3 = self.lin3(self.hid2)
        result = self.sig(lin3)
        #output = 0*input[:,0] # CHANGE CODE HERE
        return result

class ShortNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()
        self.lin1 = nn.Linear(2, num_hid)
        self.lin2 = nn.Linear(num_hid, num_hid)
        self.lin3 = nn.Linear(num_hid, 1)
        self.lin4 = nn.Linear(2, 1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        # INSERT CODE HERE

    def forward(self, input):
        self.hid1 = self.tanh(self.lin1(input))
        self.hid2 = self.tanh(self.lin2(self.hid1)+self.lin1(input))
        result = self.sig(self.lin3(self.hid2)+self.lin3(self.hid1)+self.lin4(input))
        #output = 0*input[:,0] # CHANGE CODE HERE
        return result


def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1)

    with torch.no_grad():  # suppress updating of gradients
        net.eval()  # toggle batch norm, dropout
        net(grid)
        if layer == 1:
            result = net.hid1[:, node]
            pred = (result >= 0).float()
        elif layer == 2:
            result = net.hid2[:, node]
            pred = (result >= 0).float()
        #net.train()  # toggle batch norm, dropout back again

        #pred = (output >= 0.5).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange, yrange, pred.cpu().view(yrange.size()[0], xrange.size()[0]), cmap='Wistia')
