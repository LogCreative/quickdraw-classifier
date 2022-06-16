from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNNet(torch.nn.Module):
    """
    Sketch-a-net is the evaluation method for RPCL-pix2seq.
    Yu et al. Sketch-a-Net that Beats Humans. 2015.
    """
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(1, 64, (15, 15), stride=3)
        self.conv2 = torch.nn.Conv2d(64, 128, (5, 5), stride=1)
        self.conv3 = torch.nn.Conv2d(128, 256, (3, 3), stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(256, 512, (7, 7), stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(512, 512, (1, 1), stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (3, 3), stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (3, 3), stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, (3, 3), stride=2)
        x = F.dropout(F.relu(self.conv6(x)))
        x = F.dropout(F.relu(self.conv7(x)))
        x = x.view(-1, 512)
        
        return x

class RNNNet(torch.nn.Module):
    """A Bidirectional RNN
    introduced from A Neural Representation of Sketch Drawings arxiv:1704.03477
    """
    def __init__(self, opt=None):
        super().__init__()

        self.hidden_size = 256
        self.batch_size = 64
        self.device = "cpu"
        if opt is not None:
            self.batch_size = opt['batch_size']
            self.device = opt['device']

        self.lstm = torch.nn.LSTM(input_size=5, hidden_size=self.hidden_size,bias=True,
        # dropout=0.3, 
        bidirectional=True)

    def forward(self, x, hidden=None):
        if hidden is None:
            h0 = torch.autograd.Variable(torch.zeros(2, x.size(1), self.hidden_size)).to(self.device)
            c0 = torch.autograd.Variable(torch.zeros(2, x.size(1), self.hidden_size)).to(self.device)
        else:
            h0 = hidden[0]
            c0 = hidden[1]
        
        out, _ = self.lstm(x.float(), (h0, c0))

        # # decode the hidden state of the last time step
        # out = self.linear(out[-1])
        return out[-1] # HIDDEN*2=512

class MergerMLP(torch.nn.Module):
    def __init__(self, opt=None) -> None:
        super().__init__()

        self.num_classes = 25
        if opt is not None:
            self.num_classes = opt['num_classes']

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class Net(torch.nn.Module):
    def __init__(self,opt=None) -> None:
        super().__init__()

        self.cnn = CNNNet()
        self.rnn = RNNNet(opt)
        self.mlp = MergerMLP(opt)
    
    def forward(self, img, seq):
        cnn_x = self.cnn(img) # img
        rnn_x = self.rnn(seq) # seq
        x = self.mlp(torch.concat([cnn_x,rnn_x], dim=1))
        return x

if __name__=='__main__':
    pass