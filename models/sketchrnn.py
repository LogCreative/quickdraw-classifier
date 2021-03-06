import torch
import torch.nn.functional as F
import numpy as np

class Net(torch.nn.Module):
    """A Bidirectional RNN
    introduced from A Neural Representation of Sketch Drawings arxiv:1704.03477
    """
    def __init__(self, opt=None):
        super().__init__()

        self.num_classes = 25
        self.hidden_size = 256
        self.Nz = 128
        self.cls_hidden_size = 128
        self.dropout = 0.8
        self.device = "cpu"
        if opt is not None:
            self.num_classes = opt['num_classes']
            self.hidden_size = opt['hidden_size']
            self.Nz = opt['Nz']
            self.cls_hidden_size = opt['cls_hidden_size']
            self.dropout = opt['dropout']
            self.device = opt['device']

        self.lstm = torch.nn.LSTM(input_size=5, hidden_size=self.hidden_size,bias=True,
        # dropout=0.3, 
        bidirectional=True)

        self.fc = torch.nn.Linear(self.hidden_size*2, self.num_classes)

        # self.fc_mu = torch.nn.Linear(self.hidden_size * 2, self.Nz)
        # self.fc_sigma = torch.nn.Linear(self.hidden_size * 2, self.Nz)

        # # classifier network
        # self.fc1 = torch.nn.Linear(self.Nz, self.cls_hidden_size)
        # self.drop1 = torch.nn.Dropout(self.dropout)
        # self.fc2 = torch.nn.Linear(self.cls_hidden_size, self.cls_hidden_size)
        # self.drop2 = torch.nn.Dropout(self.dropout)
        # self.fc3 = torch.nn.Linear(self.cls_hidden_size, self.num_classes)

    def forward(self, x, hidden=None):
        if hidden is None:
            h0 = torch.autograd.Variable(torch.zeros(2, x.size(1), self.hidden_size)).to(self.device)
            c0 = torch.autograd.Variable(torch.zeros(2, x.size(1), self.hidden_size)).to(self.device)
        else:
            h0 = hidden[0]
            c0 = hidden[1]

        out, _ = self.lstm(x.float(), (h0, c0))
        # decode the hidden state of the last time step
        out = out[-1]

        # mu = self.fc_mu(out)
        # sigma = torch.exp(self.fc_sigma(out) / 2.0)
        # noise = torch.normal(torch.zeros(mu.shape), torch.ones(mu.shape)).to(self.device)
        # out = mu + sigma * noise

        # out = self.drop1(self.fc1(out))
        # out = self.drop2(self.fc2(out))
        # out = self.fc3(out)

        out = self.fc(out)

        return out


def create_model(opt):
    return Net(opt)