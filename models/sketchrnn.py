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
        self.hidden_size = 10
        self.device = "cpu"
        if opt is not None:
            self.num_classes = opt['num_classes']
            self.hidden_size = opt['hidden_size']
            self.device = opt['device']

        self.lstm = torch.nn.LSTM(input_size=5, hidden_size=self.hidden_size,bias=True,
        # dropout=0.3, 
        bidirectional=True)
        self.linear = torch.nn.Linear(self.hidden_size*2, self.num_classes)

    def forward(self, x, hidden):
        if hidden is None:
            h0 = torch.autograd.Variable(torch.zeros(2, x.size(1), self.hidden_size)).to(self.device)
            c0 = torch.autograd.Variable(torch.zeros(2, x.size(1), self.hidden_size)).to(self.device)
        else:
            h0 = hidden[0]
            c0 = hidden[1]

        out, _ = self.lstm(x.float(), (h0, c0))

        out = out[-1]

        # decode the hidden state of the last time step
        out = self.linear(out[-1])
        return out


def create_model(opt):
    return Net(opt)