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
        self.batch_size = 100
        self.device = "cpu"
        if opt is not None:
            self.num_classes = opt['num_classes']
            self.hidden_size = opt['hidden_size']
            self.batch_size = opt['batch_size']
            self.device = opt['device']

        self.lstm = torch.nn.LSTM(input_size=5, hidden_size=self.hidden_size,bias=True,
        # dropout=0.3, 
        bidirectional=True)
        self.linear = torch.nn.Linear(self.hidden_size*2, self.num_classes)

    def forward(self, x):
        x = self.get_batch(x, self.batch_size)
        h0 = torch.autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_size)).to(self.device)
        c0 = torch.autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_size)).to(self.device)

        out, _ = self.lstm(x.float(), (h0, c0))

        # decode the hidden state of the last time step
        out = self.linear(out[-1])
        return out

    def get_batch(self, data, batch_size):
        # Sampling
        idxs = np.random.choice(len(data),batch_size)
        batch_strokes = [data[idx] for idx in idxs]
        batch = torch.autograd.Variable(torch.from_numpy(np.stack(batch_strokes, 1)).float())
        # return batch, lengths
        return batch


def create_model(opt):
    return Net(opt)