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

    def forward(self, x, batch_size):
        print(x.size(0))
        h0 = torch.autograd.Variable(torch.zeros(2, batch_size, self.hidden_size)).to(self.device)
        c0 = torch.autograd.Variable(torch.zeros(2, batch_size, self.hidden_size)).to(self.device)

        out, _ = self.lstm(x.float(), (h0, c0))

        # decode the hidden state of the last time step
        out = self.linear(out[:, -1, :])
        return out


def create_model(opt):
    return Net(opt)

def get_batch(data, batch_size):
    idxs = np.random.choice(len(data),batch_size)
    batch_strokes = [data[idx] for idx in idxs]
    strokes = []
    lengths = []
    for seq in batch_strokes:
        len_seq = len(seq[:, 0])  # I think this is how many lines in the image
        # Seq is always of shape (n,3) where the three dimensions
        # ∆x, ∆y, and a binary value representing whether the pen is lifted away from the paper
        new_seq = np.zeros((200, 5))  # New seq of max length, all zeros
        new_seq[:len_seq, :2] = seq[:, :2]  # fill in x:y co-ords in first two dims
        new_seq[:len_seq - 1, 2] = 1 - seq[:-1, 2]  # inverse of pen binary up to second-to-last point in third dim
        new_seq[:len_seq, 3] = seq[:, 2]  # pen binary in fourth dim
        new_seq[(len_seq - 1):, 4] = 1  # ones from second-to-last point to end of max length in fifth dim
        new_seq[len_seq - 1, 2:4] = 0  # zeros in last point for dims three and four
        strokes.append(new_seq)  # Record the sequence
        lengths.append(len(seq[:, 0]))  # Record the length of the actual sequence

    batch = torch.autograd.Variable(torch.from_numpy(np.stack(strokes, 1)).float())

    return batch, lengths

if __name__ == '__main__':
    opt = {'num_classes': 25, 'hidden_size': 70, 'device': 'cpu'}
    net = create_model(opt)
    
    bear_seq = np.load("dataset/seq/sketchrnn_bear.npz", allow_pickle=True, encoding="latin1")
    bear_seq_train = bear_seq["train"]
    batch_size = 10
    x, _ = get_batch(bear_seq_train, batch_size)

    out = net(x,batch_size)
    print(f"out put of net is {out}")

    with torch.no_grad():
        _, predicted = torch.max(out.data, 1)
        print(predicted)