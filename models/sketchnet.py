import torch
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self,opt=None):
        super().__init__()

        self.num_classes = 25
        if opt:
            self.num_classes = opt['num_classes']

        self.conv1 = torch.nn.Conv2d(1, 64, (15, 15), stride=3)
        self.conv2 = torch.nn.Conv2d(64, 128, (5, 5), stride=1)
        self.conv3 = torch.nn.Conv2d(128, 256, (3, 3), stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(256, 512, (7, 7), stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(512, 512, (1, 1), stride=1, padding=0)

        self.linear = torch.nn.Linear(512, self.num_classes)

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
        
        return self.linear(x)

def create_model(opt):
    return Net(opt)

if __name__ == '__main__':
    size = 225
    opt = {'num_classes':25}

    net = create_model(opt)
    x = torch.autograd.Variable(torch.FloatTensor(2,1,size,size).uniform_(-1,1))

    out = net(x)

    print(f"out put of net is {out}")