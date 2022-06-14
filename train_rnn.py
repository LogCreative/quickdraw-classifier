from tabnanny import verbose
import torch, os
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from torch.utils import tensorboard as tb
import torch
from torch.utils.data import DataLoader, RandomSampler
from stroke_loader import GetDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.nn as nn

class HParams():
    def __init__(self):
        self.dataroot = 'dataset/seq/'
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.num_class = 25
        self.M = 20
        self.dropout = 0.9
        self.batch_size = 256
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.5
        self.lr = 0.001
        self.weight_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.temperature = 0.4
        self.max_seq_length = 200
        self.device = "cpu"
        self.small_data = True
        self.val_batch_size = self.batch_size
        self.epochs = 10
        self.log_interval = 100
        self.model = "rnn"

hp = HParams()

# class EncoderRNN(nn.Module):
#     def __init__(self,batch_size):
#         super(EncoderRNN, self).__init__()
#         # bidirectional lstm:
#         self.lstm = nn.LSTM(5, hp.enc_hidden_size, \
#             dropout=hp.dropout, bidirectional=True)
#         # create mu and sigma from lstm's last output:
#         self.fc_mu = nn.Linear(2*hp.enc_hidden_size, hp.num_class)
#         self.fc_sigma = nn.Linear(2*hp.enc_hidden_size, hp.num_class)
#         # active dropout:
#         self.batch_size = batch_size
#         self.train()

#     def forward(self, inputs, hidden_cell=None):
#         if hidden_cell is None:
#             # then must init with zeros
#             if hp.device>=0:
#                 hidden = torch.zeros(2, self.batch_size, hp.enc_hidden_size).cuda(hp.device)
#                 cell = torch.zeros(2, self.batch_size, hp.enc_hidden_size).cuda(hp.device)
#             else:
#                 hidden = torch.zeros(2, self.batch_size, hp.enc_hidden_size)
#                 cell = torch.zeros(2, self.batch_size, hp.enc_hidden_size)
#             hidden_cell = (hidden, cell)
#         _, (hidden,cell) = self.lstm(inputs.float(), hidden_cell)
#         # hidden is (2, batch_size, hidden_size), we want (batch_size, 2*hidden_size):
#         hidden_forward, hidden_backward = torch.split(hidden,1,0)
#         hidden_cat = torch.cat([hidden_forward.squeeze(0), hidden_backward.squeeze(0)],1)
#         #!TODO:revise code here for classfication
#         """
#         last_h = tf.concat([last_h_fw, last_h_bw], 1)
#         return last_h
#         output = tf.nn.xw_plus_b(self.batch_z, output_w, output_b)
#         """ 
#         output = self.fc_mu(hidden_cat)
#         return output

from models.sketchrnn import Net
    
def main(hp):
    trainset,valset = GetDataset(hp.dataroot,None,hp.small_data)
    train_loader = DataLoader(trainset, batch_size=hp.batch_size, sampler=RandomSampler(trainset, replacement=True, num_samples=((len(trainset) // hp.batch_size + 1) * hp.batch_size)), num_workers=4)
    val_loader = DataLoader(valset, batch_size=hp.val_batch_size, sampler=RandomSampler(valset, replacement=True, num_samples=((len(valset) // hp.batch_size + 1) * hp.batch_size)), num_workers=4)
    # A RandomSampler with fixed batch size is used for fully using the whole dataset. If using drop_last, some data may missing.
    
    model = Net({'num_classes': 25, 'hidden_size': 70, 'device': hp.device, 'batch_size': hp.batch_size})
    device = torch.device(hp.device)
    
    crit = torch.nn.CrossEntropyLoss().to(device)
    optim = torch.optim.AdamW(model.parameters(),lr=hp.lr,weight_decay = hp.weight_decay)
    scheduler = ReduceLROnPlateau(optim, 'min', factor=0.3, patience=2,verbose=True)
    # Tensorboard stuff
    print("begin training")
    count = 0
    best_accuracy = 0
    model.to(device)
    for e in range(hp.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0
        for i, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)

            optim.zero_grad()

            output = model(X)
            loss = crit(output, Y)
            
            if i % hp.log_interval == 0:
                print(f'[Training] {i}/{e+1}/{hp.epochs} -> Loss: {loss.item()}')
            
            loss.backward()
            optim.step()

            count += 1
            epoch_loss += loss.item() * len(output)
            _, predicted = torch.max(output, 1)
            epoch_acc += (predicted == Y).sum().item()
        
        data_size = len(train_loader.dataset)
        epoch_loss = epoch_loss / data_size
        epoch_acc = float(epoch_acc) / data_size
        print(f'Epoch {e + 1}/{hp.epochs} | trainning | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
        scheduler.step(epoch_loss)

        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for i, (X, Y) in enumerate(val_loader):
                X, Y = X.to(device), Y.to(device)
                output = model(X)
                _, predicted = torch.max(output, 1)                
                correct += (predicted == Y).sum().item()
        accuracy = (float(correct) / len(val_loader.dataset)) * 100
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'best_{hp.model}.pth')
        print(f'[validation] -/{e+1}/{hp.epochs} -> Accuracy: {accuracy} %')




if __name__ == '__main__':
    print(f"begin training and val with {hp}") 
    main( hp )