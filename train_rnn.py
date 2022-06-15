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
        self.dataroot = 'dataset/seq'
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.num_class = 25
        self.M = 20
        self.dropout = 0.9
        self.batch_size = 64
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
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.small_data = False
        self.val_batch_size = self.batch_size
        self.epochs = 10
        self.log_interval = 100
        self.model = "rnn"

hp = HParams()

from models.sketchrnn import Net
    
def main(hp):
    trainset,valset,testset = GetDataset(hp.dataroot,None,hp.small_data)
    train_loader = DataLoader(trainset, batch_size=hp.batch_size, sampler=RandomSampler(trainset, replacement=True, num_samples=((len(trainset) // hp.batch_size + 1) * hp.batch_size)), num_workers=4)
    # A RandomSampler with fixed batch size is used for fully using the whole dataset.
    val_loader = DataLoader(valset, batch_size=hp.val_batch_size, num_workers=4)
    test_loader = DataLoader(testset, batch_size=hp.val_batch_size, num_workers=4)
    
    model = Net({'num_classes': 25, 'hidden_size': hp.enc_hidden_size, 'device': hp.device})
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
            X = X.transpose(0,1) # X [seq_len, batch_size, 5]

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

        correct = 0
        model.eval()
        with torch.no_grad():
            for i, (X, Y) in enumerate(val_loader):
                X = X.transpose(0,1)
                X, Y = X.to(device), Y.to(device)
                output = model(X)
                _, predicted = torch.max(output, 1)                
                correct += (predicted == Y).sum().item()
        accuracy = (float(correct) / len(val_loader.dataset)) * 100
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'best_{hp.model}.pth')
        print(f'[validation] -/{e+1}/{hp.epochs} -> Accuracy: {accuracy} %')

    correct = 0
    model.load_state_dict(torch.load(f'best_{hp.model}.pth')) # the best
    model.eval()
    with torch.no_grad():
        for i, (X,Y) in enumerate(test_loader):
            X = X.transpose(0,1)
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            _, predicted = torch.max(output, 1)                
            correct += (predicted == Y).sum().item()
    accuracy = (float(correct) / len(test_loader.dataset)) * 100
    print(f'[test] - {hp.model} -> Accuracy: {accuracy} %')


if __name__ == '__main__':
    print(f"begin training and val with {hp}") 
    main( hp )