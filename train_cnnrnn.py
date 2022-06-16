from torch.optim.lr_scheduler import ReduceLROnPlateau
import image_loader
import stroke_loader
import torch
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from torch.utils.data import Dataset, DataLoader

class ImgSeqDataset(Dataset):
    def __init__(self, imgset, seqset):
        assert len(imgset) == len(seqset) 
        self.imgset = imgset
        self.seqset = seqset

    def __getitem__(self, index):
        img = self.imgset[index]
        seq = self.seqset[index]
        assert img[1] == seq[1]
        return img[0], seq[0], img[1]

    def __len__(self):
        assert len(self.imgset) == len(self.seqset)
        return len(self.imgset)

from models.cnnrnn import Net

class Args:
    batch_size = 64
    val_batch_size = 64
    log_interval = 1000
    epochs = 10
    num_classes = 25
    dataroot_img = 'dataset/png'
    dataroot_seq = 'dataset/seq'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = 'cnnrnn'
    small_data = False
    weight_decay = 1e-3
    lr = 1e-3
    def __init__(self) -> None:
        pass
    def __str__(self) -> str:
        return f"batch_size: {self.batch_size}, val_batch_size: {self.val_batch_size}, log_interval: {self.log_interval}, epochs: {self.epochs}, num_classes: {self.num_classes}, dataroot: img {self.dataroot_img}, seq {self.dataroot_seq}, device: {self.device}, model: {self.model}, small_data: {self.small_data}, weight_decay: {self.weight_decay}, lr: {self.lr}"
args = Args()

def main( args ):
    transform=Compose([Resize([225, 225]), Grayscale(), ToTensor()])
    train_img, val_img, test_img = image_loader.GetDataset(args.dataroot_img,transform,args.small_data)
    train_seq, val_seq, test_seq = stroke_loader.GetDataset(args.dataroot_seq,None,args.small_data)
    trainset,valset,testset = ImgSeqDataset(train_img, train_seq),ImgSeqDataset(val_img, val_seq),ImgSeqDataset(test_img, test_seq)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size)
    test_loader = DataLoader(testset, batch_size=args.batch_size)

    model = Net({'num_classes': args.num_classes, 'batch_size': args.batch_size, 'device': args.device})
    device = torch.device(args.device)

    crit = torch.nn.CrossEntropyLoss().to(device)
    optim = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,)
    scheduler = ReduceLROnPlateau(optim, 'min', factor=0.3, patience=2, verbose=True)
    print("begin training")
    count = 0
    best_accuracy = 0
    model.to(device)
    for e in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0
        for i, (X0, X1, Y) in enumerate(train_loader):
            X1 = X1.transpose(0,1) # X [seq_len, batch_size, 5]
            X0, X1, Y = X0.to(device), X1.to(device), Y.to(device)

            optim.zero_grad()

            output = model(X0, X1)
            loss = crit(output, Y)
            
            if i % args.log_interval == 0:
                print(f'[Training] {i}/{e+1}/{args.epochs} -> Loss: {loss.item()}')
            
            loss.backward()
            optim.step()

            count += 1
            epoch_loss += loss.item() * len(output)
            _, predicted = torch.max(output, 1)
            epoch_acc += (predicted == Y).sum().item()
        
        data_size = len(train_loader.dataset)
        epoch_loss = epoch_loss / data_size
        epoch_acc = float(epoch_acc) / data_size
        print(f'Epoch {e + 1}/{args.epochs} | trainning | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
        scheduler.step(epoch_loss)

        print("Testing...")
        correct = 0
        model.eval()
        with torch.no_grad():
            for i, (X0, X1, Y) in enumerate(val_loader):
                X1 = X1.transpose(0,1)
                X0, X1, Y = X0.to(device), X1.to(device), Y.to(device)
                output = model(X0, X1)
                _, predicted = torch.max(output, 1)                
                correct += (predicted == Y).sum().item()
        accuracy = (float(correct) / len(val_loader.dataset)) * 100
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'best_{args.model}.pth')
        print(f'[validation] -/{e+1}/{args.epochs} -> Accuracy: {accuracy} %')

    correct = 0
    model.load_state_dict(torch.load(f'best_{args.model}.pth')) # the best
    model.eval()
    with torch.no_grad():
        for i, (X0, X1, Y) in enumerate(test_loader):
            X1 = X1.transpose(0,1)
            X0, X1, Y = X0.to(device), X1.to(device), Y.to(device)
            output = model(X0, X1)
            _, predicted = torch.max(output, 1)                
            correct += (predicted == Y).sum().item()
    accuracy = (float(correct) / len(test_loader.dataset)) * 100
    print(f'[test] - {args.model} -> Accuracy: {accuracy} %')


if __name__ == '__main__':
    print(f"begin training and val with {args}") 
    main( args )