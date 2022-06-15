from tabnanny import verbose
import torch, os
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from torch.utils import tensorboard as tb
import torch
from torch.utils.data import DataLoader
from config_train_cnn import args
from image_loader import GetDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main( args ):
    transform=Compose([Resize([225, 225]), Grayscale(), ToTensor()])
    trainset,valset,testset = GetDataset(args.dataroot,transform,args.small_data)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=args.val_batch_size, num_workers=4)
    test_loader = DataLoader(testset, batch_size=args.val_batch_size, num_workers=4)
    
    if args.model == 'sketchnet':
        from models.sketchnet import create_model
    elif args.model == 'AlexNet':
        from models.AlexNet import create_model
    elif args.model == 'resnet18':
        from models.resnet18 import create_model
    opt = {'num_classes':args.num_classes}
    model = create_model(opt)
    device = torch.device(args.device)
    
    crit = torch.nn.CrossEntropyLoss().to(device)
    optim = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay = args.weight_decay)
    scheduler = ReduceLROnPlateau(optim, 'min', factor=0.3, patience=2,verbose=True)
    # Tensorboard stuff
    print("begin training")
    count = 0
    best_accuracy = 0
    model.to(device)
    for e in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0
        for i, (X, Y) in enumerate(train_loader):            
            X, Y = X.to(device), Y.to(device)

            optim.zero_grad()

            output = model(X)
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

        correct = 0
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
            torch.save(model.state_dict(), f'best_{args.model}.pth')
        print(f'[validation] -/{e+1}/{args.epochs} -> Accuracy: {accuracy} %')

    correct = 0
    model.load_state_dict(torch.load(f'best_{args.model}.pth')) # the best
    model.eval()
    with torch.no_grad():
        for i, (X,Y) in enumerate(test_loader):
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            _, predicted = torch.max(output, 1)                
            correct += (predicted == Y).sum().item()
    accuracy = (float(correct) / len(test_loader.dataset)) * 100
    print(f'[test] - {args.model} -> Accuracy: {accuracy} %')

if __name__ == '__main__':
    print(f"begin training and val with {args}") 
    main( args )