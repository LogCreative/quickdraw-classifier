import torch, os
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from torch.utils import tensorboard as tb
import torch
from torch.utils.data import Dataset, DataLoader
from config_train_cnn import args
from image_loader import GetDataset
from tqdm import tqdm
def main( args ):
    transform=Compose([Resize([225, 225]), Grayscale(), ToTensor()])
    trainset,valset = GetDataset(args.dataroot,transform,args.small_data)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=args.val_batch_size, num_workers=4)
    
    if args.model == 'sketchnet':
        from models.sketchnet import create_model
    elif args.model == 'AlexNet':
        from models.AlexNet import create_model
    opt = {'num_classes':args.num_classes}
    model = create_model(opt)
    device = torch.device(args.device)
    
    crit = torch.nn.CrossEntropyLoss().to(device)
    optim = torch.optim.Adam(model.parameters())

    # Tensorboard stuff
    # writer = tb.SummaryWriter('./logs')
    print("begin training")
    count = 0
    model.to(device)
    for e in range(args.epochs):
        model.train()
        for i, (X, Y) in tqdm(enumerate(train_loader)):            
            X, Y = X.to(device), Y.to(device)

            optim.zero_grad()

            output = model(X)
            loss = crit(output, Y)
            
            if i % args.log_interval == 0:
                print(f'[Training] {i}/{e}/{args.epochs} -> Loss: {loss.item()}')
                # writer.add_scalar('train-loss', loss.item(), count)
            
            loss.backward()
            optim.step()

            count += 1

        correct, total = 0, 0
        model.eval()
        for i, (X, Y) in enumerate(val_loader):
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            _, predicted = torch.max(output, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
            accuracy = (correct / total) * 100
        
        print(f'[validation] -/{e}/{args.epochs} -> Accuracy: {accuracy} %')
        # writer.add_scalar('validation-accuracy', accuracy/100., e)




if __name__ == '__main__':
    main( args )