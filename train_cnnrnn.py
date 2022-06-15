import image_loader
import stroke_loader
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from torch.utils.data import Dataset, DataLoader

class ImgSeqDataset(Dataset):
    def __init__(self, imgset, seqset):
        assert len(imgset) == len(seqset) 
        self.imgset = imgset
        self.seqset = seqset

    def __getitem__(self, index):
        return self.imgset[index], self.seqset[index]

    def __len__(self):
        assert len(self.imgset) == len(self.seqset)
        return len(self.imgset)


if __name__ == '__main__':
    transform=Compose([Resize([225, 225]), Grayscale(), ToTensor()])
    train_img, val_img, test_img = image_loader.GetDataset("dataset/png/",transform,True)
    train_seq, val_seq, test_seq = stroke_loader.GetDataset("dataset/seq/",None,True)
    trainset,valset,testset = ImgSeqDataset(train_img, train_seq),ImgSeqDataset(val_img, val_seq),ImgSeqDataset(test_img, test_seq)

