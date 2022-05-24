import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import gc


categories = [
    "bear",
    "camel",
    "cat",
    "cow",
    "crocodile",
    "dog",
    "elephant",
    "flamingo",
    "giraffe",
    "hedgehog",
    "horse",
    "kangaroo",
    "lion",
    "monkey",
    "owl",
    "panda",
    "penguin",
    "pig",
    "raccoon",
    "rhinoceros",
    "sheep",
    "squirrel",
    "tiger",
    "whale",
    "zebra",
]

categories_label_dict = dict(zip(categories, range(len(categories))))

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def GetDataset(dataroot:str,transform=None):
    filenames = next(os.walk(dataroot), (None, None, []))[2]
    if filenames[0] == '.gitkeep':
        filenames.remove('.gitkeep')
    filename = filenames[0]
    category = filename[:-8]
    npz_file = np.load(f'{dataroot}/{filename}',allow_pickle=True, encoding="latin1")
    train_data_array = npz_file['train']
    val_data_array = npz_file['train']
    train_label_array = np.full((len(train_data_array),),categories_label_dict[category])
    val_label_array = np.full((len(val_data_array),),categories_label_dict[category])
    for filename in filenames[1:2]:
        category = filename[:-8]
        npz_file = np.load(f'{dataroot}/{filename}',allow_pickle=True, encoding="latin1")
        train_data_array = np.concatenate((train_data_array, npz_file['train']), axis=0)
        val_data_array = np.concatenate((val_data_array, npz_file['valid']), axis=0)
        train_label_array = np.concatenate((train_label_array, np.full((len(npz_file['train']),),categories_label_dict[category])), axis=0)
        val_label_array = np.concatenate((val_label_array, np.full((len(npz_file['valid']),),categories_label_dict[category])), axis=0)
        gc.collect()
    train_data_array = train_data_array.astype(np.float32)
    val_data_array = val_data_array.astype(np.float32)
    train_label_array = train_label_array
    val_label_array = val_label_array
    trainset,valset = MyDataset(train_data_array, train_label_array,transform), MyDataset(val_data_array, val_label_array,transform)
    return trainset,valset
        


transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
trainset,valset = GetDataset(dataroot='dataset/png',transform=transform)
train_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(valset, batch_size=4, num_workers=4)



## Some sample code for loading the dataset

# bear = np.load("dataset/png/bear_png.npz", allow_pickle=True, encoding="latin1")

# ## show the structure of bear
# # bear.files

# bear_train = bear["train"]
# bear_test = bear["test"]
# bear_valid = bear["valid"]

# import matplotlib.pyplot as plt
# plt.imshow(bear_train[0], cmap="gray")
# plt.show()

# ## For the seq data

# bear_seq = np.load("dataset/seq/sketchrnn_bear.npz", allow_pickle=True, encoding="latin1")

# bear_seq_train = bear_seq["train"]
# bear_seq_test = bear_seq["test"]
# bear_seq_valid = bear_seq["valid"]

# print(bear_seq_train[0].shape)