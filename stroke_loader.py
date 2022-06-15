import numpy as np
import matplotlib.pyplot as plt
import PIL
import glob
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import gc
use_cuda = torch.cuda.is_available()

def max_size(data):
    """larger sequence length in the data set"""
    sizes = [len(seq) for seq in data]
    return max(sizes)

def purify(strokes):
    """removes to small or too long sequences + removes large gaps"""
    data = []
    for seq in strokes:
        if seq.shape[0] <= 1000 and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data

def calculate_normalizing_scale_factor(strokes):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            data.append(strokes[i][j, 0])
            data.append(strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)

def normalize(strokes):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    data = []
    scale_factor = calculate_normalizing_scale_factor(strokes)
    for seq in strokes:
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data




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

class SeqDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        data = purify(data)
        data = normalize(data)
        self.Nmax = max_size(data)
        self.data = data
        self.targets = torch.LongTensor(targets)
        
    # 补齐操作
    def __getitem__(self, index):
        seq = self.data[index]
        y = self.targets[index]
        len_seq = len(seq[:,0])
        new_seq = np.zeros((self.Nmax,5))
        new_seq[:len_seq,:2] = seq[:,:2]
        new_seq[:len_seq-1,2] = 1-seq[:-1,2]
        new_seq[:len_seq,3] = seq[:,2]
        new_seq[(len_seq-1):,4] = 1
        new_seq[len_seq-1,2:4] = 0
        return new_seq, y
    
    def __len__(self):
        return len(self.data)


def get_category_name(filename:str):
    return filename.split('_')[1][:-4]
def GetDataset(dataroot:str,transform=None,small=False):
    filepaths = glob.glob(os.path.join(dataroot, "*.npz")) 
    filenames = [path.split('/')[-1] for path in filepaths]
    filename = filenames[0]
    category = get_category_name(filename)
    npz_file = np.load(f'{dataroot}/{filename}',allow_pickle=True, encoding="latin1")
    train_data_array = npz_file['train']
    val_data_array = npz_file['valid']
    test_data_array = npz_file['test']
    train_label_array = np.full((len(train_data_array),),categories_label_dict[category])
    val_label_array = np.full((len(val_data_array),),categories_label_dict[category])
    test_label_array = np.full((len(test_data_array),),categories_label_dict[category])
    
    if small:
        filenames = filenames[:2] #!used for debug and something
        
    for filename in filenames[1:]:
        category = get_category_name(filename)
        npz_file = np.load(f'{dataroot}/{filename}',allow_pickle=True, encoding="latin1")
        train_data_array = np.concatenate((train_data_array, npz_file['train']), axis=0)
        val_data_array = np.concatenate((val_data_array, npz_file['valid']), axis=0)
        train_label_array = np.concatenate((train_label_array, np.full((len(npz_file['train']),),categories_label_dict[category])), axis=0)
        val_label_array = np.concatenate((val_label_array, np.full((len(npz_file['valid']),),categories_label_dict[category])), axis=0)
        test_label_array = np.concatenate((test_label_array, np.full((len(npz_file['test']),),categories_label_dict[category])), axis=0)
        gc.collect()
    train_label_array = train_label_array
    val_label_array = val_label_array
    test_label_array = test_label_array
    trainset,valset,testset = SeqDataset(train_data_array, train_label_array,transform), SeqDataset(val_data_array, val_label_array,transform), SeqDataset(test_data_array, test_label_array,transform)
    return trainset,valset,test_data_array



if __name__ == "__main__":
    trainset,valset,testset = GetDataset('/home/songxiufeng/tk/ml_proj/quickdraw-classifier/dataset/seq',small=True)
    
    