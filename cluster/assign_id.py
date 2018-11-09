import os

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

class_num = 38
batch_size = 8
trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader,root_path = 'data/train/'):
        fh = open(txt, 'r')
        self.root_path =root_path
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        fh.close()
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(self.root_path+fn)
        if self.transform is not None:
            img = self.transform(img)
        data = {}
        data['img'] = img
        data['id'] = fn.split('.')[0]
        return data,label

    def __len__(self):
        return len(self.imgs)