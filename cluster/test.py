import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from assign_id import MyDataset
from cluster import cluster

trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

batch_size = 16
class_num = 38

train_data=MyDataset(txt='data/train/train.txt', transform=trans,root_path='data/train/')
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
resnet18 = models.resnet18()
resnet18.fc = nn.Linear(in_features=512,out_features=class_num,bias=True)
resnet18.load_state_dict(torch.load('Yaleresnet18.pth'))
resnet18 = resnet18.cuda()
mycluster = cluster(resnet18,38,512,1e-3)
for epoch in range(10):
    mycluster.acc = 0
    mycluster.all = 0
    for batchx,batchy in train_loader:  
        mycluster.get_PL(batchx,batchy)
    print('acc :{:.4f}'.format(float(mycluster.acc)/mycluster.all))

    