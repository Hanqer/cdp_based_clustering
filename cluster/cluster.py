import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from feature_extract import extract_feature

class cluster(object):
    def __init__(self,model,class_num,fea_dim,alpha):
        self.model = model
        self.center = np.zeros([class_num,fea_dim])
        self.fea_dim = fea_dim
        self.alpha = alpha
        self.batchx = {}
        self.epoch_pl = {}
        self.batchy = {}
        self.class_num = class_num
        self.all = 0
        self.acc = 0
    def get_PL(self,batchx,batchy):
        self.batchx['id'] = batchx['id']
        self.batchx['img'] = extract_feature(batchx['img'],self.model)
        self.batchy = batchy
        self.batch_size = len(batchy) 
        for i in range(self.batch_size):
            if batchy[i] < 38:
                self.epoch_pl[batchx['id'][i]] = batchy[i]
                delta = self.get_delta(batchy[i])
               # print(delta.shape)
                #print(self.center[batchy[i]].shape)
                self.center[batchy[i]] = self.center[batchy[i]] + self.alpha*delta
            else:
                dis = self.get_dis(i)
                label = np.argmax(dis)
                self.epoch_pl[batchx['id'][i]] = label+38
        pl = []
        for i in range(self.batch_size):
            pl.append(int(self.epoch_pl[self.batchx['id'][i]]))
       # self.all =self.all + self.batch_size
        print(torch.Tensor(pl))
        print(batchy)
        print('==================================================')
        for i in range(self.batch_size):
            if batchy[i] > 37:
                self.all = self.all + 1
                if pl[i] == batchy[i]:
                    self.acc = self.acc+1

    def get_delta(self,index):
        #print(self.batchx['img'].shape)
        temp_delta = np.zeros([self.fea_dim])
        #print('======================')
       # print(temp_delta.shape)
        ssum = 0
        for x in range(self.batch_size):
            if self.batchy[x] == index:
               # print(self.batchx['img'][x].shape)
                temp_delta = temp_delta + (self.center[index]-self.batchx['img'][x])
                ssum = ssum + 1
            elif self.batchx['id'][x] in self.epoch_pl and self.epoch_pl[self.batchx['id'][x]] == index+38:
                #print("================")
                temp_delta = temp_delta + (self.center[index]-self.batchx['img'][x])
                ssum = ssum + 1
        #print(temp_delta.shape)
       # print('++++++++++++++++')
       # print((temp_delta/(ssum+1)).shape)
        return temp_delta/(ssum+1)

    def get_dis(self,index):
        distance = []
        for i in range(self.class_num):
            distance.append(self.sim(self.center[i],self.batchx['img'][index]))
        distance = np.array(distance)
        return torch.Tensor(distance)

    def sim(self,x,y):
       # print(type(x))
       # print(type(y))
        return abs(x.dot(y)/(np.sqrt(np.sum(np.square(x))*np.sum(np.square(y)))+1))

    def filter(self):
        pair = np.load(path)
        


    




    




    