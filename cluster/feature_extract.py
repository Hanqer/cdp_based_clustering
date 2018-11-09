import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import os
from sklearn import decomposition 

feature_num = 128
exact_list = ['avgpool']

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name in ["fc", "classifier"]: 
                continue
            x = module(x) 
            if name in self.extracted_layers:
               outputs.append(x)
        return outputs


def extract_feature(batchx,model):
    features = []
    exactor = FeatureExtractor(model,exact_list)
    trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
       
   # #if torch.cuda.is_available():
      #  img = trans(batchx).cuda()
    #else:
    #    img = trans(batchx)
    x = Variable(batchx.cuda()).view(-1, 3, 224, 224)
   # print(x.shape)
    fea = exactor(x)[0].view(-1,512)
    #print(fea.shape)
    if torch.cuda.is_available():
        features.append(fea.detach().cpu().numpy())
    else:
        features.append(fea.detach().numpy())
    features = np.array(features)
    features = np.reshape(features,[len(batchx),512])
    #features = np.shape(features,-1)
    #print(features.shape)
    #if features.shape[1] != feature_num:
      #  pca = decomposition.PCA(n_components=feature_num)
      #  features = pca.fit_transform(features)
    #print('exacts features finished.')
   # print(features.shape)
    return features
        #features.tofile(sava_path)



