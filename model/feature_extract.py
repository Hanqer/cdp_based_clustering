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

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x) 
            # print(name)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


def extract_feature(txt, exactor, sava_path):
    features = []
    trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    with open(txt, 'r') as f:
        for line in f:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            img = Image.open(words[0]).convert('RGB')
            img = trans(img)
            x = Variable(img).view(-1, 3, 224, 224)
            fea = exactor(x)[0].view(-1)
            features.append(fea.detach().numpy())
        features = np.array(features)
        if features.shape[1] != feature_num:
            pca = decomposition.PCA(n_components=feature_num)
            features = pca.fit_transform(features)
        features.tofile(sava_path)
        print('exacts features finished.')


### please modify this for your own.
datasets = ['train', 'unlabel', 'test']
exact_list=["avgpool"]
model_path = 'YaleResnet18.pth'

resnet = models.resnet18()
resnet.fc = nn.Linear(in_features=512, out_features=38, bias=True)
resnet.load_state_dict(torch.load(model_path, map_location='cpu'))

extractor=FeatureExtractor(resnet, exact_list)
# extractor=FeatureExtractor(resnet,exact_list).cuda()

for dataset in datasets:
    savepath = 'data/' + dataset + '/features/'
    listpath = 'data/' + dataset + '/' + dataset + '.txt'
    model_name = 'resnet18'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    extract_feature(listpath, extractor, savepath+'/'+model_name+'.bin')
    print('extract {} features ok...'.format(model_name))