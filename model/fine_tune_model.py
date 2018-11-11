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
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
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
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)


def fine_tune_resnet18():
    print('resnet18....')
    train_data=MyDataset(txt='../data/train/train.txt', transform=trans)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data=MyDataset(txt='../data/test/test.txt', transform=trans)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    model_path = 'YaleResnet18.pth'
    if os.path.exists(model_path):
        resnet_model = models.resnet18()
        if torch.cuda.is_available():
            resnet_model.load_state_dict(torch.load(model_path))
        else:
            resnet_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        resnet_model.fc = nn.Linear(in_features=512, out_features=class_num, bias=True)
        print('Load model succeed!')
    else:   
        print('Model not exists.')
        resnet_model = models.resnet18(pretrained=True)
        resnet_model.fc = nn.Linear(in_features=512, out_features=class_num, bias=True)
    if torch.cuda.is_available():
        resnet_model = resnet_model.cuda()
    print(resnet_model)
    return resnet_model, train_data, train_loader, test_data, test_loader, model_path

def fine_tune_resnet34():
    train_data=MyDataset(txt='../data/train/train.txt', transform=trans)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data=MyDataset(txt='../data/test/test.txt', transform=trans)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    model_path = 'YaleResnet34.pth'
    if os.path.exists(model_path):
        resnet_model = models.resnet34()
        if torch.cuda.is_available():
            resnet_model.load_state_dict(torch.load(model_path))
        else:
            resnet_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        resnet_model.fc = nn.Linear(in_features=512, out_features=class_num, bias=True)
        print('Load model succeed!')
    else:   
        resnet_model = models.resnet34(pretrained=True)
        resnet_model.fc = nn.Linear(in_features=512, out_features=38, bias=True)
    if torch.cuda.is_available():
        resnet_model = resnet_model.cuda()
    print(resnet_model)
    return resnet_model, train_data, train_loader, test_data, test_loader, model_path   

def fine_tune_resnet50():
    train_data=MyDataset(txt='../data/train/train.txt', transform=trans)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data=MyDataset(txt='../data/test/test.txt', transform=trans)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    model_path = 'YaleResnet50.pth'
    if os.path.exists(model_path):
        resnet_model = models.resnet50()
        if torch.cuda.is_available():
            resnet_model.load_state_dict(torch.load(model_path))
        else:
            resnet_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        resnet_model.fc = nn.Linear(in_features=2048, out_features=class_num, bias=True)
        print('Load model succeed!')
    else:   
        resnet_model = models.resnet50(pretrained=True)
        resnet_model.fc = nn.Linear(in_features=2048, out_features=class_num, bias=True)
    if torch.cuda.is_available():
        resnet_model = resnet_model.cuda()
    print(resnet_model)
    return resnet_model, train_data, train_loader, test_data, test_loader, model_path   

def fine_tune_resnet101():
    train_data=MyDataset(txt='../data/train/train.txt', transform=trans)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data=MyDataset(txt='../data/test/test.txt', transform=trans)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    model_path = 'YaleResnet101.pth'
    if os.path.exists(model_path):
        resnet_model = models.resnet101()
        if torch.cuda.is_available():
            resnet_model.load_state_dict(torch.load(model_path))
        else:
            resnet_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        resnet_model.fc = nn.Linear(in_features=2048, out_features=class_num, bias=True)
        print('Load model succeed!')
    else:   
        resnet_model = models.resnet101(pretrained=True)
        resnet_model.fc = nn.Linear(in_features=2048, out_features=class_num, bias=True)
    if torch.cuda.is_available():
        resnet_model = resnet_model.cuda()
    print(resnet_model)
    return resnet_model, train_data, train_loader, test_data, test_loader, model_path   

def fine_tune_vgg16():
    train_data=MyDataset(txt='../data/train/train.txt', transform=trans)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data=MyDataset(txt='../data/test/test.txt', transform=trans)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    model_path = 'YaleVgg16.pth'
    if os.path.exists(model_path):
        model = models.vgg16()
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.classifier = nn.Sequential(nn.Linear(25088, 4096),      #vgg16
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, class_num))
        print('Load model succeed!')
    else:   
        model = models.vgg16(pretrained=True)
        model.classifier = nn.Sequential(nn.Linear(25088, 4096),      #vgg16
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(4096, 4096),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(4096, class_num))
    if torch.cuda.is_available():
        model = model.cuda()
    print(model)
    return model, train_data, train_loader, test_data, test_loader, model_path   

def fine_tune_densenet121():
    train_data=MyDataset(txt='../data/train/train.txt', transform=trans)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data=MyDataset(txt='../data/test/test.txt', transform=trans)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    model_path = 'YaleDense121.pth'
    if os.path.exists(model_path):
        model = models.densenet121()
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.classifier = nn.Linear(1024, class_num, True)
        print('Load model succeed!')
    else:   
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, class_num, True)
    if torch.cuda.is_available():
        model = model.cuda()
    print(model)
    return model, train_data, train_loader, test_data, test_loader, model_path 

def fine_tune_densenet161():
    train_data=MyDataset(txt='../data/train/train.txt', transform=trans)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data=MyDataset(txt='../data/test/test.txt', transform=trans)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    model_path = 'YaleDense161.pth'
    if os.path.exists(model_path):
        model = models.densenet161()
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.classifier = nn.Linear(2208, class_num, True)
        print('Load model succeed!')
    else:   
        model = models.densenet161(pretrained=True)
        model.classifier = nn.Linear(2208, class_num, True)
    if torch.cuda.is_available():
        model = model.cuda()
    print(model)
    return model, train_data, train_loader, test_data, test_loader, model_path 

def fine_tune(net, train_data, train_loader, test_data, test_loader, path):
    optimizer = optim.Adam(net.parameters(), 1e-4)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(30):
        print('epoch {}'.format(epoch + 1))
        # training
        train_loss = 0.0
        train_acc = 0.0
        for batch_x, batch_y in train_loader:
            if torch.cuda.is_available():
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = net(batch_x)

            loss = loss_func(out, batch_y)
            train_loss += loss.data[0]
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += float(train_correct.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch: {}\nTrain Loss: {:.6f}, Acc: {:.6f}'.format(epoch, train_loss / (len(train_data)), train_acc / (len(train_data))))
        #evaluation
        acc = 0.0
        for batch_x, batch_y in test_loader:
            if torch.cuda.is_available():
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = net(batch_x)
            loss = loss_func(out, batch_y)
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            acc += float(train_correct.data[0])
        print('Test Acc: {:.6f}'.format(acc / (len(test_data))))

    torch.save(net.state_dict(), path)



#resnet18
net, train_data, train_loader, test_data, test_loader, path = fine_tune_resnet18()
fine_tune(net, train_data, train_loader, test_data, test_loader, path)

# resnet34
#net, train_data, train_loader, test_data, test_loader, path = fine_tune_resnet34()
#fine_tune(net, train_data, train_loader, test_data, test_loader, path)

# resnet50
#net, train_data, train_loader, test_data, test_loader, path = fine_tune_resnet50()
#fine_tune(net, train_data, train_loader, test_data, test_loader, path)

# resnet101
#net, train_data, train_loader, test_data, test_loader, path = fine_tune_resnet101()
#fine_tune(net, train_data, train_loader, test_data, test_loader, path)

#vgg16
#net, train_data, train_loader, test_data, test_loader, path = fine_tune_vgg16()
#fine_tune(net, train_data, train_loader, test_data, test_loader, path)

#densenet121
#net, train_data, train_loader, test_data, test_loader, path = fine_tune_densenet121()
#fine_tune(net, train_data, train_loader, test_data, test_loader, path)

#densenet161
#net, train_data, train_loader, test_data, test_loader, path = fine_tune_densenet161()
#fine_tune(net, train_data, train_loader, test_data, test_loader, path)
