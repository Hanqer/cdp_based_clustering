import torch
from torch import nn
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.autograd import Variable
import numpy as np
class Mediator_dataset(Dataset):
    def __init__(self, feature_input, label_input=None):
        self.data = feature_input
        self.label = label_input
        # self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), torch.from_numpy(self.label[index])

    def __len__(self):
        return self.data.shape[0]

class Mediator(nn.Module):
    def __init__(self, feature_num):
        super(Mediator, self).__init__()
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(feature_num, 50), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Dropout(0.5), nn.Linear(50, 50), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Dropout(0.5), nn.Linear(50, 1), nn.Sigmoid())
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def train_mediator(args):
    batch_size = 64
    mediator_modelPath = './mediator.pth'
    exp_root = './experiment'
    output_cdp = exp_root + '/{}/output'.format(args.data_name)
    if not os.path.isfile(output_cdp + "/mediator_input.npy"):
        get_mediator_input(args, output_cdp)
    mediator_input = np.load(output_cdp + "/mediator_input.npy")
    feature_length = mediator_input.shape[1]

    print('input_feature_length:{}.'.format(feature_length))
    pair_labels = np.load(output_cdp + '/pair_label.npy')
    train_dataset = Mediator_dataset(mediator_input, pair_labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    mediator_net = Mediator(feature_length)
    if torch.cuda.is_available():
        mediator_net = mediator_net.cuda()
    
    optimizer = optim.Adam(mediator_net.parameters())
    loss_func = nn.MSELoss()

    for epoch in range(200):
        # print('epoch {}'.format(epoch + 1))
        train_loss = 0.0
        acc = 0.0
        for batch_x, batch_y in train_loader:
            labels = batch_y.numpy()
            if torch.cuda.is_available():
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = mediator_net(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc += np.sum( ( (out.detach().numpy() >= 0.5).astype(np.float32) == labels ).astype(np.float32) )
        print('Epoch: {}, Train Loss: {:.6f}, Train acc: {:.6f}'.format(epoch, train_loss / len(train_dataset), acc / len(train_dataset)))
    torch.save(mediator_net.state_dict(), mediator_modelPath)

def test_mediator(args, savepath):
    if not os.path.isfile('./mediator.pth'):
        print('please train_mediator first.')
        exit()
    ### Must modify to our dir. 
    exp_root = './experiment'
    output_cdp = exp_root + '/{}/output'.format(args.data_name)
    if not os.path.isfile(output_cdp + '/mediator_input.npy'):
        get_mediator_input(args, output_cdp)
    mediator_input = np.load(output_cdp + '/mediator_input.npy')
    feature_length = mediator_input.shape[1]
    
    mediator_net = Mediator(feature_length)
    if torch.cuda.is_available():
        mediator_net.load_state_dict(torch.load('./mediator.pth'))
    else:
        mediator_net.load_state_dict(torch.load('./mediator.pth', map_location='cpu'))
    if torch.cuda.is_available():
        mediator_net = mediator_net.cuda()
    
    if torch.cuda.is_available():
        x = Variable(torch.from_numpy(mediator_input).cuda())
    else:
        x = Variable(torch.from_numpy(mediator_input))
    y = mediator_net(x).detach().numpy()
    np.save(savepath, y)

def get_mediator_input(args, path):
    print(path)
    relationship_feat = np.load(path + '/relationship.npy')
    affinity_feat = np.load(path + '/affinity.npy')
    distribution_feat = np.load(path + '/distribution.npy')
    mediator_input = np.hstack((relationship_feat, affinity_feat, distribution_feat))

    np.save(path + '/mediator_input.npy', mediator_input)
    print('save mediator_input.npy ok!')
