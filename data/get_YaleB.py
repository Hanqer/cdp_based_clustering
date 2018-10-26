import scipy.io as sio
import numpy as np
import os
from skimage import io
from skimage import transform

data = sio.loadmat('data/YaleB_32x32.mat')
samples = data['fea']
labels = data['gnd']
samples = np.hstack((samples, labels))
np.random.shuffle(samples)

length = samples.shape[0]
num_label = int(length*0.1)
num_unlabel = int(length*0.5)
num_test = int(length*0.4)

data_path = 'data/train/'
if(not os.path.exists(data_path)):
    os.makedirs(data_path)
with open(data_path+'train.txt', 'w') as f:
    for i, image in enumerate(samples[0:num_label, :]):
        img_path = data_path + str(i) + '.jpg'
        img = image[:-1].reshape([32,32]).transpose()
        io.imsave(img_path, img)
        f.write(img_path + ' ' + str(image[-1]-1) + '\n')


data_path = 'data/unlabel/'
if(not os.path.exists(data_path)):
    os.makedirs(data_path)
with open(data_path+'unlabel.txt', 'w') as f:
    for i, image in enumerate(samples[num_label:num_label+num_unlabel, :]):
        img_path = data_path + str(i) + '.jpg'
        img = image[:-1].reshape([32,32]).transpose()
        io.imsave(img_path, img)
        f.write(img_path + ' ' + str(image[-1]-1) + '\n')

data_path = 'data/test/'
if(not os.path.exists(data_path)):
    os.makedirs(data_path)
with open(data_path+'test.txt', 'w') as f:
    for i, image in enumerate(samples[num_label+num_unlabel:, :]):
        img_path = data_path + str(i) + '.jpg'
        img = image[:-1].reshape([32,32]).transpose()
        # img = transform.resize(img, (224, 224))
        io.imsave(img_path, img)
        f.write(img_path + ' ' + str(image[-1]-1) + '\n')

 
