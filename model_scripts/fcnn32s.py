import numpy as np
import matplotlib.pyplot as plt

caffe_root = "/home/chriswc/dev/caffe"

data_sample = "/srv/datasets/sos/img/COCO_COCO_train2014_000000288018.jpg"

gist_id = "ALEXNET_CNN_SOS"
deploy = "/models/" + gist_id + "/deploy.prototxt"
pretrained = "/models/" + gist_id + "/AlexNet_SalObjSub.caffemodel"

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os

caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(caffe_root + deploy, caffe_root + pretrained, caffe.TEST);

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
net.blobs['data'].reshape(1,3,227,227)
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(data_sample))
#plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
#plt.show()

out = net.forward();

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    plt.show()

#f = net.params['conv1'][0].data
#vis_square(f.transpose(0,2,3,1))

f = net.blobs['conv1'].data[0, :36]
vis_square(f, padval=1)

#f = net.params['conv2'][0].data
#vis_square(f[:48].reshape(48**2,5,5))

f = net.blobs['conv2'].data[0, :36]
vis_square(f, padval=1);

f = net.blobs['pool5'].data[0, :36]
vis_square(f, padval=0)

#f = net.blobs['relu6']
#vis_square(f, padval=0)
