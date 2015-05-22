import numpy as np
import matplotlib.pyplot as plt

caffe_root = "/home/chriswc/dev/caffe"

data_sample = "/srv/datasets/mscoco/test2014/COCO_test2014_000000000001.jpg"

gist_id = "FCN_32s_PASCAL"
deploy = "/models/" + gist_id + "/fcn-32s-pascal-deploy.prototxt"
pretrained = "/models/" + gist_id + "/fcn-32s-pascal.caffemodel"

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os

caffe.set_mode_cpu()
#caffe.set_device(0)

net = caffe.Net(caffe_root + deploy, caffe_root + pretrained, caffe.TEST);

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
net.blobs['data'].reshape(1,3,500,500)
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(data_sample))
plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
plt.show()

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

f = net.params['conv1_1'][0].data
vis_square(f.transpose(0,2,3,1))
