import numpy as np
import matplotlib.pyplot as plt
import caffe
import os

class MDL:
	def __init__(self, 
			caffe_root="~/dev/caffe", 
			data_location="/srv/datasets", 
			deploy_file="",
			solve_file="",
			pretrained_file="",
			output_dir="~/dev/caffe/out"):
		self.data_location = data_location
		self.caffe_root = caffe_root
		#self.data_size = data_size
		self.deploy_file = deploy_file
		self.solve_file = solve_file
		self.pretrained_file = pretrained_file
		self.output_dir = output_dir

		self.allow_gpu = True
		self.allow_cpu = True
		self.gpu_device = 0

	def use_gpu(self):
		if self.allow_gpu:
			caffe.set_mode_gpu()
			caffe.set_device(self.gpu_device)

	def use_cpu(self):
		if self.allow_cpu:
			caffe.set_mode_cpu()

	def load_data(self):
		print "Loading data"
		self.data = "/srv/datasets/mscoco/test2014/COCO_test2014_000000000001.jpg"

	def set_model_files(self, gist_id):
		print "not implemented"

	def set_model_files(self, model_root, deploy_file, solve_file, pretrained_file):
		print "not implemented"		
		gist_id = "FCN_32s_PASCAL"
		self.deploy_file = "/models/" + gist_id + "/fcn-32s-pascal-deploy.prototxt"
		self.pretrained_file = "/models/" + gist_id + "/fcn-32s-pascal.caffemodel"

	def configure(self):
		plt.rcParams['figure.figsize'] = (10, 10)
		plt.rcParams['image.interpolation'] = 'nearest'
		plt.rcParams['image.cmap'] = 'gray'

		self.use_cpu()
	def runTest(self, image):
		self.net = caffe.Net(self.caffe_root + self.deploy_file, self.caffe_root + self.pretrained_file, caffe.TEST);
		transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
		transformer.set_transpose('data', (2,0,1))
		transformer.set_raw_scale('data', 255)
		transformer.set_channel_swap('data', (2,1,0))
		self.net.blobs['data'].reshape(1,3,500,500)
		self.net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(self.data))

		out = self.net.forward();
	def runTestBatch(self, image_list):
		print "Not Implemented"

	def runTrain(self, image):
		print "Not Implemented"

	def vis_square(self, data, padsize=1, padval=0):
    		data -= data.min()
    		data /= data.max()

    		n = int(np.ceil(np.sqrt(data.shape[0])))
    		padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    		data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    		data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    		data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    		plt.imshow(data)
    		plt.show()

	def visualizeLayers(self):
		f = self.net.blobs['conv1_1'].data[0, :36]
		vis_square(f, padval=1)

		f = self.net.blobs['pool5'].data[0, :36]
		vis_square(f, padval=0)

		for k, v in self.net.blobs.items():
			if len(v) == 4:
				f = self.net.blobs[k].data[0, :36]
				vis_square(f, padval=0)

	def visualizeFilters(self):
		print "Visualizing Filters"


def FCNN_32s():
	model = MDL("~/dev/caffe/","/srv/datasets/mscoco/test2014/","models/FCN_32s_PASCAL/fcn-32s-pascal-deploy.prototxt","","models/FCN_32s_PASCAL/fcn-32s-pascal.caffemodel")
	model.configure()
	model.load_data()
	model.runTest(model.data)
