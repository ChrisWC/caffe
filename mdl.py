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
		#plt.rcParams['figure.figsize'] = (10, 10)
		#plt.rcParams['image.interpolation'] = 'nearest'
		#plt.rcParams['image.cmap'] = 'gray'

		self.use_cpu()
	def runTest(self, image):
		self.net = caffe.Net(self.caffe_root + self.deploy_file, self.caffe_root + self.pretrained_file, caffe.TEST);
		transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
		transformer.set_transpose('data', (2,0,1))
		transformer.set_raw_scale('data', 255)
		transformer.set_channel_swap('data', (2,0,1))
		transformer.set_mean('data', np.load(self.caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
		self.net.blobs['data'].reshape(1,3,227,227)
		self.net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image))

		out = self.net.forward();
	def runTestBatch(self, data_dir, nmax=1):
		from os import listdir	
		from os.path import isfile, join
		k = [ f for f in listdir(data_dir) if isfile(join(data_dir,f)) ]

		i = 0
		while i < nmax:
			print k[i]
			self.runTest(data_dir + k[i])
			self.visualizeLayers(k[i])
			i = i + 1

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
    		iplt = plt.imshow(data)

	def visualizeLayers(self, save_prefix=""):
		for k, v in self.net.blobs.items():
			if (k[0] is not 'f' or k[1] is not 'c') and (k[0] is not 'p' or k[1] is not 'r'):
				f = self.net.blobs[k].data[0, :36]
				self.vis_square(f, padval=0)
				plt.savefig("output/" + save_prefix + "_" + k + ".jpg", format='jpg')

	def visualizeFilters(self):
		print "Visualizing Filters"


def FCNN_32s():
	model = MDL(caffe_root="/home/chriswc/dev/caffe/",data_location="/srv/datasets/mscoco/test2014/",deploy_file="models/ALEXNET_CNN_SOS/deploy.prototxt",pretrained_file="models/ALEXNET_CNN_SOS/AlexNet_SalObjSub.caffemodel")
	model.configure()
	model.load_data()
	model.use_gpu()
	model.runTestBatch("/srv/datasets/mscoco/test2014/", 5)
	
	#model.visualizeLayers()
