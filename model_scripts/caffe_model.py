from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os

class MDL:
    def __init__(self,
                caffe_root="~/dev/caffe",
                data_location="/srv/datasets",
                deploy_file="",
                solve_file="",
                pretrained_file="",
                output_dir="~/dev/caffe/out",
                model_dir="./models/"):
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
        self.mean = None
        self.default_image_format = "jpg"
        self.solver = None
        self.data_dim = [1,3,227,227]

        import sys
        sys.path.insert(0, caffe_root + '/python')
        global caffe
        import caffe

        self.transformer = None
        self.model_dir= model_dir
    def use_gpu(self):
        if self.allow_gpu:
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_device)

    def use_cpu(self):
        if self.allow_cpu:
            caffe.set_mode_cpu()

    def set_mean(self, mean):
        self.mean = mean

    def load_data(self):
        print "Loading data"
        self.data = "/srv/datasets/mscoco/test2014/COCO_test2014_000000000001.jpg"

    #modified from https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045
    def loadData_asLmdb(self, file_listing=None, data_folder=None, lmdb_name="lmdb-temp", count=1000, imformat="RGB", option=""):
        print "Loading data in directory: " + self.data_location + " into lmdb"
        import lmdb
        import os
        import glob
        from PIL import Image
        inputs = None
        if file_listing is not None:
            inputs = [glob.glob(os.path.join(self.data_location + data_folder, fListing.rstrip('\n') + '.*'))[0] for fListing in open(file_listing)]

        from os import listdir
        from os.path import isfile, join

        if file_listing is None:
            inputs = [ self.data_location + data_folder + f for f in listdir(self.data_location + data_folder) if isfile(join(self.data_location + data_folder, f)) ]

        in_db = lmdb.open(lmdb_name, map_size=int(1e12))
        with in_db.begin(write=True) as in_txn:
            for in_idx, in_ in enumerate(inputs):
                print "", in_idx, " " + in_
                nim = Image.open(in_).resize((400, 400), Image.ANTIALIAS).convert(imformat)
                im = np.array(nim)
                print im.shape
                if len( im.shape ) > 0:
                    if imformat is "RGB":
                        im = im[:,:,::-1]
                        im = im.transpose((2,0,1))
                    elif imformat is "L":
                        im = im[np.newaxis, :]
                        #im = im.transpose((0,,1))

                    #if option is "r4":
                        #im = np.array(1, im)
                    print im.shape
                    im_dat = caffe.io.array_to_datum(im)
                    in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
                else:
                    print im
        in_db.close()

    def set_model_files(self, gist_id):
        print "not implemented"

    def set_model_files(self, model_root, deploy_file, solve_file, pretrained_file):
        print "not implemented"
        gist_id = "FCN_32s_PASCAL"
        self.deploy_file = "/models/" + gist_id + "/fcn-32s-pascal-deploy.prototxt"
        self.pretrained_file = "/models/" + gist_id + "/fcn-32s-pascal.caffemodel"

	# from https://gist.github.com/shelhamer/91eece041c19ff8968ee#file-solve-py
	# credit @longjon
    def upsample_filter(size):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        og = np.ogrid[:size, :size]

        return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

	# from https://gist.github.com/shelhamer/91eece041c19ff8968ee#file-solve-py
    def interp_surgery(net, layers):
        for l in layers:
            m, k, h, w = net.params[l][0].data.shape
            if m != k:
                print 'input and ouput channels need to be the same'
                raise
            if h != w:
                print 'filters need to be square'
                raise
            fltr = upsample_filter(h)
            net.params[l][0].data[range(m), range(k), :, :] = fltr

    def configureForTest(self):
        #plt.rcParams['figure.figsize'] = (10, 10)
        #plt.rcParams['image.interpolation'] = 'nearest'
        #plt.rcParams['image.cmap'] = 'gray'

        self.use_cpu()

        self.net = caffe.Net(self.caffe_root + self.deploy_file, self.caffe_root + self.pretrained_file, caffe.TEST);

        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_channel_swap('data', (2,1,0))
        if self.mean is not None:
            self.transformer.set_mean('data', self.mean)

    def configureForTrain(self):
        #plt.rcParams['figure.figsize'] = (10, 10)
        #plt.rcParams['image.interpolation'] = 'nearest'
        #plt.rcParams['image.cmap'] = 'gray'

        self.use_cpu()

        base_weights = self.caffe_root + self.pretrained_file
        solver = caffe.SGDSolver(self.solve_file)

    def runTest(self, image):
        self.net.blobs['data'].reshape(self.data_dim[0],self.data_dim[1],self.data_dim[2],self.data_dim[3])
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', caffe.io.load_image(image))

        self.out = self.net.forward();
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

    def runTrainS(self, step_count):
        print "Training"
        cpath = os.getcwd()
        os.chdir(self.caffe_root + self.model_dir)
        self.solver = caffe.SGDSolver(self.caffe_root + self.solve_file)
        #interp_layers = [k for k in self.solver.net.params.keys() if 'up' in k]
        #interp_surgery(self.solver.net, #interp_layers)

        #self.solver.net.copy_from(base_weights)

        #self.solver.step(step_count)
        os.chdir(cpath)

    def vis_square(self, data, padsize=1, padval=0):
        data -= data.min()
        data /= data.max()

        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

        return data

    def visualizeLayers(self, save_prefix=""):
        for k, v in self.net.blobs.items():
            if (k[0] is not 'f' or k[1] is not 'c') and (k[0] is not 'p' or k[1] is not 'r'):
                f = self.net.blobs[k].data[0, :36]
                fig = self.vis_square(f, padval=1)
                plt.imsave(fname="output/" + save_prefix + "_" + k + "." + self.default_image_format, arr=fig, format=self.default_image_format)

    def visualizeLayers_conv(self, filename, data):
        print "Not implemented"
    def visualizeLayers_pool(self, filename, data):
        print "Not implemented"
    def visualizeLayers_relu(self, filename, data):
        print "Not implemented"

    def visualizeFilters(self):
        for k, v in self.net.params.items():
            print k + " ", len(v[0].data.shape)
            if (len(v[0].data.shape) == 4) and (k[0] != 'f') and (k[0] != 's') and (k[0] != 'u'):
                print "(", v[0].data.shape[0], ")"
                fig = self.vis_square(v[0].data[0, :40])
                plt.imsave(fname="output/" + "filter_" + k + "." + self.default_image_format, arr=fig, format=self.default_image_format);

        print "Visualizing Filters"


def FCN_32s():
	model = MDL(caffe_root="/home/chriswc/dev/caffe/",data_location="/home/chriswc/VOCdevkit2010/VOC2010/JPEGImages/",deploy_file="models/FCN_32s_PASCAL/deploy.prototxt",pretrained_file="models/FCN_32s_PASCAL/pretrained.caffemodel", model_dir="models/FCN_32s_PASCAL/")
	model.configureForTest()

	model.data_dim = [1,3,500,500]
	model.load_data()
	model.set_mean([104.00698793, 116.66876762, 122.67891434])
	#model.use_gpu()
	model.runTestBatch(model.data_location, 1)

	#model.visualizeLayers()
	model.visualizeFilters()
	#plt.imshow(model.out['upscore'][0, 20])
	return model

def FCN_8s_Context(genDB=False):
    model = MDL(caffe_root="/home/chriswc/dev/caffe/",data_location="/home/chriswc/VOCdevkit2010/VOC2010/",solve_file="models/FCN_8s_PASCAL_CONTEXT/solver.prototxt", deploy_file="models/FCN_8s_PASCAL_CONTEXT/deploy.prototxt",pretrained_file="models/FCN_8s_PASCAL_CONTEXT/pretrained.caffemodel", model_dir="models/FCN_8s_PASCAL_CONTEXT/")
    #model.configureForTest()
    path = os.getcwd()
    os.chdir(model.caffe_root + model.model_dir)
    model.data_dim = [1,3,500,500]
	#model.load_data()
    model.set_mean([104.00698793, 116.66876762, 122.67891434])
    model.use_gpu()
	#model.runTestBatch(model.data_location, 1)

	#model.visualizeLayers()
	#model.visualizeFilters()
	#plt.imshow(model.out['upscore'][0, 20])

    if genDB is True:
        model.loadData_asLmdb(file_listing="/home/chriswc/VOCdevkit2010/VOC2010/ImageSets/Segmentation/train.txt", data_folder="JPEGImages/", lmdb_name="input-lmdb", imformat="RGB")
        model.loadData_asLmdb(file_listing="/home/chriswc/VOCdevkit2010/VOC2010/ImageSets/Segmentation/train.txt", data_folder="SegmentationClass/", lmdb_name="output-lmdb", imformat="L")
        #model.loadData_asLmdb(file_listing="/home/chriswc/VOCdevkit2010/VOC2010/ImageSets/Segmentation/val.txt", data_folder="JPEGImages/", lmdb_name="input-val-lmdb")
        #model.loadData_asLmdb(file_listing="/home/chriswc/VOCdevkit2010/VOC2010/ImageSets/Segmentation/val.txt", data_folder="SegmentationClass/", lmdb_name="output-val-lmdb")

    print "Finished Loading Data. Now running training"
    model.runTrainS(1000)

    print "Completed Training"
    os.chdir(path)
    return model
def SOS_ALEXNET():
	model = MDL(caffe_root="/home/chriswc/dev/caffe/", data_location="srv/datasets/mscoco/test2014/", deploy_file="models/ALEXNET_CNN_SOS/deploy.prototxt", pretrained_file="models/ALEXNET_CNN_SOS/pretrained.caffemodel")
	model.set_mean( np.load(model.caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1) )
	model.configureForTest()

	model.data_dim = [1,3,227,227]
	model.load_data()
	model.use_gpu()
	model.runTestBatch("/srv/datasets/mscoco/test2014/", 1)

	model.visualizeFilters()
	return model
