from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
class MDL:
    def __init__(self,
                caffe_root=None,
                data_location=None,
                deploy_file=None,
                solve_file=None,
                pretrained_file=None,
                output_dir=None,
                model_root=None,
                model_dir=None,
                gist_id=None):
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
        self.dataset_root = None

        self.transformer = None
        self.model_dir= model_dir
        self.gist_id = gist_id
        self.allowModelBranchSpecification = True
        self.setCaffeDefaults()
        import sys
<<<<<<< HEAD
        sys.path.insert(0, self.caffe_root + '/python')
        global caffe
        import caffe
=======
        #sys.path.insert(0, self.caffe_root + '/python')
        #global caffe
        #import caffe
>>>>>>> b5c97b63d8f7fffda41b3a8ac10bbaf008a2f61c

    def loadModel(self, modelConfigFile, outputConfig=False, run=True):
        import json
        import os

        with open(modelConfigFile) as modelConfig:
            model = json.load(modelConfig)

            """load info"""
            self.name = model["Model Name"]
            self.short_name = model["Short Name"]
            self.model_authors = model["Model Author"]
            self.paper_authors = model["Paper Author"]

            """search for directory"""
            self.model_dir = os.getenv(model["Directory Path"][0]["Prefix"], "") + model["Directory Path"][0]["Directory"] + model["Directory Name"][0]
            os.path.isdir(self.model_dir)

            """load default dataset info, search for info"""
            self.dataset_locations = list()
            #self.data_location = self.getFromDatasets(type="directory", functions=["Input", "Testing"], database_name="PASCAL VOC 2010", name=["Input Images"])
            """load file locations"""
            if "deploy" in model["Files"]:
                self.deploy_file = model["Files"]["deploy"]["Name"]
            if "pretrained" in model["Files"]:
                self.pretrained_file = model["Files"]["pretrained"]["Name"]
<<<<<<< HEAD
            if "solver" in model["Files"]:
                self.solver_file = model["Files"]["solver"]["Name"]
=======
>>>>>>> b5c97b63d8f7fffda41b3a8ac10bbaf008a2f61c

            """load in pipeline (check that files exsist"""

            """ouput data"""
            if outputConfig:
                """output"""
                print "Loading Model: " + self.name + "(" + self.short_name + ")"
                print "as defined in: " + self.paper

            """download files if necessary"""

            """Run Model"""
            if run:
                """run"""

    def setCaffeDefaults(self, config_file=None):
        import json
        import os

        if config_file == None:
            config_file = 'caffe.config.json'

        with open(config_file) as caffe_config:
            config = json.load(caffe_config)

            print "Loading Defaults"
            if config["DefaultRoot"] == "$USER":
                self.caffe_root = "/home/" + os.environ.get('USER') + config["UserRoot"][config["DefaultBranch"]]["Location"]
                self.model_root = "/home/" + os.environ.get('USER') + config["UserModelRoot"]
            elif config["DefaultRoot"] == "$SYSTEM":
                self.caffe_root = config["SystemRoot"][config["DefaultBranch"]]["Location"]
                self.model_root = config["SystemModelRoot"]

            if config["DefaultDatasetRoot"] == "$USER":
                self.dataset_root = "/home/" + os.environ.get('USER') + config["UserDatasetRoot"]
            elif config["DefaultDatasetRoot"] == "$SYSTEM":
                self.dataset_root = config["SystemDatasetRoot"]

            self.allowModelBranchSpecification = config["AllowModelBranchSpecification"]

            print self.caffe_root
            print self.allowModelBranchSpecification
            print self.model_root
            print self.dataset_root
    def listModels(self, model_list_file=None, keywords=[], authors=[], showFields=["Model Name"], returnType=[]):
        import json

        if model_list_file == None:
            model_list_file = 'caffe.models.json'

        print "Trying to Read List of Models: " + model_list_file
        with open(model_list_file) as model_list:
            models = json.load(model_list)

            print "Listing Models"
            for sName, model in models.items():
                print "Model (short name): " + sName

                if "Model Name" in showFields:
                    print "Model Name: " + model["Model Name"]

                if "Paper Name" in showFields:
                    print "Paper Name: " + model["Paper Name"]

    """
        find -- e.g. [Directory, Images, Original Images] will give you a list of normal image sets by Directory, Datasets

        returns None, Directory, or Filename
    """
    def listDatasets(self, keywords=None, dataset_name=None, directory_name=None, file_name=None, directory_function=None):
        import json

        with open('caffe.datasets.json') as json_datasets:
            datasets = json.load(json_datasets)

            if dataset_name == None:
                for key, value in datasets.items():
                    print key
            else:
                if directory_name == None:
                    print dataset_name
                    for item in datasets[dataset_name]["Directories"]:
                        if "Directory" in item and "Name" in item:
                            print "\t- " + item["Name"][0] + "\t" + item["Directory"][0]
                else:
                    return datasets[dataset_name]["Directories"][directory_name]["Directory"]

    def getFromDatasets(self, type="", functions=[], database_name="", name=None, returntype="directory"):
        if type.lower() == "file":
            return searchDatasetByFile(self, database_name, name, functions, returntype)
        if type.lower() == "directory":
            return searchDatasetByDirectory(self, database_name, name, functions, returntype)

    def searchDatasetByFile(self, dataset_name="", name=None, functions=[], returntype="directory"):
        import json

        match = list()
        with open('caffe.datasets.json') as json_datasets:
            datasets = json.load(json_datasets)

            if dataset_name == "":
                for key, value in datasets.items():
                    print "Dataset: " + key

                    for directory in datasets[key]["Directories"]:
                        if "Files" in directory:
                            for fname, fitem in directory["Files"].items():
                                if name != None and fname.lower() == name:
                                    if returntype == "directory" and "Directory" in directory:
                                        l = list()
                                        for d in directory["Directory"]:
                                            l.append(d + "/" + fname);

                                        return l

                                l = None
                                if "Function" in fitem:
                                    l = set(functions).intersection(fitem["Function"])

                                if l != None and len(l) == len(functions):
                                    print "\t- " + fname
                                    if returntype == "directory" and "Directory" in directory:
                                        for d in directory["Directory"]:
                                            match.append(d + "/" + fname);
            else:
                for directory in datasets[dataset_name]["Directories"]:
                    if "Files" in directory:
                        for fname, fitem in directory["Files"].items():
                            if name != None and fname.lower() == name:
                                if returntype == "directory" and "Directory" in directory:
                                    l = list()
                                    for d in directory["Directory"]:
                                        l.append(d + "/" + fname);

                                    return l
                            l = None
                            if "Function" in fitem:
                                l = set(functions).intersection(fitem["Function"])

                            if l != None and len(l) == len(functions):
                                print "\t- " + fname
                                if returntype == "directory" and "Directory" in directory:
                                    for d in directory["Directory"]:
                                        match.append(d + "/" + fname);

        return match
    def searchDatasetByDirectory(self, dataset_name="", name=[], functions=[], returntype="directory"):
        import json

        match = list()
        with open('caffe.datasets.json') as json_datasets:
            datasets = json.load(json_datasets)

            if dataset_name == "":
                for key, value in datasets.items():
                    print "Dataset: " + key

                    for directory in datasets[key]["Directories"]:
                        if name != None and "Name" in directory and len(set(name).intersection(directory["Name"])) >= 1:
                            if returntype == "directory" and "Directory" in directory:
                                return directory["Directory"]
                            else:
                                return [key, directory]

                        l = None
                        if "Function" in directory:
                            l = set(functions).intersection(directory["Function"])

                        if l != None and len(l) == len(functions) and "Name" in directory:
                            print "\t- " + directory["Name"][0]
                            if returntype == "directory" and "Directory" in directory:
                                match.append(directory["Directory"])
                            else:
                                match.append([key, directory])
            else:
                for directory in datasets[dataset_name]["Directories"]:
                    if name != None and "Name" in directory and len(set(name).intersection(directory["Name"])) >= 1:
                        if returntype == "directory" and "Directory" in directory:
                            return directory["Directory"]
                        else:
                            return [dataset_name, directory]

                    l = None
                    if "Function" in directory:
                        l = set(functions).intersection(directory["Function"])

                    if l != None and len(l) == len(functions) and "Name" in directory:
                        print "\t- " + directory["Name"][0]
                        if returntype == "directory" and "Directory" in directory:
                            match.append(directory["Directory"])
                        else:
                            match.append([key, directory])

        return match

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
        print "Loading data E"
        self.data = "/srv/datasets/mscoco/test2014/COCO_test2014_000000000001.jpg"

    #modified from https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045
    def loadData_asLmdb(self, file_listing=None, data_folder=None, lmdb_name="lmdb-temp", count=1000, section=None, imformat="RGB", option=""):
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

        print self.data_location + data_folder

        if section is not None:
            print("Test")
            inputs=inputs[section[0]:section[1]]

        print "N of Inputs: ", len(inputs)

        in_db = lmdb.open(lmdb_name, map_size=int(1e12))
        with in_db.begin(write=True) as in_txn:
            for in_idx, in_ in enumerate(inputs):
                print "", in_idx, " " + in_
                nim = Image.open(in_).resize((500, 500), Image.ANTIALIAS).convert(imformat)
                im = None
                if imformat == "RGB":
                    im = np.array(nim, dtype=np.float)
                elif imformat == "L":
                    im = np.array(nim, dtype=np.int)
                #print im.shape
                if len( im.shape ) > 0:
                    if imformat is "RGB":
                        im = im[:,:,::-1]
                        im = im.transpose((2,0,1))
                    elif imformat is "L":
                        im = im[np.newaxis, :]
                        #im = im.transpose((0,,1))

                    #if option is "r4":
                        #im = np.array(1, im)
                    #print im.shape
                    im_dat = caffe.io.array_to_datum(im)
                    in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
                else:
                    print im
        in_db.close()

    """
    Set Gist ID to Download Files From
    """
    def set_gist(self, gist_id):
        print "not implemented"
    def download_gist(self):
        from subprocess import call

        "download gist"
        if self.gist_id is not None:
            call([self.caffe_root + "/scripts/download_model_from_gist.sh", self.gist_id])
            call(["mv", self.caffe_root + "/models/" + self.gist_id, self.model_root + self.model_dir])

    def download_weights(self):
        from subprocess import call

        call(["python", self.caffe_root + "/scripts/download_model_binary.py",
                self.caffe_root + self.model_root])

    def set_model_files(self, model_root, deploy_file, solve_file, pretrained_file):
        print "not implemented"
        gist_id = "FCN_32s_PASCAL"
        self.deploy_file = "/models/" + gist_id + "/fcn-32s-pascal-deploy.prototxt"
        self.pretrained_file = "/models/" + gist_id + "/fcn-32s-pascal.caffemodel"

    # from https://gist.github.com/shelhamer/91eece041c19ff8968ee#file-solve-py
    # credit @longjon
    def upsample_filter(self, size):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        og = np.ogrid[:size, :size]

        return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

    # from https://gist.github.com/shelhamer/91eece041c19ff8968ee#file-solve-py
    def interp_surgery(self, net, layers):
        for l in layers:
            m, k, h, w = net.params[l][0].data.shape
            if m != k:
                print 'input and ouput channels need to be the same'
                raise
            if h != w:
                print 'filters need to be square'
                raise
            fltr = self.upsample_filter(h)
            net.params[l][0].data[range(m), range(k), :, :] = fltr

    def configureForTest(self):
        #plt.rcParams['figure.figsize'] = (10, 10)
<<<<<<< HEAD
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'
=======
        #plt.rcParams['image.interpolation'] = 'nearest'
        #plt.rcParams['image.cmap'] = 'gray'
>>>>>>> b5c97b63d8f7fffda41b3a8ac10bbaf008a2f61c
        import os.path

        self.use_cpu()
        deploy = self.caffe_root + "/" + self.model_dir + "/" + self.deploy_file
        pretrained = self.caffe_root + "/" + self.model_dir + "/" + self.pretrained_file
        print deploy
        print pretrained
        print os.path.isfile(deploy)
        print os.path.isfile(pretrained)

        """deploy = "/home/chrwc/dev/caffe/models/FCN_32s_PASCAL/deploy.prototxt" """
        self.net = caffe.Net( deploy.encode("utf-8"), pretrained.encode("utf-8"), caffe.TEST)

        #self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        #self.transformer.set_transpose('data', (2,0,1))
        #self.transformer.set_raw_scale('data', 255)
        #self.transformer.set_channel_swap('data', (2,1,0))
        #if self.mean is not None:
        #    self.transformer.set_mean('data', self.mean)

    def configureForTrain(self):
        #plt.rcParams['figure.figsize'] = (10, 10)
        #plt.rcParams['image.interpolation'] = 'nearest'
        #plt.rcParams['image.cmap'] = 'gray'

        self.use_cpu()

        deploy = self.caffe_root + "/" + self.model_dir + "/" + self.deploy_file
        pretrained = self.caffe_root + "/" + self.model_dir + "/" + self.pretrained_file
        solver = self.caffe_root + "/" + self.model_dir + "/" + self.solver_file

        self.base_weights = pretrained.encode("utf-8")
        self.solver = caffe.SGDSolver(solver.encode("utf-8"))

    def runTest(self, image):
        self.net.blobs['data'].reshape(self.data_dim[0],self.data_dim[1],self.data_dim[2],self.data_dim[3])
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', caffe.io.load_image(image))

        self.out = self.net.forward();
    def runTestBatch(self, data_dir, nmax=1, file_listing=None, op=""):
        print "Running Tests"
        import os
        import glob
        from PIL import Image

        path = os.getcwd()
        os.chdir(self.data_location)
        if file_listing is not None:
            k = [glob.glob(os.path.join(fListing.rstrip('\n') + '.*'))[0] for fListing in open(file_listing)]
        os.chdir(path)
        from os import listdir
        from os.path import isfile, join

        if file_listing is None:
            k = [ f for f in listdir(self.data_location) if isfile(join(self.data_location, f)) ]

        i = 0
        self.transformer = None

        if op == "":
            self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
            self.transformer.set_transpose('data', (2,0,1))
            self.transformer.set_raw_scale('data', 255)
            self.transformer.set_channel_swap('data', (2,1,0))

<<<<<<< HEAD
        while (nmax > 0 and i < nmax) or (nmax == 0 and i < len(k)) or file_listing is not None:
            print "file: " + k[i]
            if op == "fcn":
                image = Image.open(self.data_location + k[i])
=======
        while i < nmax:
            print k[i]
            if op == "fcn":
                image = Image.open(data_dir + k[i])
>>>>>>> b5c97b63d8f7fffda41b3a8ac10bbaf008a2f61c
                in_ = np.array(image, dtype=np.float32)
                in_ = in_[:,:,::-1]
                in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
                in_ = in_.transpose((2,0,1))
                self.net.blobs['data'].reshape(1, *in_.shape)
                self.net.blobs['data'].data[...] = in_
                self.out = self.net.forward()
            elif op is '':
                self.runTest(k[i])
            else:
                print "Unrecognized command \"op=" + op + "\""
            self.visualizeLayers(k[i])
            self.visualizeOutput(k[i], blob_name='score')
            #self.visualizeOutput(k[i], blob_name='upscore')
            #self.visualizeOutput(k[i], blob_name='bigscore')
            #self.visualizeOutput(k[i], blob_name='pool5')
            i = i + 1

        print "End of Testing"

    def runTrainS(self, step_count):
        print "Training"
        cpath = os.getcwd()

        solver = self.caffe_root + "/" + self.model_dir + "/" + self.solver_file
        os.chdir(self.caffe_root + "/" + self.model_dir)
        self.solver = caffe.SGDSolver(solver.encode("utf-8"))
        interp_layers = [k for k in self.solver.net.params.keys() if 'up' in k]
        self.interp_surgery(self.solver.net, interp_layers)

        self.solver.net.copy_from(self.base_weights)

        self.solver.step(step_count)
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

    def visualizeOutput(self, save_prefix="output", blob_name=None):
        if blob_name is not None:
            f = self.net.blobs[blob_name].data[0].argmax(axis=0)
            if len(f.shape) == 2:
                plt.imsave(fname=self.caffe_root + "/" + "output/"+save_prefix+"_"+blob_name+"."+self.default_image_format, arr=f, format=self.default_image_format)

    def visualizeLayers(self, save_prefix=""):
        for k, v in self.net.blobs.items():
            if (k[0] is not 'f' or k[1] is not 'c') and (k[0] is not 'p' or k[1] is not 'r'):
                f = self.net.blobs[k].data[0, :36]
                fig = self.vis_square(f, padval=1)
                plt.imsave(fname="output/" + save_prefix + "_" + k + "." + self.default_image_format, arr=fig, format=self.default_image_format)

    def visualizeFilters(self):
        for k, v in self.net.params.items():
            print k + " ", len(v[0].data.shape)
            if (len(v[0].data.shape) == 4) and (k[0] != 'f') and (k[0] != 's') and (k[0] != 'u'):
                print "(", v[0].data.shape[0], ")"
                fig = self.vis_square(v[0].data[0, :40])
                plt.imsave(fname="output/" + "filter_" + k + "." + self.default_image_format, arr=fig, format=self.default_image_format);

        print "Visualizing Filters"
