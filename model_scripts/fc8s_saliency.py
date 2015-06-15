from interface import MDL
import os
from subprocess import call

model = MDL()
model.loadModel("fcn8s_context.model.json")
path = os.getcwd()
npath = model.caffe_root + "/" + model.model_dir
os.chdir(npath)
print "PATH: " + npath + " " + os.getcwd()

model.data_location = "/srv/datasets/salObj/datasets/"
model.data_dim = [1,3,500,500]
#model.load_data_as_lmdb("/home/chriswc/data/lmdb/pascal2010")

model.set_mean((104.00698793, 116.66876762, 122.67891434))

genDB = True
if genDB is True:
    call(["rm", "-rf", "*lmdb/"])
    model.loadData_asLmdb(data_folder="imgs/pascal/", lmdb_name="input-lmdb", section=[0,200], imformat="RGB")
    model.loadData_asLmdb(data_folder="masks/pascal/", lmdb_name="output-lmdb", section=[0,200], imformat="L")
    model.loadData_asLmdb(data_folder="imgs/pascal/", lmdb_name="input-val-lmdb", section=[200,400], imformat="RGB")
    model.loadData_asLmdb(data_folder="masks/pascal/", lmdb_name="output-val-lmdb", section=[200,400], imformat="L")

model.set_mean((104.00698793, 116.66876762, 122.67891434))

Train = True
if Train == True:
    model.configureForTrain()
    model.use_cpu()

    model.runTrainS(1000)

    model.solver.net.save("pascalSInt100.caffemodel")
os.chdir(path)
Test = False
if Test == True:
    model.configureForTest()
    model.data_location = "/srv/datasets/pascal/VOCdevkit/VOC2010/JPEGImages/"
    image_set = "/srv/datasets/pascal/VOCdevkit/VOC2010/ImageSets/Segmentation/train.txt"
    model.load_data()
    model.use_gpu()
    model.runTestBatch(model.data_location, 5, image_set, op="fcn")
os.chdir(path)
#model.runTestBatch(model.data_location, 5)

        #model.visualizeLayers()
#model.visualizeFilters()

#plt.imshow(model.out['upscore'][0, 20])
