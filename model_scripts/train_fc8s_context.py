from interface import MDL
import os

model = MDL()
model.loadModel("fcn8s_context.model.json")
path = os.getcwd()
npath = model.caffe_root + "/" + model.model_dir
os.chdir(npath)
print "PATH: " + npath + " " + os.getcwd()

model.data_location = "/srv/datasets/pascal/VOCdevkit/VOC2010/"
model.data_dim = [1,3,500,500]
#model.load_data_as_lmdb("/home/chriswc/data/lmdb/pascal2010")

model.set_mean((104.00698793, 116.66876762, 122.67891434))

genDB = False
if genDB is True:
            model.loadData_asLmdb(file_listing="/srv/datasets/pascal/VOCdevkit/VOC2010/ImageSets/Segmentation/train.txt", data_folder="JPEGImages/", section=[0,200], lmdb_name="input-lmdb", imformat="RGB")
            model.loadData_asLmdb(file_listing="/srv/datasets/pascal/VOCdevkit/VOC2010/ImageSets/Segmentation/train.txt", data_folder="SegmentationClass/", section=[0,200], lmdb_name="output-lmdb", imformat="L")
            model.loadData_asLmdb(file_listing="/srv/datasets/pascal/VOCdevkit/VOC2010/ImageSets/Segmentation/val.txt", data_folder="JPEGImages/", section=[200,400], lmdb_name="input-val-lmdb", imformat="RGB")
            model.loadData_asLmdb(file_listing="/srv/datasets/pascal/VOCdevkit/VOC2010/ImageSets/Segmentation/val.txt", data_folder="SegmentationClass/", section=[200,400], lmdb_name="output-val-lmdb", imformat="L")

model.set_mean((104.00698793, 116.66876762, 122.67891434))

Train = False
if Train == True:
    model.configureForTrain()
    model.use_cpu()

    model.runTrainS(100)

    model.solver.net.save("pascalSInt100.caffemodel")
os.chdir(path)
Test = True
if Test == True:
    model.pretrained_file = "pretrained.caffemodel"
    model.configureForTest()
    model.data_location = "/srv/datasets/pascal/VOCdevkit/VOC2010/JPEGImages/"
    image_set = "/srv/datasets/pascal/VOCdevkit/VOC2010/ImageSets/Segmentation/train.txt"
    model.load_data()
    model.use_gpu()
    model.set_mean((104.00698793, 116.66876762, 122.67891434))
    model.runTestBatch(model.data_location, 5, image_set, op="fcn")
os.chdir(path)
#model.runTestBatch(model.data_location, 5)

        #model.visualizeLayers()
#model.visualizeFilters()

#plt.imshow(model.out['upscore'][0, 20])
