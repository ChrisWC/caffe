from caffe_model import MDL


model = MDL(caffe_root="/home/chriswc/dev/caffe/",data_location="/home/chriswc/VOCdevkit2010/VOC2010/JPEGImages/",deploy_file="models/FCN_32s_PASCAL/deploy.prototxt",pretrained_file="models/FCN_32s_PASCAL/pretrained.caffemodel")
model.configureForTest()

model.data_dim = [1,3,500,500]
model.load_data()
model.set_mean([104.00698793, 116.66876762, 122.67891434])
#model.use_gpu()
model.runTestBatch(model.data_location, 5)

        #model.visualizeLayers()
model.visualizeFilters()

#plt.imshow(model.out['upscore'][0, 20])
