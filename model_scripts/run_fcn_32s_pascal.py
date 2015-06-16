from interface import MDL

model = MDL()

model.loadModel("fcn32s.model.json")

model.configureForTest()

model.data_location = "/srv/datasets/pascal/VOCdevkit/VOC2010/JPEGImages/"
model.data_dim = [1,3,500,500]
model.load_data()
model.set_mean((104.00698793, 116.66876762, 122.67891434))

model.use_gpu()
model.runTestBatch(model.data_location, 5, op="fcn")

        #model.visualizeLayers()
model.visualizeFilters()

#plt.imshow(model.out['upscore'][0, 20])
