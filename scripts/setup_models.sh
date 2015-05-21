#!/bin/bash

ALEXNET_CNN_SOS_GIST=0585ed9428dc5222981f
ALEXNET_CNN_SOS_DIR=models/ALEXNET_CNN_SOS

VGG16_CNN_SOS_GIST=27c1c0a7736ba66c2395
VGG16_CNN_SOS_DIR=models/VGG16_CNN_SOS

DIRECTORY=$ALEXNET_CNN_SOS_DIR
if [ ! -d "$DIRECTORY" ]; then
	./scripts/download_model_from_gist.sh $ALEXNET_CNN_SOS_GIST
	#./scripts/download_model_binary.py models/$ALEXNET_CNN_SOS_GIST
	wget -P models/$ALEXNET_CNN_SOS_GIST http://www.cs.bu.edu/groups/ivc/data/SOS/AlexNet_SalObjSub.caffemodel

	mv ./models/$ALEXNET_CNN_SOS_GIST ./$DIRECTORY
fi

DIRECTORY=$VGG16_CNN_SOS_DIR
if [ ! -d "$DIRECTORY" ]; then
	./scripts/download_model_from_gist.sh $VGG16_CNN_SOS_GIST
	#./scripts/download_model_binary.py models/$VGG16_CNN_SOS_GIST

	wget -P models/$VGG16_CNN_SOS_GIST http://www.cs.bu.edu/groups/ivc/data/SOS/VGG16_SalObjSub.caffemodel
	mv models/$VGG16_CNN_SOS_GIST $DIRECTORY
fi
