#!/bin/bash

FCN_32s_PASCAL=ac410cad48a088710872
FCN_16s_PASCAL=d24098e083bec05e456e
FCN_8s_PASCAL=1bf3aa1e0b8e788d7e1d
FCN_16s_SIFT_FLOW=f35e3a101e1478f721f5
FCN_32s_NYUDv2=16db1e4ad3afc2614067
FCN_16s_NYUDv2=dd1f5097af6b531bddcc
FCN_32s_PASCAL_CONTEXT=80667189b218ad570e82
FCN_16s_PASCAL_CONTEXT=08652f2ba191f64e619a
FCN_8s_PASCAL_CONTEXT=91eece041c19ff8968ee

DIRECTORY=models/FCN_16s_PASCAL
if [ ! -d "$DIRECTORY" ]; then
	./scripts/download_ model_from_gist.sh $FCN_32s_PASCAL
	./scripts/download_model_binary.py models/$FCN_32s_PASCAL

	mv models/$FCN_32s_PASCAL models/FCN_32s_PASCAL
	rm -rf $FCN_32s_PASCAL
fi


DIRECTORY=models/FCN_16s_PASCAL
if [ ! -d "$DIRECTORY" ]; then
	./scripts/download_model_from_gist.sh $FCN_16s_PASCAL
	./scripts/download_model_binary.py models/$FCN_16s_PASCAL

	mv models/$FCN_16s_PASCAL models/FCN_16s_PASCAL
	rm -rf $FCN_16s_PASCAL
fi


DIRECTORY=models/FCN_8s_PASCAL_CONTEXT
if [ ! -d "$DIRECTORY" ]; then
	./scripts/download_model_from_gist.sh $FCN_8s_PASCAL_CONTEXT
	./scripts/download_model_binary.py models/$FCN_8s_PASCAL_CONTEXT

	mv models/$FCN_8s_PASCAL_CONTEXT models/FCN_8s_PASCAL_CONTEXT
	rm -rf $FCN_8s_PASCAL_CONTEXT
fi
