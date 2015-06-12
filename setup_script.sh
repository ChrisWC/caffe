#install latest driver from nvidia
cd ~

sudo apt-get install tmux vim ctags cmake git-all \
libboost-all-dev libopencv-dev libopencv-core-dev libopencv-gpu-dev \
libatlas-dev libatlas-base-dev libprotobuf-dev libleveldb-dev libopencv-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler libsnappy-dev dconf-cli

#install cuda
if [ ! -f cuda-repo-ubuntu1204_7.0-28_amd64.deb ]; then
	wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1204/x86_64/cuda-repo-ubuntu1204_7.0-28_amd64.deb

	sudo dpkg -i cuda-repo-ubuntu1204_7.0-28_amd64.deb

	sudo apt-get update

	sudo apt-get install cuda

	export PATH=/usr/local/cuda-7.0/bin:\$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
	echo "PATH=\"/usr/local/cuda-7.0/bin:\$PATH\"" | sudo tee -a /etc/bash.bashrc
	echo "LD_LIBRARY_PATH=\"/usr/local/cuda-7.0/lib64:\$LD_LIBRARY_PATH\"" | sudo tee -a /etc/bash.bashrc

	cuda-install-samples-7.0.sh ~/samples/NVIDIA_CUDA-7.0_Samples/

	cd ~/samples/

	sudo make
	
	echo "You must restart to be able to run samples"
	
	cd ~
fi

#install anaconda-python
if [ ! -f Anaconda-2.2.0-Linux-x86_64.sh ]; then
	wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda-2.2.0-Linux-x86_64.sh
	sudo bash Anaconda-2.2.0-Linux-x86_64.sh
	echo "PATH=\"/opt/anaconda/bin:\$PATH\"" | sudo tee -a /etc/bash.bashrc
fi



if [ ! -d ~/dev/caffe ]; then
	#install caffe
	HOME=~/
	mkdir -p ~/dev
	git clone -b future https://github.com/ChrisWC/caffe.git ~/dev/caffe

	echo "Please go to ~/dev/caffe/ and build and make targets"

	mkdir -p ~/dev/caffe/build
	cmake -B$HOME/dev/caffe/build -H$HOME/dev/caffe

	make -C $HOME/dev/caffe/build all
	make -C $HOME/dev/caffe/build runtest

	echo "export PYTHONPATH=\"/path/to/caffe/python:$PYTHONPATH\"" | sudo tee -a ~/bashrc
fi

#install vim plugins -- OPTIONAL
if [ ! -d ~/.vim_runtime ]; then
	git clone git://github.com/amix/vimrc.git ~/.vim_runtime
	sh ~/.vim_runtime/install_awesome_vimrc.sh
fi

sudo apt-get install 

#install terminal theme -- OPTIONAL
if [ ! -d ~/solarized ]; then
	git clone https://github.com/Anthony25/gnome-terminal-colors-solarized.git ~/solarized

	~/solarized/install.sh
fi

#install additional python libraries -- do not use apt-get since we use a 3rd party python implementation/libraries
CMD=`which easy_install`
echo $CMD
sudo $CMD protobuf

PSCL_CMP_DIR=/srv/datasets/pascal/compressed
PSCL_DIR=/srv/datasets/pascal

sudo mkdir -p $PSCL_CMP_DIR
sudo mkdir -p $PSCL_DIR
#PASCAL VOC 2005
#URL: http://host.robots.ox.ac.uk/pascal/VOC/voc2005/index.html

#DATASET 1
#INFORMATION: http://host.robots.ox.ac.uk/pascal/VOC/databases.html#VOC2005_1
#DATABASE URL: http://www.pascal-network.org/challenges/VOC/databases.html
if [ ! -f $PSCL_CMP_DIR/voc2005_1.tar.gz ]; then
	sudo wget -P $PSCL_CMP_DIR/ http://host.robots.ox.ac.uk/pascal/VOC/download/voc2005_1.tar.gz
fi
if [ -f $PSCL_CMP_DIR/voc2005_1.tar.gz  ]; then
	sudo tar -xvf -k $PSCL_CMP_DIR/voc2005_1.tar.gz  -C $PSCL_DIR
fi
#DATASET 2
#INFORMATION: http://host.robots.ox.ac.uk/pascal/VOC/databases.html#VOC2005_2
#DATABASE URL: http://www.pascal-network.org/challenges/VOC/databases.html
if [ ! -f $PSCL_CMP_DIR/voc2005_2.tar.gz ]; then
	sudo wget -P $PSCL_CMP_DIR/ http://host.robots.ox.ac.uk/pascal/VOC/download/voc2005_2.tar.gz
fi
if [ -f $PSCL_CMP_DIR/voc2005_2.tar.gz  ]; then
	sudo tar -xvf -k $PSCL_CMP_DIR/voc2005_2.tar.gz  -C $PSCL_DIR
fi
#PASCAL VOC 2006
#URL: http://host.robots.ox.ac.uk/pascal/VOC/voc2006/index.html

#DATASET, train+val sets
if [ ! -f /srv/datasets/pascal/compressed/voc2006_trainval.tar ]; then
	sudo wget -P /srv/datasets/pascal/compressed/ http://host.robots.ox.ac.uk/pascal/VOC/download/voc2006_trainval.tar
fi
if [ -f $PSCL_CMP_DIR/voc2006_trainval.tar  ]; then
	sudo tar -xvf -k $PSCL_CMP_DIR/voc2006_trainval.tar  -C $PSCL_DIR
fi
#DATASET, test set
if [ ! -f /srv/datasets/pascal/compressed/voc2006_test.tar ]; then
	sudo wget -P /srv/datasets/pascal/compressed/ http://host.robots.ox.ac.uk/pascal/VOC/download/voc2006_test.tar
fi
if [ -f $PSCL_CMP_DIR/voc2006_test.tar ]; then
	sudo tar -xvf -k $PSCL_CMP_DIR/voc2006_test.tar -C $PSCL_DIR
fi
#PASCAL VOC 2007
#URL: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html

#DATASET training/validation
if [ ! -f /srv/datasets/pascal/compressed/VOCtrainval_06-Nov-2007.tar ]; then
	sudo wget -P /srv/datasets/pascal/compressed/ http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
fi
if [ -f $PSCL_CMP_DIR/VOCtrainval_06-Nov-2007.tar ]; then
	sudo tar -xvf -k $PSCL_CMP_DIR/VOCtrainval_06-Nov-2007.tar -C $PSCL_DIR
fi
#DATASET test
if [ ! -f /srv/datasets/pascal/compressed/VOCtest_06-Nov-2007.tar ]; then
	sudo wget -P /srv/datasets/pascal/compressed/ http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
fi
if [ -f $PSCL_CMP_DIR/VOCtest_06-Nov-2007.tar ]; then
	sudo tar -xvf -k $PSCL_CMP_DIR/VOCtest_06-Nov-2007.tar -C $PSCL_DIR
fi
if [ ! -f /srv/datasets/pascal/compressed/VOCtestnoimgs_06-Nov-2007.tar ]; then
	sudo wget -P /srv/datasets/pascal/compressed/ http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar
fi
if [ -f $PSCL_CMP_DIR/VOCtestnoimgs_06-Nov-2007.tar ]; then
	sudo tar -xvf -k $PSCL_CMP_DIR/VOCtestnoimgs_06-Nov-2007.tar -C $PSCL_DIR
fi
#PASCAL VOC 2008
#URL: http://host.robots.ox.ac.uk/pascal/VOC/voc2008/index.html

#DATASET
if [ ! -f /srv/datasets/pascal/compressed/VOCtrainval_14-Jul-2008.tar ]; then
	sudo wget -P /srv/datasets/pascal/compressed/ http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar
fi
if [ -f $PSCL_CMP_DIR/VOCtrainval_14-Jul-2008.tar ]; then
	sudo tar -xvf -k $PSCL_CMP_DIR/VOCtrainval_14-Jul-2008.tar -C $PSCL_DIR
fi
#PASCAL VOC 2009
#URL: http://host.robots.ox.ac.uk/pascal/VOC/voc2009/index.html

#DATASET training/validation
if [ ! -f /srv/datasets/pascal/compressed/VOCtrainval_11-May-2009.tar ]; then
	sudo wget -P /srv/datasets/pascal/compressed/ http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar
fi
if [ -f $PSCL_CMP_DIR/VOCtrainval_11-May-2009.tar ]; then
	sudo tar -xvf -k $PSCL_CMP_DIR/VOCtrainval_11-May-2009.tar -C $PSCL_DIR
fi
#PASCAL VOC 2010
#URL: http://host.robots.ox.ac.uk/pascal/VOC/voc2010/index.html

#DATASET
if [ ! -f /srv/datasets/pascal/compressed/VOCtrainval_03-May-2010.tar ]; then
	sudo wget -P /srv/datasets/pascal/compressed/ http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
fi
if [ -f $PSCL_CMP_DIR/VOCtrainval_03-May-2010.tar ]; then
	sudo tar -xvf -k $PSCL_CMP_DIR/VOCtrainval_03-May-2010.tar -C $PSCL_DIR
fi
#PASCAL VOC 2011
#URL: http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html

#DATASET
if [ ! -f /srv/datasets/pascal/compressed/VOCtrainval_25-May-2011.tar ]; then
	sudo wget -P /srv/datasets/pascal/compressed/ http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar
fi

if [ -f $PSCL_CMP_DIR/VOCtrainval_25-May-2011.tar ] && [ ! -d $PSCL_DIR/TrainVal ]; then
	sudo tar -xvf -k $PSCL_CMP_DIR/VOCtrainval_25-May-2011.tar -C $PSCL_DIR
fi

#PASCAL VOC 2012
#URL: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html

#DATASET, TRAINING/VALIDATION
if [ ! -f $PSCL_CMP_DIR/VOCtrainval_11-May-2012.tar ]; then
	sudo wget -P /srv/datasets/pascal/compressed/ http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
fi

if [ -f $PSCL_CMP_DIR/VOCtrainval_11-May-2012.tar ] && [ ! -d $PSCL_DIR/VOCdevkit/VOC2012 ]; then
	sudo tar -xvf -k $PSCL_CMP_DIR/VOCtrainval_11-May-2012.tar -C $PSCL_DIR
fi

#MSCOCO
MSCOCO_CMP_DIR=/srv/datasets/mscoco/compressed
MSCOCO_DIR=/srv/datasets/mscoco
mkdir -p $MSCOCO_CMP_DIR/
mkdir -p $MSCOCO_DIR/

#TOOLS
if [ ! -d ~/dev/coco ]; then
	git clone https://github.com/pdollar/coco.git ~/dev/coco
fi

#IMAGE DATASETS, TRAINING IMAGES
if [ ! -f $MSCOCO_CMP_DIR/train2014.zip ]; then
	sudo wget -P $MSCOCO_CMP_DIR http://msvocds.blob.core.windows.net/coco2014/train2014.zip
fi
sudo unzip -n $MSCOCO_CMP_DIR/train2014.zip -d $MSCOCO_DIR/

#IMAGE DATASETS, VALIDATION IMAGES
if [ ! -f $MSCOCO_CMP_DIR/val2014.zip ]; then
	sudo wget -P $MSCOCO_CMP_DIR http://msvocds.blob.core.windows.net/coco2014/val2014.zip
fi
sudo unzip -n $MSCOCO_CMP_DIR/val2014.zip -d $MSCOCO_DIR

#IMAGE DATASETS, TEST IMAGES
if [ ! -f $MSCOCO_CMP_DIR/test2014.zip ]; then
	sudo wget -P $MSCOCO_CMP_DIR http://msvocds.blob.core.windows.net/coco2014/test2014.zip
fi
sudo unzip -n $MSCOCO_CMP_DIR/test2014.zip -d $MSCOCO_DIR

#ANNOTATIONS, TRAINING ANNOTATIONS
if [ ! -f $MSCOCO_CMP_DIR/instances_train2014.json ]; then
	sudo wget -P $MSCOCO_CMP_DIR http://msvocds.blob.core.windows.net/annotations-1-0-2/instances_train2014.json
fi

#ANNOTATIONS, VALIDATION ANNOTATIONS
if [ ! -f $MSCOCO_CMP_DIR/instances_val2014.json ]; then
	sudo wget -P $MSCOCO_CMP_DIR http://msvocds.blob.core.windows.net/annotations-1-0-2/instances_val2014.json
fi

#ANNOTATIONS, TRAINING CAPTIONS
if [ ! -f $MSCOCO_CMP_DIR/captions_train2014.json ]; then
	sudo wget -P $MSCOCO_CMP_DIR http://msvocds.blob.core.windows.net/annotations-1-0-2/captions_train2014.json
fi

#ANNOTATIONS, VALIDATION CAPTIONS
if [ ! -f $MSCOCO_CMP_DIR/captions_val2014.json ]; then
	sudo wget -P $MSCOCO_CMP_DIR http://msvocds.blob.core.windows.net/annotations-1-0-2/captions_val2014.json
fi

