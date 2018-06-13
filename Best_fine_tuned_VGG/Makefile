###################################
#
#
#    Saliency Estimation Network 
#    (conv with euclidean loss)
#
#
###################################

# experiment information
DATASET=salicon
CAFFE=caffe
GPUID=0
ALPHA=0.5
LRATE=0.001
CPD=19
DMODE=sal-$(CPD)
NSLVR=best_fine_tuned

# std paths:
PTH_LIST=list
PTH_DS=/opt/CV_db/image_databases/pub/SALICON
PTH_GSCRIPT=/home/$(USER)/workspace/pyscripts/general
PTH_CSCRIPT=/home/$(USER)/workspace/DITC/pyscripts
PTH_EXPSCRIPTS=/home/$(USER)/projects/saliency/cs_deepsaliency/scripts
# case defined:
PTH_SHR=/local/$(USER)
PTH_EXP=$(PTH_SHR)/$(DATASET)
PTH_EDAT=/dev/shm/$(USER)/salicon/data
TEST_MODEL=best.caffemodel

BIN_PYTHON=python

######################
# DEPLOY NET 
######################
PTH_SLVR=$(PTH_EXP)/VGG_sal/solverset_$(NSLVR)

MAPDIR_TST=fortest_maps
MATDIR_TST=fortest_mats
JSONDIR_TST=$(PTH_SLVR)/fortest_rjson
JSONNAME=valresult

deploy:
	mkdir -p $(MAPDIR_TST)/
	mkdir -p $(MATDIR_TST)/
	mkdir -p $(JSONDIR_TST)/
	rm -f $(MAPDIR_TST)/*
	rm -f $(MATDIR_TST)/*
	rm -f $(JSONDIR_TST)/*

	$(BIN_PYTHON) -c "import numpy as np; np.save('alpha',$(ALPHA))"
	$(BIN_PYTHON) -u deploy_net.py \
	$(TEST_MODEL) \
	sal_train_val.prototxt \
	$(MAPDIR_TST) \
	$(MATDIR_TST) \
	$(JSONDIR_TST) \
	$(JSONNAME) \
	$(GPUID) # > $(PTH_SLVR)/deploy.log 2>&1
