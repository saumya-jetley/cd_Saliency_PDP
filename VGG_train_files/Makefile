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
LRATE=1.0
CPD=19
DMODE=sal-$(CPD)
NSLVR=vgg_gbd$(ALPHA)_$(LRATE)_$(DMODE)_salicon

# solver params:
MAX_ITER=50000
DEPLOY_ITER=$(MAX_ITER)
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
PTH_EDAT=/opt/CV_db/Users/$(USER)/salicon/data
PRETRNED_MODEL=/home/$(USER)/projects/saliency/cs_deepsaliency/data/VGG_ILSVRC_16_layers.caffemodel

BIN_PYTHON=/opt/CV_tools/conda/envs/$(USER)/bin/python
BIN_PYTHON=python

all:
	make lmdbs
	make trn_tst

######################
# CREATE LMDB DATASET 
######################
LMDBS=lmdb_$(DMODE)_
PTH_LMDB=$(PTH_EDAT)/$(LMDBS)$(ISET)2014/VGG_sal

lmdbs:
	make store_lmdb ISET=train
	make store_lmdb ISET=val

store_lmdb:
	make $(PTH_LMDB)/lmdb.log

$(PTH_LMDB)/lmdb.log:
	rm -rf   $(PTH_LMDB)/
	mkdir -p $(PTH_LMDB)/
	$(BIN_PYTHON) -u $(PTH_EXPSCRIPTS)/save_salicon_saliency_lmdbs.py $(PTH_LMDB) $(ISET) $(DMODE) > $@

meanproto:
	make $(PTH_LMDB)/meanproto.log

$(PTH_LMDB)/meanproto.log:
	sh make_salicon_mean.sh $(PTH_LMDB)/imgs $(ISET) > $@


######################
# CREATE FT LMDB 
######################
PTH_FT_LMDB=$(PTH_EDAT)/$(LMDBS)_cat2000/VGG_sal

ft_lmdb:
	make $(PTH_FT_LMDB)/lmdb.log

$(PTH_FT_LMDB)/lmdb.log:
	rm -rf   $(PTH_FT_LMDB)/
	mkdir -p $(PTH_FT_LMDB)/
	$(BIN_PYTHON) -u $(PTH_EXPSCRIPTS)/save_cat2000_saliency_lmdbs.py $(PTH_FT_LMDB) $(DMODE) > $@
	# $(BIN_PYTHON) -m pdb $(PTH_EXPSCRIPTS)/save_cat2000_saliency_lmdbs.py $(PTH_FT_LMDB) $(DMODE)

######################
# TRAIN & DEPLOY NET 
######################
PTH_SLVR=$(PTH_EXP)/VGG_sal/solverset_$(NSLVR)
MAPDIR=fortest_maps
JSONDIR=$(PTH_SLVR)/fortest_rjson
PREDFOLDER=$(PTH_SLVR)/pred_images
PRETRND_NETS=$(PTH_SLVR)/pretrained_nets
JSONNAME=valresult

MAPDIR_TST=fortest_maps_tst
JSONDIR_TST=$(PTH_SLVR)/fortest_rjson

trn_tst:
	rm -f $(PTH_SLVR)/sal_train_val.prototxt
	make $(PTH_SLVR)/sal_train_val.prototxt
	make vggnet

$(PTH_SLVR)/sal_train_val.prototxt: sal_train_val_default_conv.prototxt
	mkdir -p $(PTH_SLVR)
	cat $< | sed \
	-e "s@replacewithimglmdbfile_train@$(PTH_EDAT)/$(LMDBS)train2014/VGG_sal/imgs/lmdb@g" \
	-e "s@replacewithfmplmdbfile_train@$(PTH_EDAT)/$(LMDBS)train2014/VGG_sal/fmaps_sal/lmdb@g" \
	-e "s@replacewithimglmdbfile_test@/local/$(USER)/MIT1003/data/lmdb_val/imgs@g" \
	-e "s@replacewithfmplmdbfile_test@/local/$(USER)/MIT1003/data/lmdb_val/fmaps@g" \
	> $@

vggnet:
	mkdir -p $(MAPDIR)
	mkdir -p $(JSONDIR)
	mkdir -p $(PREDFOLDER)
	mkdir -p $(PRETRND_NETS)

	rm -f $(MAPDIR)/*
	rm -f $(JSONDIR)/*
	rm -f $(PREDFOLDER)/*
	rm -f $(PRETRND_NETS)/*

	$(BIN_PYTHON) -c "import numpy as np; np.save('alpha',$(ALPHA))"
	$(BIN_PYTHON) -u train_net.py \
	$(PRETRNED_MODEL) \
	$(PTH_SLVR)/sal_train_val.prototxt \
	$(PTH_SLVR)/solver.prototxt \
	$(PTH_SLVR)/sal_vgg16.caffemodel \
	$(PTH_SLVR) \
	$(MAPDIR) \
	$(JSONDIR) \
	$(JSONNAME) \
	$(PREDFOLDER) \
	$(PRETRND_NETS) \
	$(DMODE) \
	$(GPUID) > $(PTH_SLVR)/train.log 2>&1
