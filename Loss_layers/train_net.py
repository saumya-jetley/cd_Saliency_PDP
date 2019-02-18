import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import skimage
import skimage.io as skio
import pdb
import re
import subprocess
import sys
import os
import lmdb
import scipy.ndimage.filters as scifilt
import scipy.misc as scimisc
import datetime
import json
import base64
import cStringIO
import itertools
import random
import tempfile
from shutil import rmtree # to remove non-empty folders - fails for readonly files

# sys.path.insert(1,'/home/sjetley/workspace_eclipse/salicon_papi')
sys.path.insert(1,'/home/$(USER)/workspace/pyscripts/caffe')
sys.path.insert(1,'/home/$(USER)/workspace/pyscripts/salicon/salicon-evaluation/saliconeval')
sys.path.insert(1,'/home/$(USER)/workspace/pyscripts/salicon/salicon-evaluation/')
from salicon.salicon import SALICON
from saliconeval.eval import SALICONEval
from image2json import ImageTools 
import caffe
# import filter_fns as ffns
# import frcnn_fns as frcnnf
import sal_helpers as salh

def augment_data(mats):
    # randomly flip:
    if np.random.rand(1) < 0.5:
        new_mats = [np.fliplr(mat) for mat in mats]
        # print 'flip'
    else:
        new_mats = mats
        # print 'noflip'

    # # randomly add noise:
    # if np.random.rand(1) < 0.5:
    #     new_mats = 

    return new_mats

def get_inputs_from_lmdb(dbs, im_means, datum, idx=None, aug=None):
    # get sample:
    if not idx == None:
        db      = dbs
        imid    = db['imid'][idx]
    else:
        rnd_db  = np.random.choice(len(dbs), 1)
        db      = dbs.items()[rnd_db][1]
        imid    = random.choice(db['imid'])

    # load sample data:
    img      = db['anns'].loadImgs(imid)[0]
    imgname  = img['file_name']

    # create key:
    if 'salicon' in db['db_name']:
        key = 'orig_name' + imgname + '_m480_n640'
    else:
        key = imgname

    val      = db['img_csr'].get(key.encode())
    datum.ParseFromString(val)
    darray   = caffe.io.datum_to_array(datum)
    I        = np.transpose(darray,(1,2,0))

    val      = db['map_csr'].get(key.encode())
    datum.ParseFromString(val)
    darray   = caffe.io.datum_to_array(datum)
    fmap     = np.transpose(darray,(1,2,0)).squeeze()

    if 'mit' in db['db_name']:
        I = I[:,:,::-1]
        # print 'mit'

    if not aug == None:
        I, fmap = augment_data((I, fmap))

    # fig, (ax1, ax2) = plt.subplots(1,2,figsize=(18,10))
    # ax1.imshow(I[:,:,::-1])
    # ax2.imshow(fmap)
    # plt.show()

    # modify to caffe format:
    I = I.astype(np.float32)    
    I = I - im_means[::-1]
    I = np.transpose(I[:,:,:,np.newaxis], (3,2,0,1)).astype(np.float32)
    fmap = np.transpose(fmap[:,:,np.newaxis,np.newaxis], (3,2,0,1)).astype(np.float32)

    return [I, fmap], imgname

def do_validation(mapdir, jsondir, jsonname, db, im_means, datum, solver, rst_blob_names, map_name, resfile_name, tst_cnt, test_loss, seval_cc, predfolder, mx_tst_itr):

    # ready the fmap_prediction folder and the resultjson folder - REFRESH before every run test run
    rmtree(mapdir)
    os.mkdir(mapdir)
    rmtree(jsondir)
    os.mkdir(jsondir)

    bcdiv_loss = 0 # for this particular test run
    num_val    = len(db['imid'])
    for vali in range(num_val):
        #run the test net 
        net_inputs, imgname = get_inputs_from_lmdb(db, im_means, datum, idx=vali, aug=None)
        salh.copy_blobs_to_net(solver.net, rst_blob_names, net_inputs)
        solver.net.forward()

        #get the image
        pmap = np.transpose(solver.net.blobs[map_name].data,(2,3,1,0))[:,:,:,0].squeeze()

        # softmax normalization for prob. distribution
        bcdiv_loss = bcdiv_loss + solver.net.blobs['loss_fmap'].data
        pmap_exp   = np.exp(gamma*pmap-np.max(pmap))
        pmap_snorm = pmap_exp/np.sum(pmap_exp)
        scimisc.imsave('{}/{}'.format(mapdir,imgname),pmap_snorm)

    bcdiv_loss = bcdiv_loss / num_val
    print 'No of images - {}'.format(num_val)		

    # call the image2json python script
    sp = ImageTools('{}'.format(mapdir),'{}/{}.json'.format(jsondir,jsonname))
    sp.convert()
    sp.dumpRes()

    subtypes=['results', 'evalImgs', 'eval']
    resFile, evalImgsFile, evalFile = ['{}/{}.json'.format(jsondir,jsonname) for subtype in subtypes]

    mitRes = db['anns'].loadRes(resFile)

    # eval setup
    mitEval = SALICONEval(db['anns'], mitRes)

    # get all the images in resultjson
    mitEval.params['image_id'] = mitRes.getImgIds()

    # evaluate results
    if 'salicon' in db['db_name']:
        mitEval.evaluate()
    else:
        metrics = ['AUC', 'NSS', 'CC']
        mitEval.evaluate(metrics, filterAnns=True)

    # print output evaluation scores
    print "Final result for each metric being written in " + resfile_name
    resultfile = open(resfile_name,'a+')

    if tst_cnt == 0:
        resultfile.write("\ndataset\titer\t")
        for metric,score in mitEval.eval.items():
            resultfile.write('{}\t'.format(metric))

        resultfile.write("test_loss\n")

    resultfile.write('{}\t{}\t'.format(db['db_name'], solver.iter))
    for metric, score in mitEval.eval.items():
        if(metric=='CC'):
            met_cc = score
        resultfile.write('{:.5f}\t'.format(score))

    resultfile.write('{:.5f}\n'.format(bcdiv_loss[0]))
    resultfile.close()

    # Enter all the losses and metrics in result arrays and plot
    x = stepsz * np.arange(mx_tst_itr)
    test_loss[tst_cnt]  = bcdiv_loss
    seval_cc[tst_cnt]   = met_cc

    tst_cnt += 1
    
    return tst_cnt, test_loss, seval_cc

    # plt.plot(x,test_loss)
    # plt.savefig('{}/testloss_{:06d}.jpg'.format(predfolder, solver.iter), bbox_inches='tight')
    # plt.close()

    # plt.plot(x,seval_sauc)
    # plt.plot(x,seval_cc)
    # plt.savefig('{}/seval_{:06d}.jpg'.format(predfolder, solver.iter), bbox_inches='tight')
    # plt.close()

def do_mit_validation(mapdir, jsondir, jsonname, db, im_means, datum, net, rst_blob_names, map_name, resfile_name, gamma, fixmap=False):
    rename_dir = mapdir + 'renamed'

    # 1. rename your maps back to original MIT names:
    rename_command = '/home/$(USER)/workspace/experiments/salicon_sal/renameSalMapsForMIT1003.sh ' + \
                      mapdir + ' ' + \
                      rename_dir

    # 2. run matlab evaluation (on Matterhorn):
    code_path    = '/home/$(USER)/tools/cvzoya/saliency/code_forOptimization'
    eval_command = 'ssh matterhorn \'set -vx; matlab -nodesktop -nosplash -r "addpath(\'"\'"\'{}\'"\'"\'); computeMetricsMIT1003origGT(\'"\'"\'{}\'"\'"\', \'"\'"\'{}\'"\'"\'); quit"\' '.format(code_path, rename_dir, resfile_name)

    os.system(rename_command)
    os.system(eval_command)

#-----------------------------MAIN starts here--------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pretrned_file",
            help="")
    parser.add_argument("trnnet_proto",
            help="")
    parser.add_argument("trnnet_solver",
            help="")
    parser.add_argument("trnnet_file",
            help="")
    parser.add_argument("out_path",
            help="")
    parser.add_argument("mapdir",
            help="")
    parser.add_argument("jsondir",
            help="")
    parser.add_argument("jsonname",
            help="")
    parser.add_argument("predfolder",
            help="")
    parser.add_argument("pretrnd_nets",
            help="")
    parser.add_argument("dMode",
            help="smap mode: sal<sigma> or ag<cpd> (salicon version or MIT version)")
    parser.add_argument("gpuid",type=int,
            help="GPU ID")
    args   = parser.parse_args()

    dmode, fparam  = args.dMode.split('-')
    fparam         = float(fparam)
    rst_blob_names = ['imdata','fmap']
    im_means       = [123.6800, 116.7790, 103.9390]
    datum          = caffe.proto.caffe_pb2.Datum()

    #####################
    # datasets:
    #####################

    slcn_impath      = '/opt/CV_db/image_databases/pub/SALICON/train'
    lmdbpath         = '/opt/CV_db/Users/$(USER)'
    lmdbfile         = '{}/salicon/data/lmdb_sal-19_train2014/VGG_sal/imgs/lmdb'.format(lmdbpath)
    slcn_trn_img_csr = lmdb.open(lmdbfile, readonly=True).begin().cursor()
    lmdbfile         = '{}/salicon/data/lmdb_sal-19_train2014/VGG_sal/fmaps_sal/lmdb'.format(lmdbpath)
    slcn_trn_map_csr = lmdb.open(lmdbfile, readonly=True).begin().cursor()

    lmdbfile         = '{}/salicon/data/lmdb_sal-19_val2014/VGG_sal/imgs/lmdb'.format(lmdbpath)
    slcn_val_img_csr = lmdb.open(lmdbfile, readonly=True).begin().cursor()
    lmdbfile         = '{}/salicon/data/lmdb_sal-19_val2014/VGG_sal/fmaps_sal/lmdb'.format(lmdbpath)
    slcn_val_map_csr = lmdb.open(lmdbfile, readonly=True).begin().cursor()

    # For Salicon
    dataDirS    = '/opt/CV_db/image_databases/pub/SALICON'
    dataTypeS   = 'train2014'
    annFileS    = '%s/fixations_%s.json'%(dataDirS,dataTypeS)
    slcn_trn    = SALICON(annFileS)
    slcn_trnids = slcn_trn.getImgIds()

    dataDirS    = '/opt/CV_db/image_databases/pub/SALICON'
    dataTypeS   = 'val2014'
    annFileS    = '%s/fixations_%s.json'%(dataDirS,dataTypeS)
    slcn_val    = SALICON(annFileS)
    slcn_valids = slcn_val.getImgIds()
    slcn_valtrn, slcn_valval = np.split(np.random.permutation(len(slcn_valids)), [4000]) # use 4000 for val for training, the rest for validation

    # create dbs:
    trn_dbs     = {'salicon_trn': {'img_path': slcn_impath , 'anns': slcn_trn, 'imid': slcn_trnids, 'img_csr': slcn_trn_img_csr, 'map_csr': slcn_trn_map_csr, 'db_name': 'salicon_trn'}}
    val_dbs     = {'salicon_val': {'img_path': slcn_impath , 'anns': slcn_val, 'imid': [slcn_valids[idx] for idx in slcn_valtrn], 'img_csr': slcn_val_img_csr, 'map_csr': slcn_val_map_csr, 'db_name': 'salicon_val'}}

    #####################

    # configure caffe:
    caffe.set_device(args.gpuid)
    caffe.set_mode_gpu()

    pretrned_file  = args.pretrned_file
    trnnet_proto   = args.trnnet_proto
    trnnet_solver  = args.trnnet_solver
    trnnet_file    = args.trnnet_file

    # foldernames
    mapdir       = '{}/{}'.format(os.getcwd(), args.mapdir)
    jsondir      = args.jsondir
    jsonname     = args.jsonname
    predfolder   = args.predfolder
    pretrnd_nets = args.pretrnd_nets
    resfile_name = 'saleval.txt'

    # initialize solver
    slvr = open(trnnet_solver, 'w')
    slvr.write("""net: '""" + trnnet_proto + """' 
	base_lr: 0.01
	lr_policy: 'step'
        gamma: 0.1
        stepsize: 20000
	display: 50
	max_iter: 50000
	momentum: 0.99
	weight_decay: 0.0005
	snapshot: 1000
	solver_mode: GPU
	snapshot_prefix:'""" + args.out_path + """' """)
    slvr.close()

    gamma      = 1.0
    max_iter   = 50000
    val_iter   = 1000
    stepsz     = 1
    map_name   = 'predmap'
    tst_cnt    = 0
    trn_btchsz = 8
    solver     = caffe.SGDSolver(trnnet_solver)

    # result arrays
    mx_tst_itr = np.ceil(max_iter / val_iter)
    test_loss  = np.zeros(mx_tst_itr,'float32')
    seval_sauc = np.zeros(mx_tst_itr,'float32')
    seval_cc   = np.zeros(mx_tst_itr,'float32')

    # initialize network:
    if pretrned_file.endswith('.solverstate'):
        solver.restore(pretrned_file)
    else:
        # initialize deconv filters to bilinear filter:
        interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
        salh.interp_surgery(solver.net, interp_layers)
        solver.net.copy_from(pretrned_file)

    # run solver:
    for i in range(max_iter):
        # copy train batch data:
        net_inputs = []
        for inum in range(trn_btchsz):
            net_inputs.append(get_inputs_from_lmdb(trn_dbs, im_means, datum, idx=None, aug=None)[0])
        net_inputs = [np.concatenate(x,axis=0) for x in zip(*net_inputs)]
        salh.copy_blobs_to_net(solver.net, rst_blob_names, net_inputs)

        # train:
	solver.step(stepsz)

        if not np.mod(solver.iter,val_iter):
            tst_cnt, test_loss, seval_cc = do_validation(mapdir, jsondir, jsonname, val_dbs['salicon_val'], im_means, datum, solver, rst_blob_names, map_name, resfile_name, tst_cnt, test_loss, seval_cc, predfolder, mx_tst_itr)

    # Close the link to lmdbs
    slcn_trn_img_csr.close()	
    slcn_trn_map_csr.close()	
    slcn_val_img_csr.close()	
    slcn_val_map_csr.close()	
