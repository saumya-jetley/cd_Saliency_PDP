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
import scipy.io as sio
from shutil import rmtree # to remove non-empty folders - fails for readonly files

# sys.path.insert(1,'/home/sjetley/workspace_eclipse/salicon_papi')
sys.path.insert(1,'/home/$(USER)/workspace/pyscripts/caffe')
sys.path.insert(1,'/home/$(USER)/workspace/pyscripts/salicon/salicon-evaluation/saliconeval')
sys.path.insert(1,'/home/$(USER)/workspace/pyscripts/salicon/salicon-evaluation/')
from salicon.salicon import SALICON
from saliconeval.eval import SALICONEval
from image2json import ImageTools 
import caffe
import filter_fns as ffns
import frcnn_fns as frcnnf

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

    if not aug == None:
        I, fmap = augment_data((I, fmap))

    if 'mit' in db['db_name']:
        I = I[:,:,::-1]
        
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

def examine_lmdb(csr, datum):
    while csr.next():
        key,val = csr.item()
        datum.ParseFromString(val)
        darray  = caffe.io.datum_to_array(datum)
        I       = np.transpose(darray,(1,2,0))

        plt.imshow(I)
        plt.title(key)
        plt.show()


def examine_lmdbs(csrs, datum):
    csrs[1].next()
    key,val = csrs[1].item()
    print key
    csrs[0].set_key(key)
    while csrs[0].next():
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(18,10))

        key,val = csrs[0].item()
        datum.ParseFromString(val)
        darray  = caffe.io.datum_to_array(datum)
        I       = np.transpose(darray,(1,2,0))
        if I.shape[2] == 1:
            I = np.tile(I,(1,1,3))

        ax1.imshow(I)
        ax1.set_title(key)

        csrs[1].next()
        key,val = csrs[1].item()
        datum.ParseFromString(val)
        darray  = caffe.io.datum_to_array(datum)
        I2      = np.transpose(darray,(1,2,0))
        if I2.shape[2] == 1:
            I2 = np.tile(I2,(1,1,3))

        ax2.imshow(I2)
        ax2.set_title(key)

        print np.array_equal(I,I2)

        plt.show()

def do_sal_validation(mapdir, jsondir, jsonname, db, im_means, datum, net, rst_blob_names, map_name, resfile_name, gamma, matdir, fixmap=False):

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
        frcnnf.copy_blobs_to_net(net, rst_blob_names, net_inputs)
        net.forward()

        #get the image
        pmap = np.transpose(net.blobs[map_name].data,(2,3,1,0))[:,:,:,0].squeeze()

        # softmax normalization for prob. distribution
        bcdiv_loss = bcdiv_loss + net.blobs['loss_fmap'].data
        pmap_exp   = np.exp(gamma*pmap-np.max(pmap))
        pmap_snorm = pmap_exp/np.sum(pmap_exp)
        scimisc.imsave('{}/{}'.format(mapdir,imgname),pmap_snorm)
        sio.savemat('{}/{}.mat'.format(matdir,os.path.splitext(imgname)[0]), {'pmap':pmap_snorm})

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
    elif fixmap:
        metrics = ['AUC', 'NSS', 'CC']
        mitEval.evaluate(metrics, fixmap=fixmap)        
    else:
        metrics = ['AUC', 'NSS', 'CC']
        mitEval.evaluate(metrics, filterAnns=True)

    # print output evaluation scores
    print "Final result for each metric being written in " + resfile_name
    resultfile = open(resfile_name,'a+')

    resultfile.write("\ndataset\t")
    for metric,score in mitEval.eval.items():
        resultfile.write('{}\t'.format(metric))

    resultfile.write("test_loss\n")

    resultfile.write('{}\t'.format(db['db_name']))
    for metric, score in mitEval.eval.items():
        if(metric=='CC'):
            met_cc = score
        resultfile.write('{:.5f}\t'.format(score))

    resultfile.write('{:.5f}\n'.format(bcdiv_loss[0]))
    resultfile.close()


def do_mit_validation(mapdir, jsondir, jsonname, db, im_means, datum, net, rst_blob_names, map_name, resfile_name, gamma, matdir, fixmap=False):
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

def do_mit_validation_mat(mapdir, jsondir, jsonname, db, im_means, datum, net, rst_blob_names, map_name, resfile_name, gamma, matdir, fixmap=False):
    rename_dir = matdir + 'renamed'

    # 1. rename your maps back to original MIT names:
    rename_command = '/home/$(USER)/workspace/experiments/salicon_sal/renameSalMapsForMIT1003.sh ' + \
                      matdir + ' ' + \
                      rename_dir

    # 2. run matlab evaluation (on Matterhorn):
    code_path    = '/home/$(USER)/tools/cvzoya/saliency/code_forOptimization'
    eval_command = 'ssh matterhorn \'set -vx; matlab -nodesktop -nosplash -r "addpath(\'"\'"\'{}\'"\'"\'); computeMetricsMIT1003origGT(\'"\'"\'{}\'"\'"\', \'"\'"\'{}\'"\'"\', \'"\'"\'{}\'"\'"\'); quit"\' '.format(code_path, rename_dir, resfile_name, 'mat')

    os.system(rename_command)
    os.system(eval_command)

def shell_quotes(string):
    return '\'"\'"\'test\'"\'"\''

#-----------------------------MAIN starts here--------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trnnet_file",
            help="")
    parser.add_argument("trnnet_proto",
            help="")
    parser.add_argument("mapdir",
            help="")
    parser.add_argument("matdir",
            help="")
    parser.add_argument("jsondir",
            help="")
    parser.add_argument("jsonname",
            help="")
    parser.add_argument("gpuid",type=int,
            help="GPU ID")
    args   = parser.parse_args()

    rst_blob_names = ['imdata','fmap']
    im_means       = [123.6800, 116.7790, 103.9390]
    datum          = caffe.proto.caffe_pb2.Datum()

    #####################
    # val datasets
    #####################

    ##### For MIT
    mit_impath = '/opt/CV_db/image_databases/pub/MIT1003/json_stimuli'
    annFileS   = '/opt/CV_db/Users/$(USER)/MIT1003/salformat/gtjson/mit1003.json'
    mit_anns   = SALICON(annFileS)
    mit_ids    = mit_anns.getImgIds()
    mit_trnids = mit_ids[0:900]
    mit_valids = mit_ids[900:]
    mit_anns.dataset['type'] = 'fixations'

    lmdbpath        = '/dev/shm/$(USER)/MIT1003/data_origsize_900_103/lmdb'
    lmdbfile        = '{}_tr/imgs'.format(lmdbpath)
    mit_trn_img_csr = lmdb.open(lmdbfile, readonly=True).begin().cursor()
    lmdbfile        = '{}_tr/fmaps'.format(lmdbpath)
    mit_trn_map_csr = lmdb.open(lmdbfile, readonly=True).begin().cursor()

    lmdbfile        = '{}_val/imgs'.format(lmdbpath)
    mit_val_img_csr = lmdb.open(lmdbfile, readonly=True).begin().cursor()
    lmdbfile        = '{}_val/fmaps'.format(lmdbpath)
    mit_val_map_csr = lmdb.open(lmdbfile, readonly=True).begin().cursor()

    trn_dbs     = {'mit_trn': {'img_path': mit_impath , 'anns': mit_anns, 'imid': mit_trnids, 'img_csr': mit_trn_img_csr, 'map_csr': mit_trn_map_csr, 'db_name': 'mit_trn'}}

    val_dbs     = {'mit_val': {'img_path': mit_impath , 'anns': mit_anns, 'imid': mit_valids, 'img_csr': mit_val_img_csr, 'map_csr': mit_val_map_csr, 'db_name': 'mit_val'}}

    #################################

    # configure caffe:
    caffe.set_device(args.gpuid)
    caffe.set_mode_gpu()

    trnnet_file  = args.trnnet_file
    trnnet_proto = args.trnnet_proto

    # foldernames
    mapdir       = os.getcwd() + '/' + args.mapdir
    matdir       = os.getcwd() + '/' + args.matdir
    jsondir      = args.jsondir
    jsonname     = args.jsonname
    resfile_name = os.getcwd() + '/deploy_res.txt'
    resfile_mit  = os.getcwd() + '/deploy_res_mit.txt'
    map_name     = 'predmap'
    gamma        = 1.0

    # load network:
    testnet = caffe.Net(trnnet_proto, trnnet_file, caffe.TEST)

    # run "salicon"-style evaluation:
    do_sal_validation(mapdir, jsondir, jsonname, val_dbs['mit_val'], im_means, datum, testnet, rst_blob_names, map_name, resfile_name, gamma, matdir)

    # # run "mit"-style evaluation:
    # do_mit_validation(mapdir, jsondir, jsonname, val_dbs['mit_val'], im_means, datum, testnet, rst_blob_names, map_name, resfile_mit, gamma, matdir)

    # run "mit"-style evaluation:
    do_mit_validation_mat(mapdir, jsondir, jsonname, val_dbs['mit_val'], im_means, datum, testnet, rst_blob_names, map_name, resfile_mit, gamma, matdir)
    
    # Close the link to lmdbs
    mit_trn_img_csr.close()	
    mit_trn_map_csr.close()	
    mit_val_img_csr.close()	
    mit_val_map_csr.close()	
