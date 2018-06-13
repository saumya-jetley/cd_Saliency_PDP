import caffe
import numpy as np
# import warnings
import pdb
import scipy.misc as scimisc
import cv2
import matplotlib.pyplot as plt

def softmax(x):
    x_exp = np.exp(x - np.max(x))
    return x_exp / np.sum(x_exp)

class BCLossLayer(caffe.Layer):
    """
    Compute the Euclidean loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute distance.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # softmax and reshaping for the ground truth heatmap
        gts = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]):
            # pdb.set_trace()
	    gt = np.transpose(bottom[1].data[i,:,:,:],(1,2,0)).squeeze()
            gt = scimisc.imresize(gt,bottom[0].data.shape[2:4],interp='bilinear')
            gt = gt/255.0
	    # softmax normalization to obtain probability distribution
	    gt_exp = np.exp(gt-np.max(gt))
	    gt_snorm = gt_exp/np.sum(gt_exp)
	    # back into original block
	    gt = np.transpose(gt_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
	    gts[i,:,:,:] = gt
        gt = gts
	# softmax for the predicted heatmap
        preds = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]): # batch size
            # pdb.set_trace()
            pmap = np.transpose(bottom[0].data[i,:,:,:],(1,2,0)).squeeze()

            # if np.any(np.isnan(pmap)) | np.any(np.isinf(pmap)) | (len(np.unique(pmap)) < 100):
            #     pdb.set_trace()

            # apply softmax normalization to obtain a probability distribution
            pmap_exp = np.exp(pmap-np.max(pmap)) # range is now 0 to 1
            pmap_snorm = pmap_exp/np.sum(pmap_exp)
            # back into original block
            pmap = np.transpose(pmap_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
            preds[i,:,:,:] = pmap
        pmap = preds

        # pdb.set_trace()

        # compute log and difference of log values
	epsilon       = np.finfo(np.float).eps # epsilon (float or float 32)
	prod_sqrt     = np.sqrt(pmap * gt)
        prod_sqrt_sum = np.sum(np.sum(prod_sqrt, axis=3), axis=2)
	loss          = -np.log(np.maximum(prod_sqrt_sum,epsilon))

	# compute combined loss
        top[0].data[...] = np.mean(loss) #averaged per image in the batch

	# calculate value for bkward pass - self.diff = dl/dpk
	const      = -1/(2*(np.sum(np.sum(prod_sqrt,axis=3),axis=2)))

	# for hind in range(pmap.shape[2]):
	# 	for wind in range(pmap.shape[3]):
	# 		# for each pixel in the distribution - 2D map 
	# 		iequalk = np.zeros(pmap.shape)
	# 		inotequalk = np.ones(pmap.shape)
	# 		iequalk[:,:,hind,wind] = 1	
	# 		inotequalk[:,:,hind,wind] = 0

	# 		self.diff[:,:,hind,wind] = const * ( np.sum(np.sum(prod_sqrt*(1-pmap)*iequalk,axis=3),axis=2) - pmap[:,:,hind,wind]*np.sum(np.sum(prod_sqrt*inotequalk,axis=3),axis=2) )

        # pdb.set_trace()
        self.diff = const[:,:,np.newaxis,np.newaxis] * (prod_sqrt * (1 - pmap) - (prod_sqrt_sum[:,:,np.newaxis,np.newaxis] - prod_sqrt) * pmap)

    def backward(self, top, propagate_down, bottom):    
	loss_wgt = top[0].diff
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / bottom[i].num

class GBCLossLayer(caffe.Layer):
    """
    Compute the Euclidean loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute distance.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # pdb.set_trace()
        # softmax and reshaping for the ground truth heatmap
        gts = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]):
            # pdb.set_trace()
	    gt = np.transpose(bottom[1].data[i,:,:,:],(1,2,0)).squeeze()
            gt = scimisc.imresize(gt,bottom[0].data.shape[2:4],interp='bilinear')
            gt = gt/255.0
	    # softmax normalization to obtain probability distribution
	    gt_exp = np.exp(gt-np.max(gt))
	    gt_snorm = gt_exp/np.sum(gt_exp)
	    # back into original block
	    gt = np.transpose(gt_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
	    gts[i,:,:,:] = gt
        gt = gts
	# softmax for the predicted heatmap
        preds = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]): # batch size
            # pdb.set_trace()
            pmap = np.transpose(bottom[0].data[i,:,:,:],(1,2,0)).squeeze()

            # if np.any(np.isnan(pmap)) | np.any(np.isinf(pmap)) | (len(np.unique(pmap)) < 100):
            #     pdb.set_trace()

            # apply softmax normalization to obtain a probability distribution
            pmap_exp = np.exp(pmap-np.max(pmap)) # range is now 0 to 1
            pmap_snorm = pmap_exp/np.sum(pmap_exp)
            # back into original block
            pmap = np.transpose(pmap_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
            preds[i,:,:,:] = pmap
        pmap = preds

        # get alpha parameter:
        alpha = np.asscalar(np.load('alpha.npy'))

        # compute log and difference of log values
	epsilon        = np.finfo(np.float).eps # epsilon (float or float 32)
	prod_alpha     = (pmap * gt) ** alpha
        prod_alpha_sum = np.sum(np.sum(prod_alpha, axis=3), axis=2)
	loss           = -np.log(np.maximum(prod_alpha_sum,epsilon))

	# compute combined loss
        top[0].data[...] = np.mean(loss) #averaged per image in the batch

	# calculate value for bkward pass - self.diff = dl/dpk
	const     = -alpha / prod_alpha_sum
        const     = const[:,:,np.newaxis,np.newaxis]
        self.diff = const * (prod_alpha * (1 - pmap) - (prod_alpha_sum[:,:,np.newaxis,np.newaxis] - prod_alpha) * pmap)

        # print 'alpha = {:.2f}, min diff = {:.2e}, max diff = {:.2e}, range = {:.2e}'.format(alpha, np.min(self.diff), np.max(self.diff), np.max(self.diff) - np.min(self.diff))
        # fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(22,10))
        # ax1.imshow(gt[0,0,:,:])
        # ax2.imshow(pmap[0,0,:,:])
        # ax3.imshow(self.diff[0,0,:,:])
        # plt.show()

    def backward(self, top, propagate_down, bottom):    
	loss_wgt = top[0].diff
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / bottom[i].num


class GBDLossLayer(caffe.Layer):
    """
    Compute the generalized Bhattacharyya Distance loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute distance.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # pdb.set_trace()
        # softmax and reshaping for the ground truth heatmap
        gts = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]):
            # pdb.set_trace()
	    gt = np.transpose(bottom[1].data[i,:,:,:],(1,2,0)).squeeze()
            gt = scimisc.imresize(gt,bottom[0].data.shape[2:4],interp='bilinear')
            gt = gt / 255.
	    # softmax normalization to obtain probability distribution
	    gt_exp = np.exp(gt-np.max(gt))
	    gt_snorm = gt_exp/np.sum(gt_exp)
	    # back into original block
	    gt = np.transpose(gt_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
	    gts[i,:,:,:] = gt
        gt = gts
	# softmax for the predicted heatmap
        preds = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]): # batch size
            # pdb.set_trace()
            pmap = np.transpose(bottom[0].data[i,:,:,:],(1,2,0)).squeeze()

            # if np.any(np.isnan(pmap)) | np.any(np.isinf(pmap)) | (len(np.unique(pmap)) < 100):
            #     pdb.set_trace()

            # apply softmax normalization to obtain a probability distribution
            pmap_exp = np.exp(pmap-np.max(pmap)) # range is now 0 to 1
            pmap_snorm = pmap_exp/np.sum(pmap_exp)
            # back into original block
            pmap = np.transpose(pmap_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
            preds[i,:,:,:] = pmap
        pmap = preds

        # get alpha parameter:
        alpha = np.asscalar(np.load('alpha.npy'))

        # compute log and difference of log values
	epsilon        = np.finfo(np.float).eps # epsilon (float or float 32)
	# prod_alpha     = (pmap * gt) ** alpha
	prod_alpha     = pmap ** alpha *  gt ** (1 - alpha)
        prod_alpha_sum = np.sum(np.sum(prod_alpha, axis=3), axis=2)
	loss           = -np.log(np.maximum(prod_alpha_sum,epsilon))

	# compute combined loss
        top[0].data[...] = np.mean(loss) #averaged per image in the batch

	# calculate value for bkward pass - self.diff = dl/dpk
	const     = -alpha / prod_alpha_sum
        const     = const[:,:,np.newaxis,np.newaxis]
        self.diff = const * (prod_alpha * (1 - pmap) - (prod_alpha_sum[:,:,np.newaxis,np.newaxis] - prod_alpha) * pmap)

        # print 'alpha = {:.2f}, min diff = {:.2e}, max diff = {:.2e}, range = {:.2e}'.format(alpha, np.min(self.diff), np.max(self.diff), np.max(self.diff) - np.min(self.diff))
        # fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(22,10))
        # ax1.imshow(gt[0,0,:,:])
        # ax2.imshow(pmap[0,0,:,:])
        # ax3.imshow(self.diff[0,0,:,:])
        # plt.show()

    def backward(self, top, propagate_down, bottom):    
	loss_wgt = top[0].diff
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / bottom[i].num


class gGBCLossLayer(caffe.Layer):
    """
    Compute the Euclidean loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute distance.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # get alpha and gamma parameters:
        alpha = np.asscalar(np.load('alpha.npy'))
        gamma = np.asscalar(np.load('gamma.npy'))

        # softmax and reshaping for the ground truth heatmap
        gts = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]):
            # pdb.set_trace()
	    gt = np.transpose(bottom[1].data[i,:,:,:],(1,2,0)).squeeze()
            gt = scimisc.imresize(gt,bottom[0].data.shape[2:4],interp='bilinear')
            gt = gt/255.0
	    # softmax normalization to obtain probability distribution
	    gt_exp = np.exp(gamma * gt-np.max(gt))
	    gt_snorm = gt_exp/np.sum(gt_exp)
	    # back into original block
	    gt = np.transpose(gt_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
	    gts[i,:,:,:] = gt
        gt = gts
	# softmax for the predicted heatmap
        preds = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]): # batch size
            # pdb.set_trace()
            pmap = np.transpose(bottom[0].data[i,:,:,:],(1,2,0)).squeeze()
            # apply softmax normalization to obtain a probability distribution
            pmap_exp = np.exp(gamma * pmap-np.max(pmap)) # range is now 0 to 1
            pmap_snorm = pmap_exp/np.sum(pmap_exp)
            # back into original block
            pmap = np.transpose(pmap_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
            preds[i,:,:,:] = pmap
        pmap = preds

        # compute log and difference of log values
	epsilon        = np.finfo(np.float).eps # epsilon (float or float 32)
	prod_alpha     = (pmap * gt) ** alpha
        prod_alpha_sum = np.sum(np.sum(prod_alpha, axis=3), axis=2)
	loss           = -np.log(np.maximum(prod_alpha_sum,epsilon))

	# compute combined loss
        top[0].data[...] = np.mean(loss) #averaged per image in the batch

	# calculate value for bkward pass - self.diff = dl/dpk
	const     = -alpha * gamma / prod_alpha_sum
        const     = const[:,:,np.newaxis,np.newaxis]
        self.diff = const * (prod_alpha * (1 - pmap) - (prod_alpha_sum[:,:,np.newaxis,np.newaxis] - prod_alpha) * pmap)

    def backward(self, top, propagate_down, bottom):    
	loss_wgt = top[0].diff
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / bottom[i].num

class PPKGainLayer(caffe.Layer):
    """
    Compute the Euclidean loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute distance.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # softmax and reshaping for the ground truth heatmap
        gts = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]):
            # pdb.set_trace()
	    gt = np.transpose(bottom[1].data[i,:,:,:],(1,2,0)).squeeze()
            gt = scimisc.imresize(gt,bottom[0].data.shape[2:4],interp='bilinear')
            gt = gt/255.0
	    # softmax normalization to obtain probability distribution
	    gt_exp = np.exp(gt-np.max(gt))
	    gt_snorm = gt_exp/np.sum(gt_exp)
	    # back into original block
	    gt = np.transpose(gt_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
	    gts[i,:,:,:] = gt
        gt = gts
	# softmax for the predicted heatmap
        preds = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]): # batch size
            # pdb.set_trace()
            pmap = np.transpose(bottom[0].data[i,:,:,:],(1,2,0)).squeeze()
            # apply softmax normalization to obtain a probability distribution
            pmap_exp = np.exp(pmap-np.max(pmap)) # range is now 0 to 1
            pmap_snorm = pmap_exp/np.sum(pmap_exp)
            # back into original block
            pmap = np.transpose(pmap_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
            preds[i,:,:,:] = pmap
        pmap = preds

        # get alpha parameter:
        alpha = np.asscalar(np.load('alpha.npy'))

        # compute log and difference of log values
	#pdb.set_trace()
	epsilon    = np.finfo(np.float).eps # epsilon (float or float 32)
	prod_alpha = (pmap * gt) ** alpha
	prod_sum   = np.sum(np.sum(prod_alpha, axis=3), axis=2)
	gain       = np.maximum(prod_sum, epsilon)
	# compute combined loss
        top[0].data[...] = np.sum(gain)/bottom[0].num # averaged per image in the batch
	# calculate value for bkward pass - self.diff = dl/dpk
	for hind in range(pmap.shape[2]):
            for wind in range(pmap.shape[3]):
                # for each pixel in the distribution - 2D map 
                iequalk                   = np.zeros(pmap.shape)
                inotequalk                = np.ones(pmap.shape)
                iequalk[:,:,hind,wind]    = 1	
                inotequalk[:,:,hind,wind] = 0
                self.diff[:,:,hind,wind]  = alpha * (np.sum(np.sum(prod_alpha * (1 - pmap) * iequalk,axis=3), axis=2) - pmap[:,:,hind,wind] * np.sum(np.sum(prod_alpha * inotequalk, axis=3), axis=2))

    def backward(self, top, propagate_down, bottom):    
	loss_wgt = top[0].diff
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = -1
            else:
                sign = 1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / bottom[i].num


class gPPKGainLayer(caffe.Layer):
    """
    Compute the Euclidean loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute distance.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # softmax and reshaping for the ground truth heatmap
        gamma = 0.5
        gts   = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]):
            # pdb.set_trace()
	    gt = np.transpose(bottom[1].data[i,:,:,:],(1,2,0)).squeeze()
            gt = scimisc.imresize(gt,bottom[0].data.shape[2:4],interp='bilinear')
            gt = gt/255.0
	    # softmax normalization to obtain probability distribution
	    gt_exp = np.exp(gamma * gt-np.max(gt))
	    gt_snorm = gt_exp/np.sum(gt_exp)
	    # back into original block
	    gt = np.transpose(gt_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
	    gts[i,:,:,:] = gt
        gt = gts
	# softmax for the predicted heatmap
        preds = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]): # batch size
            # pdb.set_trace()
            pmap = np.transpose(bottom[0].data[i,:,:,:],(1,2,0)).squeeze()
            # apply softmax normalization to obtain a probability distribution
            pmap_exp = np.exp(gamma * pmap-np.max(pmap)) # range is now 0 to 1
            pmap_snorm = pmap_exp/np.sum(pmap_exp)
            # back into original block
            pmap = np.transpose(pmap_snorm[:,:,np.newaxis,np.newaxis],(3,2,0,1))
            preds[i,:,:,:] = pmap
        pmap = preds

        # get alpha parameter:
        alpha = np.asscalar(np.load('alpha.npy'))

        # compute log and difference of log values
	#pdb.set_trace()
	epsilon    = np.finfo(np.float).eps # epsilon (float or float 32)
	prod_alpha = (pmap * gt) ** alpha
	prod_sum   = np.sum(np.sum(prod_alpha, axis=3), axis=2)
	gain       = np.maximum(prod_sum, epsilon)
	# compute combined loss
        top[0].data[...] = np.sum(gain)/bottom[0].num # averaged per image in the batch
	# calculate value for bkward pass - self.diff = dl/dpk
	for hind in range(pmap.shape[2]):
            for wind in range(pmap.shape[3]):
                # for each pixel in the distribution - 2D map 
                iequalk                   = np.zeros(pmap.shape)
                inotequalk                = np.ones(pmap.shape)
                iequalk[:,:,hind,wind]    = 1	
                inotequalk[:,:,hind,wind] = 0
                self.diff[:,:,hind,wind]  = alpha * gamma * (np.sum(np.sum(prod_alpha * (1 - pmap) * iequalk,axis=3), axis=2) - pmap[:,:,hind,wind] * np.sum(np.sum(prod_alpha * inotequalk, axis=3), axis=2))

    def backward(self, top, propagate_down, bottom):    
	loss_wgt = top[0].diff
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = -1
            else:
                sign = 1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / bottom[i].num

class sGBDLossLayer(caffe.Layer):
    """
    Compute the Bhattacharyya loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute distance.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
	# softmax for the prediction
        pred = bottom[0].data

        # softmax normalization to obtain probability distribution
        pred_exp   = np.exp(pred - np.max(pred,axis=1)[:,np.newaxis])
        pred_snorm = pred_exp / np.sum(pred_exp,axis=1)[:,np.newaxis]
        pred       = pred_snorm

        # softmax for the ground truth
        gt = bottom[1].data

        # softmax normalization to obtain probability distribution
        gt_exp   = np.exp(gt-np.max(gt,axis=1)[:,np.newaxis])
        gt_snorm = gt_exp / np.sum(gt_exp,axis=1)[:,np.newaxis]
        gt       = gt_snorm

        # get alpha parameter:
        alpha = np.asscalar(np.load('alpha.npy'))

        # compute log and difference of log values
	epsilon        = np.finfo(np.float).eps # epsilon (float or float 32)
	prod_alpha     = (pred * gt) ** alpha
        prod_alpha_sum = np.sum(prod_alpha, axis=1)
	loss           = -np.log(np.maximum(prod_alpha_sum,epsilon))

	# compute combined loss
        top[0].data[...] = np.mean(loss) #averaged per image in the batch

	# calculate value for bkward pass - self.diff = dl/dpk
	const     = -alpha / prod_alpha_sum
        self.diff = const[:,np.newaxis] * (prod_alpha * (1 - pred) - (prod_alpha_sum[:,np.newaxis] - prod_alpha) * pred)

    def backward(self, top, propagate_down, bottom):    
	loss_wgt = top[0].diff
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / bottom[i].num

class SGBDLossLayer(caffe.Layer):
    """
    Compute the Bhattacharyya loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute distance.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # get ground truth:
        gt = bottom[1].data

	# softmax for the prediction
        pred = bottom[0].data
        pred = softmax(pred)

        # get alpha parameter:
        alpha = np.asscalar(np.load('alpha.npy'))

        # compute log and difference of log values
	epsilon        = np.finfo(np.float).eps # epsilon (float or float 32)
	prod_alpha     = (pred * gt) ** alpha
        prod_alpha_sum = np.sum(prod_alpha, axis=1)
	loss           = -np.log(np.maximum(prod_alpha_sum,epsilon))

	# compute combined loss
        top[0].data[...] = np.mean(loss) #averaged per image in the batch

	# calculate value for bkward pass - self.diff = dl/dpk
	const     = -alpha / prod_alpha_sum
        self.diff = const[:,np.newaxis] * (prod_alpha * (1 - pred) - (prod_alpha_sum[:,np.newaxis] - prod_alpha) * pred)

    def backward(self, top, propagate_down, bottom):    
	loss_wgt = top[0].diff
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / bottom[i].num

class SPGBDLossLayer(caffe.Layer):
    """
    Compute the Bhattacharyya loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute distance.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # create sample mask:
        pos_mask = bottom[1].data > 0.0
        npos     = np.sum(pos_mask)
        neg_m, neg_n  = np.where(bottom[1].data == 0.0)
        nneg     = neg_m.shape[0]
        neg_smp  = np.random.choice(nneg, (1,npos))
        neg_mask = np.zeros_like(pos_mask)
        neg_mask[neg_m[neg_smp], neg_n[neg_smp]] = 1.0
        mask = pos_mask | neg_mask

        # get ground truth:
        # pdb.set_trace()
        # gt = sigmoid(bottom[1].data)
        gt = bottom[1].data

	# get prediction:
        # pred = sigmoid(bottom[0].data)
        pred = bottom[0].data

        # compute loss
        diff             = pred - gt
        absdiff          = np.abs(diff)
        d_ind_lt         = absdiff < 1
        d_ind_gt         = absdiff >= 1
        floss            = np.zeros_like(diff, dtype=np.float32)
        floss[d_ind_lt]  = 0.5 * (diff[d_ind_lt] ** 2)
        # floss[d_ind_gt]  = 0.5 * (self.diff[d_ind_gt] ** 2)
        floss[d_ind_gt]  = absdiff[d_ind_gt] - 0.5
        # floss  = 0.5 * (self.diff ** 2)
        top[0].data[...] = np.sum(floss * mask) / np.sum(mask)

        # compute loss gradient:
        bloss            = np.zeros_like(diff, dtype=np.float32)
        bloss[d_ind_lt]  = diff[d_ind_lt]
        bloss[d_ind_gt]  = (0.0 < diff[d_ind_gt]).astype(np.float32) - (diff[d_ind_gt] < 0.0).astype(np.float32)
        self.diff        = bloss * mask

    def backward(self, top, propagate_down, bottom):    
	loss_wgt = top[0].diff
        nsmp     = 2.0 * np.sum(bottom[1].data > 0.0)

        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / nsmp

class PSPGBDLossLayer(caffe.Layer):
    """
    Compute the Bhattacharyya loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute distance.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # create sample mask:
        pos_mask = bottom[1].data > 0.0
        npos     = np.sum(pos_mask)
        neg_m, neg_n = np.where(bottom[1].data == 0.0)
        nneg     = neg_m.shape[0]
        neg_smp  = np.random.choice(nneg, (1,npos))
        neg_mask = np.zeros_like(pos_mask)
        neg_mask[neg_m[neg_smp], neg_n[neg_smp]] = 1.0
        mask = pos_mask | neg_mask
        mask = pos_mask

        # get ground truth:
        # pdb.set_trace()
        # gt = sigmoid(bottom[1].data)
        gt = bottom[1].data

	# get prediction:
        # pred = sigmoid(bottom[0].data)
        pred = bottom[0].data

        # compute loss
        diff             = pred - gt
        absdiff          = np.abs(diff)
        d_ind_lt         = absdiff < 1
        d_ind_gt         = absdiff >= 1
        floss            = np.zeros_like(diff, dtype=np.float32)
        floss[d_ind_lt]  = 0.5 * (diff[d_ind_lt] ** 2)
        # floss[d_ind_gt]  = 0.5 * (self.diff[d_ind_gt] ** 2)
        floss[d_ind_gt]  = absdiff[d_ind_gt] - 0.5
        # floss  = 0.5 * (self.diff ** 2)
        top[0].data[...] = np.sum(floss * mask) / np.sum(mask)

        # compute loss gradient:
        bloss            = np.zeros_like(diff, dtype=np.float32)
        bloss[d_ind_lt]  = diff[d_ind_lt]
        bloss[d_ind_gt]  = (0.0 < diff[d_ind_gt]).astype(np.float32) - (diff[d_ind_gt] < 0.0).astype(np.float32)
        self.diff        = bloss * mask

    def backward(self, top, propagate_down, bottom):    
	loss_wgt = top[0].diff
        nsmp     = 2.0 * np.sum(bottom[1].data > 0.0)

        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / nsmp

class MGBDLossLayer(caffe.Layer):
    """
    Compute the Bhattacharyya loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need three inputs (pred, gt, mask) to compute distance.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # get sample mask:
        mask = bottom[2].data

        # get ground truth:
        # gt = sigmoid(bottom[1].data)
        gt = bottom[1].data

	# get prediction:
        # pred = sigmoid(bottom[0].data)
        pred = bottom[0].data

        # compute loss
        diff             = pred - gt
        absdiff          = np.abs(diff)
        d_ind_lt         = absdiff < 1
        d_ind_gt         = absdiff >= 1
        floss            = np.zeros_like(diff, dtype=np.float32)
        floss[d_ind_lt]  = 0.5 * (diff[d_ind_lt] ** 2)
        # floss[d_ind_gt]  = 0.5 * (self.diff[d_ind_gt] ** 2)
        floss[d_ind_gt]  = absdiff[d_ind_gt] - 0.5
        # floss  = 0.5 * (self.diff ** 2)
        top[0].data[...] = np.sum(floss * mask) / np.sum(mask)

        # compute loss gradient:
        bloss            = np.zeros_like(diff, dtype=np.float32)
        bloss[d_ind_lt]  = diff[d_ind_lt]
        bloss[d_ind_gt]  = (0.0 < diff[d_ind_gt]).astype(np.float32) - (diff[d_ind_gt] < 0.0).astype(np.float32)
        self.diff        = bloss * mask

    def backward(self, top, propagate_down, bottom):    
	loss_wgt = top[0].diff
        nsmp     = 2.0 * np.sum(bottom[1].data > 0.0)

        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / nsmp

class wHuberLossLayer(caffe.Layer):
    """
    Compute the weighted Huber loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need three inputs (pred, gt, wgt) to compute loss.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
	# get prediction:
        pred = bottom[0].data

        # get ground truth:
        gt = bottom[1].data

	# get sample weights:
        wgts = bottom[2].data

        # compute loss
        diff             = pred - gt
        absdiff          = np.abs(diff)
        d_ind_lt         = absdiff < 1
        d_ind_gt         = absdiff >= 1
        floss            = np.zeros_like(diff, dtype=np.float32)
        floss[d_ind_lt]  = 0.5 * (diff[d_ind_lt] ** 2)
        floss[d_ind_gt]  = absdiff[d_ind_gt] - 0.5
        top[0].data[...] = np.sum(floss * wgts) / bottom[0].num

        # compute loss gradient:
        bloss            = np.zeros_like(diff, dtype=np.float32)
        bloss[d_ind_lt]  = diff[d_ind_lt]
        bloss[d_ind_gt]  = (0.0 < diff[d_ind_gt]).astype(np.float32) - (diff[d_ind_gt] < 0.0).astype(np.float32)
        self.diff        = bloss * wgts

    def backward(self, top, propagate_down, bottom):    
	loss_wgt = top[0].diff

        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / bottom[i].num


class HuberLossLayer(caffe.Layer):
    """
    Compute the weighted Huber loss.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need three inputs (pred, gt) to compute loss.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
	# get prediction:
        pred = bottom[0].data

        # get ground truth:
        gt = bottom[1].data

        # compute loss
        diff             = pred - gt
        absdiff          = np.abs(diff)
        d_ind_lt         = absdiff < 1
        d_ind_gt         = absdiff >= 1
        floss            = np.zeros_like(diff, dtype=np.float32)
        floss[d_ind_lt]  = 0.5 * (diff[d_ind_lt] ** 2)
        floss[d_ind_gt]  = absdiff[d_ind_gt] - 0.5
        top[0].data[...] = np.sum(floss) / bottom[0].num

        # compute loss gradient:
        bloss            = np.zeros_like(diff, dtype=np.float32)
        bloss[d_ind_lt]  = diff[d_ind_lt]
        bloss[d_ind_gt]  = (0.0 < diff[d_ind_gt]).astype(np.float32) - (diff[d_ind_gt] < 0.0).astype(np.float32)
        self.diff        = bloss

    def backward(self, top, propagate_down, bottom):    
	loss_wgt = top[0].diff

        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * self.diff / bottom[i].num
