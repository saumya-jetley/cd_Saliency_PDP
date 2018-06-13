import caffe
import numpy as np
# import warnings
import pdb
import scipy.misc as scimisc

class SoftmaxCrossEntropyLossLayer(caffe.Layer):
    """
    Compute the cross-entropy loss
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute divergence.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # softmax normalization to obtain probability distribution
        pred       = bottom[0].data.copy()
        pred_exp   = np.exp(pred - np.max(pred,axis=1)[:,np.newaxis])
        pred_snorm = pred_exp / np.sum(pred_exp,axis=1)[:,np.newaxis]
        pred       = pred_snorm

        # ground truth
        gt = bottom[1].data.copy()

        # pdb.set_trace()

        # T
        T = float(self.param_str)

        # compute log and difference of log values
	epsilon = np.finfo(np.float).eps # epsilon (float or float 32)
	pred_ln = np.log(np.maximum(pred,epsilon))
	loss    = -gt * pred_ln

	# compute batch loss
        top[0].data[...] = np.mean(np.sum(loss,axis=1)) #averaged per image in the batch

        if gt.ndim == 4: gt = gt[:,:,0,0]

        # scale by T^2 for distillation:
        self.diff[...] = T * (pred - gt) 

        # self.diff[...] = pred - gt
        # top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

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


class SphericalSoftmaxCrossEntropyLossLayer(caffe.Layer):
    """
    Compute the cross-entropy loss
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute divergence.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        #eps
        eps = float(self.param_str)

        # softmax normalization to obtain probability distribution
        opred       = bottom[0].data.copy()
        pred_sq     = opred ** 2 + eps
        pred_sq_sum = np.sum(pred_sq,axis=1)[:,np.newaxis]
        pred        = (pred_sq / pred_sq_sum)

        # ground truth
        gt = bottom[1].data.copy()

        # pdb.set_trace()

        # print 'sum pred: {}'.format(np.sum(pred))

        # compute log and difference of log values
	pred_ln = np.log(pred)
	loss    = -gt * pred_ln

	# compute batch loss
        top[0].data[...] = np.mean(np.sum(loss,axis=1))

        self.diff[...] = gt * ( np.sum(2 * opred / pred_sq_sum) - (2 * opred / pred_sq) ) 

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


class CrossEntropyLossLayer(caffe.Layer):
    """
    Compute the cross-entropy loss
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute divergence.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output  is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        pred       = bottom[0].data.copy()

        # ground truth
        gt = bottom[1].data.copy()

        # pdb.set_trace()

        # compute log and difference of log values
	epsilon = np.finfo(np.float).eps # epsilon (float or float 32)
	pred_ln = np.log(np.maximum(pred,epsilon))
	loss    = -gt * pred_ln

	# compute batch loss
        top[0].data[...] = np.mean(np.sum(loss,axis=1)) #averaged per image in the batch

        self.diff[...] = -gt / pred

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
