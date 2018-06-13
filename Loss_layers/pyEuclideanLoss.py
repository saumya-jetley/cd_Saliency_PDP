import caffe
import numpy as np
import pdb
import scipy.misc as scimisc

class L2LossLayer(caffe.Layer):
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

        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # reshape gt to shape of prediction
        gts = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]):
            gt = np.transpose(bottom[1].data[i,:,:,:],(1,2,0)).squeeze()
            gt = scimisc.imresize(gt,bottom[0].data.shape[2:4],interp='bilinear')
            gt = np.transpose(gt[:,:,np.newaxis,np.newaxis],(3,2,0,1))
            gt = gt / 255.0
            gts[i,:,:,:] = gt

        gt = gts

        # compute loss
        self.diff[...] = bottom[0].data - gt
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

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
