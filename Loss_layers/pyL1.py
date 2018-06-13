import caffe
import numpy as np
# import warnings
import pdb
import scipy.misc as scimisc

class L1LossLayer(caffe.Layer):
    """
    Compute the L1 loss.
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
        pred = bottom[0].data.copy()
        gt   = bottom[1].data.copy()
        diff = pred - gt

        # compute derivative
        self.diff[...] = (0. < diff) - (diff < 0.)

        # compute loss:
        top[0].data[...] = np.mean(np.abs(diff))

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
