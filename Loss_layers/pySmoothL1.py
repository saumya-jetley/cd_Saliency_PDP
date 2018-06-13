import caffe
import numpy as np
# import warnings
import pdb
import scipy.misc as scimisc

class L1LossLayer(caffe.Layer):
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
        # reshape gt to shape of prediction
        gts = np.empty_like(bottom[0].data)
        for i in range(bottom[0].data.shape[0]):
            gt = np.transpose(bottom[1].data[i,:,:,:],(1,2,0)).squeeze()
            gt = scimisc.imresize(gt,bottom[0].data.shape[2:4],interp='bilinear')
            gt = np.transpose(gt[:,:,np.newaxis,np.newaxis],(3,2,0,1))
            gt = gt / 255.0
            gts[i,:,:,:] = gt
        gt = gts

        # compute difference
	#pdb.set_trace()
        self.diff[...] = bottom[0].data - gt
	# absolute difference
	abs_diff = np.abs(self.diff)
	# evaluate weight-maps for 2 loss metrics
	adiff_gte_1 = abs_diff >=1
 	adiff_ls_1 = abs_diff<1
	# evaluate 2 loss metrics
	gte_loss = abs_diff - 0.5
	lt_loss = (abs_diff**2)/2.0
	# compute combined loss
	loss = ( np.multiply(adiff_gte_1, gte_loss) + np.multiply(adiff_ls_1, lt_loss) ) #-----------element wise
        top[0].data[...] = np.sum(loss) / bottom[0].num

    def backward(self, top, propagate_down, bottom):
	# absolute difference
	abs_diff = np.abs(self.diff)
	# evaluate weight-maps for 2 loss metrics
	adiff_gte_1 = abs_diff >=1
 	adiff_ls_1 = abs_diff<1
	# evaluate 2 loss metrics
	gte_diff = np.divide(self.diff, abs_diff) #------------ element wise
	gte_diff = np.nan_to_num(gte_diff); # converts nan to 0 and +/- inf to +/- finite large number
	lt_diff = self.diff
        
	loss_wgt = top[0].diff
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * loss_wgt * np.add( np.multiply(adiff_gte_1, gte_diff), np.multiply(adiff_ls_1, lt_diff) ) / bottom[i].num

