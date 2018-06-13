import numpy as np
import pdb
import sys
# import matplotlib.pyplot as plt
# import warnings
# warnings.simplefilter(action = "ignore", category = FutureWarning)
import caffe

def softmax(x):
    x_exp = np.exp(x - np.max(x))
    return x_exp / np.sum(x_exp)

def taylor_softmax(x):
    x_tay = 1 + x + 0.5 * x ** 2
    return x_tay / np.sum(x_tay)

class BDistLayer(caffe.Layer):
    """A layer that computes Bhattacharyya distance using autograd"""

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute loss.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)

        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):

        # compute loss:
        yp = np.array(bottom[0].data.copy())
        y  = np.array(bottom[1].data.copy())

        yp = softmax(yp)
        
	epsilon       = np.finfo(np.float).eps # epsilon (float 32)
	prod_sqrt     = np.sqrt(yp * y)
        prod_sqrt_sum = np.sum(prod_sqrt)
	loss          = -np.log(np.maximum(prod_sqrt_sum,epsilon))

	# compute combined loss
        top[0].data[...] = loss

        # compute diffs:
        # pdb.set_trace()
	const = -0.5 / prod_sqrt_sum
        self.diff[...] = const * ((prod_sqrt_sum - prod_sqrt) * yp - prod_sqrt * (1 - yp))

    def backward(self, top, propagate_down, bottom):
        loss_wgt = top[0].diff
        bottom[0].diff[...] = loss_wgt * self.diff / bottom[0].num

class PBDistLayer(caffe.Layer):
    """A layer that computes Bhattacharyya distance"""

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute loss.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)

        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):

        # compute loss:
        yp = np.array(bottom[0].data.copy())
        y  = np.array(bottom[1].data.copy())
        
	epsilon       = np.finfo(np.float).eps # epsilon (float 32)
	prod_sqrt     = np.sqrt(yp * y)
        prod_sqrt_sum = np.sum(prod_sqrt)
	loss          = -np.log(np.maximum(prod_sqrt_sum,epsilon))

	# compute combined loss
        top[0].data[...] = loss

        # compute diffs:
        # pdb.set_trace()
	const = -0.5 / prod_sqrt_sum
        self.diff[...] = const * np.sqrt(y / yp)

    def backward(self, top, propagate_down, bottom):
        loss_wgt = top[0].diff
        bottom[0].diff[...] = loss_wgt * self.diff / bottom[0].num

class BDistTempLayer(caffe.Layer):
    """A layer that computes Bhattacharyya distance using autograd"""

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute loss.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)

        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):

        # compute loss:
        ypo = np.array(bottom[0].data.copy())
        y   = np.array(bottom[1].data.copy())

        # T
        T = float(self.param_str)

        yp = softmax(ypo / T)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ind     = np.arange(np.prod(y.shape))
        # width   = 0.2
        # r1 = ax.bar(ind, y.squeeze(), width, color='g')
        # r2 = ax.bar(ind + width, ypo.squeeze(), width, color='b')
        # r3 = ax.bar(ind + 2 * width, yp.squeeze(), width, color='r')
        # ax.set_ylabel('Score Distribution')
        # ax.legend((r1[0], r2[0], r3[0]), ('GT', 'Pred', 'sft Pred'))
        # plt.show()

        pdb.set_trace()

	epsilon       = np.finfo(np.float).eps # epsilon (float 32)
	prod_sqrt     = np.sqrt(yp * y)
        prod_sqrt_sum = np.sum(prod_sqrt)
	loss          = -np.log(np.maximum(prod_sqrt_sum,epsilon))

	# compute combined loss
        top[0].data[...] = loss

        # compute diffs:
        # pdb.set_trace()
	const = -0.5 / prod_sqrt_sum / T
        self.diff[...] = const * ((prod_sqrt_sum - prod_sqrt) * yp - prod_sqrt * (1 - yp))

    def backward(self, top, propagate_down, bottom):
        loss_wgt = top[0].diff
        bottom[0].diff[...] = loss_wgt * self.diff / bottom[0].num


class TSMBDistTempLayer(caffe.Layer):
    """A layer that computes Bhattacharyya distance using autograd"""

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (pred, gt) to compute loss.")

    def reshape(self, bottom, top):
        # difference is shape of prediction
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)

        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):

        # compute loss:
        ypo = np.array(bottom[0].data.copy())
        y   = np.array(bottom[1].data.copy())

        # T
        T = float(self.param_str)

        yp = taylor_softmax(ypo / T)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ind     = np.arange(np.prod(y.shape))
        # width   = 0.2
        # r1 = ax.bar(ind, y.squeeze(), width, color='g')
        # r2 = ax.bar(ind + width, ypo.squeeze(), width, color='b')
        # r3 = ax.bar(ind + 2 * width, yp.squeeze(), width, color='r')
        # ax.set_ylabel('Score Distribution')
        # ax.legend((r1[0], r2[0], r3[0]), ('GT', 'Pred', 'sft Pred'))
        # plt.show()

        pdb.set_trace()

        # TODO: correct gradient

	epsilon       = np.finfo(np.float).eps # epsilon (float 32)
	prod_sqrt     = np.sqrt(yp * y)
        prod_sqrt_sum = np.sum(prod_sqrt)
	loss          = -np.log(np.maximum(prod_sqrt_sum,epsilon))

	# compute combined loss
        top[0].data[...] = loss

        # compute diffs:
        # pdb.set_trace()
	const = -0.5 / prod_sqrt_sum / T
        self.diff[...] = const * ((prod_sqrt_sum - prod_sqrt) * yp - prod_sqrt * (1 - yp))

    def backward(self, top, propagate_down, bottom):
        loss_wgt = top[0].diff
        bottom[0].diff[...] = loss_wgt * self.diff / bottom[0].num

