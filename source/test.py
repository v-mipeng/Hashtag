import theano
from theano import tensor
import numpy

def expand_index(idxes):
    left_idxes = []
    right_idxes = []

    def expand(l_idx, r_idx, l_idxes, r_idxes):
        l_idxes = tensor.concatenate([l_idxes, tensor.arange(r_idx.shape[0]) * 0 + l_idx], axis=0)
        r_idxes = tensor.concatenate([r_idxes, r_idx], axis=0)
        return l_idxes, r_idxes

    left_idxes, right_idxes, updates = theano.scan(expand,
                                                   sequences=[tensor.arange(idxes.shape[0]),
                                                              idxes],
                                                   outputs_info=[numpy.array([], dtype=idxes.dtype),
                                                                 numpy.array([], dtype=idxes.dtype)])