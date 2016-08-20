import numpy
import theano
import theano.tensor as tensor
from blocks.bricks.lookup import LookupTable
from blocks.initialization import IsotropicGaussian


class Lookup(LookupTable):
    def __init__(self, length, dim, **kwargs):
        super(Lookup, self).__init__(length, dim, **kwargs)
    

    def initialize_with_pretrain(self, index_value_pairs):
        l = []
        for i in range(min(100, len(index_value_pairs))):
            l.append(index_value_pairs[i][1])
        narray = numpy.asarray(l)
        mean = numpy.mean(narray)
        var = numpy.var(narray)
        self.weights_init = IsotropicGaussian(mean = mean, std = numpy.sqrt(var))
        self.initialize()
        w = self.W.get_value()
        for index , value in index_value_pairs:
            w[index] = value
        self.W.set_value(w)

