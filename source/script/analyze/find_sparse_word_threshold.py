from dataset import RUTHD
from config import UTHC
import numpy

def find_sparse_threshold():
    config = UTHC
    dataset = RUTHD(config)
    dataset.prepare()
    raw_dataset = dataset.raw_dataset
    texts = zip(*raw_dataset)[config.text_index]
    texts = numpy.concatenate(numpy.array(texts))
    words, count = numpy.unique(texts, return_counts=True)
    print('unique word number:{0}\n'.format(len(words)))
    print('word number with frequency less than 5:{0}\n'.format(numpy.sum(count < 5)))
    print('word number with frequency less than 10:{0}\n'.format(numpy.sum(count < 10)))
