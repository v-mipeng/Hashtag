# -*- coding : utf-8 -*-
import theano
import numpy
from fuel.transformers import Transformer
from util.exception import *

#region Transformer
class _balanced_batch_helper(object):
    def __init__(self, key):
        self.key = key
    def __call__(self, data):
        return data[self.key].shape[0]


class NegativeSample(Transformer):

    def __init__(self, data_stream, dist_tables, sample_sources, sample_sizes, **kwargs):
        # produces_examples = False: invoke transform_batch() otherwise transform_example()
        super(NegativeSample, self).__init__(
            data_stream, produces_examples=False, **kwargs)
        self.dist_tables = dist_tables
        self.sample_sources = sample_sources
        self.sample_sizes = sample_sizes
        self._check_dist_table()

    def _check_dist_table(self):
        for i in range(len(self.dist_tables)):
            _,count = self.dist_tables[i]
            if not isinstance(count, numpy.ndarray):
                count = numpy.array(count)
            if sum(count == count.sum()) > 0:
                raise ValueError('Cannot apply negtive sampling for the probability of one element is 1.0')

    @property
    def sources(self):
        sources = []
        for source in self.data_stream.sources:
            sources.append(source)
            if source in self.sample_sources:
                sources.append(source + '_negtive_sample')
        return tuple(sources)

    def transform_batch(self, batch):
        batch_with_samplings = []
        i = 0
        for source, source_batch in zip(self.data_stream.sources, batch):
            if source not in self.sample_sources:
                batch_with_samplings.append(source_batch)
                continue

            neg_samples = []
            for source_example in source_batch:
                neg_sample = []
                while len(neg_sample) < self.sample_sizes[i]:
                    ids = self.sample_id(self.dist_tables[i], self.sample_sizes[i])
                    for id in ids:
                        if len(numpy.where(source_example == id)[0]) == 0:
                            neg_sample.append(id)
                            if len(neg_sample) == self.sample_sizes[i]:
                                break
                neg_samples.append(neg_sample)
            neg_samples = numpy.array(neg_samples, dtype= source_batch.dtype)
            batch_with_samplings.append(source_batch)
            batch_with_samplings.append(neg_samples)
        i = i+1
        return tuple(batch_with_samplings)


    def sample_id(self, num_by_id, sample_size = 1):
        # bisect search
        def bisect_search(sorted_na, value):
            '''
            Do bisect search
            :param sorted_na: cumulated sum array
            :param value: random value
            :return: the index that sorted_na[index-1]<=value<sorted_na[index] with defining sorted_na[-1] = -1
            '''
            if len(sorted_na) == 1:
                return 0
            left_index = 0
            right_index = len(sorted_na)-1

            while right_index-left_index > 1:
                mid_index = (left_index + right_index) / 2
                # in right part
                if value > sorted_na[mid_index]:
                    left_index = mid_index
                elif value < sorted_na[mid_index]:
                    right_index = mid_index
                else:
                    return min(mid_index+1,right_index)
            return right_index
        id, num = num_by_id
        cum_num = num.cumsum()
        rvs = numpy.random.uniform(low = 0.0, high = cum_num[-1], size=(sample_size,))
        ids = []
        for rv in rvs:
            if len(id) < 20000: # This value is obtained by test
                index = numpy.argmin(numpy.abs(cum_num-rv))
                if rv >= cum_num[index]:
                    index += 1
                else:
                    pass
            else:
                index = bisect_search(cum_num, rv)
            ids.append(id[index])
        return ids


class SparseIndex(Transformer):
    def __init__(self, data_stream, sparse_pairs, **kwargs):
        # produces_examples = False: invoke transform_batch() otherwise transform_example()
        super(SparseIndex, self).__init__(
            data_stream, produces_examples=False, **kwargs)
        self.sparse_sources, self.sparse_idxes = zip(*sparse_pairs)

    @property
    def sources(self):
        sources = []
        for source in self.data_stream.sources:
            sources.append(source)
            if source in self.sparse_sources:
                sources.append(source+"_sparse_mask")
            if source in self.sparse_idxes:
                sources.append(source + '_left_idx')
                sources.append(source + '_right_idx')
        return tuple(sources)

    def transform_batch(self, batch):
        new_batch = []
        for source, source_batch in zip(self.data_stream.sources, batch):
            if source in self.sparse_sources:
                # turn list of ndarray to one ndarray
                tmp = numpy.concatenate(source_batch, axis=0)
                if len(tmp) > 0:
                    mask = numpy.ones(len(tmp), dtype=theano.config.floatX)
                else:
                    tmp = numpy.array([0], dtype=source_batch.dtype)
                    mask = numpy.zeros(1, dtype=theano.config.floatX)
                new_batch.append(tmp)
                new_batch.append(mask)
            elif source in self.sparse_idxes:
                new_batch.append(source_batch)
                i = 0
                left_idxes = []
                right_idxes = []
                for idxes in source_batch:
                    left_idxes += [i]*len(idxes)
                    right_idxes += idxes.tolist()
                    i += 1
                if len(left_idxes) == 0:
                    left_idxes=[0]
                    right_idxes=[0]
                new_batch.append(numpy.array(left_idxes, dtype=source_batch[0].dtype))
                new_batch.append(numpy.array(right_idxes, dtype=source_batch[0].dtype))
            else:
                new_batch.append(source_batch)
        return tuple(new_batch)


class CharEmbedding(Transformer):
    def __init__(self, data_stream, char_source, char_idx_source, **kwargs):
        super(CharEmbedding, self).__init__(data_stream=data_stream, produces_examples = False, **kwargs)
        self.char_source = char_source
        self.char_idx_source = char_idx_source


    @property
    def sources(self):
        sources = []
        for source in self.data_stream.sources:
            sources.append(source)
            if source in self.char_source:
                sources.append(source + '_mask')
                sources.append(source + '_sparse_mask')
            elif source in self.char_idx_source:
                sources.append(source + '_left_idx')
                sources.append(source + '_right_idx')
            else:
                pass
        return tuple(sources)

    def transform_batch(self, batch):
        new_batch = []
        for source, source_batch in zip(self.data_stream.sources, batch):
            if source in self.char_source:
                # turn list of ndarray to one ndarray
                char_batch = []
                for item in source_batch:
                    try:
                        char_batch += list(item)
                    except Exception as e:
                        print(type(item))
                if len(char_batch) == 0:
                    padded_batch = numpy.array([[0,0]], dtype="int32")
                    batch_mask = numpy.array([[1.,1.]], dtype=theano.config.floatX)
                    mask = numpy.zeros(1, dtype=theano.config.floatX)
                else:
                    padded_batch, batch_mask = self._padding(numpy.asarray(char_batch))
                    mask = numpy.ones(len(padded_batch), dtype=theano.config.floatX)
                new_batch.append(padded_batch)
                new_batch.append(batch_mask)
                new_batch.append(mask)
            elif source in self.char_idx_source:
                new_batch.append(source_batch)
                i = 0
                left_idxes = []
                right_idxes = []
                for idxes in source_batch:
                    left_idxes += [i] * len(idxes)
                    right_idxes += idxes.tolist()
                    i += 1
                if len(left_idxes) == 0:
                    left_idxes = [0]
                    right_idxes = [0]
                new_batch.append(numpy.array(left_idxes, dtype=source_batch[0].dtype))
                new_batch.append(numpy.array(right_idxes, dtype=source_batch[0].dtype))
            else:
                new_batch.append(source_batch)
        return tuple(new_batch)

    def _padding(self, source_batch):
        shapes = [numpy.asarray(sample).shape for sample in source_batch]
        lengths = [shape[0] for shape in shapes]
        max_sequence_length = max(lengths)
        rest_shape = shapes[0][1:]
        if not all([shape[1:] == rest_shape for shape in shapes]):
            raise ValueError("All dimensions except length must be equal")
        dtype = numpy.asarray(source_batch[0]).dtype

        padded_batch = numpy.zeros(
            (len(source_batch), max_sequence_length) + rest_shape,
            dtype=dtype)
        for i, sample in enumerate(source_batch):
            padded_batch[i, :len(sample)] = sample

        mask = numpy.zeros((len(source_batch), max_sequence_length),
                           dtype = theano.config.floatX)
        for i, sequence_length in enumerate(lengths):
            mask[i, :sequence_length] = 1
        return padded_batch, mask

#endregion

def split_train_valid(data , valid_portion):
    '''
    Split dataset into training set and validation dataset
    '''
    idxes = range(len(data))
    numpy.random.shuffle(idxes)
    train_data = []
    valid_data = []
    for idx in range(int(numpy.floor(valid_portion*len(data)))):
        valid_data.append(data[idxes[idx]])
    for idx in range(int(numpy.floor(valid_portion*len(data))),len(data)):
        train_data.append(data[idxes[idx]])
    return train_data, valid_data


def get_sparse_threshold(freq, sparse_percent):
    '''
    Get the frequency threshold. Word with its frequency below the threshold will be treated as sparse word
    '''
    assert  sparse_percent < 1.
    num = numpy.array(freq)
    num.sort()
    total = num.sum()
    cum_num = num.cumsum()
    threshold = int(total * sparse_percent)
    min_index = numpy.argmin(numpy.abs(threshold - cum_num))
    sparse_threshold = 0.
    if min_index == 0:
        if threshold < cum_num[0]:
            return num[0]-1
        else:
            if num[0] == num[1]:
                return num[0]-1
            else:
                return num[0]
    elif cum_num[min_index] > threshold:
        if num[min_index] == num[min_index-1]:
            return num[min_index]-1
        else:
            return num[min_index-1]
    else:
        if num[min_index] == num[min_index+1]:
            return num[min_index]-1
        else:
            return num[min_index]
