# -*- coding : utf-8 -*-

import codecs
import ntpath
import os
import sys

import numpy
from fuel.transformers import Transformer
from nltk.tokenize import TweetTokenizer
twtk = TweetTokenizer()
from util.exception import *


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
            if source in self.sparse_idxes:
                sources.append(source + '_left_idx')
                sources.append(source + '_right_idx')
        return tuple(sources)

    def transform_batch(self, batch):
        new_batch = []
        for source, source_batch in zip(self.data_stream.sources, batch):
            if source in self.sparse_sources:
                # turn list of ndarray to one ndarray
                new_batch.append(numpy.concatenate(source_batch,axis = 0))
            elif source in self.sparse_idxes:
                new_batch.append(source_batch)
                i = 0
                left_idxes = []
                right_idxes = []
                for idxes in source_batch:
                    left_idxes += [i]*len(idxes)
                    right_idxes += idxes.tolist()
                    i += 1
                new_batch.append(numpy.array(left_idxes, dtype=source_batch.dtype))
                new_batch.append(numpy.array(right_idxes, dtype=source_batch.dtype))
            else:
                new_batch.append(source_batch)
        return tuple(new_batch)


class CharEmbedding(Transformer):
    def __init__(self, data_stream, char_source, char_idx_source, char_mask_dtype, **kwargs):
        super(CharEmbedding, self).__init__(data_stream=data_stream, produces_examples = False, **kwargs)
        self.char_source = char_source
        self.char_idx_source = char_idx_source
        self.char_mask_dtype = char_mask_dtype


    @property
    def sources(self):
        sources = []
        for source in self.data_stream.sources:
            sources.append(source)
            if source in self.char_source:
                sources.append(source + '_mask')
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
                for sample in source_batch:
                    char_batch += sample
                padded_batch, batch_mask = self._padding(numpy.array(char_batch))
                new_batch.append(padded_batch)
                new_batch.append(batch_mask)
            elif source in self.char_idx_source:
                new_batch.append(source_batch)
                i = 0
                left_idxes = []
                right_idxes = []
                for idxes in source_batch:
                    left_idxes += [i] * len(idxes)
                    right_idxes += idxes.tolist()
                    i += 1
                new_batch.append(numpy.array(left_idxes, dtype=source_batch.dtype))
                new_batch.append(numpy.array(right_idxes, dtype=source_batch.dtype))
            else:
                new_batch.append(source_batch)
        return tuple(new_batch)

    def _padding(self, source_batch):
        if len(source_batch) == 0:
            return numpy.array(source_batch), numpy.array([], dtype=self.char_mask_dtype)
        else:
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
                               self.char_mask_dtype)
            for i, sequence_length in enumerate(lengths):
                mask[i, :sequence_length] = 1
            return padded_batch, mask

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


def load_dic(path, mode = "debug"):
    if not os.path.exists(path):
        raise FileNotExistError()
    dic = {}
    count = 0
    with codecs.open(path, "r", encoding = "UTF-8", errors = "ignore") as f:
        for line in f:
            count += 1
            try:
                array = line.split('\t')
                if len(array) != 2:
                    raise FileFormatError("Encounter format error at %dth line" % count)
                dic[array[0]] = int(array[1])
            except Exception as error:
                if mode == "debug":
                    print(error.message)
                    choice = raw_input("Skip this error?y|n")
                    if choice.lower() == "n":
                        sys.exit(1)
    return dic


def read_file_by_line(file_path, delimiter="\t", field_num = None, mode="debug"):
    '''
    Read file by line and split it into fields with given delimiter.
    :param file_path: The path of the file
    :param delimiter: delimiter applied to split line into fields
    :param field_num: designed field number, if it does not match, error will raise in debug mode
    :param mode: running mode: debug or run, if it is run, ignore file format error else system will raise a hint
    :return: [[field1_line1,field2_line2..],[field1_line2,...]...]
    '''
    if not os.path.exists(file_path):
        raise FileNotExistError()
    dataset = []
    count = 0
    with codecs.open(file_path, mode = "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            count += 1
            try:
                if field_num is not None:
                    array = line.split(delimiter)
                    if len(array) != field_num:
                        raise FileFormatError("Only find %d fields in line %d with delimiter %s" %(len(array),count,delimiter))
                    else:
                        dataset.append(array)
                else:
                    dataset.append(line.split(delimiter))
            except FileFormatError as error:
                if mode == "debug":
                    print(error.message)
                    choice = raw_input("Skip this error?y|n")
                    if choice.lower() == "n":
                        sys.exit(1)
                else:
                    #pass the line. Debug mode should be run firstly
                    pass
    return dataset


def save_dic(path, dictionary):
    dir = ntpath.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with codecs.open(path, "w+", encoding = "UTF-8") as f:
        for key, value in dictionary.iteritems():
            f.write("%s\t%s\n" % (key, value))


def tokenize_sentence(sentence):
    global twtk
    if sentence is None:
        return []
    else:
        return twtk._tokenize(sentence)


