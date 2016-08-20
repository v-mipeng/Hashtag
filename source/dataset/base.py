# -*- coding : utf-8 -*-

import sys
sys.path.append("..")
import logging
import numpy
import re
import nltk

import ntpath
import codecs
import os
from abc import abstractmethod, ABCMeta
from error import MentionNotFoundError
from error import *


class _balanced_batch_helper(object):
    def __init__(self, key):
        self.key = key
    def __call__(self, data):
        return data[self.key].shape[0]


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


if __name__ == "main":
    c = _balanced_batch_helper()

