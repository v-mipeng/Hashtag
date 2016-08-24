﻿# -*- coding : utf-8 -*-
'''
This module provide the following datasets:
 1. User-graph of the given day. Format: user_id    followee_id
 2. User-post_hashtags on the given day. Format: user_id    post_hashtag    post_hashtag_times
 3. User-post-hashtag on the given day. Format: user_id     post_content    post_hashtags
'''
# import third-party module
import numpy
import theano
from fuel.datasets import Dataset, IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, ConstantScheme, IndexScheme, ShuffledExampleScheme, SequentialScheme
from fuel.transformers import Batch, Mapping, SortMapping, Unpack, Padding, Transformer
from nltk.tokenize import TweetTokenizer

#import system module
import sys
import os
import os.path as path
import re
import codecs
from collections import OrderedDict
from abc import abstractmethod, ABCMeta
import cPickle
from picklable_itertools import iter_
import logging
import random
import datetime


#import self-defined module
import base
from base import *
from base import _balanced_batch_helper
from error import *
from error import *
import time
from __builtin__ import super

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


class RUGD(object):
    '''
    Interface of raw user-graph dataset.
    This class provide user and corresponding follewees information on given day.

    '''
    def __init__(self, config):
        # Root path of user graph dataset
        self.user_graph_dir = config.user_graph_dir
        self.user_id2index_dir = config.user_id2index_dir
        self.mode = config.mode

    def get_user_graph(self, date):
        '''
        Get user graph of given day

        :param date: datetime.date object
        :return: A dictionary format user graph.
                The data format is: [(user_id, Set(user_followees)),...].
                Every field is an integer value.
        '''
        user_graph_path = path.join(self.user_graph_dir, str(date))
        raw_user_graph =  self._load_user_graph(user_graph_path)
        id2index_path = path.join(self.user_id2index_dir, str(date))
        if path.exists(id2index_path):
            id2index = self._load_id2index(id2index_path)
        else:
            id2index = self._extract_id2index(raw_user_graph)
        return self._turn_id2index(raw_user_graph, id2index),id2index

    def _load_user_graph(self, user_graph_path):
        '''
        Load raw user graph.
        Default data format:
        user_id     TAB     followee1   TAB     followee2...

        :param time: year_month_day format time, load given day's user graph
        :return:{(user, [followees]),...} format user graph. All the fields are string format
        '''
        raw_user_graph = {}
        count = 0
        with open(user_graph_path, "r",encoding="utf-8",errors="ignore") as f:
            for line in f:
                count += 1
                try:
                    array = line.split('\t')
                    user = array[0]
                    followees = array[1:]
                    raw_user_graph[user]=followees
                except Exception as error:
                    if self.mode == "debug":
                        print(error.message)
                        choice = raw_input("Skip this error?y|n")
                        if choice.lower() == "n":
                            sys.exit(1)
        return raw_user_graph

    def _load_id2index(self, id2index_path):
        '''
        Load user id to index information
        :param id2index_path: file path of id2index. file format: user_id   TAB     integer_index
        :return: A dictionary storing the id and corresponding index pairs
        '''
        return load_dic(id2index_path, self.mode)

    def _turn_id2index(self, raw_user_graph, id2index):
        user_graph = {}
        for user_id, user_followees in raw_user_graph.iteritems():
            user_index = id2index(user_id)
            followee_indexes = set()
            for user_followee in user_followees:
                followee_indexes.add(id2index(user_followee));
            user_graph[user_index] = followee_indexes
        return user_graph

    def _extract_id2index(self, raw_user_graph):
        '''
        Extract id2index from raw_user_graph
        :param raw_user_graph:
        :return:
        '''
        id2index = {}
        for user_id, user_followees in raw_user_graph.iteritems():
            if user_id not in id2index:
                id2index[user_id] = len(id2index)
            for user_followee in user_followees:
                if user_followee not in id2index:
                    id2index[user_followee] = len(id2index)
        save_dic(self.user_id2index_path, id2index)
        return id2index


class UGD(object):
    '''
    Turn raw user graph into stream.
    For each user and its followee pair, sample given number of users which are not of its followees.
    '''
    def __init__(self, config):
        self.user_sample_size = config.user_sample_size
        self.time = config.time
        self.rugd = RUGD(config)
        self.provide_souces = ('user','followees')

    def get_train_stream(self, time = None):
        '''
        Load dataset from given data_path, and split it into training dataset and validation dataset.
        Validation dataset size = total_dataset_size*valid_protion

        :param time: datetime.date object, if not given config.time will be used
        '''
        if time is None:
            time = self.time
        dataset = self._construct_dataset(time)
        return self._construct_shuffled_stream(dataset)

    def _construct_dataset(self, time):
        '''
        Do negtive sampling for user-followee pairs. The sample size is defined in the config file by field: user_hashtag_sample_size

        :return: block style dataset
        '''
        user_graph, id2index = self.rugd.get_user_graph(time)
        users = []
        followees = []
        rand_indexes = range(0,len(id2index))
        numpy.random.shuffle(rand_indexes)
        for user_index, followee_indexes in user_graph.iteritems():
            rand_begins = numpy.random.randint(0,len(id2index),size = len(followee_indexes))
            for followee_index, rand_begin in zip(followee_indexes, rand_begins):
                # sample for each user-followee pair
                users.append(user_index)
                samples = []
                samples.append(followee_index)
                i = 0
                while len(samples)<self.user_sample_size+1:
                    if (rand_begin+i%len(id2index)) not in followee_indexes:
                        samples.append(rand_begin+i%len(id2index))
                    i += 1
                followees.append(samples)
        #Construct block dataset
        return IndexableDataset(indexables={self.provide_souces[0] : users, self.provide_souces[1] : followees})

    def _construct_shuffled_stream(self, dataset):
        '''
        Construct shuffled stream.
        '''
        it = ShuffledExampleScheme(dataset.num_examples)
        stream = DataStream(dataset, iteration_scheme=it)
        # Group into batch
        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size))
        return stream


class RUHGD(object):
    '''
    Interface of raw user-hashtag-graph dataset.
    This class provide user and its post hashtag graph information during the last given_time-config.time_span days to the given_time

    '''
    def __init__(self,config):
        self.user_hashtag_graph_dir = config.user_hashtag_graph_dir
        self.user_id2index_dir = config.user_id2index_dir
        self.user_hashtag_time_span = config.user_hashtag_time_span
        self.hashtag2index_dir = config.hashtag2index_dir
        self.mode = config.mode

    def get_user_hashtag_graph(self, date, span): # do statistic work
        from datetime import timedelta
        assert date is not None and span is not None
        user_hashtag_graph_pathes = []
        for i in range(0, span):
            day = date - timedelta(days=i)
            user_hashtag_graph_pathes.append(path.join(self.user_hashtag_graph_dir, str(day)))
        raw_user_hashtag_graph = self._load_user_hashtag_graph(user_hashtag_graph_pathes)
        user_id2index = self._load_user_id2index(path.join(self.user_id2index_dir, str(date)))
        hashtag2index_path = path.join(self.hashtag2index_dir, str(date))
        if path.exists(hashtag2index_path):
            # Load hashtag2index information if it has been extracted for given date
            hashtag2index = self._load_hashtag2index(hashtag2index_path)
        else:
            if path.exists(path.join(self.hashtag2index_dir, str(date-timedelta(days=1)))):
                # Load hashtag2index information of last day for given date, and update it to the give date
                hashtag2index = self._load_hashtag2index(path.join(self.hashtag2index_dir, str(date-timedelta(days=1))))
            else:
                hashtag2index = {}
            self._extract_hashtag2index(raw_user_hashtag_graph, hashtag2index)
            save_dic(hashtag2index_path, hashtag2index)
        user_hashtag_graph = self._turn_id2index(user_id2index, hashtag2index, raw_user_hashtag_graph)
        return user_hashtag_graph, user_id2index, hashtag2index

    def _load_user_hashtag_graph(self, user_hashtag_graph_pathes):
        '''
        File format: user_id    TAB     hashtag1;hashtag2...    TAB     post_content    TAB     time
        :param time:
        :return: format {(user_id, List[hashtags])...}
        '''
        raw_user_hashtag_graph = {}

        for user_hashtag_graph_path in user_hashtag_graph_pathes:
            self._load_one_file(user_hashtag_graph_path, raw_user_hashtag_graph)
        return raw_user_hashtag_graph

    def _load_one_file(self, user_hashtag_graph_path, raw_user_hashtag_graph):
        with codecs.open(user_hashtag_graph_path, "r", encoding="utf-8", errors="ignore") as f:
            count = 0
            try:
                for line in f:
                    count += 1
                    array = line.split(line)
                    if len(array) != 4:
                        raise FileFormatError(
                            "Encounter format error at %dth line in file %s" % (count, user_hashtag_graph_path))
                    hashtags = array[1].split(";")
                    if array[0] not in raw_user_hashtag_graph:
                        raw_user_hashtag_graph[array[0]] = []
                    for hashtag in hashtags:
                        raw_user_hashtag_graph[array[0]].append(hashtag)
            except Exception as error:
                if self.mode == "debug":
                    print(error.message)
                    choice = raw_input("Skip this error?y|n")
                    if choice.lower() == "n":
                        sys.exit(1)

    def _load_user_id2index(self, user_id2index_path):
        return load_dic(user_id2index_path)


    def _load_hashtag2index(self, hashtag2index_path):
        return load_dic(hashtag2index_path)

    def _turn_id2index(self, user_id2index, hashtag2index, raw_user_hashtag_graph):
        user_hashtag_graph = {}
        for user_id, hashtags in raw_user_hashtag_graph.iteritems():
            hashtag_indexes = []
            user_index = user_id2index(user_id)
            for hashtag in hashtags:
                hashtag_indexes = hashtag2index(hashtag)
            user_hashtag_graph[user_index] = hashtag2index
        return user_hashtag_graph

    def _extract_hashtag2index(self, raw_user_hashtag_graph, hashtag2index):
        for user_id, hashtags in raw_user_hashtag_graph.iteritems():
            for hashtag in hashtags:
                if hashtag not in hashtag2index:
                    hashtag2index[hashtag] = len(hashtag2index)
        return hashtag2index


class UHGD(object):
    '''
    Turn raw user graph into stream.
    For each user and its followee pair, sample given number of users which are not of its followees.
    '''
    def __init__(self, config):
        self.user_hashtag_sample_size = config.user_sample_size
        self.date = config.date
        self.ruhgd = RUHGD(config)
        self.provide_souces = ('user','posts')

    def get_train_stream(self, time = None):
        '''
        Load dataset from given data_path, and split it into training dataset and validation dataset.
        Validation dataset size = total_dataset_size*valid_protion

        :param time: datetime.date object, if not given config.time will be used
        '''
        if time is None:
            time = self.time
        dataset = self._construct_dataset(time)
        return self._construct_shuffled_stream(dataset)

    def _construct_dataset(self, time):
        '''
        Do negtive sampling for user-post pairs. The sample size is defined in the config file by field: user_hashtag_sample_size

        :return: block style dataset
        '''
        user_graph, id2index = self.rugd.get_user_graph(time)
        users = []
        followees = []
        rand_indexes = range(0,len(id2index))
        numpy.random.shuffle(rand_indexes)
        for user_index, followee_indexes in user_graph.iteritems():
            rand_begins = numpy.random.randint(0,len(id2index),size = len(followee_indexes))
            for followee_index, rand_begin in zip(followee_indexes, rand_begins):
                # sample for each user-followee pair
                users.append(user_index)
                samples = []
                samples.append(followee_index)
                i = 0
                while len(samples)<self.user_hashtag_sample_size+1:
                    if (rand_begin+i%len(id2index)) not in followee_indexes:
                        samples.append(rand_begin+i%len(id2index))
                    i += 1
                followees.append(samples)
        #Construct block dataset
        return IndexableDataset(indexables={self.provide_souces[0] : users, self.provide_souces[1] : followees})

    def _construct_shuffled_stream(self, dataset):
        '''
        Construct shuffled stream.
        '''
        it = ShuffledExampleScheme(dataset.num_examples)
        stream = DataStream(dataset, iteration_scheme=it)
        # Group into batch
        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size))
        return stream

    def _count_hashtag_distribution(self, user_hashtag_graph):
        '''
        Get hashtag distribution: Statistic hashtag number
        For negtive sampling
        :param user_hashtag_graph:
        :return:
        '''
        hashtag_list = []
        for _, hashtags in user_hashtag_graph.iteritems():
            hashtag_list = hashtag_list+hashtags


class BUTHD(object):
    '''
    Basic dataset with user-text-time-hashtag information.

    load dataset --> parse string type date --> map user, hashtag, word to id --> provide samples of given date -->
    --> construct indexable dataset --> construct shuffled or sequencial fuel stream
    '''
    def __init__(self, config):
        self.config = config
        # Dictionary
        self.user2index = None
        self.hashtag2index = None
        self.word2index = None
        self.word2freq = None
        # Integer. Word whose frequency is less than the threshold will be stemmed
        self.sparse_word_threshold = None
        # 1D numpy array. Storing the user ids in file reading order
        self.users = None
        # 2D numpy array. Storing the post words in file reading order
        self.words = None
        # 1D numpy array. Storing the hashtags in file reading order
        self.hashtags = None
        # 1D numpy array. Storing the post dates in file reading order
        self.dates = None
        # Integer. Date span of the posted tweets
        self.date_span = 0
        # (1D numpy array, 1D numpy array). Storing hashtag id and hashtag normed number pair
        self.hashtag_dis_table = None
        self.provide_souces = ('user', 'text', 'hashtag')
        self.need_mask_sources = {'text':config.int_type}
        self.compare_source = 'text'
        self.LAST_DAY = "LAST_DAY"
        self.FIRST_DAY = "FIRST_DAY"

    def get_parameter_to_save(self):
        '''
        Return parameters that need to be saved with model
        :return:
        '''
        return OrderedDict({'hashtag2index' : self.hashtag2index, 'word2index' : self.word2index, 'user2index' : self.user2index})

    def prepare(self, data_path = None):
        '''
        Prepare dataset
        :param data_path:
        :return:
        '''
        if data_path is None and self.users is None:
            data_path = self.config.train_path
        if data_path is not None:
            raw_dataset = self._load_dataset(data_path)
            raw_dataset = self._turn_str_date2obj(raw_dataset)
            self._turn_str2index(raw_dataset)
            self._construct_hashtag_distribution()

    def get_shuffled_stream(self, data_path = None, reference_date = "LAST_DAY", date_offset = 0, duration = 3):
        '''
        Get shuffled stream of the dataset constructed with samples posted within given date duration
        date duration :
            if reference_date is FIRST_DAY:
                duration = 9first_day + date_offset - duration, first_day + date_offset]
            else:
                duration = 9last_day - date_offset - duration, last_day - date_offset)
        :param data_path: string type path of the dataset. If not given, get data from the dataset last loaded
        :param reference_date: 'FIRST_DAY' OR 'LAST_DAT', 'FIRST_DAY' (by default)
        :param date_offset: integer type, 0 (by default)
        :param duration: integer type, 3 (by default)
        :return: a shuffled stream constructed from the items of the given day
        '''
        dataset = self._get_dataset(data_path, reference_date = "LAST_DAY", date_offset = 0, duration = 3)
        return self._construct_shuffled_stream(dataset)

    def get_sequencial_stream(self, data_path = None, reference_date = "LAST_DAY", date_offset = 0, duration = 3):

        '''
        Get sequencial stream of the dataset constructed with samples posted within given date duration
        date duration :
            if reference_date is FIRST_DAY:
                duration = 9first_day + date_offset - duration, first_day + date_offset]
            else:
                duration = 9last_day - date_offset - duration, last_day - date_offset]
        :param data_path: string type path of the dataset. If not given, get data from the dataset last loaded
        :param reference_date: 'FIRST_DAY' OR 'LAST_DAT', 'LAST_DAT' (by default)
        :param date_offset: integer type, 0 (by default)
        :param duration: integer type, 3 (by default)
        :return: a sequencial stream constructed from the items of the given day
        '''
        dataset = self._get_dataset(data_path, reference_date = "LAST_DAY", date_offset = 0, duration = 3)
        return self._construct_sequencial_stream(dataset)

    def _get_dataset(self, data_path = None, reference_date = "LAST_DAY", date_offset = 0, duration = 3):
        self.prepare(data_path)
        # Get duration of the dataset to load
        if reference_date == self.FIRST_DAY:
            date_end = self.dates.min() + datetime.timedelta(days= date_offset)
            date_begin = date_end - datetime.timedelta(days= duration)
        elif reference_date == self.LAST_DAY:
            date_end = self.dates.max() - datetime.timedelta(days=date_offset)
            date_begin = date_end - datetime.timedelta(days=duration)
        else:
            raise ValueError('reference_date should only be "FIRST_DAY" or "LAST_DAY"!')

        dataset = self._find_data_by_date(date_begin, date_end)
        return self._construct_dataset(dataset)

    def _load_dataset(self, data_path):
        '''
        Load the raw dataset from given file
        :param data_path: string type path of the dataset file
        :return: [[field1_line1,field2_line2..],[field1_line2,...]...] format dataset
        '''
        dataset = base.read_file_by_line(data_path, delimiter=self.config.delimiter,field_num=self.config.field_num,mode=self.config.mode)
        fields = zip(*dataset)
        tokenized_posts = []
        for post in fields[self.config.text_index]:
            tokens = post.split(' ')
            tokenized_posts.append(tokens)
        fields[self.config.text_index] = tokenized_posts
        return zip(*fields)

    def _turn_str_date2obj(self, raw_dataset):
        '''
        Turn string type dates into datetime.date objects
        :param dataset:
        :return:
        '''
        fields = zip(*raw_dataset)
        dates = []
        for item in fields[self.config.date_index]:
            dates.append(self._parse_date(item))
        fields[self.config.date_index] = dates
        return zip(*fields)

    def _parse_date(self, str_date):
        '''
        Parse string format date.

        Reference: https://docs.python.org/2/library/time.html or http://strftime.org/
        :return: A datetime.date object
        '''
        return datetime.datetime.strptime(str_date, "%a %b %d %H:%M:%S +0000 %Y").date()

    def _turn_str2index(self, raw_dataset):
        '''
        Turn string type user, words of context, hashtag representation into index representation.
        load the dictionaries if existing otherwise extract from dataset and store them
        '''
        # try to load mapping dictionaries
        self.user2index = {}
        self.word2index = {}
        self.hashtag2index = {}
        fields = zip(*raw_dataset)
        if path.exists(self.config.user2index_path):
            self.user2index = base.load_dic(self.config.user2index_path, mode=self.config.mode)
        else:
            self.user2index = self._extract_user2index(fields[self.config.user_index])
            base.save_dic(self.config.user2index_path, self.user2index)
        if path.exists(self.config.word2freq_path):
            self.word2freq = base.load_dic(self.config.word2freq_path, mode=self.config.mode)
        else:
            self.word2freq = self._extract_word2freq(fields[self.config.text_index])
            base.save_dic(self.config.word2freq_path, self.word2freq)
        if path.exists(self.config.word2index_path):
            self.word2index = base.load_dic(self.config.word2index_path, mode=self.config.mode)
        else:
            self.word2index = self._extract_word2index(fields[self.config.text_index])
            base.save_dic(self.config.word2index_path, self.word2index)

        if path.exists(self.config.hashtag2index_path):
            self.hashtag2index = base.load_dic(self.config.hashtag2index_path, mode=self.config.mode)
        else:
            self.hashtag2index = self._extract_hashtag2index(fields[self.config.hashtag_index])
            base.save_dic(self.config.hashtag2index_path, self.hashtag2index)

        self.users = numpy.array([self.user2index[user] for user in fields[self.config.user_index]], dtype = self.config.int_type)
        self.hashtags = numpy.array([self.hashtag2index[hashtag] for hashtag in fields[self.config.hashtag_index]], dtype = self.config.int_type)
        self.texts = numpy.array([numpy.array([self.word2index[self._stem(word)] for word in text], dtype = self.config.int_type) for text in fields[self.config.text_index]])
        self.dates = numpy.asarray(fields[self.config.date_index])
        self.date_span = self.dates.max() - self.dates.min()+1

    def _extract_word2freq(self, texts):
        '''
        Count word frequency
        :param texts:
        :return:
        '''
        assert  texts is not None
        word2freq = {}
        for words in texts:
            for word in words:
                if word not in word2freq:
                    word2freq[word] = 1
                else:
                    word2freq[word] += 1
        return word2freq

    def _extract_user2freq(self, users):
        assert users is not None
        user2freq = {}
        for user in users:
            if user not in user2freq:
                user2freq[user] = 1
            else:
                user2freq[user] += 1
        return user2freq

    def _extract_hashtag2freq(self, hashtags):
        assert hashtags is not None
        hashtag2freq = {}
        for hashtag in hashtags:
            if hashtag not in hashtag2freq:
                hashtag2freq[hashtag] = 1
            else:
                hashtag2freq[hashtag] += 1
        return hashtag2freq

    def _get_sparse_word_threshold(self):
        '''
        Get the frequency threshold. Word with its frequency below the threshold will be treated as sparse word
        '''
        num = numpy.array(self.word2freq.values())
        num.sort()
        total = num.sum()
        cum_num = num.cumsum()
        threshold = int(total*self.config.sparse_word_percent)
        min_index = numpy.argmin(numpy.abs(threshold-cum_num))
        if cum_num[min_index] > threshold:
            self.sparse_word_threshold = num[numpy.max(min_index-1,0)]
        else:
            self.sparse_word_threshold = num[min_index]

    def _extract_user2index(self, users):
        assert  users is not None
        user2index = {}
        for user in users:
            if user not in user2index:
                user2index[user] = len(user2index)
        return user2index

    def _extract_word2index(self, texts):
        assert  texts is not None
        word2index = {}
        for words in texts:
            for word in words:
                word = self._stem(word)
                if word not in word2index:
                    word2index[word] = len(word2index)
        return word2index

    def _extract_hashtag2index(self, hashtags):
        assert  hashtags is not None
        hashtag2index = {}
        for user in hashtags:
            if user not in hashtag2index:
                hashtag2index[user] = len(hashtag2index)
        return hashtag2index

    def _find_data_by_date(self, date_begin, date_end):
        '''
        Find samples posted during (data_begin,data_end)
        :param date_begin:
        :param date_end:
        :return: [[field1_line1,field2_line2..],[field1_line2,...]...] format dataset for given date
        '''

        idxes = (self.dates > date_begin) * (self.dates <= date_end)
        return (self.users[idxes], self.texts[idxes], self.hashtags[idxes])

    def _construct_dataset(self, dataset):
        return IndexableDataset(indexables= OrderedDict(zip(self.provide_souces, dataset)))

    def _construct_shuffled_stream(self, dataset):
        '''
        Construc a shuffled stream from given dataset
        :param dataset:
        :return:
        '''
        it = ShuffledExampleScheme(dataset.num_examples)
        stream = DataStream(dataset, iteration_scheme=it)
        # Sort sets of multiple batches to make batches of similar sizes
        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size * self.config.sort_batch_count))
        comparison = _balanced_batch_helper(stream.sources.index(self.compare_source))
        stream = Mapping(stream, SortMapping(comparison))
        stream = Unpack(stream)
        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size))
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype= source[1])
        stream = base.NegativeSample(stream,
                                     [self.hashtag_dis_table],
                                     sample_sources= ["hashtag"],
                                     sample_sizes= [self.config.hashtag_sample_size])
        return stream

    def _construct_sequencial_stream(self, dataset):
        '''Construct sequencial stream.
        This is usually used for testing and prediction.
        '''
        it = SequentialScheme(dataset.num_examples, self.config.batch_size)
        stream = DataStream(dataset, iteration_scheme=it)
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype= source[1])
        stream = base.NegativeSample(stream,
                                     [self.hashtag_dis_table],
                                     sample_sources=["hashtag"],
                                     sample_sizes=[self.config.hashtag_sample_size])
        return stream

    def _construct_hashtag_distribution(self):
        '''
        Build hashtag distribution table
        :return:
        '''
        count,_ = numpy.histogram(self.hashtags, bins = len(self.hashtag2index))
        count = count**(3.0/4)
        count = count/count.sum()
        self.hashtag_dis_table = (numpy.arange(len(self.hashtag2index)), count)

    def _stem(self, word):
        '''
        Do word stemming
        :param word: original string type of word
        :return: stemmed word
        '''
        if self.sparse_word_threshold is None:
            self._get_sparse_word_threshold()
        if word in self.word2freq and self.word2freq[word] > self.sparse_word_threshold:
            return word
        elif word.startwith('http'):
            return 'URL'
        else:
            return '<unk>'

class SUTHD(BUTHD):
    '''
    Sequential user-text-hahstag dataset.Provide dataset in time order.
    '''

    def __init__(self, config, duration):

        self.duration = duration
        super(SUTHD, self).__init__(config)

    def get_stream(self, data_path):
        pass



if __name__ == "__main__":
    from config.hashtag_config import UTHC
    config = UTHC()
    dataset = BUTHD(config)
    stream = dataset.get_shuffled_stream()
    print("\t".join(stream.sources))
    for batch in stream.get_epoch_iterator():
        print(batch[stream.sources.index('hashtag')])
        print(batch[stream.sources.index('hashtag_negtive_sample')])
        if raw_input("continue?y|n") == 'n':
            break

