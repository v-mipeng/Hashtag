# -*- coding : utf-8 -*-
#region Import
'''
This module provide the following datasets:
 1. User-graph of the given day. Format: user_id    followee_id
 2. User-post_hashtags on the given day. Format: user_id    post_hashtag    post_hashtag_times
 3. User-post-hashtag on the given day. Format: user_id     post_content    post_hashtags
'''
# import third-party module
import cPickle
import datetime
import os.path as path
from collections import OrderedDict

from fuel.datasets import IndexableDataset
from fuel.schemes import ConstantScheme, ShuffledExampleScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Batch, Mapping, SortMapping, Unpack, Padding


from util.dataset import *
from util.dataset import _balanced_batch_helper
import theano

from abc import ABCMeta, abstractmethod, abstractproperty
#endregion

#region Develop
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
#endregion

class RUTHD(object):
    '''
       Basic dataset with user-text-time-hashtag information.

       load dataset --> parse string type date --> provide samples of given date --> map user, hashtag, word to id -->
       --> construct indexable dataset --> construct shuffled or sequencial fuel stream
       '''

    def __init__(self, config):
        # Dictionary
        self.config = config
        self.raw_dataset = None
        self.dates = None
        self.first_date = None
        self.last_date = None
        self.date_span = None
        self.LAST_DAY = "LAST_DAY"
        self.FIRST_DAY = "FIRST_DAY"

    def prepare(self, raw_dataset = None, data_path=None):
        '''
        Prepare dataset
        :param data_path:
        :return:
        '''
        if data_path is None and self.raw_dataset is None:
            data_path = self.config.train_path
        print("Preparing dataset...")
        # Load pickled dataset
        #TODO: if raw_dataset is not None
        if raw_dataset is None:
            with open(data_path, 'rb') as f:
                self.raw_dataset = cPickle.load(f)
        else:
            self.raw_dataset = raw_dataset
        fields = zip(*self.raw_dataset)
        self.dates = numpy.array(fields[self.config.date_index])
        self.first_date = self.dates.min()
        self.last_date = self.dates.max()
        self.date_span = (self.last_date - self.first_date).days + 1
        print("Done!")

    def get_dataset(self, data_path=None, reference_date="LAST_DAY", date_offset=0, duration=3):
        '''
        Return dataset within given days
        :param data_path:
        :param reference_date:
        :param date_offset:
        :param duration:
        :return: A ndarray: array([[[],[]..],...])
        '''
        # Get duration of the dataset to load
        if reference_date == self.FIRST_DAY:
            date_end = self.first_date + datetime.timedelta(days=date_offset)
            date_begin = date_end - datetime.timedelta(days=duration)
        elif reference_date == self.LAST_DAY:
            date_end = self.last_date - datetime.timedelta(days=date_offset)
            date_begin = date_end - datetime.timedelta(days=duration)
        else:
            raise ValueError('reference_date should only be "FIRST_DAY" or "LAST_DAY"!')

        raw_dataset = self._find_data_by_date(date_begin, date_end)

        return raw_dataset

    def _find_data_by_date(self, date_begin, date_end):
        '''
        Find samples posted during (data_begin,data_end)
        :param date_begin:
        :param date_end:
        :return: [[field1_line1,field2_line2..],[field1_line2,...]...] format dataset for given date
        '''
        idxes = numpy.logical_and(self.dates > date_begin, self.dates <= date_end)
        return self.raw_dataset[idxes]


class UTHD(object):

    def __init__(self, config, raw_dataset=None):
        self.config = config
        self.provide_souces = ('user', 'text', 'hashtag')
        self.need_mask_sources = {'text': theano.config.floatX}
        self.compare_source = 'text'
        if raw_dataset is None:
            self.raw_dataset = RUTHD(config)
        else:
            self.raw_dataset = raw_dataset
        self._initialize()

    @abstractmethod
    def _initialize(self):
        '''
        Initialize dataset information
        '''
        raise NotImplementedError

    @abstractmethod
    def get_parameter_to_save(self):
        '''
        Return parameters that need to be saved with model
        :return: OrderedDict of{name:data,...}
        '''
        raise NotImplementedError

    def get_shuffled_stream_by_date(self, date, update=False):
        self.raw_dataset.prepare()
        return self.get_shuffled_stream(reference_date="FIRST_DAY",
                                        date_offset=(date - self.raw_dataset.first_date).days,
                                        duration=1, update=update)

    def get_train_stream(self, raw_dataset, it='shuffled'):
        return self._get_stream(raw_dataset, it, for_type='train')

    def get_test_stream(self, raw_dataset, it='shuffled'):
        return self._get_stream(raw_dataset, it, for_type='test')

    def _get_stream(self, raw_dataset, it = 'shuffled', for_type = 'train'):
        raw_dataset = self._update_before_transform(raw_dataset, for_type)
        dataset = self._turn_str2index(raw_dataset, for_type)
        dataset = self._update_after_transform(dataset, for_type)
        if it == 'shuffled':
            return self._construct_shuffled_stream(dataset)
        elif it == 'sequencial':
            return self._construct_sequencial_stream(dataset)
        else:
            raise ValueError('it should be "shuffled" or "sequencial"!')

    @abstractmethod
    def _update_before_transform(self, raw_dataset, for_type = 'train'):
        '''
        Do updation beform transform raw_dataset into index representation dataset
        :param raw_dataset:
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :return: a new raw_dataset
        '''
        return raw_dataset

    @abstractmethod
    def _update_after_transform(self, dataset, for_type = 'train'):
        '''
        Do updation after transform raw_dataset into index representation dataset
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :param dataset: tranformed dataset
        :return: a new transformed dataset
        '''
        return dataset

    @abstractmethod
    def _update(self, raw_dataset):
        '''
        Update dataset information.
        :param raw_dataset: ndarray or list
        '''
        raise NotImplementedError

    @abstractmethod
    def _turn_str2index(self, raw_dataset, for_type='train'):
        '''
        Turn string type user, words of context, hashtag representation into index representation.

        Note: Implement this function in subclass
        '''

        raise NotImplementedError

    def _construct_dataset(self, dataset):
        '''
        Construct an fule indexable dataset.
        Every data corresponds to the name of self.provide_sources
        :param dataset: A tuple of data
        :return:
        '''
        return IndexableDataset(indexables=OrderedDict(zip(self.provide_souces, dataset)))

    def _construct_shuffled_stream(self, dataset):
        '''
        Construc a shuffled stream from given dataset
        :param dataset: fuel Indexable dataset
        :return: A fuel shuffled stream with basic transformations:
        1.Sort data by self.compare_source
        2.Batch dataset
        3.Add mask on self.need_mask_sources
        '''
        it = ShuffledExampleScheme(dataset.num_examples)
        stream = DataStream(dataset, iteration_scheme=it)
        # Sort sets of multiple batches to make batches of similar sizes
        stream = Batch(stream,
                       iteration_scheme=ConstantScheme(self.config.batch_size * self.config.sort_batch_count))
        comparison = _balanced_batch_helper(stream.sources.index(self.compare_source))
        stream = Mapping(stream, SortMapping(comparison))
        stream = Unpack(stream)
        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size))
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        return stream

    def _construct_sequencial_stream(self, dataset):
        '''
        Construc a sequencial stream from given dataset
        :param dataset: fuel Indexable dataset
        :return: A fuel sequencial stream with basic transformations:
        1.Sort data by self.compare_source
        2.Batch dataset
        3.Add mask on self.need_mask_sources
        '''
        it = SequentialScheme(dataset.num_examples, self.config.batch_size)
        stream = DataStream(dataset, iteration_scheme=it)
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        return stream


class BUTHD(object):
    __metaclass__ = ABCMeta

    def __init__(self, config, raw_dataset = None):
        self.config = config
        self.provide_souces = ('user', 'text', 'hashtag')
        self.need_mask_sources = {'text': theano.config.floatX}
        self.compare_source = 'text'
        if raw_dataset is None:
            self.raw_dataset = RUTHD(config)
        else:
            self.raw_dataset = raw_dataset
        self._initialize()

    @abstractmethod
    def _initialize(self):
        '''
        Initialize dataset information
        '''
        raise NotImplementedError

    @abstractmethod
    def get_parameter_to_save(self):
        '''
        Return parameters that need to be saved with model
        :return: OrderedDict of{name:data,...}
        '''
        raise NotImplementedError

    def get_shuffled_stream_by_date(self, date, update=False):
        self.raw_dataset.prepare()
        return self.get_shuffled_stream(reference_date="FIRST_DAY",
                                        date_offset=(date-self.raw_dataset.first_date).days,
                                        duration=1,update=update)

    def get_shuffled_stream(self, reference_date="LAST_DAY", date_offset=0, duration=3, update=True):
        '''
        Get shuffled stream of the dataset constructed with samples posted within given date duration
        date duration :
            if reference_date is FIRST_DAY:
                duration = 9first_day + date_offset - duration, first_day + date_offset]
            else:
                duration = 9last_day - date_offset - duration, last_day - date_offset)
        :param reference_date: 'FIRST_DAY' OR 'LAST_DAT', 'FIRST_DAY' (by default)
        :param date_offset: integer type, 0 (by default)
        :param duration: integer type, 3 (by default)
        :return: a shuffled stream constructed from the items of the given day
        '''
        dataset = self._get_dataset(reference_date=reference_date, date_offset=date_offset,
                                    duration=duration, update=update)
        return self._construct_shuffled_stream(dataset)

    def get_sequencial_stream(self, reference_date="LAST_DAY", date_offset=0, duration=3, update=True):

        '''
        Get sequencial stream of the dataset constructed with samples posted within given date duration
        date duration :
            if reference_date is FIRST_DAY:
                duration = 9first_day + date_offset - duration, first_day + date_offset]
            else:
                duration = 9last_day - date_offset - duration, last_day - date_offset]
        :param reference_date: 'FIRST_DAY' OR 'LAST_DAT', 'LAST_DAT' (by default)
        :param date_offset: integer type, 0 (by default)
        :param duration: integer type, 3 (by default)
        :return: a sequencial stream constructed from the items of the given day
        '''
        dataset = self._get_dataset(reference_date=reference_date, date_offset=date_offset,
                                    duration=duration, update=update)
        return self._construct_sequencial_stream(dataset)

    def _get_dataset(self, reference_date="LAST_DAY", date_offset=0, duration=3, update=True):
        raw_dataset = self.raw_dataset.get_dataset(reference_date=reference_date, date_offset=date_offset,
                                                   duration=duration)
        if update:
            raw_dataset = self._update(raw_dataset)
        else:
            raw_dataset = self._not_update(raw_dataset)
        dataset = self._turn_str2index(raw_dataset, update)
        return self._construct_dataset(dataset)

    @abstractmethod
    def _not_update(self, raw_dataset):
        '''
        Deal with dataset when testing, i.e., update = False
        :param raw_dataset:
        :return:
        '''
        return raw_dataset

    @abstractmethod
    def _update(self, raw_dataset):
        '''
        Update dataset information.
        :param raw_dataset: ndarray or list
        '''
        raise NotImplementedError

    @abstractmethod
    def _turn_str2index(self, raw_dataset, update=True):
        '''
        Turn string type user, words of context, hashtag representation into index representation.

        Note: Implement this function in subclass
        '''

        raise NotImplementedError

    def _construct_dataset(self, dataset):
        '''
        Construct an fule indexable dataset.
        Every data corresponds to the name of self.provide_sources
        :param dataset: A tuple of data
        :return:
        '''
        return IndexableDataset(indexables=OrderedDict(zip(self.provide_souces, dataset)))

    def _construct_shuffled_stream(self, dataset):
        '''
        Construc a shuffled stream from given dataset
        :param dataset: fuel Indexable dataset
        :return: A fuel shuffled stream with basic transformations:
        1.Sort data by self.compare_source
        2.Batch dataset
        3.Add mask on self.need_mask_sources
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
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        return stream

    def _construct_sequencial_stream(self, dataset):
        '''
        Construc a sequencial stream from given dataset
        :param dataset: fuel Indexable dataset
        :return: A fuel sequencial stream with basic transformations:
        1.Sort data by self.compare_source
        2.Batch dataset
        3.Add mask on self.need_mask_sources
        '''
        it = SequentialScheme(dataset.num_examples, self.config.batch_size)
        stream = DataStream(dataset, iteration_scheme=it)
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        return stream


class NUTHD(BUTHD):
    '''
    Negtive sampling UTHD
    '''
    def __init__(self, config, raw_dataset = None):
        super(NUTHD, self).__init__(config, raw_dataset)
        # Dictionary
        self.user2index = {}
        self.hashtag2index = {}
        self.word2index = {}
        # Frequency
        self.word2freq = {}
        self.user2freq = {}
        self.hashtag2freq = {}
        self.last_hashtag2freq = {}
        self.cum_hashtag2freq = {}
        # Integer. Word whose frequency is less than the threshold will be stemmed
        self.sparse_word_threshold = 0
        self.sparse_hashtag_threshold = 0
        self.sparse_user_threshold = 0
        # (1D numpy array, 1D numpy array). Storing hashtag id and hashtag normed number pair
        self.hashtag_dis_table = None
        self.sample_from = []
        self.sample_sources = ['hashtag']
        self.sample_sizes = [config.hashtag_sample_size]
        self.raw_dataset = RUTHD(config)
        self._initialize()

    def _initialize(self):
        '''
        Load dataset information of last day
        Initialize fields: user2index, hashtag2index and so on
        '''
        if os.path.exists(self.config.model_path):
            with open(self.config.model_path, 'rb') as f:
                cPickle.load(f)
                dataset_prarms = cPickle.load(f)
                self.user2index = dataset_prarms['user2index']
                self.word2index = dataset_prarms['word2index']
                self.hashtag2index = dataset_prarms['hashtag2index']
                self.last_hashtag2freq = dataset_prarms['cum_hashtag2freq']

    def get_parameter_to_save(self):
        '''
        Return parameters that need to be saved with model
        :return:
        '''
        return OrderedDict(
            {'hashtag2index': self.hashtag2index, 'word2index': self.word2index, 'user2index': self.user2index,
             'word2freq': self.word2freq, 'hashtag2freq': self.hashtag2freq, 'cum_hashtag2freq': self.cum_hashtag2freq, 'user2freq': self.user2freq})

    def get_shuffled_stream_by_date(self, date, update=False):
        self.raw_dataset.prepare()
        return self.get_shuffled_stream(reference_date="FIRST_DAY",
                                        date_offset=(date-self.raw_dataset.first_date).days,
                                        duration=1,update=update)

    def get_shuffled_stream(self, reference_date="LAST_DAY", date_offset=0, duration=3, update=True):
        '''
        Get shuffled stream of the dataset constructed with samples posted within given date duration
        date duration :
            if reference_date is FIRST_DAY:
                duration = 9first_day + date_offset - duration, first_day + date_offset]
            else:
                duration = 9last_day - date_offset - duration, last_day - date_offset)
        :param reference_date: 'FIRST_DAY' OR 'LAST_DAT', 'FIRST_DAY' (by default)
        :param date_offset: integer type, 0 (by default)
        :param duration: integer type, 3 (by default)
        :return: a shuffled stream constructed from the items of the given day
        '''
        dataset = self._get_dataset(reference_date=reference_date, date_offset=date_offset,
                                    duration=duration, update=update)
        self.sample_from.append(self.hashtag_dis_table)
        return self._construct_shuffled_stream(dataset)

    def get_sequencial_stream(self, reference_date="LAST_DAY", date_offset=0, duration=3, update=True):

        '''
        Get sequencial stream of the dataset constructed with samples posted within given date duration
        date duration :
            if reference_date is FIRST_DAY:
                duration = 9first_day + date_offset - duration, first_day + date_offset]
            else:
                duration = 9last_day - date_offset - duration, last_day - date_offset]
        :param reference_date: 'FIRST_DAY' OR 'LAST_DAT', 'LAST_DAT' (by default)
        :param date_offset: integer type, 0 (by default)
        :param duration: integer type, 3 (by default)
        :return: a sequencial stream constructed from the items of the given day
        '''
        dataset = self._get_dataset(reference_date=reference_date, date_offset=date_offset,
                                    duration=duration, update=update)
        self.sample_from.append(self.hashtag_dis_table)
        return self._construct_sequencial_stream(dataset)

    def _get_dataset(self, reference_date="LAST_DAY", date_offset=0, duration=3, update=True):
        raw_dataset = self.raw_dataset.get_dataset(reference_date=reference_date, date_offset=date_offset,
                                                   duration=duration)
        if update:
            self._update(raw_dataset)
        dataset = self._turn_str2index(raw_dataset, update)
        return self._construct_dataset(dataset)

    def _update(self, raw_dataset):
        '''
        Update dataset information.
        Extend user2index, word2index, hashtag2index
        Statistic user, word and hashtag frequence within the raw_dataset
        :param raw_dataset: ndarray or list
        '''
        fields = zip(*raw_dataset)
        self.word2freq = NUTHD._extract_word2freq(fields[self.config.text_index])

        self.user2freq = NUTHD._extract_user2freq(fields[self.config.user_index])

        self.hashtag2freq = NUTHD._extract_hashtag2freq(fields[self.config.hashtag_index])

        self.cum_hashtag2freq = self._extend_dic(self.last_hashtag2freq, self.hashtag2freq)

        self.sparse_word_threshold = NUTHD._get_sparse_threshold(self.word2freq.values(),
                                                                self.config.sparse_word_percent)
        self.sparse_hashtag_threshold = NUTHD._get_sparse_threshold(self.hashtag2freq.values(),
                                                                   self.config.sparse_hashtag_percent)
        self.sparse_user_threshold = NUTHD._get_sparse_threshold(self.user2freq.values(),
                                                                self.config.sparse_hashtag_percent)

        return raw_dataset

    def _extend_dic(self, original_dic, add_dic):
        key,value = zip(*add_dic.items())
        tmp = numpy.array(value)
        value = (1.0*tmp/tmp.sum()).tolist()
        add_dic = dict(zip(key,value))
        new_dic = original_dic.copy()
        for key, value in add_dic.iteritems():
            new_dic[key] = value
        return new_dic

    def _turn_str2index(self, raw_dataset, update=True):
        '''
        Turn string type user, words of context, hashtag representation into index representation.

        Note: Implement this function in subclass
        '''
        assert raw_dataset is not None or len(raw_dataset) > 0
        fields = zip(*raw_dataset)
        users = numpy.array(
            [self._get_user_index(self._stem_user(user), update) for user in fields[self.config.user_index]],
            dtype=self.config.int_type)
        hashtags = numpy.array([self._get_hashtag_index(self._stem_hashtag(hashtag), update) for hashtag in
                                fields[self.config.hashtag_index]],
                               dtype=self.config.int_type)
        texts = [numpy.array([self._get_word_index(self._stem_word(word), update) for word in text],
                             dtype=self.config.int_type)
                 for text in fields[self.config.text_index]]
        if update:
            self.hashtag_dis_table = self._construct_hashtag_distribution(hashtags)
        return (users, texts, hashtags)

    def _get_user_index(self, user, update=True):
        return self._get_index(user, self.user2index, update)

    def _get_hashtag_index(self, hashtag, update=True):
        return self._get_index(hashtag, self.hashtag2index, update)

    def _get_word_index(self, word, update=True):
        return self._get_index(word, self.word2index, update)

    def _get_index(self, item, _dic, update=True):
        if item not in _dic:
            if update:
                _dic[item] = len(_dic)
                return len(_dic) - 1
            else:
                # return a value randomly
                return _dic.values()[numpy.random.randint(low = 0,high=len(_dic),size=1)[0]]
        else:
            return _dic[item]

    @classmethod
    def _extract_word2freq(cls, texts):
        '''
        Count word frequency
        :param texts:
        :return:
        '''
        assert texts is not None
        word2freq = {}
        for words in texts:
            for word in words:
                if word not in word2freq:
                    word2freq[word] = 1
                else:
                    word2freq[word] += 1
        return word2freq

    @classmethod
    def _extract_user2freq(cls, users):
        return cls._extract_freq(users)

    @classmethod
    def _extract_hashtag2freq(cls, hashtags):
        return cls._extract_freq(hashtags)

    @classmethod
    def _extract_freq(cls, items):
        assert items is not None
        if not isinstance(items, numpy.ndarray):
            item2freq = {}
            for item in items:
                if item not in item2freq:
                    item2freq[item] = 1
                else:
                    item2freq[item] += 1
            return item2freq
        else:
            return dict(zip(*numpy.unique(items, return_counts=True)))

    @classmethod
    def _get_sparse_threshold(cls, freq, percent):
        '''
        Get the frequency threshold. Word with its frequency below the threshold will be treated as sparse word
        '''
        num = numpy.array(freq)
        num.sort()
        total = num.sum()
        cum_num = num.cumsum()
        threshold = int(total * percent)
        min_index = numpy.argmin(numpy.abs(threshold - cum_num))
        if cum_num[min_index] > threshold:
            return num[numpy.max(min_index - 1, 0)]
        else:
            return num[min_index]

    def _construct_dataset(self, dataset):
        return IndexableDataset(indexables=OrderedDict(zip(self.provide_souces, dataset)))

    def _construct_shuffled_stream(self, dataset):
        '''
        Construc a shuffled stream from given dataset
        :param dataset:
        :return:
        '''
        stream = super(NUTHD, self)._construct_sequencial_stream(dataset)
        self.sample_from.append(self.hashtag_dis_table)
        stream = NegativeSample(stream,
                                dist_tables=self.sample_from,
                                sample_sources=self.sample_sources,
                                sample_sizes=self.sample_sizes)
        return stream

    def _construct_sequencial_stream(self, dataset):
        '''Construct sequencial stream.
        This is usually used for testing and prediction.
        '''
        stream = super(NUTHD, self)._construct_sequencial_stream(dataset)
        stream = NegativeSample(stream,
                                dist_tables=self.sample_from,
                                sample_sources=self.sample_sources,
                                sample_sizes=self.sample_sizes)
        return stream

    def _construct_hashtag_distribution(self, *args):
        '''
        Build hashtag distribution table
        '''
        id = []
        count = []
        _dic = {}
        for hashtag, freq in self.cum_hashtag2freq.iteritems():
            index = self._get_hashtag_index(self._stem_hashtag(hashtag), update=False)
            if index in _dic:
                _dic[index] += freq
            else:
                _dic[index] = freq
        id, count = zip(*_dic.items())
        id = numpy.array(id)
        count = numpy.array(count)
        count = count ** (3.0 / 4)
        return (id, count)

    def _stem_word(self, word, update = True):
        '''
        Do word stemming
        :param word: original string type of word
        :return: stemmed word
        '''
        assert word is not None
        if word in self.word2index:
            return word
        elif word in self.word2freq:
            if self.word2freq[word] > self.sparse_word_threshold:
                return word
            else:
                return '<unk>'
        else:
            return '<unk>'

    def _stem_hashtag(self, hashtag, update = True):
        assert hashtag is not None
        if hashtag in self.hashtag2index:
            return hashtag
        elif hashtag in self.hashtag2freq:
            if self.hashtag2freq[hashtag] > self.sparse_hashtag_threshold:
                return hashtag
            else:
                return '<unk>'
        else:
            return '<unk>'

    def _stem_user(self, user, update = True):
        assert user is not None
        if user in self.user2index:
            return user
        elif user in self.user2freq:
            if self.user2freq[user] > self.sparse_user_threshold:
                return user
            else:
                return '<unk>'
        else:
            return '<unk>'


class FUTHD(BUTHD):
    '''
    Full UTHD
    '''
    def __init__(self, config, raw_dataset = None):
        super(FUTHD, self).__init__(config, raw_dataset)
        # Dictionary
        self.user2index = {'<unk>':0}   # Deal with OV when testing
        self.hashtag2index = {}
        self.word2index = {}
        self.char2index = {'<unk>':0}
        # Frequency
        self.word2freq = {}
        self.hashtag2day = {}
        self.hashtag_coverage = 1.
        # Integer. Word whose frequency is less than the threshold will be stemmed
        self.sparse_word_threshold = 0
        # (1D numpy array, 1D numpy array). Storing hashtag id and hashtag normed number pair
        self.provide_souces = ('user', 'text', 'user_word', 'user_word_idx',
                               'hashtag_word', 'hashtag_word_idx', 'sparse_word','sparse_word_idx', 'hashtag')
        self.sparse_pairs = [('user_word','user_word_idx'),
                             ('hashtag_word', 'hashtag_word_idx')]
        self.char_sources = ('sparse_word',)
        self.char_idx_sources = ('sparse_word_idx',)

    def _initialize(self):
        if os.path.exists(self.config.model_path):
            with open(self.config.model_path, 'rb') as f:
                cPickle.load(f)
                dataset_prarms = cPickle.load(f)
                self.user2index = dataset_prarms['user2index']
                self.word2index = dataset_prarms['word2index']
                self.hashtag2index = dataset_prarms['hashtag2index']
                self.hashtag2ay = dataset_prarms['hashtag2day']
                self.char2index = dataset_prarms['char2index']
        with open(self.config.user_name2id_path, 'rb') as f:
            self.user_name2id = cPickle.load(f)

    def get_parameter_to_save(self):
        '''
        Return parameters that need to be saved with model
        :return: OrderedDict
        '''
        return OrderedDict(
            {'hashtag2index': self.hashtag2index, 'word2index': self.word2index, 'user2index': self.user2index,
             'char2index':self.char2index, 'hashtag2day': self.hashtag2day, })

    def _not_update(self, raw_dataset):
        '''
        Delete items with hashtag out of vocabulary
        :param raw_dataset:
        :return:
        '''
        tmp = []
        for sample in raw_dataset:
            if sample[self.config.hashtag_index] in self.hashtag2index:
                tmp.append(sample)
            else:
                pass
        self.hashtag_coverage = 1.0*len(tmp)/len(raw_dataset)
        return numpy.asarray(tmp)

    def _update(self, raw_dataset):
        '''
        Update dataset information.
        Extend user2index, word2index, hashtag2index
        Statistic user, word and hashtag frequence within the raw_dataset
        :param raw_dataset: ndarray or list
        '''
        fields = zip(*raw_dataset)
        self.word2freq = FUTHD._extract_word2freq(fields[self.config.text_index])

        self.sparse_word_threshold = NUTHD._get_sparse_threshold(self.word2freq.values(),
                                                                self.config.sparse_word_percent)

        new_hashtags = set(fields[self.config.hashtag_index])
        for hashtag in new_hashtags:
            self.hashtag2day[hashtag] = 0
        for hashtag, expire_days in self.hashtag2day.items():
            expire_days += 1
            if expire_days > self.config.time_window:
                self.hashtag2index.pop(hashtag,None)
                self.hashtag2day.pop(hashtag, None)
            else:
                self.hashtag2day[hashtag] = expire_days
        self.hashtag_coverage = 1.
        return raw_dataset

    @classmethod
    def _extract_word2freq(cls, texts):
        '''
        Count word frequency
        :param texts:
        :return:
        '''
        assert texts is not None
        word2freq = {}
        for words in texts:
            for word in words:
                if (not word.startswith('#')) and (not word.startswith('@')):
                    if word not in word2freq:
                        word2freq[word] = 1
                    else:
                        word2freq[word] += 1
                else:
                    continue
        if 'Coooooooome' in word2freq:
            print('yes!')
        return word2freq

    def _turn_str2index(self, raw_dataset, update=True):
        assert raw_dataset is not None or len(raw_dataset) > 0
        fields = zip(*raw_dataset)
        users = numpy.array(
            [self._get_user_index(user, update) for user in fields[self.config.user_index]],
            dtype=self.config.int_type)
        hashtags = numpy.array([self._get_hashtag_index(hashtag, update) for hashtag in
                                fields[self.config.hashtag_index]],
                               dtype=self.config.int_type)

        text_idxes = self._turn_word2index(fields[self.config.text_index], update)
        return (users,)+text_idxes+(hashtags,)

    def _turn_word2index(self, texts, update = True):
        user_words = []
        user_word_idxes = []
        hashtag_words = []
        hashtag_word_idxes = []
        sparse_words = []
        sparse_word_idxes = []
        text_idxes = []
        for words in texts:
            users = []
            user_idx=[]
            hashtags = []
            hashtag_idx = []
            sparse_word = []
            sparse_word_idx = []
            text = []
            for i in range(len(words)):
                if words[i].startswith('@'):
                    user_name = words[i][1:]
                    if user_name in self.user_name2id:
                        users.append(self._get_user_index(self.user_name2id[user_name],update=update))
                        text.append(0)
                        user_idx.append(i)
                    else:
                        sparse_word.append(numpy.array([self._get_char_index(c) for c in words[i]],
                                                   dtype = self.config.int_type))
                        sparse_word_idx.append(i)
                        text.append(0)
                elif words[i].startswith('#'):
                    if len(words[i]) > 1:
                        hashtag = words[i][1:]
                        if hashtag in self.hashtag2index:
                            hashtags.append(self.hashtag2index[hashtag])
                            text.append(0)
                            hashtag_idx.append(i)
                        else:
                            sparse_word.append(numpy.array([self._get_char_index(c) for c in words[i]],
                                                       dtype = self.config.int_type))
                            sparse_word_idx.append(i)
                            text.append(0)
                    else:
                        text.append(self._get_word_index(words[i], update))
                elif self._is_sparse_word(words[i]):
                    sparse_word.append(numpy.array([self._get_char_index(c) for c in words[i]],
                                                   dtype = self.config.int_type))
                    sparse_word_idx.append(i)
                    text.append(0)
                else:
                    text.append(self._get_word_index(words[i], update))
            user_words.append(numpy.array(users, dtype=self.config.int_type))
            hashtag_words.append(numpy.array(hashtags, dtype=self.config.int_type))
            text_idxes.append(numpy.array(text, dtype=self.config.int_type))
            user_word_idxes.append(numpy.array(user_idx, dtype=self.config.int_type))
            hashtag_word_idxes.append(numpy.array(hashtag_idx, dtype=self.config.int_type))
            sparse_word_idxes.append(numpy.array(sparse_word_idx, dtype=self.config.int_type))
            sparse_words.append(sparse_word)
        return (text_idxes, user_words, user_word_idxes, hashtag_words, hashtag_word_idxes, sparse_words, sparse_word_idxes)

    def _is_sparse_word(self, word):
        if word in self.word2freq and self.word2freq[word] > self.sparse_word_threshold:
            return False
        else:
            return True

    def _get_char_index(self, char, update = True):
        return self._get_index(char, self.char2index, update=update)

    def _get_user_index(self, user, update=True):
        return self._get_index(user, self.user2index, update)

    def _get_hashtag_index(self, hashtag, update=True):
        if not update and hashtag not in self.hashtag2index:
            raise ValueError
        return self._get_index(hashtag, self.hashtag2index, update)

    def _get_word_index(self, word, update=True):
        return self._get_index(word, self.word2index, update)

    def _get_index(self, item, _dic, update=True):
        if item not in _dic:
            if update:
                _dic[item] = len(_dic)
                return len(_dic) - 1
            else:
                return _dic['<unk>']
        else:
            return _dic[item]

    def _construct_shuffled_stream(self, dataset):
        '''
        Construc a shuffled stream from given dataset
        :param dataset:
        :return:
        '''
        stream = super(FUTHD, self)._construct_shuffled_stream(dataset)
        stream = SparseIndex(stream, self.sparse_pairs)
        stream = CharEmbedding(stream, char_source=self.char_sources, char_idx_source= self.char_idx_sources)
        return stream

    def _construct_sequencial_stream(self, dataset):
        '''Construct sequencial stream.
        This is usually used for testing and prediction.
        '''
        stream = super(FUTHD, self)._construct_sequencial_stream(dataset)
        stream = SparseIndex(stream,self.sparse_pairs)
        stream = CharEmbedding(stream, char_source=self.char_sources, char_idx_source= self.char_idx_sources)
        return stream


class SUTHD(object):
    '''
    Sequential user-text-hahstag dataset.
    '''
    def __init__(self, config, dataset):
        '''
        Sequentialize given dataset with given config
        :param config: class in config module
        :param dataset: subclass of BUTHD
        '''
        self.config = config
        self.date_iterator = None
        self.begin_date = config.begin_date
        self.dataset = dataset

    def __iter__(self, ):
        self.dataset.raw_dataset.prepare()
        return self

    def next(self):
        if self.date_iterator is None:
            if self.begin_date is not None:
                self.begin_offset = (self.begin_date -self.dataset.raw_dataset.first_date).days
            else:
                self.begin_offset = self.config.duration
            if self.config.mode == 'debug':
                self.date_iterator = iter(range(self.begin_offset, min(self.config.duration+10,self.dataset.raw_dataset.date_span)))
            else:
                self.date_iterator = iter(range(self.begin_offset, self.dataset.raw_dataset.date_span))
        try:
            date_offset = self.date_iterator.next()
            return self.get_stream(date_offset)
        except StopIteration as e:
            self.date_iterator = None
            raise e

    def get_stream(self, date_offset):
        train_stream = self.dataset.get_shuffled_stream(reference_date= self.dataset.raw_dataset.FIRST_DAY,
                                                        date_offset = date_offset-1,
                                                        duration = self.config.duration,
                                                        update= True)
        valid_stream = self.dataset.get_shuffled_stream(reference_date= self.dataset.raw_dataset.FIRST_DAY,
                                                        date_offset = date_offset,
                                                        duration = 1,
                                                        update = False)

        return train_stream, valid_stream, self.dataset.raw_dataset.first_date+datetime.timedelta(days = date_offset)


class TUTHD(NUTHD):
    def __init__(self, config):
        super(TUTHD, self).__init__(config)

    def _construct_sequencial_stream(self, dataset):
        '''Construct sequencial stream.
        This is usually used for testing and prediction.
        '''
        it = SequentialScheme(dataset.num_examples, self.config.batch_size)
        stream = DataStream(dataset, iteration_scheme=it)
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        return stream

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
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        return stream