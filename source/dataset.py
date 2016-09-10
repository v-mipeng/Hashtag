# -*- coding : utf-8 -*-
# region Import
import os
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


# endregion

# region Develop
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
        raw_user_graph = self._load_user_graph(user_graph_path)
        id2index_path = path.join(self.user_id2index_dir, str(date))
        if path.exists(id2index_path):
            id2index = self._load_id2index(id2index_path)
        else:
            id2index = self._extract_id2index(raw_user_graph)
        return self._turn_id2index(raw_user_graph, id2index), id2index

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
        with open(user_graph_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                count += 1
                try:
                    array = line.split('\t')
                    user = array[0]
                    followees = array[1:]
                    raw_user_graph[user] = followees
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
        self.provide_souces = ('user', 'followees')

    def get_train_stream(self, time=None):
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
        rand_indexes = range(0, len(id2index))
        numpy.random.shuffle(rand_indexes)
        for user_index, followee_indexes in user_graph.iteritems():
            rand_begins = numpy.random.randint(0, len(id2index), size=len(followee_indexes))
            for followee_index, rand_begin in zip(followee_indexes, rand_begins):
                # sample for each user-followee pair
                users.append(user_index)
                samples = []
                samples.append(followee_index)
                i = 0
                while len(samples) < self.user_sample_size + 1:
                    if (rand_begin + i % len(id2index)) not in followee_indexes:
                        samples.append(rand_begin + i % len(id2index))
                    i += 1
                followees.append(samples)
        # Construct block dataset
        return IndexableDataset(indexables={self.provide_souces[0]: users, self.provide_souces[1]: followees})

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

    def __init__(self, config):
        self.user_hashtag_graph_dir = config.user_hashtag_graph_dir
        self.user_id2index_dir = config.user_id2index_dir
        self.user_hashtag_time_span = config.user_hashtag_time_span
        self.hashtag2index_dir = config.hashtag2index_dir
        self.mode = config.mode

    def get_user_hashtag_graph(self, date, span):  # do statistic work
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
            if path.exists(path.join(self.hashtag2index_dir, str(date - timedelta(days=1)))):
                # Load hashtag2index information of last day for given date, and update it to the give date
                hashtag2index = self._load_hashtag2index(
                    path.join(self.hashtag2index_dir, str(date - timedelta(days=1))))
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
        self.provide_souces = ('user', 'posts')

    def get_train_stream(self, time=None):
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
        rand_indexes = range(0, len(id2index))
        numpy.random.shuffle(rand_indexes)
        for user_index, followee_indexes in user_graph.iteritems():
            rand_begins = numpy.random.randint(0, len(id2index), size=len(followee_indexes))
            for followee_index, rand_begin in zip(followee_indexes, rand_begins):
                # sample for each user-followee pair
                users.append(user_index)
                samples = []
                samples.append(followee_index)
                i = 0
                while len(samples) < self.user_hashtag_sample_size + 1:
                    if (rand_begin + i % len(id2index)) not in followee_indexes:
                        samples.append(rand_begin + i % len(id2index))
                    i += 1
                followees.append(samples)
        # Construct block dataset
        return IndexableDataset(indexables={self.provide_souces[0]: users, self.provide_souces[1]: followees})

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
            hashtag_list = hashtag_list + hashtags


# endregion


class RUTHD(object):
    '''
       Basic dataset with user-text-time-hashtag information.

       load dataset --> parse string type date --> provide samples of given date --> map user, hashtag, word to id -->
       --> construct indexable dataset --> construct shuffled or sequencial fuel stream
       '''

    def __init__(self, config, raw_dataset=None):
        # Dictionary
        self.config = config
        self.raw_dataset = None
        self.dates = None
        self.first_date = None
        self.last_date = None
        self.date_span = None
        self.LAST_DAY = "LAST_DAY"
        self.FIRST_DAY = "FIRST_DAY"
        if raw_dataset is not None:
            self.prepare(raw_dataset=raw_dataset)

    def prepare(self, raw_dataset=None, data_path=None):
        '''
        Prepare dataset
        :param data_path:
        :return:
        '''
        if data_path is None and self.raw_dataset is None:
            data_path = self.config.train_path
        print("Preparing dataset...")
        # Load pickled dataset
        # TODO: if raw_dataset is not None
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
        if data_path is not None:
            self.prepare(data_path)
        elif self.dates is None or len(self.dates) == 0:
            self.prepare()
        else:
            pass
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


class BUTHD(object):
    def __init__(self, config):
        self.config = config
        self.provide_souces = ('user', 'text', 'hashtag')
        self.need_mask_sources = {'text': theano.config.floatX}
        self.compare_source = 'text'
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

    def get_train_stream(self, raw_dataset, it='shuffled'):
        return self._get_stream(raw_dataset, it, for_type='train')

    def get_test_stream(self, raw_dataset, it='shuffled'):
        return self._get_stream(raw_dataset, it, for_type='test')

    def _get_stream(self, raw_dataset, it='shuffled', for_type='train'):
        raw_dataset = self._update_before_transform(raw_dataset, for_type)
        dataset = self._map(raw_dataset, for_type)
        dataset = self._update_after_transform(dataset, for_type)
        dataset = self._construct_dataset(dataset)
        if it == 'shuffled':
            return self._construct_shuffled_stream(dataset)
        elif it == 'sequencial':
            return self._construct_sequencial_stream(dataset)
        else:
            raise ValueError('it should be "shuffled" or "sequencial"!')

    @abstractmethod
    def _update_before_transform(self, raw_dataset, for_type='train'):
        '''
        Do updation beform transform raw_dataset into index representation dataset
        :param raw_dataset:
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :return: a new raw_dataset
        '''
        return raw_dataset

    @abstractmethod
    def _update_after_transform(self, dataset, for_type='train'):
        '''
        Do updation after transform raw_dataset into index representation dataset
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :param dataset: tranformed dataset
        :return: a new transformed dataset
        '''
        return dataset

    @abstractmethod
    def _map(self, raw_dataset, for_type='train'):
        '''
        Turn raw_dataset into index representation dataset.

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


class UTHD(BUTHD):
    '''
    UTHD with only user, word and hashtag embeddings
    '''

    def __init__(self, config):
        super(UTHD, self).__init__(config)
        self.hashtag_coverage = 1.
        # Integer. Word whose frequency is less than the threshold will be stemmed
        # (1D numpy array, 1D numpy array). Storing hashtag id and hashtag normed number pair
        self.provide_souces = ('user', 'text', 'hashtag')

    @abstractmethod
    def _initialize(self):
        if os.path.exists(self.config.model_path):
            with open(self.config.model_path, 'rb') as f:
                cPickle.load(f)
                dataset_prarms = cPickle.load(f)
                self.user2index = dataset_prarms['user2index']
                self.word2index = dataset_prarms['word2index']
                self.hashtag2index = dataset_prarms['hashtag2index']
                self.user2freq = dataset_prarms['user2freq']
                self.word2freq = dataset_prarms['word2freq']
                self.hashtag2freq = dataset_prarms['hashtag2freq']
                self.sparse_word_threshold = dataset_prarms['sparse_word_threshold']
                self.sparse_user_threshold = dataset_prarms['sparse_user_threshold']
                self.sparse_hashtag_threshold = dataset_prarms['sparse_hashtag_threshold']
                return dataset_prarms
        else:
            # Dictionary
            self.user2index = {'<unk>': 0}  # Deal with OV when testing
            self.hashtag2index = {'<unk>': 0}
            self.word2index = {'<unk>': 0}
            self.word2freq = {}
            self.user2freq = {}
            self.hashtag2freq = {}
            self.sparse_word_threshold = 0
            self.sparse_user_threshold = 0
            self.sparse_hashtag_threshold = 0
            return None

    @abstractmethod
    def get_parameter_to_save(self):
        '''
        Return parameters that need to be saved with model
        :return: OrderedDict
        '''
        return OrderedDict(
            {'hashtag2index': self.hashtag2index, 'word2index': self.word2index, 'user2index': self.user2index,
             'user2freq': self.user2freq, 'word2freq': self.word2freq, 'hashtag2freq': self.hashtag2freq,
             'sparse_word_threshold': self.sparse_word_threshold, 'sparse_user_threshold': self.sparse_user_threshold,
             'sparse_hashtag_threshold': self.sparse_hashtag_threshold})

    @abstractmethod
    def _update_before_transform(self, raw_dataset, for_type='train'):
        '''
        Do updation beform transform raw_dataset into index representation dataset
        :param raw_dataset:
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :return: a new raw_dataset
        '''
        if for_type == 'train':
            fields = zip(*raw_dataset)
            self.word2freq = self._extract_word2freq(fields[self.config.text_index])
            self.user2freq = self._extract_user2freq(fields[self.config.user_index])
            self.hashtag2freq = self._extract_hashtag2freq(fields[self.config.hashtag_index])
            self.sparse_word_threshold = get_sparse_threshold(self.word2freq.values(), self.config.sparse_word_percent)
            self.sparse_user_threshold = get_sparse_threshold(self.user2freq.values(), self.config.sparse_user_percent)
            self.sparse_hashtag_threshold = get_sparse_threshold(self.hashtag2freq.values(),
                                                                 self.config.sparse_hashtag_percent)
            return raw_dataset
            # Implement more updation
        elif for_type == 'test':
            tmp = []
            for sample in raw_dataset:
                if sample[self.config.hashtag_index] in self.hashtag2index:
                    tmp.append(sample)
                else:
                    pass
            self.hashtag_coverage = 1.0 * len(tmp) / len(raw_dataset)
            return numpy.asarray(tmp)
        else:
            raise ValueError('for_type should be either "train" or "test"')

    @abstractmethod
    def _update_after_transform(self, dataset, for_type='train'):
        '''
        Do updation after transform raw_dataset into index representation dataset
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :param dataset: tranformed dataset
        :return: a new transformed dataset
        '''
        return dataset

    def _map(self, raw_dataset, for_type='train'):
        '''
        Turn string type user, words of context, hashtag representation into index representation.

        Note: Implement this function in subclass
        '''
        assert raw_dataset is not None or len(raw_dataset) > 0
        fields = zip(*raw_dataset)
        users = numpy.array(
            [self._get_user_index(user, for_type) for user in fields[self.config.user_index]],
            dtype=self.config.int_type)
        hashtags = numpy.array([self._get_hashtag_index(hashtag, for_type) for hashtag in
                                fields[self.config.hashtag_index]],
                               dtype=self.config.int_type)

        texts = [numpy.array([self._get_word_index(word, for_type) for word in text],
                             dtype=self.config.int_type)
                 for text in fields[self.config.text_index]]
        return (users, texts, hashtags)

    def _extract_word2freq(self, texts):
        '''
        Count word frequency
        :param texts:
        :return:
        '''
        id, count = numpy.unique(numpy.concatenate(numpy.array(texts)), return_counts=True)
        return dict(zip(id, count))

    def _extract_user2freq(self, users):
        assert users is not None
        id, count = numpy.unique(numpy.array(users), return_counts=True)
        return dict(zip(id, count))

    def _extract_hashtag2freq(self, hashtag):
        assert hashtag is not None
        id, count = numpy.unique(numpy.array(hashtag), return_counts=True)
        return dict(zip(id, count))

    def _is_sparse_hashtag(self, hashtag):
        if hashtag in self.hashtag2freq and self.hashtag2freq[hashtag] > self.sparse_hashtag_threshold:
            return False
        else:
            return True

    def _is_sparse_word(self, word):
        if word in self.word2freq and self.word2freq[word] > self.sparse_word_threshold:
            return False
        else:
            return True

    def _is_sparse_user(self, user):
        if user in self.user2freq and self.user2freq[user] > self.sparse_user_threshold:
            return False
        else:
            return True

    def _get_hashtag_index(self, hashtag, for_type='train'):
        if for_type == 'test':
            if hashtag not in self.hashtag2index:
                raise ValueError('The subclass of EUTHD should remove samples'
                                 ' for testing whose hashtag is not in training dataset!')
            else:
                return self.hashtag2index[hashtag]
        elif self._is_sparse_hashtag(hashtag):
            return self.hashtag2index['<unk>']
        else:
            return self._get_index(hashtag, self.hashtag2index, for_type)

    def _get_user_index(self, user, for_type='train'):
        if self._is_sparse_user(user):
            return self.user2index['<unk>']
        else:
            return self._get_index(user, self.user2index, for_type)

    def _get_word_index(self, word, for_type='train'):
        if self._is_sparse_word(word):
            return self.word2index['<unk>']
        else:
            return self._get_index(word, self.word2index, for_type)

    def _get_index(self, item, _dic, for_type='train'):
        if item not in _dic:
            _dic[item] = len(_dic)
            return len(_dic) - 1
        else:
            return _dic[item]


class EUTHD(UTHD):
    '''
    To Deal with OV:
    1. when OV input occurs, force it to be zeros, i.e., make it not sense, and get output with left information
    2. train a OV input, i.e., select some inputs of training samples and treat it as OV input, train this representation.
    '''

    def __init__(self, config):
        super(EUTHD, self).__init__(config)
        # (1D numpy array, 1D numpy array). Storing hashtag id and hashtag normed number pair
        self.provide_souces = ('user', 'text', 'user_word', 'user_word_idx',
                               'hashtag_word', 'hashtag_word_idx', 'sparse_word', 'sparse_word_idx', 'hashtag')
        self.sparse_pairs = [('user_word', 'user_word_idx'),
                             ('hashtag_word', 'hashtag_word_idx')]
        self.char_sources = ('sparse_word',)
        self.char_idx_sources = ('sparse_word_idx',)

    @abstractmethod
    def _initialize(self):
        with open(self.config.user_name2id_path, 'rb') as f:
            self.user_name2id = cPickle.load(f)
        dataset_prarms = super(EUTHD, self)._initialize()
        if os.path.exists(self.config.model_path):
            self.char2index = dataset_prarms['char2index']
            return dataset_prarms
        else:
            self.word2index = {}
            self.char2index = {'<unk>': 0}
            return None

    @abstractmethod
    def get_parameter_to_save(self):
        '''
        Return parameters that need to be saved with model
        :return: OrderedDict
        '''
        dic = super(EUTHD, self).get_parameter_to_save()
        dic['char2index'] = self.char2index
        return dic

    @abstractmethod
    def _update_before_transform(self, raw_dataset, for_type='train'):
        return super(EUTHD, self)._update_before_transform(raw_dataset, for_type)

    @abstractmethod
    def _update_after_transform(self, dataset, for_type='train'):
        '''
        Do updation after transform raw_dataset into index representation dataset
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :param dataset: tranformed dataset
        :return: a new transformed dataset
        '''
        return dataset

    def _extract_word2freq(self, texts):
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
        return word2freq

    def _map(self, raw_dataset, for_type='train'):
        assert raw_dataset is not None or len(raw_dataset) > 0
        fields = zip(*raw_dataset)
        users = numpy.array(
            [self._get_user_index(user, for_type) for user in fields[self.config.user_index]],
            dtype=self.config.int_type)
        hashtags = numpy.array([self._get_hashtag_index(hashtag, for_type) for hashtag in
                                fields[self.config.hashtag_index]],
                               dtype=self.config.int_type)

        text_idxes = self._turn_word2index(fields[self.config.text_index], for_type)
        return (users,) + text_idxes + (hashtags,)

    def _turn_word2index(self, texts, for_type='train'):
        user_words = []
        user_word_idxes = []
        hashtag_words = []
        hashtag_word_idxes = []
        sparse_words = []
        sparse_word_idxes = []
        text_idxes = []
        for words in texts:
            users = []
            user_idx = []
            hashtags = []
            hashtag_idx = []
            sparse_word = []
            sparse_word_idx = []
            text = []
            for i in range(len(words)):
                if words[i].startswith('@'):
                    user_name = words[i][1:]
                    if user_name in self.user_name2id:
                        users.append(self._get_user_index(self.user_name2id[user_name], for_type))
                        text.append(0)
                        user_idx.append(i)
                    else:
                        sparse_word.append(numpy.array([self._get_char_index(c, for_type) for c in words[i]],
                                                       dtype=self.config.int_type))
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
                                                           dtype=self.config.int_type))
                            sparse_word_idx.append(i)
                            text.append(0)
                    else:
                        text.append(self._get_word_index(words[i], for_type))
                elif self._is_sparse_word(words[i]):
                    sparse_word.append(numpy.array([self._get_char_index(c, for_type) for c in words[i]],
                                                   dtype=self.config.int_type))
                    sparse_word_idx.append(i)
                    text.append(0)
                else:
                    text.append(self._get_word_index(words[i], for_type))
            user_words.append(numpy.array(users, dtype=self.config.int_type))
            hashtag_words.append(numpy.array(hashtags, dtype=self.config.int_type))
            text_idxes.append(numpy.array(text, dtype=self.config.int_type))
            user_word_idxes.append(numpy.array(user_idx, dtype=self.config.int_type))
            hashtag_word_idxes.append(numpy.array(hashtag_idx, dtype=self.config.int_type))
            sparse_word_idxes.append(numpy.array(sparse_word_idx, dtype=self.config.int_type))
            sparse_words.append(sparse_word)
        return (
        text_idxes, user_words, user_word_idxes, hashtag_words, hashtag_word_idxes, sparse_words, sparse_word_idxes)

    def _get_word_index(self, word, for_type='train'):
        if for_type != 'train' and word not in self.word2index:
            raise ValueError('Sparse word should not occure in _get_word_index!')
        else:
            return self._get_index(word, self.word2index, for_type)

    def _get_char_index(self, char, for_type='train'):
        if for_type == 'train':
            return self._get_index(char, self.char2index, for_type)
        elif char in self.char2index:
            return self.char2index[char]
        else:
            return self.char2index['<unk>']

    def _construct_shuffled_stream(self, dataset):
        '''
        Construc a shuffled stream from given dataset
        :param dataset:
        :return:
        '''
        stream = super(EUTHD, self)._construct_shuffled_stream(dataset)
        stream = SparseIndex(stream, self.sparse_pairs)
        stream = CharEmbedding(stream, char_source=self.char_sources, char_idx_source=self.char_idx_sources)
        return stream

    def _construct_sequencial_stream(self, dataset):
        '''Construct sequencial stream.
        This is usually used for testing and prediction.
        '''
        stream = super(EUTHD, self)._construct_sequencial_stream(dataset)
        stream = SparseIndex(stream, self.sparse_pairs)
        stream = CharEmbedding(stream, char_source=self.char_sources, char_idx_source=self.char_idx_sources)
        return stream


class TimeLineEUTHD(EUTHD):
    '''
    Dataset for training day by day
    '''

    def __init__(self, config):
        super(TimeLineEUTHD, self).__init__(config)

    def _initialize(self):
        dataset_prarms = super(TimeLineEUTHD, self)._initialize()
        if os.path.exists(self.config.model_path):
            self.hashtag2date = dataset_prarms['hashtag2date']
            return dataset_prarms
        else:
            self.hashtag2date = {}
            return None

    def get_parameter_to_save(self):
        '''
        Return parameters that need to be saved with model
        :return: OrderedDict
        '''
        dic = super(TimeLineEUTHD, self).get_parameter_to_save()
        dic['hashtag2date'] = self.hashtag2date
        return dic

    def _update_before_transform(self, raw_dataset, for_type='train'):
        '''

        :param raw_dataset: The dataset of current day
        :param for_type:
        :return:
        '''
        raw_dataset = super(TimeLineEUTHD, self)._update_before_transform(raw_dataset, for_type)
        if for_type == 'train':
            fields = zip(*raw_dataset)
            dates = fields[self.config.date_index]
            assert (numpy.array(dates) != dates[0]).sum() == 0
            current_date = fields[self.config.date_index][0]
            expire_date = current_date - datetime.timedelta(days=self.config.time_window)
            hashtags = fields[self.config.hashtag_index]
            for hashtag in hashtags:
                self.hashtag2date[hashtag] = current_date
            for hashtag, date in self.hashtag2date.items():
                if date <= expire_date:
                    self.hashtag2index.pop(hashtag, None)
                    self.hashtag2date.pop(hashtag, None)
                else:
                    pass
            self.hashtag_coverage = 1.
        else:
            pass
        return raw_dataset


class NegTimeLineEUTHD(TimeLineEUTHD):
    def __init__(self, config):
        super(NegTimeLineEUTHD, self).__init__(config)

    def _initialize(self):
        dataset_prarms = super(NegTimeLineEUTHD, self)._initialize()
        if os.path.exists(self.config.model_path):
            self.date2hashtag_freq = dataset_prarms['date2hashtag_freq']
            self.hashtag_distribution = self._get_hashtag_distribution()
            return dataset_prarms
        else:
            self.date2hashtag_freq = {}
            return None

    def get_parameter_to_save(self):
        dic = super(NegTimeLineEUTHD, self).get_parameter_to_save()
        dic['date2hashtag_freq'] = self.date2hashtag_freq
        return dic

    def _update_before_transform(self, raw_dataset, for_type='train'):
        raw_dataset = super(NegTimeLineEUTHD, self)._update_before_transform(raw_dataset, for_type)
        if for_type == 'train':
            fields = zip(*raw_dataset)
            current_date = fields[self.config.date_index][0]
            expire_date = current_date - datetime.timedelta(days=self.config.time_window)
            for date, _dic in self.date2hashtag_freq.items():
                if date <= expire_date:
                    self.date2hashtag_freq.pop(date)
                else:
                    pass
            self.date2hashtag_freq[current_date] = self.hashtag2freq
        else:
            pass
        return raw_dataset

    def _update_after_transform(self, dataset, for_type='train'):
        if for_type == 'train':
            self.hashtag_distribution = self._get_hashtag_distribution()
        else:
            pass
        return dataset

    def _get_hashtag_distribution(self):
        _dic = {}
        for date, hashtag2freq in self.date2hashtag_freq.iteritems():
            for hashtag, freq in hashtag2freq.iteritems():
                id = self.hashtag2index.get(hashtag)
                if id is not None:
                    if id in _dic:
                        _dic[id] += freq
                    else:
                        _dic[id] = freq
        id, count = zip(*_dic.items())
        count = numpy.array(count) ** (3.0 / 4)
        return (numpy.array(id, dtype='int32'), count)

    def _construct_shuffled_stream(self, dataset):
        '''
        Construc a shuffled stream from given dataset
        :param dataset: fuel Indexable dataset
        :return: A fuel shuffled stream with basic transformations:
        1.Sort data by self.compare_source
        2.Batch dataset
        3.Add mask on self.need_mask_sources
        '''
        stream = super(NegTimeLineEUTHD, self)._construct_shuffled_stream(dataset)
        sample_from = [self.hashtag_distribution]
        sample_sources = ['hashtag']
        sample_sizes = [self.config.hashtag_sample_size]
        stream = NegativeSample(stream, sample_from, sample_sources, sample_sizes)
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
        stream = super(NegTimeLineEUTHD, self)._construct_shuffled_stream(dataset)
        sample_from = [self.hashtag_distribution]
        sample_sources = ['hashtag']
        sample_sizes = [self.config.hashtag_sample_size]
        stream = NegativeSample(stream, sample_from, sample_sources, sample_sizes)
        return stream


class FDUTHD(object):
    def __init__(self, config):
        self.config = config
        self.index2hashtag = {}
        self.hashtag2index = {}
        self.user2dic = {} # user-->{hashtag index-->freq}
        self.hashtag_index2freq = None # ndarray
        self.alpha = 0.1

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_train_data(self, raw_dataset):
        fields = zip(*raw_dataset)
        users = fields[self.config.user_index]
        hashtags = fields[self.config.hashtag_index]
        self._get_hashtag2index(hashtags)
        self._get_hashtag_index2freq(hashtags)
        self._get_user_dic(users,hashtags)

    def test(self, raw_dataset):
        fields = zip(*raw_dataset)
        users = fields[self.config.user_index]
        hashtags = fields[self.config.hashtag_index]
        top1_accuracy = 0
        top10_accuracy = 0
        for user, hashtag in zip(users,hashtags):
            top1_pred, top10_pred = self.single_test(user, hashtag)
            top1_accuracy += top1_pred
            top10_accuracy += top10_pred
        top1_accuracy = 1.0*top1_accuracy/len(users)
        top10_accuracy = 1.0*top10_accuracy/len(users)
        return top1_accuracy,top10_accuracy

    def single_test(self, user, hashtag):
        if hashtag not in self.hashtag2index:
            return 0,0
        else:
            top1 = 0
            top10 = 0
            order_ids = self.single_predict(user)
            real_id = self.hashtag2index[hashtag]
            if order_ids[0] == real_id:
                top1 = 1
            else:
                pass
            if real_id in order_ids[0:10]:
                top10 = 1
            else:
                pass
            return top1, top10

    def single_predict(self, user):
        base = numpy.ones(len(self.hashtag2index))
        # base = numpy.zeros(len(self.hashtag2index))
        hashtag2freq = self.user2dic.get(user)
        if hashtag2freq is not None:
            index, freq = zip(*hashtag2freq.items())
            base[list(index)] += numpy.array(freq)
        else:
            pass
        f_u_h = base
        value = self._get_value(f_u_h)
        order_id = numpy.argsort(value)[::-1]
        return order_id

    def _get_value(self, f_u_h):
        # value =  f_u_h*self.hashtag_index2freq
        value =  (1-self.alpha)*f_u_h/f_u_h.sum()+self.alpha*self.hashtag_index2freq/self.hashtag_index2freq.sum()
        return value

    def _get_hashtag2index(self, hashtags):
        assert  hashtags is not None and len(hashtags) > 0
        for hashtag in hashtags:
            if hashtag not in self.hashtag2index:
                self.hashtag2index[hashtag] = len(self.hashtag2index)
                self.index2hashtag[len(self.hashtag2index)-1] = hashtag
            else:
                pass

    def _get_hashtag_index2freq(self, hashtags):
        l = []
        for hashtag in hashtags:
            l.append(self.hashtag2index[hashtag])
        id,count = numpy.unique(numpy.array(l), return_counts=True)
        idx = numpy.argsort(id)
        id = id[idx]
        count = count[idx]
        self.hashtag_index2freq = numpy.array(count, dtype="float32")

    def _get_user_dic(self, users, hashtags):
        assert len(users) == len(hashtags)
        for user, hashtag in zip(users, hashtags):
            hashtag = self.hashtag2index[hashtag]
            if user not in self.user2dic:
                self.user2dic[user] = {hashtag:1}
            else:
                dic = self.user2dic[user]
                if hashtag not in dic:
                    dic[hashtag] = 1
                else:
                    dic[hashtag] += 1


class TUTHD(UTHD):
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


