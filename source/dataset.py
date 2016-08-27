# -*- coding : utf-8 -*-
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

    def prepare(self, data_path=None):
        '''
        Prepare dataset
        :param data_path:
        :return:
        '''
        if data_path is None and self.raw_dataset is None:
            data_path = self.config.train_path
        print("Preparing dataset...")
        # Load pickled dataset
        with open(data_path, 'rb') as f:
            self.raw_dataset = cPickle.load(f)
        fields = zip(*self.raw_dataset)
        self.dates = numpy.array(fields[self.config.date_index])
        self.first_date = self.dates.min()
        self.last_date = self.dates.max()
        self.date_span = (self.last_date - self.first_date).days + 1
        print("Done!")

    def get_dataset(self, data_path=None, reference_date="LAST_DAY", date_offset=0, duration=3):
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
    def __init__(self, config):
        self.data_path = config.data_path
        self.config = config
        # Dictionary
        self.user2index = {}
        self.hashtag2index = {}
        self.word2index = {}
        # Frequency
        self.word2freq = {}
        self.user2freq = {}
        self.hashtag2freq = {}
        # Integer. Word whose frequency is less than the threshold will be stemmed
        self.sparse_word_threshold = 0
        self.sparse_hashtag_threshold = 0
        self.sparse_user_threshold = 0
        # (1D numpy array, 1D numpy array). Storing hashtag id and hashtag normed number pair
        self.hashtag_dis_table = None
        self.provide_souces = ('user', 'text', 'hashtag')
        self.need_mask_sources = {'text': config.int_type}
        self.compare_source = 'text'
        self.sample_from = [self.hashtag_dis_table]
        self.sample_sources = ['hashtag']
        self.sample_sizes = [config.hashtag_sample_size]
        self.raw_dataset = BUTHD(config)
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

    def get_parameter_to_save(self):
        '''
        Return parameters that need to be saved with model
        :return:
        '''
        return OrderedDict(
            {'hashtag2index': self.hashtag2index, 'word2index': self.word2index, 'user2index': self.user2index,
             'word2freq': self.word2freq, 'hashtag2freq': self.hashtag2freq, 'user2freq': self.user2freq})

    def get_shuffled_stream(self, reference_date="LAST_DAY", date_offset=0, duration=3, update=True):
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
        :param data_path: string type path of the dataset. If not given, get data from the dataset last loaded
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
        self.word2freq = self._extract_word2freq(fields[self.config.text_index])

        self.user2freq = self._extract_user2freq(fields[self.config.user_index])

        self.hashtag2freq = self._extract_hashtag2freq(fields[self.config.hashtag_index])

        self.sparse_word_threshold = self._get_sparse_threshold(self.word2freq.values(),
                                                                self.config.sparse_word_percent)
        self.sparse_hashtag_threshold = self._get_sparse_threshold(self.hashtag2freq.values(),
                                                                   self.config.sparse_hashtag_percent)
        self.sparse_user_threshold = self._get_sparse_threshold(self.user2freq.values(),
                                                                self.config.sparse_hashtag_percent)

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
            self.hashtag_dis_table = self._construct_distribution_table(hashtags)
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
            return _dic[item]

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
                if word not in word2freq:
                    word2freq[word] = 1
                else:
                    word2freq[word] += 1
        return word2freq

    def _extract_user2freq(self, users):
        return self._extract_freq(users)

    def _extract_hashtag2freq(self, hashtags):
        return self._extract_freq(hashtags)

    def _extract_freq(self, items):
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

    def _get_sparse_threshold(self, freq, percent):
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
        stream = NegativeSample(stream,
                                dist_tables=self.sample_from,
                                sample_sources=self.sample_sources,
                                sample_sizes=self.sample_sizes)
        return stream

    def _construct_sequencial_stream(self, dataset):
        '''Construct sequencial stream.
        This is usually used for testing and prediction.
        '''
        it = SequentialScheme(dataset.num_examples, self.config.batch_size)
        stream = DataStream(dataset, iteration_scheme=it)
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        stream = NegativeSample(stream,
                                dist_tables=self.sample_from,
                                sample_sources=self.sample_sources,
                                sample_sizes=self.sample_sizes)
        return stream

    def _construct_distribution_table(self, elements):
        '''
        Build hashtag distribution table
        '''
        id, count = numpy.unique(elements, return_counts=True)
        count = count ** (3.0 / 4)
        return (id, count)

    def _stem_word(self, word):
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
            elif word.startswith('http'):
                return '<url>'
            else:
                return '<unk>'
        else:
            return '<unk>'

    def _stem_hashtag(self, hashtag):
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

    def _stem_user(self, user):
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


class SUTHD(UTHD):
    '''
    Sequential user-text-hahstag dataset.Provide dataset in time order.
    '''

    def __init__(self, config):
        super(SUTHD, self).__init__(config)
        self.data_path = config.data_path
        self.config = config
        # Dictionary
        self.user2index = {}
        self.hashtag2index = {}
        self.word2index = {}
        # Frequency
        self.word2freq = {}
        self.user2freq = {}
        self.hashtag2freq = {}
        # Integer. Word whose frequency is less than the threshold will be stemmed
        self.sparse_word_threshold = 0
        self.sparse_hashtag_threshold = 0
        self.sparse_user_threshold = 0
        # (1D numpy array, 1D numpy array). Storing hashtag id and hashtag normed number pair
        self.hashtag_dis_table = None
        self.provide_souces = ('user', 'text', 'hashtag')
        self.need_mask_sources = {'text': config.int_type}
        self.compare_source = 'text'
        self.raw_dataset = BUTHD(config)
        self.date_iterator = None
        self._initialize()

    def _initialize(self):
        '''
        Initialize fields: user2index, hashtag2index and so on
        :return:
        '''
        if os.path.exists(self.config.model_path):
            with open(self.config.model_path, 'rb') as f:
                cPickle.load(f)
                dataset_prarms = cPickle.load(f)
                self.user2index = dataset_prarms['user2index']
                self.word2index = dataset_prarms['word2index']
                self.hashtag2index = dataset_prarms['hashtag2index']

    def __iter__(self):
        self.raw_dataset.prepare(self.data_path)
        return self

    def next(self):
        if self.date_iterator is None:
            if self.config.mode == 'debug':
                self.date_iterator = iter(range(self.config.duration, min(self.config.duration+10,self.raw_dataset.date_span)))
            else:
                self.date_iterator = iter(range(self.config.duration, self.raw_dataset.date_span))
        try:
            date_offset = self.date_iterator.next()
            return self.get_stream(date_offset)
        except StopIteration as e:
            self.date_iterator = None
            raise e

    def get_parameter_to_save(self):
        '''
        Return parameters that need to be saved with model
        :return:
        '''
        return OrderedDict(
            {'hashtag2index': self.hashtag2index, 'word2index': self.word2index, 'user2index': self.user2index,
             'word2freq': self.word2freq, 'hashtag2freq':self.hashtag2freq, 'user2freq':self.user2freq})

    def get_stream(self, date_offset):
        train_stream = self.get_shuffled_stream(reference_date= self.raw_dataset.FIRST_DAY,
                                                        date_offset = date_offset-1,
                                                        duration = self.config.duration,
                                                        update= True)
        valid_stream = self.get_shuffled_stream(reference_date= self.raw_dataset.FIRST_DAY,
                                                        date_offset = date_offset,
                                                        duration = 1,
                                                        update = False)

        return train_stream, valid_stream, self.raw_dataset.first_date+datetime.timedelta(days = date_offset)

    def get_shuffled_stream(self,reference_date = "LAST_DAY", date_offset = 0, duration = 3, update = True):
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
        dataset = self._get_dataset(self.data_path, reference_date = reference_date, date_offset = date_offset, duration = duration, update = update)
        return self._construct_shuffled_stream(dataset)

    def get_sequencial_stream(self, reference_date = "LAST_DAY", date_offset = 0, duration = 3, update = True):

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
        dataset = self._get_dataset(self.data_path, reference_date = reference_date, date_offset = date_offset, duration = duration, update = update)
        return self._construct_sequencial_stream(dataset)

    def _get_dataset(self, data_path = None, reference_date = "LAST_DAY", date_offset = 0, duration = 3, update = True):

        # Get duration of the dataset to load

        raw_dataset = self.raw_dataset.get_dataset(reference_date = reference_date, date_offset = date_offset, duration = duration)
        if update:
            self._update(raw_dataset)
        dataset = self._turn_str2index(raw_dataset, update)
        return self._construct_dataset(dataset)

    def _update(self, raw_dataset):
        fields = zip(*raw_dataset)
        self.word2freq = self._extract_word2freq(fields[self.config.text_index])

        self.user2freq = self._extract_user2freq(fields[self.config.user_index])

        self.hashtag2freq = self._extract_hashtag2freq(fields[self.config.hashtag_index])

        self.sparse_word_threshold = self._get_sparse_threshold(self.word2freq.values(),
                                                                self.config.sparse_word_percent)
        self.sparse_hashtag_threshold = self._get_sparse_threshold(self.hashtag2freq.values(),
                                                                   self.config.sparse_hashtag_percent)
        self.sparse_user_threshold = self._get_sparse_threshold(self.user2freq.values(),
                                                               self.config.sparse_hashtag_percent)

    def _turn_str2index(self, raw_dataset, update = True):
        '''
        Turn string type user, words of context, hashtag representation into index representation.
        load the dictionaries if existing otherwise extract from dataset and store them
        '''
        # try to load mapping dictionaries
        assert raw_dataset is not None or len(raw_dataset)>0
        fields = zip(*raw_dataset)
        users = numpy.array([self._get_user_index(self._stem_user(user), update) for user in fields[self.config.user_index]],
                            dtype = self.config.int_type)
        hashtags = numpy.array([self._get_hashtag_index(self._stem_hashtag(hashtag), update) for hashtag in fields[self.config.hashtag_index]],
                               dtype= self.config.int_type)
        texts = [numpy.array([self._get_word_index(self._stem_word(word), update) for word in text],
                             dtype = self.config.int_type)
                            for text in fields[self.config.text_index]]
        if update:
            self.hashtag_dis_table = self._construct_distribution_table(hashtags)
        return (users, texts, hashtags)

    def _get_user_index(self, user, update = True):
        return self._get_index(user, self.user2index, update)

    def _get_hashtag_index(self, hashtag, update = True):
        return self._get_index(hashtag,self.hashtag2index, update)

    def _get_word_index(self, word, update = True):
        return self._get_index(word, self.word2index, update)

    def _get_index(self, item, _dic, update = True):
        if item not in _dic:
            if update:
                _dic[item] = len(_dic)
            return len(_dic)-1
        else:
            return _dic[item]

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
                if word not in word2freq:
                    word2freq[word] = 1
                else:
                    word2freq[word] += 1
        return word2freq

    def _extract_user2freq(self, users):
        return self._extract_freq(users)

    def _extract_hashtag2freq(self, hashtags):
        return self._extract_freq(hashtags)

    def _extract_freq(self, items):
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

    def _get_sparse_threshold(self, freq, percent):
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

    def _extract_user2index(self, users):
        assert users is not None
        user2index = {}
        for user in users:
            if user not in user2index:
                user2index[user] = len(user2index)
        return user2index

    def _extract_word2index(self, texts):
        assert texts is not None
        word2index = {}
        for words in texts:
            for word in words:
                word = self._stem(word)
                if word not in word2index:
                    word2index[word] = len(word2index)
        return word2index

    def _extract_hashtag2index(self, hashtags):
        assert hashtags is not None
        hashtag2index = {}
        for user in hashtags:
            if user not in hashtag2index:
                hashtag2index[user] = len(hashtag2index)
        return hashtag2index

    def _construct_dataset(self, dataset):
        return IndexableDataset(indexables=OrderedDict(zip(self.provide_souces, dataset)))

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
        stream = NegativeSample(stream,
                                        [self.hashtag_dis_table],
                                        sample_sources=["hashtag"],
                                        sample_sizes=[self.config.hashtag_sample_size])
        return stream

    def _construct_sequencial_stream(self, dataset):
        '''Construct sequencial stream.
        This is usually used for testing and prediction.
        '''
        it = SequentialScheme(dataset.num_examples, self.config.batch_size)
        stream = DataStream(dataset, iteration_scheme=it)
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        stream = NegativeSample(stream,
                                        [self.hashtag_dis_table],
                                        sample_sources=["hashtag"],
                                        sample_sizes=[self.config.hashtag_sample_size])
        return stream

    def _construct_distribution_table(self, elements):
        '''
        Build hashtag distribution table
        :return:
        '''
        id, count = numpy.unique(elements, return_counts=True)
        count = count ** (3.0 / 4)
        return (id, count)

    def _stem_word(self, word):
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
            elif word.startswith('http'):
                return '<url>'
            else:
                return '<unk>'
        else:
            return '<unk>'

    def _stem_hashtag(self, hashtag):
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

    def _stem_user(self, user):
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


