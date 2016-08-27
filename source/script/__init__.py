import cPickle
import codecs
import datetime
import os

import numpy
from nltk.tokenize import TweetTokenizer
from pymongo import  MongoClient

import sys
sys.path.extend("..")
from config import UTHC
from util import dataset


class ScriptConfig(UTHC):
    host = '10.141.209.3'
    port = 27017
    database_name = 'tweet_contents'
    table_name = 'tweets'
    train_path = os.path.join(UTHC.project_dir, "data/train.txt")
    data_path = os.path.join(UTHC.project_dir, "data/tweet/tweet_2015.txt")


class MongoDumper(object):

    def __init__(self, config):
        self.config = config
        self.client = MongoClient(config.host, config.port)
        # Get database
        self.database = self.client.get_database(name=config.database_name)
        # Get table
        self.table = self.database.get_collection(name=config.table_name)

    def dump(self, sample_size = 1000000, save_to = None):
        # Get items with field hashtags not empty.
        # Filter is a dictionary. '$key' is a property key and 'key' is a field key
        # Reference: https://docs.mongodb.com/manual/reference/
        # Examples:
        #     one_item = tweet_content_tb.find_one({'$where':'this.entities.hashtags.length > 0'})
        #     one_item = tweet_content_tb.find_one({'entities.hashtags.1':{'$exists':True}})
        # An item is a dictionary storing its information
        cursor = self.table.find(
            {'entities.hashtags.1': {'$exists': True}, 'lang': 'en', 'retweeted_status': {'$exists': False}})
        if save_to is None:
            save_to = self.config.data_path
        with codecs.open(save_to, "w+", encoding="ascii", errors="strict") as writer:
            count = 0
            for post in cursor:
                if count % 10000 == 0:
                    print("%d posts dumped" % count)
                time = post.get('created_at')
                if not time.endswith('2015'):
                    continue
                user_id = post.get('user').get('id')
                # replace \n within post
                text = post.get('text').replace("\n", " ")
                hashtags = []
                l = post.get('entities').get('hashtags')
                for item in l:
                    hashtags.append(item.get('text'))
                self._save_one_sample(writer, user_id, text, hashtags, time)
                count += 1
                if count > sample_size:
                    break

    def _save_one_sample(self, writer, user_id, text, hashtags, time):
        line = "%d\t%s\t%s\t%s\n" % (user_id, text, ";".join(hashtags), time)
        try:
            line.encode('ascii')
            writer.write(line)
        except Exception as error:
            pass


class UserNameDumper(MongoDumper):

    def __init__(self, config):
        super(UserNameDumper,self).__init__(config)

    def dump(self, sample_size = 1000000, save_to = None):
        cursor = self.table.find(
            {'entities.hashtags.1': {'$exists': True}, 'lang': 'en', 'retweeted_status': {'$exists': False}})
        if save_to is None:
            save_to = self.config.data_path
        with codecs.open(save_to, "w+", encoding="ascii", errors="strict") as writer:
            count = 0
            for post in cursor:
                if count % 10000 == 0:
                    print("%d posts dumped" % count)
                time = post.get('created_at')
                if not time.endswith('2015'):
                    continue
                user_id = post.get('user').get('id')
                user_name = post.get('user').get('screen_name')
                writer.write("{0}\t{1}\n".format(user_name, user_id))
                count += 1
                if count > sample_size:
                    break


class BUTHD(object):
    '''
       Basic dataset with user-text-time-hashtag information.

       load dataset --> parse string type date --> provide samples of given date --> map user, hashtag, word to id -->
       --> construct indexable dataset --> construct shuffled or sequencial fuel stream
       '''

    def __init__(self, data_path):
        # Dictionary
        self.index = 0
        self.data_path  = data_path


    def get_dataset(self):
        '''
        Prepare dataset
        '''
        print("Preparing dataset...")
        raw_dataset = dataset.read_file_by_line(self.data_path, delimiter="\t", field_num=4, mode="debug")
        for sample in raw_dataset:
            sample[1] = sample[1].split(' ')
            sample[3] = self._parse_date(sample[3])
        return numpy.array(raw_dataset)
        print("Done!")

    def _parse_date(self, str_date):
        '''
        Parse string format date.

        Reference: https://docs.python.org/2/library/time.html or http://strftime.org/
        :return: A datetime.date object
        '''
        return datetime.datetime.strptime(str_date, "%a %b %d %H:%M:%S +0000 %Y").date()


def tokenize_text(config, source_file, des_file):
    '''
    Tokenize post content
    '''
    twtk = TweetTokenizer()
    with open(des_file, 'w+') as writer:
        with open(source_file, 'r') as reader:
            for line in reader:
                array = line.strip().split(config.delimiter)
                array[config.text_index] = " ".join(twtk.tokenize(array[config.text_index]))
                writer.write("\t".join(array)+"\n")


def format2UTHD(config, read_from, save_to):
    '''
    Pre-process UTHD dataset.
    1. split text by hashtag
    :param read_from:
    :param save_to:
    :return:
    '''
    with open(save_to, 'w+') as writer:
        with open(read_from, 'r') as reader:
            for line in reader:
                array = line.strip().split(config.delimiter)
                hashtags = set(array[config.hashtag_index].split(';'))
                user_id = array[config.user_index]
                time = array[config.date_index]
                tokens = array[config.text_index].split(' ')
                for hashtag in hashtags:
                    try:
                        index = tokens.index("#"+hashtag)
                    except Exception as error:
                        continue
                    writer.write("{0}\t{1}\t{2}\t{3}\n".format(user_id, " ".join(tokens[0:index]+['#']), hashtag, time))


def pickle_dataset(config = None, read_from = "path_to_read", save_to = "path_to_save") :
    dataset = BUTHD(read_from).get_dataset()
    with open(save_to, 'wb+') as f:
        cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    config = ScriptConfig()
    project_dir = config.project_dir
    dumper = UserNameDumper(config)
    dumper.dump(numpy.inf,save_to=os.path.join(project_dir, "data/tweet/user_name2id.txt"))

