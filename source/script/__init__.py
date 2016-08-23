from config.hashtag_config import UTHC

class ScriptConfig(UTHC):
    host = '10.141.209.3'
    port = 27017
    database_name = 'tweet_contents'
    table_name = 'tweets'


from pymongo import  MongoClient
import codecs

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
                count += 1
                if count % 10000 == 0:
                    print("%d posts dumped" % count)
                user_id = post.get('user').get('id')
                # replace \n within post
                text = post.get('text').replace("\n", " ")
                hashtags = []
                l = post.get('entities').get('hashtags')
                for item in l:
                    hashtags.append(item.get('text'))
                time = post.get('created_at')
                self._save_one_sample(writer, user_id, text, hashtags, time)
                if count > sample_size:
                    break

    def _save_one_sample(self, writer, user_id, text, hashtags, time):
        line = "%d\t%s\t%s\t%s\n" % (user_id, text, ";".join(hashtags), time)
        try:
            line.encode('ascii')
            writer.write(line)
        except Exception as error:
            pass


from nltk.tokenize import TweetTokenizer

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


from numpy.random import RandomState


def split_dataset(config,read_from, save_train_to, save_dev_to, save_test_to):
    '''
    Split dataset into training dataset, developing dataset and test dataset.
    :param config: ScriptConfig object
    :param read_from: original file
    :param save_train_to: file path to save training dataset
    :param save_dev_to:  file path to save developt dataset
    :param save_test_to: file path to save test dataset
    '''
    train_portion = 1-config.dev_portion-config.test_portion
    dev_up = train_portion+config.dev_portion
    rvg = RandomState()
    with open(save_train_to, 'w+') as train_writer:
        with open(save_dev_to, 'w+') as dev_writer:
            with open(save_test_to, 'w+') as test_writer:
                with open(read_from, 'r') as reader:
                    for line in reader:
                        rv = rvg.uniform(size = 1)
                        if rv<train_portion:
                            train_writer.write(line)
                        elif rv < dev_up:
                            dev_writer.write(line)
                        else:
                            test_writer.write(line)