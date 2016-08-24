import logging
import io
import os

import numpy

import cPickle

from blocks.extensions import SimpleExtension

logging.basicConfig(level='INFO')
logger = logging.getLogger('extensions.SaveLoadParams')

class SaveLoadParams(SimpleExtension):
    def __init__(self, path, model, dataset, **kwargs):
        super(SaveLoadParams, self).__init__(**kwargs)

        self.path = path
        self.model = model
        self.dataset = dataset

    def do_save(self):
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))
        with open(self.path, 'wb') as f:
            logger.info('Saving parameters to %s...'%self.path)
            dumper = cPickle.Pickler(f, protocol=cPickle.HIGHEST_PROTOCOL)
            # Save model and necessary dataset information
            dumper.dump(self.model.get_parameter_values())
            dumper.dump(self.dataset.get_parameter_to_save())


    def do_load(self):
        try:
            with open(self.path, 'rb') as f:
                logger.info('Loading parameters from %s...'%self.path)
                last_model_params = cPickle.load(f)
                last_dataset_params = cPickle.load(f)
                self.do_initialize(last_model_params, last_dataset_params)
        except IOError as e:
            print("Cannot load parameters!")

    def do_initialize(self, last_model_params, last_dataset_params):
        cur_dataset_params = self.dataset.get_parameter_to_save()
        cur_model_params = self.model.get_parameter_values()
        # Initialize LSTM params
        for key, value in last_dataset_params.iteritems():
            if key != "/hashtag_embed.W" and key != "/user_embed.W" and key != "word_embed.W":
                cur_model_params[key] = value

        #region Initialize embedding params
        # Initialize hashtag embedding
        last_hashtag_embed = last_model_params['/hashtag_embed.W']
        cur_hashtag_embed = cur_model_params['/hashtag_embed.W']
        last_hashtag2index = last_dataset_params['hashtag2index']
        cur_hashtag2index = cur_dataset_params['hashtag2index']
        for hashtag, index in last_hashtag2index.iteritems():
            if hashtag in cur_hashtag2index:
                cur_hashtag_embed[cur_hashtag2index[hashtag]] = last_hashtag_embed[index]
        # Initialize user embedding
        last_user_embed = last_model_params['/user_embed.W']
        cur_user_embed = cur_model_params['/user_embed.W']
        last_user2index = last_dataset_params['user2index']
        cur_user2index = cur_dataset_params['user2index']
        for user, index in last_user2index.iteritems():
            if user in cur_user2index:
                cur_user_embed[cur_user2index[user]] = last_user_embed[index]
        # Initialize word embedding
        last_word_embed = last_model_params['/word_embed.W']
        cur_word_embed = cur_model_params['/word_embed.W']
        last_word2index = last_dataset_params['word2index']
        cur_word2index = cur_dataset_params['word2index']
        for word, index in last_word2index.iteritems():
            if word in cur_word2index:
                cur_word_embed[cur_word2index[word]] = last_word_embed[index]
        #endregion

        pass

    def do(self, which_callback, *args):
        if which_callback == 'before_training':
            self.do_load()
        else:
            self.do_save()


if __name__ == "__main__":
    loader = SaveLoadParams(path = r"E:\projects\Hashtag\output\model\UTHC_lstm.pkl", model = None, dataset = None)
    loader.do_load()