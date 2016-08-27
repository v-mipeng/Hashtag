import os
import codecs
import ntpath
from blocks.extensions import SimpleExtension
import logging
import logging
import cPickle

from blocks.extensions import SimpleExtension

logger = logging.getLogger('extensions.SaveLoadParams')


class EpochMonitor(SimpleExtension):
    def __init__(self, max_epoch, **kwargs):
        super(EpochMonitor, self).__init__(after_epoch = True, **kwargs)

        self.cur_epoch = 0
        self.max_epoch = max_epoch

    def do(self, which_callback, *args):
        if which_callback == "after_epoch":
            self.cur_epoch += 1
            if self.cur_epoch >= self.max_epoch:
                self.main_loop.status['epoch_interrupt_received'] = True



class SaveLoadParams(SimpleExtension):
    def __init__(self, load_from, save_to, model, dataset, **kwargs):
        super(SaveLoadParams, self).__init__(**kwargs)

        self.load_from = load_from
        self.save_to = save_to
        self.model = model
        self.dataset = dataset

    def do_save(self):
        if not os.path.exists(os.path.dirname(self.save_to)):
            os.makedirs(os.path.dirname(self.save_to))
        with open(self.save_to, 'wb+') as f:
            logger.info('Saving parameters to %s...'%self.save_to)
            # Save model and necessary dataset information
            cPickle.dump(self.model.get_parameter_values(), f)
            cPickle.dump(self.dataset.get_parameter_to_save(), f)

    def do_load(self):
        try:
            with open(self.load_from, 'rb') as f:
                logger.info('Loading parameters from %s...'%self.load_from)
                last_model_params = cPickle.load(f)
                last_dataset_params = cPickle.load(f)
                self.do_initialize(last_model_params, last_dataset_params)
        except IOError as e:
            print("Cannot load parameters!")

    def do_initialize(self, last_model_params, last_dataset_params):
        cur_dataset_params = self.dataset.get_parameter_to_save()
        cur_model_params = self.model.get_parameter_values()
        # Initialize LSTM params
        for key, value in last_model_params.iteritems():
            if key != "/hashtag_embed.W" and key != "/user_embed.W" and key != "/word_embed.W":
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
        self.model.set_parameter_values(cur_model_params)
        pass

    def do(self, which_callback, *args):
        if which_callback == 'before_training':
            self.do_load()
        else:
            self.do_save()



def save_result(file_path, results):
    assert file_path is not None
    dir = ntpath.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    writer = codecs.open(file_path,"w+")
    for result in results:
        writer.write("%s\n" %"\t".join(map(str,result)))
    writer.close()


def get_in_out_files(input_path, output_path):
    input_files = []
    output_files = []
    if isinstance(input_path,list):
        input_files = input_path
        output_files = output_path
    elif isinstance(input_path, str):
        if os.path.isfile(input_path):
            input_files = [input_path]
            output_files = [output_path]
        elif os.path.isdir(input_path):
            input_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.txt')]
            if input_path == output_path:
                output_files = [(f+".result") for f in input_files]
            else:
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                output_files = [os.path.join(output_path, f) for f in os.listdir(input_path) if f.endswith('.txt')]
        else:
            Exception("Test file can only be defined by a list of file paths or a directory paths!")
    else:
        raise Exception("Test file can only be defined by a list of file paths or a directory paths!")
    return input_files, output_files


