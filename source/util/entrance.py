import os
import codecs
import ntpath
import logging
import logging
import cPickle
import theano.tensor as tensor
import theano
from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.theano_expressions import l2_norm
from blocks.algorithms import Scale
from picklable_itertools.extras import equizip

logger = logging.getLogger('extensions.SaveLoadParams')

#region Extension
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


class BasicSaveLoadParams(SimpleExtension):
    '''
    Only initialize word, user and haashtag embeddings and bricks params
    '''
    def __init__(self, load_from, save_to, model, dataset,  **kwargs):
        super(BasicSaveLoadParams, self).__init__(**kwargs)

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
        self._initialize_other(last_model_params,last_dataset_params, cur_model_params, cur_dataset_params)
        #region Initialize embedding params
        # Initialize hashtag embedding
        self._initialize_hashtag(last_model_params,last_dataset_params,cur_model_params, cur_dataset_params)
        # Initialize user embedding
        self._initialize_user(last_model_params,last_dataset_params,cur_model_params, cur_dataset_params)
        # Initialize word embedding
        self._initialize_word(last_model_params,last_dataset_params,cur_model_params, cur_dataset_params)
        #endregion
        self.model.set_parameter_values(cur_model_params)

    def _initialize_hashtag(self, last_model_params, last_dataset_params,
                            cur_model_params, cur_dataset_params):
        last_hashtag_embed = last_model_params['/hashtag_embed.W']
        cur_hashtag_embed = cur_model_params['/hashtag_embed.W']
        last_hashtag2index = last_dataset_params['hashtag2index']
        cur_hashtag2index = cur_dataset_params['hashtag2index']

        for hashtag, index in last_hashtag2index.iteritems():
            if hashtag in cur_hashtag2index:
                cur_hashtag_embed[cur_hashtag2index[hashtag]] = last_hashtag_embed[index]

    def _initialize_user(self,last_model_params, last_dataset_params,
                            cur_model_params, cur_dataset_params):
        last_user_embed = last_model_params['/user_embed.W']
        cur_user_embed = cur_model_params['/user_embed.W']
        last_user2index = last_dataset_params['user2index']
        cur_user2index = cur_dataset_params['user2index']
        for user, index in last_user2index.iteritems():
            if user in cur_user2index:
                cur_user_embed[cur_user2index[user]] = last_user_embed[index]

    def _initialize_word(self,last_model_params, last_dataset_params,
                            cur_model_params, cur_dataset_params):
        last_word_embed = last_model_params['/word_embed.W']
        cur_word_embed = cur_model_params['/word_embed.W']
        last_word2index = last_dataset_params['word2index']
        cur_word2index = cur_dataset_params['word2index']
        for word, index in last_word2index.iteritems():
            if word in cur_word2index:
                cur_word_embed[cur_word2index[word]] = last_word_embed[index]

    def _initialize_other(self, last_model_params, last_dataset_params,
                          cur_model_params, cur_dataset_params):
        for key, value in last_model_params.iteritems():
            if key != "/hashtag_embed.W" and key != "/user_embed.W" and key != '/word_embed.W':
                cur_model_params[key] = value

    def do(self, which_callback, *args):
        if which_callback == 'before_training':
            self.do_load()
        else:
            self.do_save()


class BiasSaveLoadParams(BasicSaveLoadParams):
    '''
    Initialize hashtag bias
    '''
    def __init__(self, load_from, save_to, model, dataset,  **kwargs):
        super(BiasSaveLoadParams, self).__init__(load_from, save_to, model, dataset,**kwargs)

    def _initialize_hashtag(self, last_model_params, last_dataset_params,
                            cur_model_params, cur_dataset_params):
        last_hashtag_embed = last_model_params['/hashtag_embed.W']
        cur_hashtag_embed = cur_model_params['/hashtag_embed.W']
        last_hashtag2index = last_dataset_params['hashtag2index']
        cur_hashtag2index = cur_dataset_params['hashtag2index']
        last_hashtag_bias = last_model_params['/hashtag_bias.b']
        cur_hashtag_bias = cur_model_params['/hashtag_bias.b']
        for hashtag, index in last_hashtag2index.iteritems():
            if hashtag in cur_hashtag2index:
                cur_hashtag_embed[cur_hashtag2index[hashtag]] = last_hashtag_embed[index]
                cur_hashtag_bias[cur_hashtag2index[hashtag]] = last_hashtag_bias[index]


class ExtendSaveLoadParams(BiasSaveLoadParams):
    def __init__(self, load_from, save_to, model, dataset, **kwargs):
        super(ExtendSaveLoadParams, self).__init__(load_from, save_to, model, dataset,**kwargs)

    def _initialize_other(self, last_model_params, last_dataset_params,
                          cur_model_params, cur_dataset_params):
        last_char_embed = last_model_params['/char_embed.W']
        cur_char_embed = cur_model_params['/char_embed.W']
        last_char2index = last_dataset_params['char2index']
        cur_char2index = cur_dataset_params['char2index']
        for char, index in last_char2index.iteritems():
            if char in cur_char2index:
                cur_char_embed[cur_char2index[char]] = last_char_embed[index]
        for key, value in last_model_params.iteritems():
            if key not in ("/hashtag_embed.W",'/hashtag_bias.b', "/user_embed.W", '/word_embed.W', '/char_embed.W'):
                cur_model_params[key] = value


class EarlyStopMonitor(DataStreamMonitoring):
    PREFIX_SEPARATOR = '_'

    def __init__(self, variables, monitor_variable, data_stream, updates=None, saver=None, tolerate_time = 5, **kwargs):
        super(DataStreamMonitoring, self).__init__(**kwargs)
        self._evaluator = DatasetEvaluator(variables, updates)
        self.data_stream = data_stream
        self.saver = saver
        self.best_result = -1.0
        self.wait_time = 0
        self.tolerate_time = tolerate_time
        self.monitor_variable = monitor_variable

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        logger.info("Monitoring on auxiliary data started")
        value_dict = self._evaluator.evaluate(self.data_stream)
        self.add_records(self.main_loop.log, value_dict.items())
        logger.info("Monitoring on auxiliary data finished")
        self.check_stop(value_dict)

    def check_stop(self, value_dict):
        if value_dict[self.monitor_variable.name] > self.best_result:
            self.best_result = value_dict[self.monitor_variable.name]
            self.wait_time = 0
            if self.saver is not None:
                self.saver.do_save()
        else:
            self.wait_time += 1
        if self.wait_time > self.tolerate_time:
            self.main_loop.status['batch_interrupt_received'] = True
            self.main_loop.status['epoch_interrupt_received'] = True


class EvaluatorWithEarlyStop(EarlyStopMonitor):
    def __init__(self, coverage, **kwargs):
        super(EvaluatorWithEarlyStop, self).__init__(**kwargs)
        self.coverage = coverage

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        logger.info("Monitoring on auxiliary data started")
        value_dict = self._evaluator.evaluate(self.data_stream)
        for key in value_dict.keys():
            value_dict[key] *= self.coverage
        value_dict['coverage'] = self.coverage
        self.add_records(self.main_loop.log, value_dict.items())
        logger.info("Monitoring on auxiliary data finished")
        self.check_stop(value_dict)


class ForgetExtendSaveLoadParams(BasicSaveLoadParams):
    def __init__(self, load_from, save_to, model, dataset, hashtag_forget_rate = 1, **kwargs):
        super(ForgetExtendSaveLoadParams, self).__init__(load_from, save_to, model, dataset,**kwargs)
        self.hashtag_forget_rate = hashtag_forget_rate


    def do_initialize(self, last_model_params, last_dataset_params):
        cur_dataset_params = self.dataset.get_parameter_to_save()
        cur_model_params = self.model.get_parameter_values()
        # Initialize LSTM params
        for key, value in last_model_params.iteritems():
            if key != "/hashtag_embed.W" and key != "/user_embed.W" and key != "/word_embed.W" and key != '/char_embed.W':
                cur_model_params[key] = value

        #region Initialize embedding params
        # Initialize hashtag embedding
        last_hashtag_embed = last_model_params['/hashtag_embed.W']
        cur_hashtag_embed = cur_model_params['/hashtag_embed.W']
        last_hashtag2index = last_dataset_params['hashtag2index']
        cur_hashtag2index = cur_dataset_params['hashtag2index']
        for hashtag, index in last_hashtag2index.iteritems():
            if hashtag in cur_hashtag2index:
                cur_hashtag_embed[cur_hashtag2index[hashtag]] = last_hashtag_embed[index]*self.hashtag_forget_rate
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
        # Initialize character embedding
        last_char_embed = last_model_params['/char_embed.W']
        cur_char_embed = cur_model_params['/char_embed.W']
        last_char2index = last_dataset_params['char2index']
        cur_char2index = cur_dataset_params['char2index']
        for char, index in last_char2index.iteritems():
            if char in cur_char2index:
                cur_char_embed[cur_char2index[char]] = last_char_embed[index]
        #endregion
        self.model.set_parameter_values(cur_model_params)


class ForgetSaveLoadParams(BasicSaveLoadParams):
    def __init__(self, load_from, save_to, model, dataset, hashtag_forget_rate = 1, **kwargs):
        super(ForgetSaveLoadParams, self).__init__(load_from, save_to, model, dataset,**kwargs)
        self.hashtag_forget_rate = hashtag_forget_rate

    def do_initialize(self, last_model_params, last_dataset_params):
        cur_dataset_params = self.dataset.get_parameter_to_save()
        cur_model_params = self.model.get_parameter_values()
        # Initialize LSTM params
        for key, value in last_model_params.iteritems():
            if key != "/hashtag_embed.W" and key != "/user_embed.W" and key != "/word_embed.W" and key != '/char_embed.W':
                cur_model_params[key] = value

        #region Initialize embedding params
        # Initialize hashtag embedding
        last_hashtag_embed = last_model_params['/hashtag_embed.W']
        cur_hashtag_embed = cur_model_params['/hashtag_embed.W']
        last_hashtag2index = last_dataset_params['hashtag2index']
        cur_hashtag2index = cur_dataset_params['hashtag2index']
        for hashtag, index in last_hashtag2index.iteritems():
            if hashtag in cur_hashtag2index:
                cur_hashtag_embed[cur_hashtag2index[hashtag]] = last_hashtag_embed[index]*self.hashtag_forget_rate
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
#endregion



