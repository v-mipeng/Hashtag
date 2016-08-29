import theano
from theano import tensor

import numpy
import codecs

from blocks.bricks import Tanh, Softmax, Linear, MLP, Identity, Rectifier
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM, SimpleRecurrent

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from util.model import Lookup
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal

from util.model import MLSTM, MWLSTM


class LUTHM2(object):
    '''
    LSTM User-text-hashtag model
    '''

    def __init__(self, config, dataset):
        '''
        Define User-text-hashtag model
        :param config:
        :param dataset: User-text-hashtag dataset
        '''

        user = tensor.ivector('user')
        hashtag = tensor.ivector('hashtag')
        neg_hashtag = tensor.imatrix('hashtag_negtive_sample')
        text = tensor.imatrix('text')
        text_mask = tensor.imatrix('text_mask')

        # sample negtive hashtags

        # Transpose text
        text = text.dimshuffle(1, 0)
        text_mask = text_mask.dimshuffle(1, 0)

        # Initialize word user and hashtag embedding
        word_embed = LookupTable(len(dataset.word2index), config.word_embed_dim, name='word_embed')
        word_embed.weights_init = IsotropicGaussian(std=1 / numpy.sqrt(config.word_embed_dim))
        word_embed.initialize()

        user_embed = LookupTable(len(dataset.user2index), config.user_embed_dim, name="user_embed")
        user_embed.weights_init = IsotropicGaussian(std=1 / numpy.sqrt(config.user_embed_dim))
        user_embed.initialize()

        hashtag_embed = LookupTable(len(dataset.hashtag2index), config.lstm_dim + config.user_embed_dim,
                                    name='hashtag_embed')
        hashtag_embed.weights_init = IsotropicGaussian(std=1 / numpy.sqrt(config.lstm_dim + config.user_embed_dim))
        hashtag_embed.initialize()

        # Turn word user and hashtag into vector representation
        text_vec = word_embed.apply(text)
        user_vec = user_embed.apply(user)
        true_hashtag_vec = hashtag_embed.apply(hashtag)
        neg_hashtag_vec = hashtag_embed.apply(neg_hashtag)

        # Build and apply multiple-time LSTM
        mlstm_ins = Linear(input_dim=config.word_embed_dim, output_dim=4 * config.lstm_dim, name='mlstm_in')
        mlstm_ins.weights_init = IsotropicGaussian(
            std=numpy.sqrt(2) / numpy.sqrt(config.word_embed_dim + config.lstm_dim))
        mlstm_ins.biases_init = Constant(0)
        mlstm_ins.initialize()
        mlstm = MLSTM(config.lstm_time, config.lstm_dim, shared=False)
        mlstm.weights_init = IsotropicGaussian(std=numpy.sqrt(2) / numpy.sqrt(config.word_embed_dim + config.lstm_dim))
        mlstm.biases_init = Constant(0)
        mlstm.initialize()
        mlstm_hidden, mlstm_cell = mlstm.apply(inputs=mlstm_ins.apply(text_vec),
                                               mask=text_mask.astype(theano.config.floatX))
        text_encodes = mlstm_hidden[-1]

        # Calculate negtive samping cost
        # Reference:Mikolov T, Sutskever I, Chen K, et al.
        #           Distributed Representations of Words and Phrases and their Compositionality[J].
        #           Advances in Neural Information Processing Systems, 2013, 26:3111-3119.
        def get_cost(enc, th, nh):
            '''
            :param enc: text encoding
            :param th: true hashtag
            :param nh: negtive hashtags
            :return:
            '''
            # return tensor.dot(enc, th), tensor.dot(nh,enc)
            return enc.dot(th), nh.dot(enc)

        input_vec = tensor.concatenate([text_encodes, user_vec], axis=1)
        (t_pred, n_pred), _ = theano.scan(get_cost,
                                          sequences=[input_vec, true_hashtag_vec, neg_hashtag_vec],
                                          n_steps=hashtag.shape[0])
        cost = numpy.log(1. / (1. + numpy.exp(-t_pred))).mean() + numpy.log(1. / (1. + numpy.exp(n_pred))).sum(
            axis=1).mean()
        cost = -cost
        pred = tensor.concatenate([t_pred[:, None], n_pred], axis=1)
        max_index = pred.argmax(axis=1)
        error_rate = tensor.neq(0, max_index).mean()

        cost.name = 'cost'
        error_rate.name = 'error_rate'
        self.sgd_cost = cost
        self.monitor_vars = [[cost], [error_rate]]
        self.monitor_vars_valid = [[cost], [error_rate]]


class LUTHM(object):
    '''
    LSTM User-text-hashtag model
    '''
    def __init__(self, config, dataset):
        '''
        Define User-text-hashtag model
        :param config:
        :param dataset: User-text-hashtag dataset
        '''
        self.config = config
        self.dataset = dataset
        self.rank = min(len(self.dataset.hashtag2index), 10)
        self._build_model()

    def _build_model(self):
        # Define inputs
        self._define_inputs()
        self._build_bricks()
        # Transpose text
        self.text = self.text.dimshuffle(1, 0)
        self.text_mask = self.text_mask.dimshuffle(1, 0)
        # Turn word, user and hashtag into vector representation
        text_vec = self.word_embed.apply(self.text)
        user_vec = self.user_embed.apply(self.user)
        true_hashtag_vec = self.hashtag_embed.apply(self.hashtag)
        neg_hashtag_vec = self.hashtag_embed.apply(self.neg_hashtag)
        # Encode text
        mlstm_hidden, mlstm_cell = self.mlstm.apply(inputs=self.mlstm_ins.apply(text_vec),
                                                    mask=self.text_mask.astype(theano.config.floatX))
        text_encodes = mlstm_hidden[-1]
        input_vec = tensor.concatenate([text_encodes, user_vec], axis=1)
        self._get_cost(input_vec, true_hashtag_vec, neg_hashtag_vec)

    def _get_cost(self, input_vec, true_hashtag_vec, neg_hashtag_vec):
        # Calculate negtive samping cost
        # Reference:Mikolov T, Sutskever I, Chen K, et al.
        #           Distributed Representations of Words and Phrases and their Compositionality[J].
        #           Advances in Neural Information Processing Systems, 2013, 26:3111-3119.
        def get_cost(enc, th, nh):
            '''
            :param enc: text encoding
            :param th: true hashtag
            :param nh: negtive hashtags
            :return:
            '''
            # return tensor.dot(enc, th), tensor.dot(nh,enc)
            return enc.dot(th), nh.dot(enc)

        (t_pred, n_pred), _ = theano.scan(get_cost,
                                          sequences=[input_vec, true_hashtag_vec, neg_hashtag_vec],
                                          n_steps=self.hashtag.shape[0])
        cost = numpy.log(1. / (1. + numpy.exp(-t_pred))).mean() + numpy.log(1. / (1. + numpy.exp(n_pred))).sum(
            axis=1).mean()
        cost = -cost
        pred = tensor.concatenate([t_pred[:, None], n_pred], axis=1)
        max_index = pred.argmax(axis=1)
        error_rate = tensor.neq(0, max_index).mean()
        cost.name = 'cost'
        error_rate.name = 'error_rate'
        self.monitor_train_vars = [[cost], [error_rate]]
        top1_recall, top10_recall = self._get_test_cost(input_vec[0:tensor.ceil(input_vec.shape[0]*self.config.sample_percent_for_test).astype(self.config.int_type)])
        self.monitor_valid_vars = [[top1_recall], [top10_recall]]
        self.cg_generator = cost

    def _get_test_cost(self, input_vec):
        preds = tensor.argsort(tensor.dot(input_vec, self.hashtag_embed.W.T),axis = 1)[:,::-1]
        top1_recall = tensor.eq(self.hashtag[0:input_vec.shape[0]], preds[:,0]).mean()
        top10_recall = tensor.sum(tensor.eq(preds[:,0:self.rank], self.hashtag[0:input_vec.shape[0],None]), axis=1).mean()
        top1_recall.name = "top1_recall"
        top10_recall.name = "top10_recall"
        return top1_recall, top10_recall

    def _define_inputs(self):
        self.user = tensor.ivector('user')
        self.hashtag = tensor.ivector('hashtag')
        self.neg_hashtag = tensor.imatrix('hashtag_negtive_sample')
        self.text = tensor.imatrix('text')
        self.text_mask = tensor.matrix('text_mask', dtype=theano.config.floatX)

    def _build_bricks(self):
        # Build lookup tables
        self.word_embed = self._embed(len(self.dataset.word2index), self.config.word_embed_dim, name='word_embed')

        self.user_embed = self._embed(len(self.dataset.user2index), self.config.user_embed_dim, name="user_embed")

        self.hashtag_embed = self._embed(len(self.dataset.hashtag2index),
                                         self.config.lstm_dim + self.config.user_embed_dim,
                                         name='hashtag_embed')
        # Build text encoder
        self.mlstm_ins = Linear(input_dim=self.config.word_embed_dim, output_dim=4 * self.config.lstm_dim, name='mlstm_in')
        self.mlstm_ins.weights_init = IsotropicGaussian(
            std=numpy.sqrt(2) / numpy.sqrt(self.config.word_embed_dim + self.config.lstm_dim))
        self.mlstm_ins.biases_init = Constant(0)
        self.mlstm_ins.initialize()
        self.mlstm = MLSTM(self.config.lstm_time, self.config.lstm_dim, shared=False)
        self.mlstm.weights_init = IsotropicGaussian(
            std=numpy.sqrt(2) / numpy.sqrt(self.config.word_embed_dim + self.config.lstm_dim))
        self.mlstm.biases_init = Constant(0)
        self.mlstm.initialize()

    def _embed(self, sample_num, dim, name):
        embed = LookupTable(sample_num, dim, name= name)
        embed.weights_init = IsotropicGaussian(std=1 / numpy.sqrt(dim))
        embed.initialize()
        return embed


class EUTHM(LUTHM):
    def __init__(self, config, dataset):
        '''
        Define User-text-hashtag model
        :param config:
        :param dataset: User-text-hashtag dataset
        '''
        super(EUTHM, self).__init__(config, dataset)

    def _build_model(self):
        # Define inputs
        self._define_inputs()
        self._build_bricks()
        # Transpose text
        self.text = self.text.dimshuffle(1, 0)
        self.text_mask = self.text_mask.dimshuffle(1, 0)
        self.url = self.url.dimshuffle(1, 0)
        self.url_mask = self.url_mask.dimshuffle(1, 0)
        # Turn word, user and hashtag into vector representation
        text_vec = self.word_embed.apply(self.text)
        user_vec = self.user_embed.apply(self.user)
        true_hashtag_vec = self.hashtag_embed.apply(self.hashtag)
        neg_hashtag_vec = self.hashtag_embed.apply(self.neg_hashtag)
        # Apply user word, hashtag word and url
        self._apply_user_word(text_vec)
        self._apply_hashtag_word(text_vec)
        self._apply_url(text_vec)

        # Encode text
        mlstm_hidden, mlstm_cell = self.mlstm.apply(inputs=self.mlstm_ins.apply(text_vec),
                                                    mask=self.text_mask.astype(theano.config.floatX))
        text_encodes = mlstm_hidden[-1]
        input_vec = tensor.concatenate([text_encodes, user_vec], axis=1)
        self._get_cost(input_vec, true_hashtag_vec, neg_hashtag_vec)


    def _define_inputs(self):
        self.user = tensor.ivector('user')
        self.hashtag = tensor.ivector('hashtag')
        self.neg_hashtag = tensor.imatrix('hashtag_negtive_sample')
        self.text = tensor.imatrix('text')
        self.text_mask = tensor.matrix('text_mask', dtype=theano.config.floatX)
        self.user_word = tensor.imatrix('user_word')
        self.user_word_left_idx = tensor.imatrix('user_word_idx_left_idx')
        self.user_word_right_idx = tensor.imatrix('user_word_idx_right_idx')
        self.hashtag_word = tensor.imatrix('hashtag_word')
        self.hashtag_word_left_idx = tensor.imatrix('hashtag_word_idx_left_idx')
        self.hashtag_word_right_idx = tensor.imatrix('hashtag_word_idx_right_idx')
        self.url = tensor.imatrix('url')
        self.url_mask = tensor.matrix('url_mask', dtype=theano.config.floatX)
        self.url_left_idx = tensor.imatrix('url_idx_left_idx')
        self.url_right_idx = tensor.imatrix('url_idx_right_idx')
        self.user2word = theano.shared(
            numpy.random.randn(self.config.word_embed_dim, self.config.user_embed_dim).astype(dtype=theano.config.floatX),
            'user2word')
        self.hashtag2word = theano.shared(
            numpy.random.randn(self.config.word_embed_dim,
                               self.config.user_embed_dim + self.config.word_embed_dim).astype(dtype=theano.config.floatX),
            'hashtag2word')
        self.url2word = theano.shared(
            numpy.random.randn(self.config.word_embed_dim, self.config.url_rnn_dim).astype(dtype=theano.config.floatX),
            'url2word')

    def _build_bricks(self):
        # Build lookup tables
        self.word_embed = self._embed(len(self.dataset.word2index), self.config.word_embed_dim, name='word_embed')

        self.user_embed = self._embed(len(self.dataset.user2index), self.config.user_embed_dim, name="user_embed")

        self.hashtag_embed = self._embed(len(self.dataset.hashtag2index),
                                         self.config.lstm_dim + self.config.user_embed_dim,
                                         name='hashtag_embed')
        # TODO: set url embedding config
        self.char_embed = self._embed(len(self.dataset.char2index), self.config.char_embed_dim, name='char_embed')
        # Build url encoder
        self.rnn_ins = Linear(input_dim=self.config.char_embed_dim, output_dim=self.config.url_rnn_dim, name='rnn_in')
        self.rnn_ins.weights_init = IsotropicGaussian(
            std=numpy.sqrt(2) / numpy.sqrt(self.config.char_embed_dim + self.config.url_rnn_dim))
        self.rnn_ins.biases_init = Constant(0)
        self.rnn_ins.initialize()
        self.rnn = SimpleRecurrent(dim=self.config.url_rnn_dim, activation=Tanh())
        self.rnn.weights_init = IsotropicGaussian(std=1 / numpy.sqrt(self.config.url_rnn_dim))
        self.rnn.initialize()
        # Build text encoder
        self.mlstm_ins = Linear(input_dim=self.config.word_embed_dim, output_dim=4 * self.config.lstm_dim, name='mlstm_in')
        self.mlstm_ins.weights_init = IsotropicGaussian(
            std=numpy.sqrt(2) / numpy.sqrt(self.config.word_embed_dim + self.config.lstm_dim))
        self.mlstm_ins.biases_init = Constant(0)
        self.mlstm_ins.initialize()
        self.mlstm = MLSTM(self.config.lstm_time, self.config.lstm_dim, shared=False)
        self.mlstm.weights_init = IsotropicGaussian(
            std=numpy.sqrt(2) / numpy.sqrt(self.config.word_embed_dim + self.config.lstm_dim))
        self.mlstm.biases_init = Constant(0)
        self.mlstm.initialize()

    def _embed(self, sample_num, dim, name):
        embed = LookupTable(sample_num, dim, name= name)
        embed.weights_init = IsotropicGaussian(std=1 / numpy.sqrt(dim))
        embed.initialize()
        return embed

    def _apply_user_word(self, text_vec):
        if tensor.gt(self.user_word.shape[0],0):
            user_word_vec = tensor.dot(self.user_embed.apply(self.user_word), self.user2word.T)
            tensor.set_subtensor(text_vec[self.user_word_right_idx, self.user_word_left_idx], user_word_vec)

    def _apply_hashtag_word(self, text_vec):
        if tensor.gt(self.hashtag_word.shape[0], 0):
            hashtag_word_vec = tensor.dot(self.hashtag_embed.apply(self.hashtag_word), self.hashtag2word.T)
            tensor.set_subtensor(text_vec[self.hashtag_word_right_idx, self.hashtag_word_left_idx], hashtag_word_vec)

    def _apply_url(self, text_vec):
        url_vec = self.char_embed.apply(self.url)
        url_hiddens = self.rnn.apply(inputs=self.rnn_ins.apply(url_vec), mask=self.url_mask)
        url_word_vec = tensor.dot(url_hiddens[-1], self.hashtag2word.T)
        tensor.set_subtensor(text_vec[self.url_right_idx, self.url_left_idx], url_word_vec)


class TUTHM(LUTHM):

    def __init__(self, config, dataset):
        super(TUTHM, self).__init__(config, dataset)

    def _get_cost(self, input_vec, true_hashtag_vec, neg_hashtag_vec):

        rank = min(len(self.dataset.hashtag2index),10)
        preds = tensor.argsort(tensor.dot(input_vec, self.hashtag_embed.W.T),axis = 1)[:,::-1]
        top1_recall = tensor.eq(self.hashtag, preds[:,0]).mean()
        top10_recall = tensor.sum(tensor.eq(preds[:,0:rank], self.hashtag[:,None]), axis=1).mean()

        top1_recall.name = "top1_recall"
        top10_recall.name = "top10_recall"
        self.cg_generator = top1_recall
        self.top1_recall = top1_recall
        self.top10_recall = top10_recall
        self.monitor_vars = [[top1_recall], [top10_recall]]
