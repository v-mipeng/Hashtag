import theano
from theano import tensor
from theano.ifelse import ifelse
import numpy
import codecs

from blocks.bricks import Tanh, Softmax, Linear, MLP, Identity, Rectifier, Bias
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM, SimpleRecurrent

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from util.model import Lookup
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal

from util.model import MLSTM, MWLSTM


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
        top1_recall, top10_recall = self._get_test_cost(input_vec[0:tensor.ceil(
            input_vec.shape[0] * self.config.sample_percent_for_test).astype(self.config.int_type)])
        self.monitor_valid_vars = [[top1_recall], [top10_recall]]
        self.cg_generator = cost

    def _get_test_cost(self, input_vec):
        preds = tensor.argsort(tensor.dot(input_vec, self.hashtag_embed.W.T), axis=1)[:, ::-1]
        top1_recall = tensor.eq(self.hashtag[0:input_vec.shape[0]], preds[:, 0]).mean()
        top10_recall = tensor.sum(tensor.eq(preds[:, 0:self.rank], self.hashtag[0:input_vec.shape[0], None]),
                                  axis=1).mean()
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
        self.mlstm_ins = Linear(input_dim=self.config.word_embed_dim, output_dim=4 * self.config.lstm_dim,
                                name='mlstm_in')
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
        embed = LookupTable(sample_num, dim, name=name)
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
        text_vec = self._apply_user_word(text_vec)
        text_vec = self._apply_hashtag_word(text_vec)
        text_vec = self._apply_url(text_vec)

        # Encode text
        mlstm_hidden, mlstm_cell = self.mlstm.apply(inputs=self.mlstm_ins.apply(text_vec),
                                                    mask=self.text_mask.astype(theano.config.floatX))
        text_encodes = mlstm_hidden[-1]
        input_vec = tensor.concatenate([text_encodes, user_vec], axis=1)
        self.input_vec = input_vec
        self._get_cost(input_vec, true_hashtag_vec, neg_hashtag_vec)

    def _define_inputs(self):
        self.user = tensor.ivector('user')
        self.hashtag = tensor.ivector('hashtag')
        self.neg_hashtag = tensor.imatrix('hashtag_negtive_sample')
        self.text = tensor.imatrix('text')
        self.text_mask = tensor.matrix('text_mask', dtype=theano.config.floatX)
        self.user_word = tensor.ivector('user_word')
        self.user_word_sparse_mask = tensor.vector('user_word_sparse_mask', dtype=theano.config.floatX)
        self.user_word_left_idx = tensor.ivector('user_word_idx_left_idx')
        self.user_word_right_idx = tensor.ivector('user_word_idx_right_idx')
        self.hashtag_word = tensor.ivector('hashtag_word')
        self.hashtag_sparse_mask = tensor.vector('hashtag_word_sparse_mask', dtype=theano.config.floatX)
        self.hashtag_word_left_idx = tensor.ivector('hashtag_word_idx_left_idx')
        self.hashtag_word_right_idx = tensor.ivector('hashtag_word_idx_right_idx')
        self.url = tensor.imatrix('url')
        self.url_sparse_mask = tensor.vector('url_sparse_mask', dtype=theano.config.floatX)
        self.url_mask = tensor.matrix('url_mask', dtype=theano.config.floatX)
        self.url_left_idx = tensor.ivector('url_idx_left_idx')
        self.url_right_idx = tensor.ivector('url_idx_right_idx')
        self.user2word = theano.shared(
            numpy.random.randn(self.config.word_embed_dim, self.config.user_embed_dim).astype(
                dtype=theano.config.floatX),
            'user2word')
        self.hashtag2word = theano.shared(
            numpy.random.randn(self.config.word_embed_dim,
                               self.config.user_embed_dim + self.config.word_embed_dim).astype(
                dtype=theano.config.floatX),
            'hashtag2word')
        self.url2word = theano.shared(
            numpy.random.randn(self.config.word_embed_dim, self.config.char_rnn_dim).astype(dtype=theano.config.floatX),
            'url2word')

    def _build_bricks(self):
        # Build lookup tables
        self.word_embed = self._embed(len(self.dataset.word2index), self.config.word_embed_dim, name='word_embed')

        self.user_embed = self._embed(len(self.dataset.user2index), self.config.user_embed_dim, name="user_embed")

        self.hashtag_embed = self._embed(len(self.dataset.hashtag2index),
                                         self.config.lstm_dim + self.config.user_embed_dim,
                                         name='hashtag_embed')
        self.char_embed = self._embed(len(self.dataset.char2index), self.config.char_embed_dim, name='char_embed')
        # Build url encoder
        self.rnn_ins = Linear(input_dim=self.config.char_embed_dim, output_dim=self.config.char_rnn_dim, name='rnn_in')
        self.rnn_ins.weights_init = IsotropicGaussian(
            std=numpy.sqrt(2) / numpy.sqrt(self.config.char_embed_dim + self.config.char_rnn_dim))
        self.rnn_ins.biases_init = Constant(0)
        self.rnn_ins.initialize()
        self.rnn = SimpleRecurrent(dim=self.config.char_rnn_dim, activation=Tanh())
        self.rnn.weights_init = IsotropicGaussian(std=1 / numpy.sqrt(self.config.char_rnn_dim))
        self.rnn.initialize()
        # Build text encoder
        self.mlstm_ins = Linear(input_dim=self.config.word_embed_dim, output_dim=4 * self.config.lstm_dim,
                                name='mlstm_in')
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
        embed = LookupTable(sample_num, dim, name=name)
        embed.weights_init = IsotropicGaussian(std=1 / numpy.sqrt(dim))
        embed.initialize()
        return embed

    def _apply_user_word(self, text_vec):
        user_word_vec = tensor.dot(self.user_embed.apply(self.user_word), self.user2word.T)
        text_vec = tensor.set_subtensor(text_vec[self.user_word_right_idx, self.user_word_left_idx],
                                        text_vec[self.user_word_right_idx, self.user_word_left_idx] * (
                                        1 - self.user_word_sparse_mask[:, None]) +
                                        user_word_vec * self.user_word_sparse_mask[:, None])
        return text_vec

    def _apply_hashtag_word(self, text_vec):
        hashtag_word_vec = tensor.dot(self.hashtag_embed.apply(self.hashtag_word), self.hashtag2word.T)
        text_vec = tensor.set_subtensor(text_vec[self.hashtag_word_right_idx, self.hashtag_word_left_idx],
                                        text_vec[self.hashtag_word_right_idx, self.hashtag_word_left_idx] * (
                                        1 - self.hashtag_sparse_mask[:, None])
                                        + hashtag_word_vec * self.hashtag_sparse_mask[:, None])
        return text_vec

    def _apply_url(self, text_vec):
        url_vec = self.char_embed.apply(self.url)
        url_hiddens = self.rnn.apply(inputs=self.rnn_ins.apply(url_vec), mask=self.url_mask)
        url_word_vec = tensor.dot(url_hiddens[-1], self.url2word.T)
        text_vec = tensor.set_subtensor(text_vec[self.url_right_idx, self.url_left_idx],
                                        text_vec[self.url_right_idx, self.url_left_idx] * (
                                        1 - self.url_sparse_mask[:, None]) +
                                        url_word_vec * self.url_sparse_mask[:, None])
        return text_vec

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
        self.cost = cost
        pred = tensor.concatenate([t_pred[:, None], n_pred], axis=1)
        max_index = pred.argmax(axis=1)
        error_rate = tensor.neq(0, max_index).mean()
        cost.name = 'cost'
        error_rate.name = 'error_rate'
        self.monitor_train_vars = [[cost], [error_rate]]
        top1_recall, top10_recall = self._get_test_cost(input_vec[0:tensor.ceil(
            input_vec.shape[0] * self.config.sample_percent_for_test).astype(self.config.int_type)])
        self.monitor_valid_vars = [[top1_recall], [top10_recall]]
        self.cg_generator = cost


class FUTHM(EUTHM):
    def __init__(self, config, dataset):
        super(FUTHM, self).__init__(config, dataset)

    def _build_model(self):
        # Define inputs
        self._define_inputs()
        self._build_bricks()
        # Transpose text
        self.text = self.text.dimshuffle(1, 0)
        self.text_mask = self.text_mask.dimshuffle(1, 0)
        self.sparse_word = self.sparse_word.dimshuffle(1, 0)
        self.sparse_word_mask = self.sparse_word_mask.dimshuffle(1, 0)
        # Turn word, user and hashtag into vector representation
        text_vec = self.word_embed.apply(self.text)
        user_vec = self.user_embed.apply(self.user)
        # Apply user word, hashtag word and url
        text_vec = self._apply_user_word(text_vec)
        text_vec = self._apply_hashtag_word(text_vec)
        text_vec = self._apply_sparse_word(text_vec)

        # Encode text
        mlstm_hidden, mlstm_cell = self.mlstm.apply(inputs=self.mlstm_ins.apply(text_vec),
                                                    mask=self.text_mask.astype(theano.config.floatX))
        text_encodes = mlstm_hidden[-1]
        input_vec = tensor.concatenate([text_encodes, user_vec], axis=1)
        self._get_cost(input_vec, None, None)

    def _define_inputs(self):
        self.user = tensor.ivector('user')
        self.hashtag = tensor.ivector('hashtag')
        self.text = tensor.imatrix('text')
        self.text_mask = tensor.matrix('text_mask', dtype=theano.config.floatX)
        self.user_word = tensor.ivector('user_word')
        self.user_word_sparse_mask = tensor.vector('user_word_sparse_mask', dtype=theano.config.floatX)
        self.user_word_left_idx = tensor.ivector('user_word_idx_left_idx')
        self.user_word_right_idx = tensor.ivector('user_word_idx_right_idx')
        self.hashtag_word = tensor.ivector('hashtag_word')
        self.hashtag_sparse_mask = tensor.vector('hashtag_word_sparse_mask', dtype=theano.config.floatX)
        self.hashtag_word_left_idx = tensor.ivector('hashtag_word_idx_left_idx')
        self.hashtag_word_right_idx = tensor.ivector('hashtag_word_idx_right_idx')
        self.sparse_word = tensor.imatrix('sparse_word')
        self.sparse_word_sparse_mask = tensor.vector('sparse_word_sparse_mask', dtype=theano.config.floatX)
        self.sparse_word_mask = tensor.matrix('sparse_word_mask', dtype=theano.config.floatX)
        self.sparse_word_left_idx = tensor.ivector('sparse_word_idx_left_idx')
        self.sparse_word_right_idx = tensor.ivector('sparse_word_idx_right_idx')
        self.user2word = theano.shared(
            numpy.random.randn(self.config.word_embed_dim, self.config.user_embed_dim).astype(
                dtype=theano.config.floatX),
            'user2word')
        self.hashtag2word = theano.shared(
            numpy.random.randn(self.config.word_embed_dim,
                               self.config.user_embed_dim + self.config.word_embed_dim).astype(
                dtype=theano.config.floatX),
            'hashtag2word')
        self.char2word = theano.shared(
            numpy.random.randn(self.config.word_embed_dim, self.config.char_rnn_dim).astype(dtype=theano.config.floatX),
            'char2word')

    def _build_bricks(self):
        super(FUTHM, self)._build_bricks()
        self.hashtag_bias = Bias(dim = len(self.dataset.hashtag2index),name = 'hashtag_bias')
        self.hashtag_bias.biases_init = Constant(0)
        self.hashtag_bias.initialize()
        # Set embedding of OV characters and users
        W = self.char_embed.W.get_value()
        W[self.dataset.char2index['<unk>']] = 0.
        self.char_embed.W.set_value(W)
        W = self.user_embed.W.get_value()
        W[self.dataset.user2index['<unk>']] = 0.
        self.user_embed.W.set_value(W)

    def _apply_sparse_word(self, text_vec):
        sparse_word_vec = self.char_embed.apply(self.sparse_word)
        sparse_word_hiddens = self.rnn.apply(inputs=self.rnn_ins.apply(sparse_word_vec), mask=self.sparse_word_mask)
        tmp = tensor.dot(sparse_word_hiddens[-1], self.char2word.T)
        text_vec = tensor.set_subtensor(text_vec[self.sparse_word_right_idx, self.sparse_word_left_idx],
                                        text_vec[self.sparse_word_right_idx, self.sparse_word_left_idx] * (
                                        1 - self.sparse_word_sparse_mask[:, None]) +
                                        tmp * self.sparse_word_sparse_mask[:, None])
        return text_vec

    def _get_cost(self, input_vec, true_hashtag_vec, neg_hashtag_vec):
        # Calculate negtive samping cost
        # Reference:Mikolov T, Sutskever I, Chen K, et al.
        #           Distributed Representations of Words and Phrases and their Compositionality[J].
        #           Advances in Neural Information Processing Systems, 2013, 26:3111-3119.
        preds = self.hashtag_bias.apply(tensor.dot(input_vec, self.hashtag_embed.W.T))
        preds = preds - tensor.max(preds, axis=1)[:, None]
        cost = Softmax().categorical_cross_entropy(self.hashtag, preds).mean()
        max_index = preds.argmax(axis=1)
        error_rate = tensor.neq(self.hashtag, max_index).mean()
        cost.name = 'cost'
        error_rate.name = 'error_rate'
        ranks = tensor.argsort(preds, axis=1)[::-1]
        top1_recall = tensor.eq(self.hashtag, ranks[:, 0]).mean()
        top10_recall = tensor.sum(tensor.eq(ranks[:, 0:self.rank], self.hashtag[:, None]), axis=1).mean()
        top1_recall.name = "top1_recall"
        top10_recall.name = "top10_recall"
        self.monitor_train_vars = [[cost], [top1_recall],[top10_recall]]
        # self.monitor_valid_vars =  self._get_test_cost(input_vec)
        self.monitor_valid_vars =  [[top1_recall],[top10_recall]]
        self.cg_generator = cost
        self.stop_monitor_var = top10_recall

    def _get_test_cost(self, input_vec):
        idx = (input_vec.shape[0] * self.config.sample_percent_for_test).astype('int32')
        preds = self.hashtag_bias.apply(tensor.dot(input_vec[0:idx], self.hashtag_embed.W.T))
        ranks = tensor.argsort(preds, axis=1)[::-1]
        top1_recall = tensor.eq(self.hashtag[0:idx], ranks[:, 0]).mean()
        top10_recall = tensor.sum(tensor.eq(ranks[:, 0:self.rank], self.hashtag[0:idx, None]),
                                  axis=1).mean()
        top1_recall.name = "top1_recall"
        top10_recall.name = "top10_recall"
        return [[top1_recall], [top10_recall]]


class TUTHM(LUTHM):
    def __init__(self, config, dataset):
        super(TUTHM, self).__init__(config, dataset)

    def _get_cost(self, input_vec, true_hashtag_vec, neg_hashtag_vec):
        rank = min(len(self.dataset.hashtag2index), 10)
        preds = tensor.argsort(tensor.dot(input_vec, self.hashtag_embed.W.T), axis=1)[:, ::-1]
        top1_recall = tensor.eq(self.hashtag, preds[:, 0]).mean()
        top10_recall = tensor.sum(tensor.eq(preds[:, 0:rank], self.hashtag[:, None]), axis=1).mean()

        top1_recall.name = "top1_recall"
        top10_recall.name = "top10_recall"
        self.top1_recall = top1_recall
        self.top10_recall = top10_recall
        self.monitor_vars = [[top1_recall], [top10_recall]]
        self.cg_generator = top1_recall
