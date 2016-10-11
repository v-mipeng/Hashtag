import theano
from theano import tensor
from theano.ifelse import ifelse
import numpy
import codecs

from blocks.bricks import Tanh, Softmax, Linear, MLP, Identity, Rectifier, Bias
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM, SimpleRecurrent
from blocks import theano_expressions
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from util.model import Lookup
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.bricks.conv import *


from util.model import *
from abc import ABCMeta, abstractmethod, abstractproperty


class UTHM(object):
    '''
    Basic User-text-hashtag model, with only user, word and hashtag embedding and without bias
    '''
    def __init__(self, config, dataset, *args, **kwargs):
        '''
        Define User-text-hashtag model
        :param config:
        :param dataset: User-text-hashtag dataset
        '''
        self.config = config
        self.dataset = dataset
        self.rank = min(len(self.dataset.hashtag2index), 10)
        self._build_model()

    def _build_model(self, *args, **kwargs):
        # Define inputs
        self._define_inputs()
        self._build_bricks()
        self._set_OV_value()
        self._get_cost(self._get_classifier_input())

    def _define_inputs(self, *args, **kwargs):
        self.user = tensor.ivector('user')
        self.hashtag = tensor.ivector('hashtag')
        self.text = tensor.imatrix('text')
        self.text_mask = tensor.matrix('text_mask', dtype=theano.config.floatX)

    def _build_bricks(self, *args, **kwargs):
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

    def _set_OV_value(self, *args, **kwargs):
        # Set embedding value of out-of-vocabulary items
        pass

    def _get_classifier_input(self, *args, **kwargs):
        '''
        Get input of classifier
        :param args:
        :param kwargs:
        :return:
        '''
        doc_embed = self._get_doc_embed()
        user_vec = self.user_embed.apply(self.user)
        input_vec = tensor.concatenate([doc_embed, user_vec], axis=1)
        return input_vec

    def _get_doc_embed(self, *args, **kwargs):
        '''
        Get vector representation of tweet
        :param args:
        :param kwargs:
        :return:
        '''
        text_vec = self._get_text_vec()
        return self._encode_text_vec(text_vec)

    def _get_text_vec(self, *args, **kwargs):
        '''
        Get tweet textual matrix
        :param args:
        :param kwargs:
        :return:
        '''
        # Transpose text
        self.text = self.text.dimshuffle(1, 0)
        self.text_mask = self.text_mask.dimshuffle(1, 0)
        # Turn word, user and hashtag into vector representation
        text_vec = self.word_embed.apply(self.text)
        return text_vec

    def _encode_text_vec(self, text_vec, *args, **kwargs):
        '''
        Encode vector representation of textual
        :param text_vec:
        :param args:
        :param kwargs:
        :return:
        '''
        mlstm_hidden, mlstm_cell = self.mlstm.apply(inputs=self.mlstm_ins.apply(text_vec),
                                                    mask=self.text_mask.astype(theano.config.floatX))
        doc_embed = mlstm_hidden[-1]
        return doc_embed

    def _embed(self, sample_num, dim, name, *args, **kwargs):
        embed = LookupTable(sample_num, dim, name=name)
        embed.weights_init = IsotropicGaussian(std=1 / numpy.sqrt(dim))
        embed.initialize()
        return embed

    def _get_pred_dist(self, input_vec, *args, **kwargs):
        return tensor.dot(input_vec, self.hashtag_embed.W.T)

    def _get_cost(self, input_vec, *args, **kwargs):
        self._get_train_cost(input_vec)
        self._get_valid_cost(input_vec)
        self._get_test_cost(input_vec)
        self._get_extra_measure(input_vec)

    def _get_train_cost(self, input_vec, *args, **kwargs):
        preds = self._get_pred_dist(input_vec)
        cost = Softmax().categorical_cross_entropy(self.hashtag, preds).mean()
        #Apply regularization
        cost = self._apply_reg(cost)
        cost.name = 'cost'
        ranks = tensor.argsort(preds, axis=1)[:, ::-1]
        top1_accuracy = tensor.eq(self.hashtag, ranks[:, 0]).mean()
        top10_accuracy = tensor.sum(tensor.eq(ranks[:, 0:self.rank], self.hashtag[:, None]), axis=1).mean()
        top1_accuracy.name = "top1_accuracy"
        top10_accuracy.name = "top10_accuracy"
        #Apply dropout
        cost_drop, top1_accuracy_drop, top10_accuracy_drop = self._apply_dropout([cost, top1_accuracy, top10_accuracy])
        cost_drop.name = cost.name
        top1_accuracy_drop.name = top1_accuracy.name
        top10_accuracy_drop.name = top10_accuracy.name
        self.monitor_train_vars = [[cost_drop], [top1_accuracy_drop], [top10_accuracy_drop]]
        self.cg_generator = cost_drop

    def _get_valid_cost(self, input_vec, *args, **kwargs):
        idx = tensor.ceil(input_vec.shape[0] * self.config.sample_percent_for_test).astype('int32')
        new_input_vec = input_vec[0:idx]
        preds = self._get_pred_dist(new_input_vec)
        ranks = tensor.argsort(preds, axis=1)[:, ::-1]
        top1_accuracy = tensor.eq(self.hashtag[0:idx], ranks[:, 0]).mean()
        top10_accuracy = tensor.sum(tensor.eq(ranks[:, 0:self.rank], self.hashtag[0:idx, None]), axis=1).mean()
        top1_accuracy.name = "top1_accuracy"
        top10_accuracy.name = "top10_accuracy"
        self.monitor_valid_vars = [[top1_accuracy],[top10_accuracy]]
        self.stop_monitor_var = top10_accuracy

    def _get_test_cost(self, input_vec, *args, **kwargs):
        self.monitor_test_vars = self.monitor_valid_vars

    def _get_extra_measure(self, input_vec, *args):
        preds = self._get_pred_dist(input_vec)
        ranks = tensor.argsort(preds, axis=1)[:, ::-1]
        self.monitor_extra_vars = [ranks]

    def _apply_dropout(self, outputs, *args, **kwargs):
        # variables = [self.word_embed.W, self.user_embed.W, self.hashtag_embed.W]
        # cgs = ComputationGraph(outputs)
        # cg_dropouts = apply_dropout(cgs, variables, drop_prob=self.config.dropout_prob, seed=123).outputs
        # return cg_dropouts
        return outputs

    def _apply_reg(self, cost, params = None, *args, **kwargs):
        '''
        Apply regularization (default L2 norm) on parameters (default user, hashtag and word embedding)
        :param params: A list of parameters to which regularization applied
        :return:
        '''
        try:
            if self.config.l2_norm > 0:
                cost = cost + self.config.l2_norm*theano_expressions.l2_norm(
                    tensors = [self.user_embed.W, self.hashtag_embed.W, self.word_embed.W])**2
            else:
                pass
        except Exception:
            pass
        return cost


class EUTHM(UTHM):
    '''
    UTH model with extend information
    '''
    def __init__(self, config, dataset, *args, **kwargs):
        super(EUTHM, self).__init__(config, dataset)

    def _define_inputs(self, *args, **kwargs):
        super(EUTHM, self)._define_inputs()
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

    def _build_bricks(self, *args, **kwargs):
        # Build lookup tables
        super(EUTHM, self)._build_bricks()
        self.user2word = MLP(activations=[Tanh('user2word_tanh')], dims=[self.config.user_embed_dim, self.config.word_embed_dim], name='user2word_mlp')
        self.user2word.weights_init = IsotropicGaussian(std = 1/numpy.sqrt(self.config.word_embed_dim))
        self.user2word.biases_init = Constant(0)
        self.user2word.initialize()
        self.hashtag2word = MLP(activations=[Tanh('hashtag2word_tanh')], dims=[self.config.user_embed_dim+self.config.word_embed_dim, self.config.word_embed_dim], name = 'hashtag2word_mlp')
        self.hashtag2word.weights_init = IsotropicGaussian(std = 1/numpy.sqrt(self.config.word_embed_dim))
        self.hashtag2word.biases_init = Constant(0)
        self.hashtag2word.initialize()
        self.user2word_bias = Bias(dim=1, name='user2word_bias')
        self.user2word_bias.biases_init = Constant(0)
        self.user2word_bias.initialize()
        self.hashtag2word_bias = Bias(dim=1, name='hashtag2word_bias')
        self.hashtag2word_bias.biases_init = Constant(0)
        self.hashtag2word_bias.initialize()
        #Build character embedding
        self.char_embed = self._embed(len(self.dataset.char2index), self.config.char_embed_dim, name='char_embed')
        # Build sparse word encoder
        self.rnn_ins = Linear(input_dim=self.config.char_embed_dim, output_dim=self.config.word_embed_dim, name='rnn_in')
        self.rnn_ins.weights_init = IsotropicGaussian(
            std=numpy.sqrt(2) / numpy.sqrt(self.config.char_embed_dim + self.config.word_embed_dim))
        self.rnn_ins.biases_init = Constant(0)
        self.rnn_ins.initialize()
        self.rnn = SimpleRecurrent(dim=self.config.word_embed_dim, activation=Tanh())
        self.rnn.weights_init = IsotropicGaussian(std=1 / numpy.sqrt(self.config.word_embed_dim))
        self.rnn.initialize()

    def _set_OV_value(self, *args, **kwargs):
        '''Train a <unk> representation'''
        tensor.set_subtensor(self.char_embed.W[self.dataset.char2index['<unk>']],
                             numpy.zeros(self.config.char_embed_dim, dtype=theano.config.floatX))

    def _get_text_vec(self, *args, **kwargs):
        # Transpose text
        self.text = self.text.dimshuffle(1, 0)
        self.text_mask = self.text_mask.dimshuffle(1, 0)
        self.sparse_word = self.sparse_word.dimshuffle(1, 0)
        self.sparse_word_mask = self.sparse_word_mask.dimshuffle(1, 0)
        # Turn word, user and hashtag into vector representation
        text_vec = self.word_embed.apply(self.text)
        # Apply user word, hashtag word and url
        text_vec = self._apply_user_word(text_vec)
        text_vec = self._apply_hashtag_word(text_vec)
        text_vec = self._apply_sparse_word(text_vec)
        return text_vec

    @abstractmethod
    def _apply_user_word(self, text_vec, *args, **kwargs):
        '''
        Replace @a with transformed author vector
        :param text_vec:
        :param args:
        :param kwargs:
        :return:
        '''
        user_word_vec = self.user2word.apply(self.user_embed.apply(self.user_word)) + \
                        self.user2word_bias.parameters[0][0]
        text_vec = tensor.set_subtensor(text_vec[self.user_word_right_idx, self.user_word_left_idx],
                                        text_vec[self.user_word_right_idx, self.user_word_left_idx] * (
                                        1 - self.user_word_sparse_mask[:, None]) +
                                        user_word_vec * self.user_word_sparse_mask[:, None])
        return text_vec

    @abstractmethod
    def _apply_hashtag_word(self, text_vec, *args, **kwargs):
        '''
        Replace #h with transformed hashtag vector
        :param text_vec:
        :param args:
        :param kwargs:
        :return:
        '''
        hashtag_word_vec = self.hashtag2word.apply(self.hashtag_embed.apply(self.hashtag_word)) +\
                           self.hashtag2word_bias.parameters[0][0]
        text_vec = tensor.set_subtensor(text_vec[self.hashtag_word_right_idx, self.hashtag_word_left_idx],
                                        text_vec[self.hashtag_word_right_idx, self.hashtag_word_left_idx] * (
                                        1 - self.hashtag_sparse_mask[:, None])
                                        + hashtag_word_vec * self.hashtag_sparse_mask[:, None])
        return text_vec

    @abstractmethod
    def _apply_sparse_word(self, text_vec, *args, **kwargs):
        '''
        Replace sparse word encoding with character embedding. (maybe lstm)
        :param text_vec:
        :param args:
        :param kwargs:
        :return:
        '''
        sparse_word_vec = self.char_embed.apply(self.sparse_word)
        sparse_word_hiddens = self.rnn.apply(inputs=self.rnn_ins.apply(sparse_word_vec), mask=self.sparse_word_mask)
        tmp = sparse_word_hiddens[-1]
        text_vec = tensor.set_subtensor(text_vec[self.sparse_word_right_idx, self.sparse_word_left_idx],
                                        text_vec[self.sparse_word_right_idx, self.sparse_word_left_idx] * (
                                            1 - self.sparse_word_sparse_mask[:, None]) +
                                        tmp * self.sparse_word_sparse_mask[:, None])
        return text_vec


class NegEUTHM(EUTHM):
    '''
    UTH model with extend information and negtive sampling
    '''
    def __init__(self, config, dataset, *args, **kwargs):
        super(NegEUTHM, self).__init__(config, dataset)

    def _build_model(self, *args, **kwargs):
        # Define inputs
        self._define_inputs()
        self._build_bricks()
        self._set_OV_value()
        true_hashtag_vec = self.hashtag_embed.apply(self.hashtag)
        neg_hashtag_vec = self.hashtag_embed.apply(self.neg_hashtag)
        input_vec = self._get_classifier_input()
        self._get_cost(input_vec, true_hashtag_vec, neg_hashtag_vec)

    def _define_inputs(self, *args, **kwargs):
        super(NegEUTHM, self)._define_inputs()
        self.neg_hashtag = tensor.imatrix('hashtag_negtive_sample')

    def _get_train_cost(self, input_vec, *args, **kwargs):
        def get_cost(enc, th, nh):
            '''
            :param enc: text encoding
            :param th: true hashtag
            :param nh: negtive hashtags
            :return:
            '''
            # return tensor.dot(enc, th), tensor.dot(nh,enc)
            return enc.dot(th), nh.dot(enc)

        true_hashtag_vec = args[0]
        neg_hashtag_vec = args[1]
        (t_pred, n_pred), _ = theano.scan(get_cost,
                                          sequences=[input_vec, true_hashtag_vec, neg_hashtag_vec],
                                          n_steps=self.hashtag.shape[0])
        cost = numpy.log(1. / (1. + numpy.exp(-t_pred))).mean() + numpy.log(1. / (1. + numpy.exp(n_pred))).sum(
            axis=1).mean()
        cost = -cost
        #Apply regularization
        cost = self._apply_reg(cost)
        cost.name = 'cost'
        cost_drop= self._apply_dropout([cost])
        cost_drop.name = cost.name
        self.monitor_train_vars = [[cost_drop]]
        self.cg_generator = cost_drop


class ETHM(EUTHM):
    '''Model with only textual-hashtag information'''
    def __init__(self, config, dataset, *args, **kwargs):
        super(ETHM, self).__init__(config, dataset)

    def _build_model(self, *args, **kwargs):
        # Define inputs
        self._define_inputs()
        self._build_bricks()
        self._set_OV_value()
        # Transpose text
        self.text = self.text.dimshuffle(1, 0)
        self.text_mask = self.text_mask.dimshuffle(1, 0)
        self.sparse_word = self.sparse_word.dimshuffle(1, 0)
        self.sparse_word_mask = self.sparse_word_mask.dimshuffle(1, 0)
        # Turn word, and hashtag into vector representation
        text_vec = self.word_embed.apply(self.text)
        # Apply word and hashtag word and url
        text_vec = self._apply_hashtag_word(text_vec)
        text_vec = self._apply_sparse_word(text_vec)
        # Encode text
        mlstm_hidden, mlstm_cell = self.mlstm.apply(inputs=self.mlstm_ins.apply(text_vec),
                                                    mask=self.text_mask.astype(theano.config.floatX))
        text_encodes = mlstm_hidden[-1]
        input_vec = text_encodes
        self._get_cost(input_vec, None, None)

    def _define_inputs(self, *args, **kwargs):
        self.hashtag = tensor.ivector('hashtag')
        self.text = tensor.imatrix('text')
        self.text_mask = tensor.matrix('text_mask', dtype=theano.config.floatX)
        self.hashtag_word = tensor.ivector('hashtag_word')
        self.hashtag_sparse_mask = tensor.vector('hashtag_word_sparse_mask', dtype=theano.config.floatX)
        self.hashtag_word_left_idx = tensor.ivector('hashtag_word_idx_left_idx')
        self.hashtag_word_right_idx = tensor.ivector('hashtag_word_idx_right_idx')
        self.sparse_word = tensor.imatrix('sparse_word')
        self.sparse_word_sparse_mask = tensor.vector('sparse_word_sparse_mask', dtype=theano.config.floatX)
        self.sparse_word_mask = tensor.matrix('sparse_word_mask', dtype=theano.config.floatX)
        self.sparse_word_left_idx = tensor.ivector('sparse_word_idx_left_idx')
        self.sparse_word_right_idx = tensor.ivector('sparse_word_idx_right_idx')

    def _build_bricks(self, *args, **kwargs):
        # Build lookup tables
        self.word_embed = self._embed(len(self.dataset.word2index), self.config.word_embed_dim, name='word_embed')

        self.hashtag_embed = self._embed(len(self.dataset.hashtag2index),
                                         self.config.lstm_dim,
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
        self.hashtag2word = MLP(activations=[Tanh('hashtag2word_tanh')], dims=[self.config.lstm_dim, self.config.word_embed_dim], name = 'hashtag2word_mlp')
        self.hashtag2word.weights_init = IsotropicGaussian(std = 1/numpy.sqrt(self.config.word_embed_dim))
        self.hashtag2word.biases_init = Constant(0)
        self.hashtag2word.initialize()
        self.hashtag2word_bias = Bias(dim=1, name='hashtag2word_bias')
        self.hashtag2word_bias.biases_init = Constant(0)
        self.hashtag2word_bias.initialize()
        #Build character embedding
        self.char_embed = self._embed(len(self.dataset.char2index), self.config.char_embed_dim, name='char_embed')
        # Build sparse word encoder
        self.rnn_ins = Linear(input_dim=self.config.char_embed_dim, output_dim=self.config.word_embed_dim, name='rnn_in')
        self.rnn_ins.weights_init = IsotropicGaussian(
            std=numpy.sqrt(2) / numpy.sqrt(self.config.char_embed_dim + self.config.word_embed_dim))
        self.rnn_ins.biases_init = Constant(0)
        self.rnn_ins.initialize()
        self.rnn = SimpleRecurrent(dim=self.config.word_embed_dim, activation=Tanh())
        self.rnn.weights_init = IsotropicGaussian(std=1 / numpy.sqrt(self.config.word_embed_dim))
        self.rnn.initialize()

    def _apply_dropout(self, outputs, *args, **kwargs):
        variables = [self.word_embed.W, self.hashtag_embed.W]
        cgs = ComputationGraph(outputs)
        cg_dropouts = apply_dropout(cgs, variables, drop_prob=self.config.dropout_prob, seed=123).outputs
        return cg_dropouts

    def _apply_reg(self, cost, params = None, *args, **kwargs):
        try:
            if self.config.l2_norm > 0:
                cost = cost + self.config.l2_norm*theano_expressions.l2_norm(
                    tensors = [self.hashtag_embed.W, self.word_embed.W])**2

            else:
                pass
        except Exception:
            pass
        return cost


class NegETHM(ETHM):
    '''
    Model with textual-hashtag information and negative sampling
    '''
    def __init__(self, config, dataset, *args, **kwargs):
        super(NegETHM, self).__init__(config, dataset)

    def _build_model(self, *args, **kwargs):
        # Define inputs
        self._define_inputs()
        self._build_bricks()
        self._set_OV_value()
        # Transpose text
        self.text = self.text.dimshuffle(1, 0)
        self.text_mask = self.text_mask.dimshuffle(1, 0)
        self.sparse_word = self.sparse_word.dimshuffle(1, 0)
        self.sparse_word_mask = self.sparse_word_mask.dimshuffle(1, 0)
        # Turn word, and hashtag into vector representation
        text_vec = self.word_embed.apply(self.text)
        true_hashtag_vec = self.hashtag_embed.apply(self.hashtag)
        neg_hashtag_vec = self.hashtag_embed.apply(self.neg_hashtag)
        # Apply word and hashtag word and url
        text_vec = self._apply_hashtag_word(text_vec)
        text_vec = self._apply_sparse_word(text_vec)
        # Encode text
        mlstm_hidden, mlstm_cell = self.mlstm.apply(inputs=self.mlstm_ins.apply(text_vec),
                                                    mask=self.text_mask.astype(theano.config.floatX))
        text_encodes = mlstm_hidden[-1]
        input_vec = text_encodes
        self._get_cost(input_vec, true_hashtag_vec, neg_hashtag_vec)

    def _define_inputs(self, *args, **kwargs):
        super(NegETHM, self)._define_inputs()
        self.neg_hashtag = tensor.imatrix('hashtag_negtive_sample')

    def _get_train_cost(self, input_vec, *args, **kwargs):
        def get_cost(enc, th, nh):
            '''
            :param enc: text encoding
            :param th: true hashtag
            :param nh: negtive hashtags
            :return:
            '''
            # return tensor.dot(enc, th), tensor.dot(nh,enc)
            return enc.dot(th), nh.dot(enc)

        true_hashtag_vec = args[0]
        neg_hashtag_vec = args[1]
        (t_pred, n_pred), _ = theano.scan(get_cost,
                                          sequences=[input_vec, true_hashtag_vec, neg_hashtag_vec],
                                          n_steps=self.hashtag.shape[0])
        cost = numpy.log(1. / (1. + numpy.exp(-t_pred))).mean() + numpy.log(1. / (1. + numpy.exp(n_pred))).sum(
            axis=1).mean()
        cost = -cost
        cost.name = 'cost'
        pred = tensor.concatenate([t_pred[:, None], n_pred], axis=1)
        max_index = pred.argmax(axis=1)
        error_rate = tensor.neq(0, max_index).mean()
        error_rate.name = 'error_rate'
        cost_drop, error_rate_drop = self._apply_dropout([cost, error_rate])
        self.monitor_train_vars = [[cost_drop], [error_rate_drop]]
        self.cg_generator = cost_drop

    def _apply_dropout(self, outputs, *args, **kwargs):
        variables = [self.word_embed.W, self.hashtag_embed.W]
        cgs = ComputationGraph(outputs)
        cg_dropouts = apply_dropout(cgs, variables, drop_prob=self.config.dropout_prob, seed=123).outputs
        return cg_dropouts

    def _apply_reg(self, cost, params = None, *args, **kwargs):
        try:
            if self.config.l2_norm > 0:
                cost = cost + self.config.l2_norm*theano_expressions.l2_norm(
                    tensors = [self.hashtag_embed.W, self.word_embed.W])**2

            else:
                pass
        except Exception:
            pass
        return cost


class AttentionEUTHM(EUTHM):
    '''
    EUTHM with user attention
    '''
    def __init__(self, config, dataset, *args, **kwargs):
        super(EUTHM, self).__init__(config, dataset)

    def _get_doc_embed(self, *args, **kwargs):
        text_vec = self._get_text_vec()
        user_vec = self.user_embed.apply(self.user)
        text_vec = tensor.concatenate([text_vec, user_vec[None,:,:][tensor.zeros(shape=(text_vec.shape[0],), dtype='int32')]], axis=2)
        return self._encode_text_vec(text_vec)

    def _build_bricks(self, *args, **kwargs):
        super(AttentionEUTHM, self)._build_bricks()
        self.mlstm_ins = Linear(input_dim=self.config.word_embed_dim+self.config.user_embed_dim, output_dim=4 * self.config.lstm_dim,
                                name='mlstm_in')
        self.mlstm_ins.weights_init = IsotropicGaussian(
            std=numpy.sqrt(2) / numpy.sqrt(self.config.word_embed_dim+ self.config.user_embed_dim + self.config.lstm_dim))
        self.mlstm_ins.biases_init = Constant(0)
        self.mlstm_ins.initialize()


class NegAttentionEUTHM(NegEUTHM, AttentionEUTHM):
    def __init__(self, config, dataset, *args, **kwargs):
        '''
        Define User-text-hashtag model with negtive sampling
        :param config:
        :param dataset: User-text-hashtag dataset
        '''
        AttentionEUTHM.__init__(self, config, dataset)
        NegEUTHM.__init__(self, config, dataset)

    def _get_doc_embed(self, *args, **kwargs):
        AttentionEUTHM._get_doc_embed(self)

    def _build_bricks(self, *args, **kwargs):
        AttentionEUTHM._build_bricks(self)

    def _get_train_cost(self, input_vec, *args, **kwargs):
        NegEUTHM._get_train_cost(self, input_vec)


class AttentionEUTHM2(AttentionEUTHM):
    def __init__(self, config, dataset, *args, **kwargs):
        '''
        Define User-text-hashtag model with negtive sampling
        :param config:
        :param dataset: User-text-hashtag dataset
        '''
        super(AttentionEUTHM2, self).__init__(config, dataset)

    def _get_doc_embed(self, *args, **kwargs):
        text_vec = self._get_text_vec()
        user_vec = self.user_embed.apply(self.user)
        text_vec = tensor.concatenate([text_vec, user_vec[None,:,:][tensor.zeros(shape=(text_vec.shape[0],), dtype='int32')]], axis=2)
        text_vec = self.word_shift.apply(text_vec) + \
                        self.word_shift_bias.parameters[0][0]
        return self._encode_text_vec(text_vec)

    def _build_bricks(self, *args, **kwargs):
        super(AttentionEUTHM2, self)._build_bricks()
        self.word_shift = MLP(activations=[Tanh('word_shift_tanh')],
                              dims=[self.config.user_embed_dim+self.config.word_embed_dim, self.config.word_embed_dim],
                              name='word_shift_mlp')
        self.word_shift.weights_init = IsotropicGaussian(std = 1/numpy.sqrt(self.config.word_embed_dim+self.config.user_embed_dim))
        self.word_shift.biases_init = Constant(0)
        self.word_shift.initialize()
        self.word_shift_bias = Bias(dim=1, name='word_shift_bias')
        self.word_shift_bias.biases_init = Constant(0)
        self.word_shift_bias.initialize()


class NegAttentionEUTHM2(NegAttentionEUTHM, AttentionEUTHM2):
    def __init__(self, config, dataset, *args, **kwargs):
        '''
        Define User-text-hashtag model with negtive sampling
        :param config:
        :param dataset: User-text-hashtag dataset
        '''
        NegAttentionEUTHM.__init__(self, config, dataset)
        AttentionEUTHM2.__init__(self, config, dataset)

    def _get_doc_embed(self, *args, **kwargs):
        AttentionEUTHM2._get_doc_embed(self)

    def _build_bricks(self, *args, **kwargs):
        AttentionEUTHM2._build_bricks(self)


class ComAttentionEUTHM(NegAttentionEUTHM):
    def __init__(self, config, dataset):
        super(ComAttentionEUTHM, self).__init__(config, dataset)

    def _get_cost(self, input_vec, *args, **kwargs):
        super(ComAttentionEUTHM, self)._get_cost(input_vec, *args, **kwargs)
        self.neg_cost = self.cg_generator
        self.neg_cost.name = "neg_cost"
        self.neg_monitor_train_vars = [[self.neg_cost]]
        self._get_full_cost(input_vec)

    def _get_full_cost(self, input_vec, *args, **kwargs):
        preds = self._get_pred_dist(input_vec)
        cost = Softmax().categorical_cross_entropy(self.hashtag, preds).mean()
        max_index = preds.argmax(axis=1)
        cost.name = 'full_cost'
        ranks = tensor.argsort(preds, axis=1)[:, ::-1]
        top1_accuracy = tensor.eq(self.hashtag, ranks[:, 0]).mean()
        top10_accuracy = tensor.sum(tensor.eq(ranks[:, 0:self.rank], self.hashtag[:, None]), axis=1).mean()
        top1_accuracy.name = "top1_accuracy"
        top10_accuracy.name = "top10_accuracy"
        cost_drop, top1_accuracy_drop, top10_accuracy_drop = self._apply_dropout([cost, top1_accuracy, top10_accuracy])
        cost_drop.name = cost.name
        top1_accuracy_drop.name = top1_accuracy.name
        top10_accuracy_drop.name = top10_accuracy.name
        self.full_monitor_train_vars = [[cost_drop], [top1_accuracy_drop], [top10_accuracy_drop]]
        self.full_cost = cost_drop

