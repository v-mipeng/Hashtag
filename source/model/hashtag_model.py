import theano
from theano import tensor
from theano.tensor.raw_random import random_integers
from theano.tensor.shared_randomstreams import RandomStreams
import numpy
import codecs

from blocks.bricks import Tanh, Softmax, Linear, MLP, Identity, Rectifier
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from word_emb import Lookup
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal

from lstm import MLSTM, WLSTM, MWLSTM


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
        rvg = RandomStreams(seed = 223)
        user = tensor.ivector('user')
        hashtag = tensor.ivector('hashtag')
        text = tensor.imatrix('text')
        text_mask = tensor.imatrix('text_mask')

        # sample negtive hashtags
        def neg_sample(h, dis_table, sample_size):
            neg_hashtags = []
            while len(neg_hashtags) < sample_size:
                rvs = rvg.uniform(size = (10,), low = 0.0, high = dataset.hashtag_dis_table[-1], dtype=theano.config.floatX)
                for rv in rvs:
                    index = get_index(dis_table, rv)
                    if index != h:
                        neg_hashtags.append(index)
                    if len(neg_hashtags) == sample_size:
                        break
            return numpy.array(neg_hashtags)

        neg_hashtags = theano.scan(fn = neg_sample,
                                  sequences=[hashtag],
                                  outputs_info= None,
                                  non_sequences=[dataset.hashtag_dis_table, config.hashtag_sample_size],
                                  n_steps= hashtag.shape[0])
        # Transpose text
        text = text.dimshuffle(1,0)
        text_mask = text_mask.dimshuffle(1, 0)

        # Initialize word user and hashtag embedding
        word_embed = LookupTable(len(dataset.word2index), config.word_embed_dim, name='word_embed')
        word_embed.weights_init = IsotropicGaussian(std = 1/numpy.sqrt(config.word_embed_dim))
        word_embed.initialize()

        user_embed = LookupTable(len(dataset.user2index), config.user_embed_dim, name="user_embed")
        user_embed.weights_init = IsotropicGaussian(std = 1/numpy.sqrt(config.user_embed_dim))
        user_embed.initialize()

        hashtag_embed = LookupTable(len(dataset.hashtag2index), config.lstm_dim+config.user_embed_dim, name = 'hashtag_embed')
        hashtag_embed.weights_init = IsotropicGaussian(std = numpy.sqrt(2)/numpy.sqrt(len(dataset.hashtag2index)+config.lstm_dim+config.user_embed_dim))
        hashtag_embed.initialize()

        # Turn word user and hashtag into vector representation
        text_vec = word_embed.apply(text)
        user_vec = user_embed.apply(user)
        true_hashtag_vec = hashtag_embed.apply(hashtag)
        neg_hashtag_vec = hashtag_embed.apply(neg_hashtags)

        # Build and apply multiple-time LSTM
        mlstm_ins = Linear(input_dim=config.word_embed_dim, output_dim=4 * config.lstm_dim, name='mlstm_in')
        mlstm_ins.weights_init = IsotropicGaussian(std=numpy.sqrt(2) / numpy.sqrt(config.word_embed_dim + config.lstm_dim))
        mlstm_ins.biases_init = Constant(0)
        mlstm_ins.initialize()
        mlstm = MLSTM(config.lstm_time, config.lstm_dim, shared=False)
        mlstm.weights_init = IsotropicGaussian(std=numpy.sqrt(2) / numpy.sqrt(config.word_embed_size + config.lstm_dim))
        mlstm.biases_init = Constant(0)
        mlstm.initialize()
        mlstm_hidden, mlstm_cell = mlstm.apply(inputs=mlstm_ins.apply(text_vec),
                                               mask=text_mask.astype(theano.config.floatX))
        text_encodes = mlstm[-1]

        # Calculate negtive samping cost
        # Reference:Mikolov T, Sutskever I, Chen K, et al.
        #           Distributed Representations of Words and Phrases and their Compositionality[J].
        #           Advances in Neural Information Processing Systems, 2013, 26:3111-3119.
        def get_cost(enc, th, nh):
            return enc.dot(th), enc.dot(nh)

        ([t_pred, n_pred], _) = theano.scan(get_cost,
                            sequences=[text_encodes,true_hashtag_vec, neg_hashtag_vec],
                            n_steps=hashtag.shape[0])
        cost = numpy.log(1./(1.+numpy.exp(-t_pred))).mean()+numpy.log(1./(1.+numpy.exp(n_pred))).sum(axis = 1).mean()
        cost = -cost
        pred = numpy.concatenate([t_pred, n_pred], axis = 1)
        max_index = pred.argmax(axis = 1)
        error_rate = tensor.neq(0, max_index).mean()

        # Monitor values
        cost.name = 'cost'
        error_rate.name = 'error_rate'
        self.monitor_vars = [[cost], [error_rate]]
        self.monitor_vars_valid = [[cost], [error_rate]]


def get_index(cum_num, rv):
    # bisect search
    def bisect_search(sorted_na, value):
        '''
        Do bisect search
        :param sorted_na: cumulated sum array
        :param value: random value
        :return: the index that sorted_na[index-1]<=value<sorted_na[index] with defining sorted_na[-1] = -1
        '''
        if len(sorted_na) == 1:
            return 0
        left_index = 0
        right_index = len(sorted_na) - 1

        while right_index - left_index > 1:
            mid_index = (left_index + right_index) / 2
            # in right part
            if value > sorted_na[mid_index]:
                left_index = mid_index
            elif value < sorted_na[mid_index]:
                right_index = mid_index
            else:
                return min(mid_index + 1, right_index)
        return right_index

    if len(cum_num) < 20000:  # This value is obtained by test
        index = numpy.argmin(numpy.abs(cum_num - rv))
        if rv >= cum_num[index]:
           return index + 1
    else:
        return bisect_search(cum_num, rv)

#region Reference Model
class MTLM(object):
    '''
    Multiple Time LSTM Model
    '''
    def __init__(self, config, dataset):
        context = tensor.imatrix('context')                                 # shape: batch_size*sequence_length
        context_mask = tensor.imatrix('context_mask')
        mention_begin = tensor.ivector('mention_begin')
        mention_end = tensor.ivector('mention_end')
        label = tensor.ivector('label')
        bricks = []

        # set time as first dimension
        context = context.dimshuffle(1, 0)
        context_mask = context_mask.dimshuffle(1, 0)

        # Initialize embedding
        embed = Lookup(len(dataset.word2id), config.embed_size, name='word_embed')
        embs = initialize_embed(config, dataset.word2id)
        embed.initialize_with_pretrain(embs)                    # initialize embeding table with pre-traing values
        # Embed contexts
        context_embed = embed.apply(context)

        # Build  and apply multiple-time LSTM
        mlstm_ins = Linear(input_dim=config.embed_size, output_dim=4 * config.lstm_size, name='mlstm_in')
        mlstm_ins.weights_init = IsotropicGaussian(std= numpy.sqrt(2)/numpy.sqrt(config.embed_size+config.lstm_size))
        mlstm_ins.biases_init = Constant(0)
        mlstm_ins.initialize()
        mlstm = MLSTM(config.lstm_time, config.lstm_size, shared = False)
        mlstm.weights_init = IsotropicGaussian(std= numpy.sqrt(2)/numpy.sqrt(config.embed_size+config.lstm_size))
        mlstm.biases_init = Constant(0)
        mlstm.initialize()
        mlstm_hidden, mlstm_cell = mlstm.apply(inputs = mlstm_ins.apply(context_embed), mask = context_mask.astype(theano.config.floatX))
        # Create and apply output MLP
        out_mlp = MLP(dims = [config.lstm_size*2] + [config.n_labels],
                          activations = [Identity()],
                          name='out_mlp')
        out_mlp.weights_init = IsotropicGaussian(std = numpy.sqrt(2)/numpy.sqrt(config.lstm_size+config.n_labels))
        out_mlp.biases_init = Constant(0)
        out_mlp.initialize()
        out_mlp_inputs = tensor.concatenate([mlstm_hidden[mention_end, tensor.arange(context.shape[1]), :],
                                            mlstm_hidden[mention_begin, tensor.arange(context.shape[1]), :]],axis=1)
        probs = out_mlp.apply(out_mlp_inputs)
        self.get_output(label, probs)

    def get_output(self, label, probs):
        # Calculate prediction, cost and error rate
        prob_max = probs.max(axis = 1)
        probs = probs - prob_max[:,None]
        pred = probs.argmax(axis=1)
        cost = Softmax().categorical_cross_entropy(label, probs).mean()
        error_rate = tensor.neq(label, pred).mean()

        # Other stuff
        cost.name = 'cost'
        error_rate.name = 'error_rate'

        self.sgd_cost = cost
        self.monitor_vars = [[cost], [error_rate]]
        self.monitor_vars_valid = [[cost], [error_rate]]
        self.pred = pred
        self.pred_prob = Softmax().apply(probs).max(axis = 1)
        self.error_rate = error_rate

class MTLDM(MTLM):
    '''
    Multiple Time LSTM with DBpedia Model
    '''
    def __init__(self, config, dataset):
        context = tensor.imatrix('context')                                 # shape: batch_size*sequence_length
        context_mask = tensor.imatrix('context_mask')
        type = tensor.imatrix('type')
        type_weight = tensor.matrix('type_weight', dtype=theano.config.floatX)
        mention_begin = tensor.ivector('mention_begin')
        mention_end = tensor.ivector('mention_end')
        label = tensor.ivector('label')


        # set time as first dimension
        context = context.dimshuffle(1, 0)
        context_mask = context_mask.dimshuffle(1, 0)

        # Embed contexts
        embed = Lookup(len(dataset.word2id), config.embed_size, name='word_embed')
        embs = initialize_embed(config, dataset.word2id)
        embed.initialize_with_pretrain(embs)                    # initialize embeding table with pre-traing values

        # Embed types
        type_lookup = LookupTable(len(dataset.type2id), config.type_embed_size, name="type_embed")
        type_lookup.weights_init = IsotropicGaussian(std= 1/numpy.sqrt(config.type_embed_size))
        type_lookup.initialize()
        type_embed = (type_lookup.apply(type)*type_weight[:,:,None]).sum(axis=1)

        # Apply embedding
        context_embed = embed.apply(context)

        mlstm_ins = Linear(input_dim=config.embed_size, output_dim=4 * config.lstm_size, name='mlstm_in')
        mlstm_ins.weights_init = IsotropicGaussian(std= numpy.sqrt(2)/numpy.sqrt(config.embed_size+config.lstm_size))
        mlstm_ins.biases_init = Constant(0)
        mlstm_ins.initialize()
        mlstm = MLSTM(config.lstm_time, config.lstm_size, shared = False)
        mlstm.weights_init = IsotropicGaussian(std= numpy.sqrt(2)/numpy.sqrt(config.embed_size+config.lstm_size))
        mlstm.biases_init = Constant(0)
        mlstm.initialize()
        mlstm_hidden, mlstm_cell = mlstm.apply(inputs = mlstm_ins.apply(context_embed), mask = context_mask.astype(theano.config.floatX))
        # Create and apply output MLP
        out_mlp = MLP(dims = [config.lstm_size*2+config.type_embed_size] + [config.n_labels],
                          activations = [Identity()],
                          name='out_mlp')
        out_mlp.weights_init = IsotropicGaussian(std = numpy.sqrt(2)/numpy.sqrt(config.lstm_size*2+config.n_labels))
        out_mlp.biases_init = Constant(0)
        out_mlp.initialize()
        out_mlp_inputs = tensor.concatenate([lstm_hidden[mention_end, tensor.arange(context.shape[1]), :],
                                            lstm_hidden[mention_begin, tensor.arange(context.shape[1]), :]],axis=1)
        out_mlp_inputs = tensor.concatenate([out_mlp_inputs, type_embed],axis=1)
        probs = out_mlp.apply(out_mlp_inputs)
        # Calculate prediction, cost and error rate
        self.get_output(label, probs)

class WLSTMM(MTLM):
    '''
    Weighted Single LSTM Model
    '''
    def __init__(self, config, dataset):
        context = tensor.imatrix('context')                                 # shape: batch_size*sequence_length
        mention_begin = tensor.ivector('mention_begin')
        mention_end = tensor.ivector('mention_end')
        context_mask = tensor.imatrix('context_mask')
        distance = tensor.imatrix('distance')
        label = tensor.ivector('label')
        delta = theano.shared((10.0*numpy.sqrt(2.0)).astype(theano.config.floatX), name = 'delta')
        self.delta = delta
        weights = tensor.exp(-distance*distance/(delta*delta))
        self.weights = weights
        # set time as first dimension
        context = context.dimshuffle(1, 0)
        context_mask = context_mask.dimshuffle(1, 0)
        weights = weights.dimshuffle(1,0)

        # Initialize embedding
        embed = Lookup(len(dataset.word2id), config.embed_size, name='word_embed')
        embs = initialize_embed(config, dataset.word2id)
        embed.initialize_with_pretrain(embs)                    # initialize embeding table with pre-traing values
        # Embed contexts
        context_embed = embed.apply(context)

        lstm_ins = Linear(input_dim=config.embed_size, output_dim=4 * config.lstm_size, name='lstm_in')
        lstm_ins.weights_init = IsotropicGaussian(std= numpy.sqrt(2)/numpy.sqrt(config.embed_size+config.lstm_size*4))
        lstm_ins.biases_init = Constant(0)
        lstm_ins.initialize()
        mwlst = MWLSTM(times = 2, shared = False, dim = config.lstm_size)
        mwlst.weights_init = IsotropicGaussian(std= 1/numpy.sqrt(config.lstm_size))
        mwlst.biases_init = Constant(0)
        mwlst.initialize()
        mwlstm_hidden, _ = mwlst.apply(inputs = lstm_ins.apply(context_embed), weights = weights, mask=context_mask.astype(theano.config.floatX))

        # Create and apply output MLP
        out_mlp = MLP(dims = [config.lstm_size*2] + [config.n_labels],
                          activations = [Identity()],
                          name='out_mlp')
        out_mlp.weights_init = IsotropicGaussian(std = numpy.sqrt(2)/numpy.sqrt(config.lstm_size*2+config.n_labels))
        out_mlp.biases_init = Constant(0)
        out_mlp.initialize()
        out_mlp_inputs = tensor.concatenate([mwlstm_hidden[mention_end, tensor.arange(context.shape[1]), :],
                                            mwlstm_hidden[mention_begin, tensor.arange(context.shape[1]), :]],axis=1)
        probs = out_mlp.apply(out_mlp_inputs)
        # Calculate prediction, cost and error rate
        self.get_output(label, probs)

class BDLSTMM(MTLM):
    '''
    Bi-direction LSTM Model: order_lstm(mention_end)||reverse_lstm(mention_begin)
    '''
    def __init__(self, config, dataset):
        order_context = tensor.imatrix('order_context')                                 # shape: batch_size*sequence_length
        order_context_mask = tensor.imatrix('order_context_mask')
        reverse_context = tensor.imatrix('reverse_context')                                 # shape: batch_size*sequence_length
        reverse_context_mask = tensor.imatrix('reverse_context_mask')
        label = tensor.ivector('label')
        bricks = []

        # set time as first dimension
        order_context = order_context.dimshuffle(1, 0)
        order_context_mask = order_context_mask.dimshuffle(1, 0)
        reverse_context = reverse_context.dimshuffle(1, 0)
        reverse_context_mask = reverse_context_mask.dimshuffle(1, 0)

        # Initialize embedding
        embed = Lookup(len(dataset.word2id), config.embed_size, name='word_embed')
        embs = initialize_embed(config, dataset.word2id)
        embed.initialize_with_pretrain(embs)                    # initialize embeding table with pre-traing values
        # Embed contexts
        order_context_embed = embed.apply(order_context)
        reverse_context_embed = embed.apply(reverse_context)

        inputs = [order_context_embed, reverse_context_embed]
        masks = [order_context_mask, reverse_context_mask]
        names = ["order", "reverse"]
        hiddens = []
        for i in range(len(inputs)):
            lstm_ins = Linear(input_dim=config.embed_size, output_dim=4 * config.lstm_size, name='%s_lstm_in' % names[i])
            lstm_ins.weights_init = IsotropicGaussian(std= numpy.sqrt(2)/numpy.sqrt(config.embed_size+config.lstm_size))
            lstm_ins.biases_init = Constant(0)
            lstm_ins.initialize()
            lstm = LSTM(dim=config.lstm_size, activation=Tanh(), name='%s_lstm' % names[i])
            lstm.weights_init = IsotropicGaussian(std= 1/numpy.sqrt(config.lstm_size))
            lstm.biases_init = Constant(0)
            lstm.initialize()
            hidden, _ = lstm.apply(inputs = lstm_ins.apply(inputs[i]), mask=masks[i].astype(theano.config.floatX))
            hiddens.append(hidden)
        # Create and apply output MLP
        out_mlp = MLP(dims = [config.lstm_size*2] + [config.n_labels],
                          activations = [Identity()],
                          name='out_mlp')
        out_mlp.weights_init = IsotropicGaussian(std = numpy.sqrt(2)/numpy.sqrt(config.lstm_size*2+config.n_labels))
        out_mlp.biases_init = Constant(0)
        out_mlp.initialize()
        out_mlp_inputs = tensor.concatenate([hiddens[0][-1,:,:], hiddens[1][-1,:,:]],axis=1)
        self.mention_hidden = out_mlp_inputs
        probs = out_mlp.apply(out_mlp_inputs)
        self.get_output(label, probs)

class BDLSTMM2(MTLM):
    '''
    Bi-direction LSTM Model: order_lstm(mention_begin-1)||max_pooling(mention)||reverse_lstm(mention_end+1)
    '''
    def __init__(self, config, dataset):
        order_context = tensor.imatrix('order_context')                                 # shape: batch_size*sequence_length
        order_context_mask = tensor.imatrix('order_context_mask')
        reverse_context = tensor.imatrix('reverse_context')                                 # shape: batch_size*sequence_length
        reverse_context_mask = tensor.imatrix('reverse_context_mask')
        mention = tensor.imatrix('mention')
        mention_mask = tensor.imatrix('mention_mask')
        label = tensor.ivector('label')
        bricks = []

        # set time as first dimension
        order_context = order_context.dimshuffle(1, 0)
        order_context_mask = order_context_mask.dimshuffle(1, 0)
        reverse_context = reverse_context.dimshuffle(1, 0)
        reverse_context_mask = reverse_context_mask.dimshuffle(1, 0)
        # Initialize embedding
        embed = Lookup(len(dataset.word2id), config.embed_size, name='word_embed')
        embs = initialize_embed(config, dataset.word2id)
        embed.initialize_with_pretrain(embs)                    # initialize embeding table with pre-traing values
        # Embed contexts
        order_context_embed = embed.apply(order_context)
        reverse_context_embed = embed.apply(reverse_context)
        mention_embed = embed.apply(mention)*mention_mask[:,:,None]
        
        mention_pooled = self.max_pool(mention_embed)


        inputs = [order_context_embed, reverse_context_embed]
        masks = [order_context_mask, reverse_context_mask]
        names = ["order", "reverse"]
        hiddens = []
        for i in range(len(inputs)):
            lstm_ins = Linear(input_dim=config.embed_size, output_dim=4 * config.lstm_size, name='%s_lstm_in' % names[i])
            lstm_ins.weights_init = IsotropicGaussian(std= numpy.sqrt(2)/numpy.sqrt(config.embed_size+config.lstm_size))
            lstm_ins.biases_init = Constant(0)
            lstm_ins.initialize()
            lstm = LSTM(dim=config.lstm_size, activation=Tanh(), name='%s_lstm' % names[i])
            lstm.weights_init = IsotropicGaussian(std= 1/numpy.sqrt(config.lstm_size))
            lstm.biases_init = Constant(0)
            lstm.initialize()
            hidden, _ = lstm.apply(inputs = lstm_ins.apply(inputs[i]), mask=masks[i].astype(theano.config.floatX))
            hiddens.append(hidden)
        # Create and apply output MLP
        out_mlp = MLP(dims = [config.lstm_size*2+config.embed_size] + [config.n_labels],
                          activations = [Identity()],
                          name='out_mlp')
        out_mlp.weights_init = IsotropicGaussian(std = numpy.sqrt(2)/numpy.sqrt(config.lstm_size*2+config.embed_size+config.n_labels))
        out_mlp.biases_init = Constant(0)
        out_mlp.initialize()
        out_mlp_inputs = tensor.concatenate([hiddens[0][-1,:,:], hiddens[1][-1,:,:], mention_pooled],axis=1)
        self.mention_hidden = out_mlp_inputs
        probs = out_mlp.apply(out_mlp_inputs)
        # Calculate prediction, cost and error rate
        self.get_output(label, probs)

    def max_pool(self, embed):
        return tensor.max(tensor.abs_(embed), axis = 1)

#endregion


def initialize_embed(config, word2id):
    if config.with_pre_train and not config.debug:
        path = config.embed_path
    else:
        path = config.embed_backup_path
    embs = []
    with codecs.open(path,'r','UTF-8') as f:
        for line in f:
             for line in f:
                word = line.split(' ', 1)[0]
                if word in word2id:
                    array = line.split(' ')
                    if len(array) != config.embed_size + 1:
                        return None
                    vector = []
                    for i in range(1,len(array)):
                        vector.append(float(array[i]))
                    embs += [(word2id[array[0]], numpy.asarray(vector, theano.config.floatX))]
    return embs
