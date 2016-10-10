from blocks.algorithms import BasicMomentum, AdaDelta, AdaGrad, RMSProp, Adam, CompositeRule, StepClipping, Momentum, Scale
import os
import datetime
from model import *
from dataset import *
from util.entrance import *


class BasicConfig:
    '''
    Basic Config
    '''

    # Running mode: debug or run
    mode = "debug"

    #region raw dataset control parameters
    cur_path = os.path.abspath(__file__)
    project_dir = cur_path[0:cur_path.index('Hashtag')+len('Hashtag')]

    # GPU: "int32"; CPU: "int64"
    int_type = "int32"

    batch_size = 32
    sort_batch_count = 20

    # Step rule
    step_rule = AdaGrad()

    # Measured by batches, e.g, valid every 1000 batches
    print_freq = 100
    save_freq = 1000
    # Measured by epoch
    valid_freq = 0.2


class UTHC(BasicConfig):

    Model = UTHM

    Dataset = UTHD

    model_save_loader = BasicSaveLoadParams

    model_path = os.path.join(BasicConfig.project_dir, "output/model/UTH/UTH.pkl")

    # If is directory, read all the files with extension ".txt"
    train_path = os.path.join(BasicConfig.project_dir, "data/tweet/first_11_days.pkl")
    # train_path = os.path.join(BasicConfig.project_dir, "data/unit test/posts.pkl")

    test_path = os.path.join(BasicConfig.project_dir, "data/test/")

    test_result_path = os.path.join(BasicConfig.project_dir, "output/test/")

    predict_path = os.path.join(BasicConfig.project_dir, "data/predict/")

    predict_result_path = os.path.join(BasicConfig.project_dir, "data/predict/")

    # delimiter of line storing a post information
    delimiter = "\t"

    field_num = 4
    # user, text, hashtag and time index in the line when splitted by delimiter
    user_index = 0

    text_index = 1

    hashtag_index = 2

    date_index = 3

    #sparse word threshold
    #TODO: adjust sparse_word_percent
    sparse_word_percent = 0.05
    sparse_user_percent = 0.005
    sparse_hashtag_freq = 10
    sparse_user_freq = 10
    sparse_word_freq = 10
    # This will reduce hashtag number thus accelerating training and prediction
    sparse_hashtag_percent = 0.005


    # begin date
    begin_date = None

    time_window = 10

    # day offset to do prediction
    T = 10

    # tolerate time for validation
    tolerate_time = 5


    # percent of validation dataset
    valid_percent = 0.2


    # valid on 0.1* size of validation dataset
    sample_percent_for_test = 1.

    # region Model control parameters
    user_embed_dim = 50

    word_embed_dim = 100

    hashtag_sample_size = 10

    lstm_dim = 100

    lstm_time = 1

    dropout_prob = 0.

    # endregion


class EUTHC(UTHC):
    Model = EUTHM

    Dataset = EUTHD

    model_save_loader = ExtendSaveLoadParams

    model_path = os.path.join(BasicConfig.project_dir, 'output/model/REUTH/REUTH.pkl')

    user_name2id_path = os.path.join(BasicConfig.project_dir, "data/tweet/user_name2id.pkl")

    # character embedding dimention
    char_embed_dim = 10


class NegEUTHC(EUTHC):
    Model = NegEUTHM

    Dataset = NegEUTHD

    model_path = os.path.join(BasicConfig.project_dir, 'output/neg after full/REUTH/REUTH.pkl')


class ETHC(EUTHC):
    Model = ETHM

    Dataset = ETHD

    model_save_loader = ETHSaveLoadParams

    model_path = os.path.join(BasicConfig.project_dir, 'output/model/RETH/RETH.pkl')

    # disable dropout
    dropout_prob = 0.


class AttentionEUTHC(EUTHC):
    '''
    concatenate user and word vector
    '''
    Model = AttentionEUTHM

    model_path = os.path.join(BasicConfig.project_dir, 'output/model/RAEUTH/RAEUTH.pkl')


class NegAttentionEUTHC(AttentionEUTHC):
    Model = NegAttentionEUTHM

    model_path = os.path.join(BasicConfig.project_dir, 'output/neg after full/RAEUTH/RAEUTH.pkl')


class AttentionEUTHC2(AttentionEUTHC):
    '''
    Add author attention before lstm
    '''
    Model = AttentionEUTHM2

    model_path = os.path.join(BasicConfig.project_dir, 'output/models/AEUTH2/AEUTH2.pkl')


class ComETHC(ETHC):
    model_path = os.path.join(BasicConfig.project_dir, 'output/sig vs com/RCETH/RCETH.pkl')

    train_path = os.path.join(BasicConfig.project_dir, "data/tweet/first_11_days.pkl")

    T = 10

    neg_epoch = 20


class ComEUTHC(EUTHC):
    model_path = os.path.join(BasicConfig.project_dir, 'output/sig vs com/RCEUTH/RCEUTH.pkl')

    train_path = os.path.join(BasicConfig.project_dir, "data/tweet/first_11_days.pkl")

    T = 10

    neg_epoch = 20


class ComAttentionEUTHC(AttentionEUTHC):
    train_path = os.path.join(BasicConfig.project_dir, "data/tweet/first_11_days.pkl")

    model_path = os.path.join(BasicConfig.project_dir, 'output/sig vs com/RCAEUTH/RCAEUTH.pkl')

    T = 10

    neg_epoch = 20

