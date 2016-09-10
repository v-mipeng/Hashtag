from blocks.algorithms import BasicMomentum, AdaDelta, RMSProp, Adam, CompositeRule, StepClipping, Momentum, Scale
import os
import datetime
from model import *


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
    step_rule = AdaDelta()

    # Measured by batches, e.g, valid every 1000 batches
    print_freq = 100
    save_freq = 1000
    # Measured by epoch
    valid_freq = 0.5


class UTHC(BasicConfig):

    Model = UTHM

    model_path = os.path.join(BasicConfig.project_dir, "output/model/UTH/UTH.pkl")

    # If is directory, read all the files with extension ".txt"
    train_path = os.path.join(BasicConfig.project_dir, "data/unit test/posts.pkl")

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
    sample_percent_for_test = 0.2

    # region Model control parameters
    user_embed_dim = 50

    word_embed_dim = 100

    hashtag_sample_size = 10

    lstm_dim = 100

    lstm_time = 1

    dropout_prob = 0.2


    # endregion


class EUTHC(UTHC):
    Model = EUTHM

    model_path = os.path.join(BasicConfig.project_dir, 'output/model/EUTH/EUTH.pkl')

    user_name2id_path = os.path.join(BasicConfig.project_dir, "data/tweet/user_name2id.pkl")

    train_path = os.path.join(BasicConfig.project_dir, "data/unit test/posts.pkl")

    # character embedding dimention
    char_embed_dim = 10


class AttentionEUTHC(EUTHC):
    Model = AttentionEUTHM


class TimeLineEUTHC(EUTHC):
    Model = AttentionEUTHM

    model_path = os.path.join(BasicConfig.project_dir, 'output/model/TAEUTH/TAEUTH.pkl')

    valid_freq = 0.5

    sample_percent_for_test = 0.5

    tolerate_time = 10

    dropout_prob = 0.5


class NegTimeLineEUTHC(TimeLineEUTHC):
    Model = NegAttentionEUTHM

    model_path = os.path.join(BasicConfig.project_dir, 'output/model/NTAEUTH/NTAEUTH.pkl')

    neg_sample_size = 10

    valid_freq = 0.5

    sample_percent_for_test = 0.5

    tolerate_time = 20


class TUTHC(UTHC):
    Model = TUTHM


#region Developing
class UGC(BasicConfig):
    '''
    User graph config
    '''

    # directory storing the users' id2index files
    user_id2index_dir = ""

    # directory storing the user graph files
    user_graph_dir = ""

    # negtive sampling size for training user graph.
    # reference: Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality."?Advances in neural information processing systems. 2013.
    user_sample_size = 5

    date = datetime.date(year=2015, month=10, day=23)

    user_hashtag_time_span = 3

    # path to store the model trained on user graph
    user_graph_model_path = ""

    hashtag_embed_dim = 100
#endregion
