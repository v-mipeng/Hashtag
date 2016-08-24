from blocks.algorithms import BasicMomentum, AdaDelta, RMSProp, Adam, CompositeRule, StepClipping, Momentum, Scale
import os
import datetime
from datetime import date, timedelta

from model.step_rule import WAdaDelta

from model.hashtag_model import MTLM, MTLDM, WLSTMM, BDLSTMM, BDLSTMM2, LUTHM

class BasicConfig:
    '''
    Basic Config
    '''

    # Running mode: debug or run
    mode = "debug"

    #region raw dataset control parameters
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # GPU: "int32"; CPU: "int64"
    int_type = "int32"

    batch_size = 32
    sort_batch_count = 20

    # Step rule
    step_rule = AdaDelta(decay_rate = 0.95, epsilon = 1e-06)

    # Measured by batches, e.g, valid every 1000 batches
    valid_freq = 1000
    save_freq = 1000
    print_freq = 100

    # NLTK data path



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

class UTHC(BasicConfig):
    Model = LUTHM

    model_path = os.path.join(BasicConfig.project_dir, "output/model/UTHC_lstm.pkl")

    # If is directory, read all the files with extension ".txt"
    train_path = os.path.join(BasicConfig.project_dir, "data/unit test/posts.txt")

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
    sparse_word_percent = 0.01

    # date span for traing
    duration = 3

    # file path storing id to index dictionaries
    data_dir = os.path.join(BasicConfig.project_dir, "output/table/")

    user2index_path = os.path.join(data_dir, "user2index.txt")

    word2index_path = os.path.join(data_dir, "word2index.txt")

    word2freq_path = os.path.join(data_dir, "word2freq.txt")

    hashtag2index_path = os.path.join(data_dir, "hashtag2index.txt")

    # region Model control parameters
    user_embed_dim = 100

    word_embed_dim = 100

    hashtag_sample_size = 10

    lstm_dim = 256

    lstm_time = 1

    # endregion