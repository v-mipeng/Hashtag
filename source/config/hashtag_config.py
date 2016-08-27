from blocks.algorithms import BasicMomentum, AdaDelta, RMSProp, Adam, CompositeRule, StepClipping, Momentum, Scale
import os
import datetime


from model.hashtag_model import MTLM, MTLDM, WLSTMM, BDLSTMM, BDLSTMM2, LUTHM

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
    step_rule = RMSProp(learning_rate=1.0, decay_rate=0.9,max_scaling=1e5)

    # Measured by batches, e.g, valid every 1000 batches
    valid_freq = 10000
    save_freq = 10000
    print_freq = 1000

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
    # sparse_word_percent = 0.005
    # sparse_hashtag_percent = 0.01
    # sparse_user_percent = 0.005
    sparse_word_percent = 0.001
    sparse_hashtag_percent = 0.005
    sparse_user_percent = 0.005
    # date span for traing
    duration = 1


    # region Model control parameters
    user_embed_dim = 50

    word_embed_dim = 100

    hashtag_sample_size = 10

    lstm_dim = 100

    lstm_time = 1

    # endregion